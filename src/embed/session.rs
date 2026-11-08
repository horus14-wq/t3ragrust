//! ONNX embedding session: tokenize -> forward -> last-token pooling -> L2.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context, Result};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use tracing::{info, warn};

use super::{ModelSpec, MAX_EMBED_TOKENS};
use crate::cli::Ep;

/// Map the `--ep` choice to a list of execution providers. `auto` registers
/// every compiled-in EP with `error_on_failure(false)` so missing hardware
/// just skips to CPU; explicit choices fail loudly when unavailable.
pub(crate) fn providers_for(ep: Ep) -> Result<Vec<ExecutionProviderDispatch>> {
    fn not_built(name: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "EP '{name}' not compiled into this binary — rebuild with the matching cargo feature: --features ep-{name}"
        )
    }
    // Mutability is only exercised when an ep-* feature is compiled in.
    #[allow(unused_mut)]
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
    // ort rc.15: error_on_failure is 0-arity (t3-x9y ep-* builds).
    #[cfg(any(feature = "ep-cuda", feature = "ep-openvino", feature = "ep-webgpu"))]
    fn maybe_strict(d: ExecutionProviderDispatch, strict: bool) -> ExecutionProviderDispatch {
        if strict {
            d.error_on_failure()
        } else {
            d
        }
    }
    #[allow(unused_mut)]
    let mut push_all = |#[allow(unused_variables)] strict: bool| {
        #[cfg(feature = "ep-cuda")]
        {
            eps.push(maybe_strict(ort::ep::CUDA::default().build(), strict));
            eps.push(maybe_strict(ort::ep::TensorRT::default().build(), strict));
        }
        #[cfg(feature = "ep-openvino")]
        eps.push(maybe_strict(ort::ep::OpenVINO::default().build(), strict));
        // DirectML's builder has no error_on_failure(strict) variant —
        // registration failures are always non-fatal (falls back to CPU).
        #[cfg(feature = "ep-directml")]
        eps.push(ort::ep::DirectML::default().build().error_on_failure());
        #[cfg(feature = "ep-webgpu")]
        eps.push(maybe_strict(ort::ep::WebGPU::default().build(), strict));
    };
    match ep {
        Ep::Auto => push_all(false),
        Ep::Cpu => {}
        Ep::Cuda => {
            #[cfg(feature = "ep-cuda")]
            eps.push(ort::ep::CUDA::default().build().error_on_failure());
            #[cfg(not(feature = "ep-cuda"))]
            return Err(not_built("cuda"));
        }
        Ep::Tensorrt => {
            #[cfg(feature = "ep-cuda")]
            eps.push(ort::ep::TensorRT::default().build().error_on_failure());
            #[cfg(not(feature = "ep-cuda"))]
            return Err(not_built("cuda"));
        }
        Ep::Openvino => {
            #[cfg(feature = "ep-openvino")]
            eps.push(ort::ep::OpenVINO::default().build().error_on_failure());
            #[cfg(not(feature = "ep-openvino"))]
            return Err(not_built("openvino"));
        }
        Ep::Coreml => {
            #[cfg(feature = "ep-coreml")]
            eps.push(ort::ep::CoreML::default().build().error_on_failure());
            #[cfg(not(feature = "ep-coreml"))]
            return Err(not_built("coreml"));
        }
        Ep::Directml => {
            #[cfg(feature = "ep-directml")]
            // No strict variant for DirectML — see the push_all comment.
            eps.push(ort::ep::DirectML::default().build().error_on_failure());
            #[cfg(not(feature = "ep-directml"))]
            return Err(not_built("directml"));
        }
        Ep::Webgpu => {
            #[cfg(feature = "ep-webgpu")]
            eps.push(ort::ep::WebGPU::default().build().error_on_failure());
            #[cfg(not(feature = "ep-webgpu"))]
            return Err(not_built("webgpu"));
        }
    }
    Ok(eps)
}

/// ORT allows exactly one `Environment` per process — share it across pool
/// sessions via a lazily-initialized singleton.
pub(crate) fn ort_env() -> Result<Arc<ort::environment::Environment>> {
    static ENV: OnceLock<Arc<ort::environment::Environment>> = OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e.clone());
    }
    let env = Arc::new(
        ort::init()
            .with_name("t3rag")
            .build()
            .context("init ort environment")?,
    );
    let _ = ENV.set(env.clone()); // benign race: first writer wins
    Ok(env)
}

/// Qwen3 pad token id ("endoftext"), used for right-padding batches.
const PAD_ID: u32 = 151643;

/// A loaded ONNX embedding session: tokenizer + ORT session +
/// declared-input feeding (registry instructions, generic extra
/// inputs). Not `Sync` — pooled behind [`crate::embed::EmbedderPool`].
pub struct Embedder {
    /// ORT session; `run` takes `&mut self`, so concurrent callers must
    /// serialize (the pool's per-session `Mutex` does this).
    session: Session,
    /// HF `tokenizer.json` with truncation/padding configured.
    tokenizer: Tokenizer,
    /// Embedding width, discovered by a probe forward pass at load.
    dims: usize,
    /// Query-side instruction prefix ("" when the model isn't instruct-tuned).
    instruction: String,
    /// Passage-side prefix (e5-family `passage: `; empty for most models).
    passage_instruction: String,
    /// Input names the export declares (t3-css.1.2) — position_ids/
    /// attention_mask/token_type_ids are fed only when present; non-Qwen
    /// exports that omit them no longer crash.
    declared_inputs: std::collections::HashSet<String>,
    /// `past_key_values.*` input names + declared shapes on merged KV-cache
    /// ONNX exports; fed zero-length cache tensors at run time.
    kv_inputs: Vec<(String, Vec<i64>)>,
}

impl Embedder {
    /// Download (or reuse HF cache) and load the model: fetch ONNX +
    /// tokenizer, build the ORT session with the requested EP, configure
    /// the tokenizer, and probe dims with a warmup forward pass.
    ///
    /// # Errors
    /// Files cannot be fetched, the session/tokenizer fails to build, or
    /// the probed dims contradict `spec.dims`.
    pub fn load(spec: &ModelSpec, intra_threads: usize, ep: Ep) -> Result<Self> {
        let (onnx_path, tokenizer_path) = fetch_files(spec)?;
        let eps = providers_for(ep)?;
        let env = ort_env()?;
        // ort builder errors aren't Send+Sync — render via Debug, not `?`.
        let berr = |e: ort::Error| anyhow::anyhow!("session builder: {e:?}");
        let builder = Session::builder(env.as_ref()).map_err(berr)?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {e:?}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("intra threads: {e:?}"))?
            // Sequential transformer graphs get nothing from an inter-op
            // thread pool — pin it to 1 so intra-op threads do the work.
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("inter threads: {e:?}"))?
            // Enable the memory-pattern allocator (ORT reuses buffers across
            // runs instead of reallocating every forward).
            .with_memory_pattern(true)
            .map_err(|e| anyhow::anyhow!("memory pattern: {e:?}"))?
            .with_execution_providers(&eps)
            .map_err(|e| anyhow::anyhow!("execution providers: {e:?}"))?;
        let session = builder
            .commit_from_file(&onnx_path)
            .with_context(|| format!("load ONNX model {}", onnx_path.display()))?;
        let declared_inputs: std::collections::HashSet<String> = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let kv_inputs: Vec<(String, Vec<i64>)> = session
            .inputs()
            .iter()
            .filter_map(|o| {
                let name = o.name().to_string();
                if !name.starts_with("past_key_values.") {
                    return None;
                }
                match o.dtype() {
                    ValueType::Tensor { ty, shape, .. } => {
                        if *ty != TensorElementType::Float32 {
                            warn!(%name, ?ty, "kv input not f32; empty fill assumes f32");
                        }
                        Some((name, shape.iter().copied().collect()))
                    }
                    _ => None,
                }
            })
            .collect();
        if !kv_inputs.is_empty() {
            info!(
                count = kv_inputs.len(),
                "merged KV export; feeding empty past_key_values"
            );
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_EMBED_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("set truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: PAD_ID,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));

        let mut emb = Self {
            session,
            tokenizer,
            dims: 0,
            instruction: spec.instruction.clone(),
            passage_instruction: spec.passage_instruction.clone(),
            declared_inputs,
            kv_inputs,
        };
        // Probe dims with a trivial forward pass (also serves as warmup).
        let probe = emb.forward_batch(&["t3rag warmup"])?;
        emb.dims = probe[0].len();
        if spec.dims != 0 && spec.dims != emb.dims {
            bail!(
                "model {} produced dims {} but spec expected {}",
                spec.repo_id,
                emb.dims,
                spec.dims
            );
        }
        Ok(emb)
    }

    /// Embedding dimension of the loaded model (0 until `load` probes it).
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Embed one retrieval query (instruction prefix applied).
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let input = format!("{}{}", self.instruction, query);
        Ok(self.forward_batch(&[input.as_str()])?.remove(0))
    }

    /// Embed a batch of passages — the spec's passage prefix applies
    /// (e5-family `passage: `; empty for most models). Caller batches.
    /// `&[&str]` (t3-css.2.9): the index path used to clone every doc's
    /// text into a `Vec<String>` — borrowed slices cost nothing.
    pub fn embed_passages(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.passage_instruction.is_empty() {
            return self.forward_batch(texts);
        }
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.passage_instruction))
            .collect();
        self.forward_batch(&prefixed.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// Tokenize+pad a batch, run the ONNX forward pass, last-token-pool each
    /// sequence, L2-normalize. Returns one unit vector per input.
    fn forward_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize batch: {e}"))?;
        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&t| t as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&t| t as i64));
        }
        let mut pos = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            // Position ids = running index per sequence (right-padded batch).
            pos.extend((0..enc.get_ids().len() as i64).collect::<Vec<_>>());
        }
        let shape = vec![batch, seq];
        let ids_t = Tensor::from_array((shape.clone(), ids)).context("ids tensor")?;
        let mask_t = Tensor::from_array((shape.clone(), mask)).context("mask tensor")?;
        let pos_t = Tensor::from_array((shape, pos)).context("position_ids tensor")?;
        // Feed only what the export declares (t3-css.1.2): non-Qwen models
        // omit position_ids/attention_mask; token_type_ids gets zeros.
        let mut feeds: Vec<(String, SessionInputValue)> = vec![("input_ids".into(), ids_t.into())];
        if self.declared_inputs.contains("attention_mask") {
            feeds.push(("attention_mask".into(), mask_t.into()));
        }
        if self.declared_inputs.contains("position_ids") {
            feeds.push(("position_ids".into(), pos_t.into()));
        }
        if self.declared_inputs.contains("token_type_ids") {
            let zeros = vec![0i64; batch * seq];
            let tt =
                Tensor::from_array((vec![batch, seq], zeros)).context("token_type_ids tensor")?;
            feeds.push(("token_type_ids".into(), tt.into()));
        }
        for (name, dims) in &self.kv_inputs {
            // Keep declared static dims; dynamic dims resolve positionally:
            // dim0 -> batch, everything else -> 0 (empty cache).
            let d: Vec<usize> = dims
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if v >= 0 {
                        v as usize
                    } else if i == 0 {
                        batch
                    } else {
                        0
                    }
                })
                .collect();
            let t = Tensor::<f32>::from_array((d, Vec::new()))
                .with_context(|| format!("empty kv tensor for {name}"))?;
            feeds.push((name.clone(), t.into()));
        }
        let outputs = self.session.run(feeds).context("onnx forward")?;
        let hidden = &outputs["last_hidden_state"];
        let (shape_ref, data) = hidden
            .try_extract_tensor::<f32>()
            .context("extract last_hidden_state")?;
        let dims = *shape_ref
            .last()
            .context("last_hidden_state has no trailing dim")? as usize;
        let total = data.len() / dims;
        let expected = batch * seq;
        if total != expected {
            bail!("hidden states shape mismatch: {total} rows vs batch*seq {expected}");
        }

        let mut out = Vec::with_capacity(batch);
        for (i, enc) in encodings.iter().enumerate() {
            // Last non-pad position = last-token pooling index.
            let last = enc
                .get_attention_mask()
                .iter()
                .rposition(|&m| m == 1)
                .unwrap_or(0);
            let start = (i * seq + last) * dims;
            let mut v = data[start..start + dims].to_vec();
            l2_normalize(&mut v);
            out.push(v);
        }
        Ok(out)
    }
}

/// In-place L2 normalization; a zero vector is left untouched (norm 0
/// would produce NaNs — callers rely on non-degenerate embeddings).
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Fetch the ONNX model + tokenizer from HF, trying candidate export
/// names in order (onnx/ subdir first, root-level fallback), plus the
/// `<file>_data` external-weights sibling when the export needs one
/// (t3-css.1.1).
///
/// # Errors
/// No usable ONNX export exists, or the tokenizer cannot be fetched.
pub(crate) fn fetch_files(spec: &ModelSpec) -> Result<(PathBuf, PathBuf)> {
    // hf-hub 1.x HFClient (t3-css.4.8): Xet dedup transfers when the repo
    // is Xet-enabled; `blocking` sync surface keeps callers unchanged.
    let client = hf_hub::HFClientSync::new().context("init hf-hub client")?;
    let (owner, name) = spec
        .repo_id
        .split_once('/')
        .context("model repo_id must be owner/name")?;
    let repo = client.model(owner, name);

    // T3RAG_ONNX_FILE pins the export to probe first — e.g.
    // `onnx/model_fp16.onnx` on CUDA, where int8 ops fall back to CPU.
    let mut files = spec.onnx_files.clone();
    if let Ok(pinned) = std::env::var("T3RAG_ONNX_FILE") {
        files.retain(|f| *f != pinned);
        files.insert(0, pinned);
    }
    let mut onnx = None;
    let mut errs = Vec::new();
    for file in &files {
        match repo.download_file().filename(file.clone()).send() {
            Ok(p) => {
                onnx = Some((p, file.clone()));
                break;
            }
            Err(e) => errs.push(format!("{file}: {e}")),
        }
    }
    let (onnx, onnx_file) = onnx.with_context(|| {
        format!(
            "no usable ONNX export in {} (tried: {})",
            spec.repo_id,
            errs.join("; ")
        )
    })?;
    // External-data exports (fp32/fp16) keep weights in a `<file>_data`
    // sibling that ORT resolves adjacent to the .onnx — fetch it when the
    // repo has one (t3-css.1.1; quantized variants are self-contained).
    if let Err(e) = repo
        .download_file()
        .filename(format!("{onnx_file}_data"))
        .send()
    {
        tracing::debug!(%e, file = %onnx_file, "no external _data sibling (self-contained export)");
    }
    let tokenizer = repo
        .download_file()
        .filename(spec.tokenizer_file.clone())
        .send()
        .with_context(|| format!("fetch {} from {}", spec.tokenizer_file, spec.repo_id))?;
    Ok((onnx, tokenizer))
}
