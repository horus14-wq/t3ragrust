//! ONNX embedding session: tokenize -> forward -> last-token pooling -> L2.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use super::{ModelSpec, MAX_EMBED_TOKENS};

/// Qwen3 pad token id ("endoftext"), used for right-padding batches.
const PAD_ID: u32 = 151643;

/// One loaded embedding model: ONNX session + tokenizer. Not `Sync`-safe to
/// call concurrently (ORT `run` takes `&mut self`) — the daemon/session pool
/// clones or wraps instances behind a semaphore.
pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    dims: usize,
    instruction: String,
}

impl Embedder {
    /// Download (or reuse HF cache) and load the model.
    pub fn load(spec: &ModelSpec, intra_threads: usize) -> Result<Self> {
        let (onnx_path, tokenizer_path) = fetch_files(spec)?;
        let env = ort::init()
            .with_name("t3rag")
            .build()
            .context("init ort environment")?;
        // ort builder errors aren't Send+Sync — render via Debug, not `?`.
        let berr = |e: ort::Error| anyhow::anyhow!("session builder: {e:?}");
        let builder = Session::builder(&env).map_err(berr)?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {e:?}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("intra threads: {e:?}"))?;
        let session = builder
            .commit_from_file(&onnx_path)
            .with_context(|| format!("load ONNX model {}", onnx_path.display()))?;

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
        };
        // Probe dims with a trivial forward pass (also serves as warmup).
        let probe = emb.forward_batch(&["t3rag warmup".to_string()])?;
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

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Embed one retrieval query (instruction prefix applied).
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let input = format!("{}{}", self.instruction, query);
        Ok(self.forward_batch(&[input])?.remove(0))
    }

    /// Embed a batch of passages (no instruction prefix). Caller batches.
    pub fn embed_passages(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.forward_batch(texts)
    }

    /// Tokenize+pad a batch, run the ONNX forward pass, last-token-pool each
    /// sequence, L2-normalize. Returns one unit vector per input.
    fn forward_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(String::as_str).collect::<Vec<_>>(), true)
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
        let shape = vec![batch, seq];
        let ids_t = Tensor::from_array((shape.clone(), ids)).context("ids tensor")?;
        let mask_t = Tensor::from_array((shape, mask)).context("mask tensor")?;
        let outputs = self
            .session
            .run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t])
            .context("onnx forward")?;
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

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Fetch the ONNX model + tokenizer from HF, trying candidate export names
/// in order. Returns local cache paths.
fn fetch_files(spec: &ModelSpec) -> Result<(PathBuf, PathBuf)> {
    let api = hf_hub::api::sync::Api::new().context("init hf-hub api")?;
    let repo = api.model(spec.repo_id.clone());

    let mut onnx = None;
    let mut errs = Vec::new();
    for file in &spec.onnx_files {
        match repo.get(file) {
            Ok(p) => {
                onnx = Some(p);
                break;
            }
            Err(e) => errs.push(format!("{file}: {e}")),
        }
    }
    let onnx = onnx.with_context(|| {
        format!(
            "no usable ONNX export in {} (tried: {})",
            spec.repo_id,
            errs.join("; ")
        )
    })?;
    let tokenizer = repo
        .get(&spec.tokenizer_file)
        .with_context(|| format!("fetch {} from {}", spec.tokenizer_file, spec.repo_id))?;
    Ok((onnx, tokenizer))
}
