//! Cross-encoder reranker (t3-wkp.10): BGE-v2-m3 via ort + ONNX.
//!
//! Rerankers score (query, passage) pairs directly — much more accurate than
//! bi-encoder fusion scores but O(pair) compute, so it only runs on the fused
//! candidate list (top ~50), with pairs truncated to 512 tokens. Default
//! model: `onnx-community/bge-reranker-v2-m3-ONNX` (uint8 preferred).
//!
//! Like the embedder, extra declared inputs are filled generically
//! (token_type_ids -> zeros; past_key_values.* -> empty) so other reranker
//! exports work without code changes.

use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};
use tokenizers::{EncodeInput, PaddingParams, Tokenizer, TruncationParams};
use tracing::info;

use crate::cli::Ep;
use crate::embed::session::{fetch_files, ort_env};
use crate::embed::ModelSpec;

/// Pairs are truncated to this length — reranking reads the whole passage
/// only for the final top-few, so head-truncation is the speed/quality sweet spot.
const RERANK_MAX_TOKENS: usize = 512;
const PAD_ID: u32 = 1; // XLM-RoBERTa pad token

/// Default reranker model: BGE-v2-m3 ONNX (multilingual, strong cross-encoder).
pub fn default_spec() -> ModelSpec {
    ModelSpec::for_hf_repo("onnx-community/bge-reranker-v2-m3-ONNX")
}

/// A loaded cross-encoder session. Same concurrency model as [`crate::embed::Embedder`].
pub struct Reranker {
    session: Session,
    tokenizer: Tokenizer,
    /// Non-pair inputs declared by the export (name, declared dims, is_f32):
    /// token_type_ids -> zeros, past_key_values.* -> empty cache.
    extra_inputs: Vec<(String, Vec<i64>, bool)>,
}

impl Reranker {
    pub fn load(spec: &ModelSpec, intra_threads: usize, ep: Ep) -> Result<Self> {
        let eps = crate::embed::session::providers_for(ep)?;
        let (onnx_path, tokenizer_path) = fetch_files(spec)?;
        let env = ort_env()?;
        let berr = |e: ort::Error| anyhow::anyhow!("session builder: {e:?}");
        let builder = Session::builder(env.as_ref()).map_err(berr)?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {e:?}"))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!("intra threads: {e:?}"))?
            .with_execution_providers(&eps)
            .map_err(|e| anyhow::anyhow!("execution providers: {e:?}"))?;
        let session = builder
            .commit_from_file(&onnx_path)
            .with_context(|| format!("load ONNX model {}", onnx_path.display()))?;

        let mut extra_inputs = Vec::new();
        for o in session.inputs() {
            let name = o.name();
            if matches!(name, "input_ids" | "attention_mask") {
                continue;
            }
            if let ValueType::Tensor { ty, shape, .. } = o.dtype() {
                extra_inputs.push((
                    name.to_string(),
                    shape.iter().copied().collect(),
                    *ty == TensorElementType::Float32,
                ));
            }
        }
        if !extra_inputs.is_empty() {
            info!(?extra_inputs, "reranker: feeding generic extra inputs");
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: RERANK_MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("set truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: PAD_ID,
            pad_token: "<pad>".into(),
            ..Default::default()
        }));
        Ok(Self {
            session,
            tokenizer,
            extra_inputs,
        })
    }

    /// Score (query, passage) pairs. Returns raw logits (monotonic rank signal;
    /// sigmoid if a probability is wanted downstream).
    pub fn score_pairs(&mut self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<EncodeInput> = pairs
            .iter()
            .map(|(q, p)| EncodeInput::Dual(q.as_str().into(), p.as_str().into()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("tokenize pairs: {e}"))?;
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
        let mut feeds: Vec<(String, SessionInputValue)> = vec![
            (
                "input_ids".into(),
                Tensor::from_array((shape.clone(), ids))
                    .context("ids tensor")?
                    .into(),
            ),
            (
                "attention_mask".into(),
                Tensor::from_array((shape.clone(), mask))
                    .context("mask tensor")?
                    .into(),
            ),
        ];
        for (name, dims, is_f32) in &self.extra_inputs {
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
            let v: SessionInputValue = if *is_f32 {
                Tensor::<f32>::from_array((d, Vec::new()))
                    .with_context(|| format!("extra input {name}"))?
                    .into()
            } else {
                // token_type_ids and friends: zeros like ids.
                let zeros = vec![0i64; d.iter().product()];
                Tensor::from_array((d, zeros))
                    .with_context(|| format!("extra input {name}"))?
                    .into()
            };
            feeds.push((name.clone(), v));
        }
        let outputs = self.session.run(feeds).context("reranker forward")?;
        let out = outputs
            .iter()
            .next()
            .context("reranker produced no outputs")?;
        let (shape_ref, data) = out
            .1
            .try_extract_tensor::<f32>()
            .context("extract reranker logits")?;
        if data.is_empty() {
            bail!("reranker returned empty logits");
        }
        // logits shape [batch, 1] or [batch]; take first value per row.
        let row = data.len() / batch;
        if shape_ref.len() == 1 || row == 1 {
            Ok(data.iter().step_by(row.max(1)).copied().collect())
        } else {
            Ok(data.chunks(row).map(|c| c[0]).collect())
        }
    }
}

/// Parallel reranker pool — mirrors [`crate::embed::EmbedderPool`].
pub struct RerankerPool {
    sessions: Vec<Mutex<Reranker>>,
}

impl RerankerPool {
    pub fn load(spec: &ModelSpec, threads: usize, ep: Ep, size: usize) -> Result<Self> {
        let size = size.max(1);
        let per = if threads == 0 {
            (std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                / size)
                .max(1)
        } else {
            threads.max(1)
        };
        let mut sessions = Vec::with_capacity(size);
        for i in 0..size {
            sessions.push(Mutex::new(
                Reranker::load(spec, per, ep)
                    .with_context(|| format!("load reranker session {i}"))?,
            ));
        }
        info!(sessions = size, intra_threads = per, "reranker pool ready");
        Ok(Self { sessions })
    }

    /// Score pairs split across pool sessions; results stay in input order.
    pub fn score_pairs(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if self.sessions.len() == 1 || pairs.len() < self.sessions.len() * 2 {
            return self.sessions[0].lock().unwrap().score_pairs(pairs);
        }
        let n = self.sessions.len().min(pairs.len());
        let chunk = pairs.len().div_ceil(n);
        let chunks: Vec<&[(String, String)]> = pairs.chunks(chunk).take(n).collect();
        let mut results: Vec<Result<Vec<f32>>> = Vec::with_capacity(chunks.len());
        std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let m = &self.sessions[i];
                    s.spawn(move || m.lock().unwrap().score_pairs(c))
                })
                .collect();
            for h in handles {
                results.push(
                    h.join()
                        .map_err(|_| anyhow::anyhow!("reranker worker panicked"))?,
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
        let mut out = Vec::with_capacity(pairs.len());
        for r in results {
            out.extend(r?);
        }
        Ok(out)
    }
}
