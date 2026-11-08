//! Parallel embedder pool: N ORT sessions, each owning a private model copy
//! (~600MB per session for the 0.6B). `embed_passages` splits a batch into
//! per-session chunks and runs them concurrently on scoped threads — the
//! merged-KV model is latency-bound per call, so K sessions give ~K× index
//! throughput on multi-core boxes. The daemon reuses this as its
//! inference-semaphore mechanism (t3-wkp.12).

use std::sync::Mutex;

use anyhow::{Context, Result};

use super::{Embedder, ModelSpec};
use crate::cli::Ep;

/// A pool of [`Embedder`] sessions: `size` independent sessions for
/// parallel passage embedding, round-robin on call.
pub struct EmbedderPool {
    embedders: Vec<Mutex<Embedder>>,
}

impl EmbedderPool {
    /// Load `size` sessions of the same model. `threads == 0` divides the
    /// machine's cores evenly across sessions (min 1).
    /// Load `size` sessions of `spec` (first call downloads the model).
    ///
    /// # Errors
    /// The model/tokenizer cannot be fetched or a session fails.
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
        let mut embedders = Vec::with_capacity(size);
        for i in 0..size {
            let e = Embedder::load(spec, per, ep)
                .with_context(|| format!("load embedder session {i}"))?;
            embedders.push(Mutex::new(e));
        }
        tracing::info!(sessions = size, intra_threads = per, "embedder pool ready");
        Ok(Self { embedders })
    }

    /// Embedding dimension of the loaded model.
    pub fn dims(&self) -> usize {
        self.embedders[0].lock().unwrap().dims()
    }

    /// Embed one query (instruction prefix applied) — uses session 0.
    /// Embed `query` with the registry's query instruction prefix.
    ///
    /// # Errors
    /// Tokenization or the forward pass fails.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embedders[0].lock().unwrap().embed_query(query)
    }

    /// Embed a batch of passages, split across pool sessions and run
    /// concurrently; results stay in input order.
    /// Embed `texts` (borrowed slices, t3-css.2.9) with the passage
    /// instruction; results align one-to-one with the input.
    ///
    /// # Errors
    /// Tokenization or a session's forward pass fails.
    pub fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.embedders.len() == 1 || texts.len() < self.embedders.len() * 2 {
            return self.embedders[0].lock().unwrap().embed_passages(texts);
        }
        let n = self.embedders.len().min(texts.len());
        let chunk = texts.len().div_ceil(n);
        let chunks: Vec<&[&str]> = texts.chunks(chunk).take(n).collect();
        let mut results: Vec<Result<Vec<Vec<f32>>>> = Vec::with_capacity(chunks.len());
        std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let m = &self.embedders[i];
                    s.spawn(move || m.lock().unwrap().embed_passages(c))
                })
                .collect();
            for h in handles {
                results.push(
                    h.join()
                        .map_err(|_| anyhow::anyhow!("embedder worker panicked"))?,
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
        let mut out = Vec::with_capacity(texts.len());
        for r in results {
            out.extend(r?);
        }
        Ok(out)
    }
}
