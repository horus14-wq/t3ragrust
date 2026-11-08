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

/// Estimated resident cost of one embedder session (0.6B ONNX weights +
/// ORT session + workspace), used by the RAM guard (t3-tr2).
const MODEL_SESSION_GB: f64 = 2.0;

/// Available system RAM in GiB (Linux `/proc/meminfo` MemAvailable,
/// macOS `sysctl hw.memsize` as a total-RAM proxy; `None` elsewhere —
/// the guard then silently no-ops rather than guessing).
fn mem_available_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if let Some(kb) = line.strip_prefix("MemAvailable:") {
                if let Ok(kb) = kb.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                    return Some(kb as f64 / 1_048_576.0);
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
    {
        if let Ok(gb) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
            return Some(gb as f64 / 1_073_741_824.0);
        }
    }
    None
}

/// A pool of [`Embedder`] sessions: `size` independent sessions for
/// parallel passage embedding. Each session owns a private model copy
/// (~600MB per session for the 0.6B); `embed_passages` splits a batch
/// into per-session chunks run concurrently on scoped threads — the
/// merged-KV model is latency-bound per call, so K sessions give ~K×
/// index throughput on multi-core boxes. The daemon reuses this as its
/// inference-semaphore mechanism (t3-wkp.12).
pub struct EmbedderPool {
    embedders: Vec<Mutex<Embedder>>,
}

impl EmbedderPool {
    /// Load `size` sessions of `spec` (first call downloads the model).
    /// `threads == 0` divides the machine's cores evenly across sessions
    /// (min 1); otherwise `threads` intra-op threads per session.
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
        // RAM guard (t3-tr2): every session carries a private model copy —
        // warn BEFORE allocating when the request likely exceeds available
        // memory (the default pool is small; this guards explicit --pool).
        if let Some(avail_gb) = mem_available_gb() {
            let need_gb = size as f64 * MODEL_SESSION_GB;
            if need_gb > avail_gb {
                tracing::warn!(
                    sessions = size,
                    est_gb = need_gb,
                    avail_gb,
                    "embedder pool likely exceeds available RAM — lower --pool"
                );
            }
        }
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

    /// Embed `query` with the registry's query instruction prefix
    /// (single session — queries are latency-critical, not throughput).
    ///
    /// # Errors
    /// Tokenization or the forward pass fails.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embedders[0].lock().unwrap().embed_query(query)
    }

    /// Embed `texts` with the passage instruction prefix, split across
    /// pool sessions and run concurrently; results stay in input order
    /// (one vector per input text). Small batches run on a single
    /// session — splitting overhead isn't worth it.
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
