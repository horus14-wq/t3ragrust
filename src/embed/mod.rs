//! Embedding model wrapper (t3-wkp.5).
//!
//! Qwen3-Embedding-0.6B via `ort` + ONNX (`onnx-community` export, uint8/fp16
//! preferred). Recipe: instruction prefix for queries -> tokenize -> ONNX
//! session -> last-token pooling -> L2 normalize. Batch API for index builds.
//! `candle` is banned by design; `fastembed` is the documented fallback only.
//!
//! Model files are fetched via `hf-hub`'s sync (ureq) API and land in the
//! standard HF cache (`~/.cache/huggingface` unless `HF_HOME` is set) — the
//! cache is shared across machines/users and dedupes the ~1-2GB weights.

mod model;
mod pool;
pub(crate) mod session;

pub use model::{ModelSpec, Pooling, REGISTRY};
pub use pool::EmbedderPool;
pub use session::Embedder;

/// Instruction prefix applied to *queries only* — the Qwen3-Embedding
/// convention. Passages are embedded bare. Tuneable via [`ModelSpec`].
pub const DEFAULT_QUERY_INSTRUCTION: &str =
    "Instruct: Given a question, retrieve thinking traces whose reasoning helps answer it\nQuery: ";

/// Max tokens per passage before head-truncation at embed time.
/// Matches the chunking policy's ~8K threshold (t3-wkp.11): passages
/// above this are chunked at index time, and this bound keeps any
/// un-chunked stragglers from blowing up inference.
pub const MAX_EMBED_TOKENS: usize = 8192;

/// Resolve a `--model`/`T3RAG_MODEL` flag to a [`ModelSpec`]: known
/// registry entry or alias first, then an `owner/name` HF repo id
/// (same filename/layout conventions assumed), else the default model.
pub fn resolve_model(flag: Option<&str>) -> ModelSpec {
    match flag.unwrap_or("qwen3-0.6b") {
        "qwen3-0.6b" | "0.6b" | "qwen3" | "default" => ModelSpec::qwen3_embedding_0_6b(),
        other => ModelSpec::for_hf_repo(other),
    }
}
