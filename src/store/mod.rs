//! Packed passage store (t3-wkp.2).
//!
//! `passages.bin` (length-prefixed records) + `passages.idx` (fixed-width
//! offset table), read through `memmap2` for O(1) random access at ~0
//! resident RAM. `meta.json` records model, dims, quantization, doc_count
//! and `format_version` — bump `FORMAT_VERSION` on any layout change.

mod reader;
mod writer;

pub use reader::PassageReader;
pub use writer::PassageWriter;

use serde::{Deserialize, Serialize};

/// Bump on any corpus artifact layout change; readers refuse newer versions.
/// v2: vectors.raw.f32 records carry explicit passage ids (chunking parents
/// are stored without vectors, so row-index != passage_id).
pub const FORMAT_VERSION: u32 = 2;

/// Per-corpus metadata persisted as `meta.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusMeta {
    /// Layout version — see [`FORMAT_VERSION`]; readers refuse newer.
    pub format_version: u32,
    /// Embedder spec the vectors were built with.
    pub model: String,
    /// Vector dimensionality.
    pub dims: u32,
    /// Quantization scheme (e.g. `f32`, `i8`).
    pub quantization: String,
    /// Number of passages in the corpus.
    pub doc_count: u64,
    // Provenance (t3-css.4.7): corpora self-describe their build so
    // stats/doctor/eval can compare; Option+default keeps older
    // metas (written before these fields existed) parseable.
    /// RFC3339 build timestamp.
    #[serde(default)]
    pub created_at: Option<String>,
    /// `<host>:<user>:<input-path>` — warns on cross-origin reuse.
    #[serde(default)]
    pub origin: Option<String>,
    /// Passage instruction prefix used at embed time.
    #[serde(default)]
    pub instruction: Option<String>,
    /// Index-time knobs that produced this corpus.
    #[serde(default)]
    pub index_params: Option<serde_json::Value>,
    /// Aboutness manifest (t3-iv2): transform type, top terms, length
    /// histogram, language, centroid — corpus routing for `suggest`.
    #[serde(default)]
    pub manifest: Option<crate::manifest::CorpusManifest>,
}

/// Serialize `meta` to `<dir>/meta.json` (pretty-printed).
///
/// # Errors
/// The directory does not exist, or the write fails.
pub fn write_meta(dir: &std::path::Path, meta: &CorpusMeta) -> anyhow::Result<()> {
    let path = dir.join("meta.json");
    let s = serde_json::to_string_pretty(meta)?;
    std::fs::write(&path, s)?;
    Ok(())
}

/// Read + validate `<dir>/meta.json`, refusing `format_version`s
/// newer than [`FORMAT_VERSION`] (older ones parse via `Option`
/// defaults — no migration shim).
///
/// # Errors
/// The file is missing/unparseable, or declares a newer format.
pub fn read_meta(dir: &std::path::Path) -> anyhow::Result<CorpusMeta> {
    let path = dir.join("meta.json");
    let s = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("read {}: {e} (corpus not indexed?)", path.display()))?;
    let meta: CorpusMeta = serde_json::from_str(&s)?;
    if meta.format_version > FORMAT_VERSION {
        anyhow::bail!(
            "corpus format_version {} > supported {} — rebuild the index",
            meta.format_version,
            FORMAT_VERSION
        );
    }
    Ok(meta)
}
