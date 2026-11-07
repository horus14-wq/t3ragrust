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
pub const FORMAT_VERSION: u32 = 1;

/// Per-corpus metadata persisted as `meta.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusMeta {
    pub format_version: u32,
    pub model: String,
    pub dims: u32,
    pub quantization: String,
    pub doc_count: u64,
}

/// Write `meta.json` into a corpus dir (fsync'd).
pub fn write_meta(dir: &std::path::Path, meta: &CorpusMeta) -> anyhow::Result<()> {
    let path = dir.join("meta.json");
    let s = serde_json::to_string_pretty(meta)?;
    std::fs::write(&path, s)?;
    Ok(())
}

/// Read + validate `meta.json`; refuses newer format_version.
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
