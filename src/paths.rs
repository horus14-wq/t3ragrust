use std::path::PathBuf;

use anyhow::{anyhow, Result};

/// Root under which all t3rag artifacts live (`$XDG_CACHE_HOME/t3rag` or
/// `~/.cache/t3rag`).
pub fn cache_root() -> Result<PathBuf> {
    dirs::cache_dir()
        .map(|d| d.join("t3rag"))
        .ok_or_else(|| anyhow!("cannot resolve user cache directory"))
}

/// Directory holding every indexed corpus artifact.
pub fn corpora_root() -> Result<PathBuf> {
    Ok(cache_root()?.join("corpora"))
}

/// Artifact dir for one named corpus: passages.bin/.idx, vectors.usearch,
/// bm25/, meta.json.
pub fn corpus_dir(name: &str) -> Result<PathBuf> {
    Ok(corpora_root()?.join(name))
}
