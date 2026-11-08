use std::path::PathBuf;

use anyhow::{anyhow, Result};

/// Root under which all t3rag artifacts live (`$XDG_CACHE_HOME/t3rag` or
/// `~/.cache/t3rag`).
pub fn cache_root() -> Result<PathBuf> {
    // T3RAG_HOME overrides the XDG cache — needed for tests/CI/
    // multi-instance dev (t3-css.3.1).
    if let Some(h) = std::env::var_os("T3RAG_HOME") {
        return Ok(PathBuf::from(h));
    }
    dirs::cache_dir()
        .map(|d| d.join("t3rag"))
        .ok_or_else(|| anyhow!("cannot resolve user cache directory"))
}

/// Directory holding every indexed corpus artifact.
/// Directory holding every indexed corpus artifact (`<cache>/corpora`).
pub fn corpora_root() -> Result<PathBuf> {
    Ok(cache_root()?.join("corpora"))
}

/// Artifact dir for one named corpus: passages.bin/.idx, vectors.usearch,
/// bm25/, meta.json.
/// `<corpora_root>/<name>` — the artifact dir for corpus `name`.
pub fn corpus_dir(name: &str) -> Result<PathBuf> {
    Ok(corpora_root()?.join(name))
}
