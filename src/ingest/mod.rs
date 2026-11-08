//! Corpus ingest readers (t3-wkp.3).
//!
//! JSONL (one passage/line; `text` required, `question`/`title`/`source`
//! optional), directory of `.md`/`.txt` (filename -> title), and parquet
//! via `parquet`+`arrow` streaming row groups (`--text-col` or auto-detect).
//! User-provided corpora only — no network fetches.

mod dir;
mod jsonl;
#[cfg(feature = "parquet")]
mod parquet_reader;
mod stats;

pub use stats::SkipStats;

use std::path::Path;

use anyhow::{bail, Result};

use crate::passage::Passage;

/// Boxed passage stream produced by an ingest reader.
pub type RawIter = Box<dyn Iterator<Item = Result<RawPassage>> + Send>;

/// A normalized ingest record prior to id assignment.
#[derive(Debug)]
pub struct RawPassage {
    pub text: String,
    pub title: Option<String>,
    pub question: Option<String>,
    pub source: Option<String>,
}

/// Which reader handled the input (for stats logging).
#[derive(Clone, Copy, Debug)]
pub enum InputKind {
    Jsonl,
    Directory,
    #[cfg(feature = "parquet")]
    Parquet,
}

/// Open `input`, auto-detecting the format: directory -> dir reader,
/// `.parquet` -> parquet (feature-gated), otherwise JSONL. `text_col`
/// overrides parquet column auto-detection.
///
/// # Errors
/// The input cannot be opened, or its format is unrecognized (or a
/// `.parquet` without the `parquet` feature).
///
/// Returns the iterator of raw records, the detected kind, and the shared
/// skip-stats counter (updated as the iterator is consumed).
pub fn open(input: &Path, text_col: Option<&str>) -> Result<(RawIter, InputKind, SkipStats)> {
    if input.is_dir() {
        let (it, stats) = dir::open(input)?;
        return Ok((it, InputKind::Directory, stats));
    }
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");
    #[cfg(not(feature = "parquet"))]
    let _ = text_col;
    match ext.to_ascii_lowercase().as_str() {
        "parquet" => {
            #[cfg(feature = "parquet")]
            {
                let (it, stats) = parquet_reader::open(input, text_col)?;
                Ok((it, InputKind::Parquet, stats))
            }
            #[cfg(not(feature = "parquet"))]
            bail!(
                ".parquet input but this binary was built without the `parquet` \
                 feature — rebuild with --features parquet"
            )
        }
        "jsonl" | "json" | "ndjson" | "" => {
            let (it, stats) = jsonl::open(input)?;
            Ok((it, InputKind::Jsonl, stats))
        }
        other => bail!(
            "unrecognized input format '.{other}' for {} — expected .jsonl, a directory of .md/.txt, or .parquet",
            input.display()
        ),
    }
}

/// Materialize a raw record into a `Passage` with its dense ordinal id.
/// Returns `None` (and counts a skip) for empty/whitespace text.
/// FxHash of a passage's text — the `--dedup` key (t3-css.2.8).
/// Fast, not cryptographic: exact duplicates only.
pub fn hash_text(text: &str) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = rustc_hash::FxBuildHasher.build_hasher();
    h.write(text.as_bytes());
    h.finish()
}

/// Materialize a [`RawPassage`] into a [`Passage`] with dense ordinal
/// `id`. Returns `None` (and counts a skip) for empty/whitespace text.
pub fn to_passage(raw: RawPassage, id: u64, stats: &SkipStats) -> Option<Passage> {
    if raw.text.trim().is_empty() {
        stats.skip("empty_text");
        return None;
    }
    let mut p = Passage::new(id, raw.text);
    p.title = raw.title;
    p.question = raw.question;
    p.source = raw.source;
    Some(p)
}

/// Shared extension check for the directory reader.
fn is_text_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("md") | Some("txt") | Some("markdown")
    )
}
