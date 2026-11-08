//! Directory reader: one passage per `.md`/`.markdown`/`.txt` file, recursive,
//! deterministic order (sorted paths). Filename stem becomes `title`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{is_text_file, RawIter, RawPassage, SkipStats};

pub fn open(dir: &Path) -> Result<(RawIter, SkipStats)> {
    // Collect + sort up front for reproducible re-index order.
    let mut files: Vec<PathBuf> = Vec::new();
    collect(dir, &mut files)?;
    files.sort();
    let stats = SkipStats::default();
    let st = stats.clone();
    // Parallel read (t3-css.2.9): rayon over the sorted path list —
    // order is preserved by par_iter's collect, keeping doc ids stable.
    use rayon::prelude::*;
    let results: Vec<_> = files
        .par_iter()
        .map(|path| (path.clone(), read_one(path)))
        .collect();
    let it = results.into_iter().filter_map(move |(path, r)| {
        match r {
            Ok((text, title)) => {
                st.record_read(text.len() as u64);
                Some(Ok(RawPassage {
                    text,
                    title: Some(title),
                    question: None,
                    source: Some(path.display().to_string()),
                }))
            }
            Err(e) => {
                // One unreadable file must not kill a large corpus — count
                // and skip, reporting the error at warn level.
                st.skip("unreadable_file");
                tracing::warn!("skipping {}: {e:#}", path.display());
                None
            }
        }
    });
    Ok((Box::new(it), stats))
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out)?;
        } else if is_text_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn read_one(path: &Path) -> Result<(String, String)> {
    // lossy-convert: corpus text is user-supplied; invalid UTF-8 becomes the
    // replacement char rather than a hard failure (documented choice).
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let title = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    Ok((text, title))
}
