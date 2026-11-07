//! JSONL reader: one JSON object per line; `text` required, optional
//! `question` / `title` / `source`. Malformed/blank/missing-text lines are
//! counted in SkipStats and skipped; only real read failures yield `Err`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{RawIter, RawPassage, SkipStats};

#[derive(Deserialize)]
struct Line {
    text: Option<String>,
    title: Option<String>,
    question: Option<String>,
    source: Option<String>,
}

pub fn open(path: &Path) -> Result<(RawIter, SkipStats)> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let stats = SkipStats::default();
    let st = stats.clone();
    let src = path.to_path_buf();
    let it =
        BufReader::new(file)
            .lines()
            .enumerate()
            .filter_map(move |(lineno, line)| match line {
                Err(e) => Some(
                    Err(e).with_context(|| format!("read {} line {}", src.display(), lineno + 1)),
                ),
                Ok(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        return None;
                    }
                    st.record_read(trimmed.len() as u64);
                    match serde_json::from_str::<Line>(trimmed) {
                        Ok(rec) => match rec.text {
                            Some(text) => Some(Ok(RawPassage {
                                text,
                                title: rec.title,
                                question: rec.question,
                                source: rec.source.or_else(|| Some(default_source(&src, lineno))),
                            })),
                            None => {
                                st.skip("missing_text_field");
                                None
                            }
                        },
                        Err(_) => {
                            st.skip("malformed_json");
                            None
                        }
                    }
                }
            });
    Ok((Box::new(it), stats))
}

fn default_source(path: &Path, lineno: usize) -> String {
    format!("{}:{}", path.display(), lineno + 1)
}
