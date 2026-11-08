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
    /// `output` is the passage field used by the T3 transform pipeline
    /// (t3data `f5_v2_t3_*.jsonl`); `text` wins when both are present.
    output: Option<String>,
    title: Option<String>,
    question: Option<String>,
    source: Option<String>,
    /// Upstream record index (T3 transforms carry `source_idx` pointing
    /// into the source corpus; the raw f5 file spells it `Source_IDX`).
    /// Folded into `source` as `path:line#src_idx=N` — reusing the
    /// existing string field avoids a passages.bin schema change, so
    /// old corpora stay readable (t3-dyt).
    #[serde(alias = "Source_IDX")]
    source_idx: Option<serde_json::Value>,
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
                        Ok(rec) => match rec.text.or(rec.output) {
                            Some(text) => Some(Ok(RawPassage {
                                text,
                                title: rec.title,
                                question: rec.question,
                                source: rec.source.or_else(|| {
                                    let idx = rec.source_idx.as_ref().map(|v| match v {
                                        serde_json::Value::String(s) => s.clone(),
                                        other => other.to_string(),
                                    });
                                    Some(match idx {
                                        Some(i) => {
                                            format!("{}#src_idx={i}", default_source(&src, lineno))
                                        }
                                        None => default_source(&src, lineno),
                                    })
                                }),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(lines: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("t3jsonl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("in.jsonl");
        std::fs::write(&p, lines.join("\n") + "\n").unwrap();
        p
    }

    #[test]
    fn source_idx_folds_into_source() {
        let p = fixture(&[
            r#"{"text":"a","source_idx":42}"#,
            r#"{"text":"b","Source_IDX":"7"}"#,
            r#"{"text":"c"}"#,
            r#"{"text":"d","source":"explicit:9","source_idx":5}"#,
        ]);
        let (it, _stats) = open(&p).unwrap();
        let recs: Vec<RawPassage> = it.map(|r| r.unwrap()).collect();
        assert_eq!(recs.len(), 4);
        assert!(recs[0].source.as_deref().unwrap().ends_with("#src_idx=42"));
        assert!(recs[1].source.as_deref().unwrap().ends_with("#src_idx=7"));
        assert!(!recs[2].source.as_deref().unwrap().contains("src_idx"));
        // An explicit `source` field always wins over source_idx.
        assert_eq!(recs[3].source.as_deref(), Some("explicit:9"));
    }
}
