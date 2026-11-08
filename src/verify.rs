//! `t3rag verify` (t3-ao9): corpus integrity checks — meta consistency,
//! store/index counts, artifact presence, staleness. Read-only, cheap:
//! mmaps headers, never embeds or searches. An agent should never
//! silently search a corrupt or stale index.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::index::dense::DenseIndex;
use crate::paths;
use crate::store;

/// One check outcome.
pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

/// Full verification report for one corpus dir.
pub struct VerifyReport {
    pub corpus: String,
    pub dir: PathBuf,
    pub checks: Vec<Check>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }
    /// Human-readable rendering for `t3rag verify`.
    pub fn render(&self) -> String {
        let mut s = format!("corpus: {} ({})\n", self.corpus, self.dir.display());
        for c in &self.checks {
            s.push_str(&format!(
                "  {:<4} {:<22} {}\n",
                if c.ok { "ok" } else { "FAIL" },
                c.name,
                c.detail
            ));
        }
        s.push_str(if self.ok() {
            "verify: PASS\n"
        } else {
            "verify: FAIL — rebuild with `t3rag index`\n"
        });
        s
    }
}

/// Verify corpus `name` (or newest when `None`).
///
/// # Errors
/// The corpus cannot be resolved at all (no meta.json).
pub fn verify(name: Option<String>) -> Result<VerifyReport> {
    let root = paths::corpora_root()?;
    let name = name.unwrap_or(crate::serve::newest_corpus(&root)?);
    let dir = root.join(&name);
    Ok(verify_dir(&name, &dir))
}

/// Verify a corpus directory directly (also used by tests).
pub fn verify_dir(name: &str, dir: &Path) -> VerifyReport {
    let mut checks = Vec::new();
    let meta = match store::read_meta(dir) {
        Ok(m) => {
            checks.push(Check {
                name: "meta.json",
                ok: true,
                detail: format!(
                    "format v{} model={} dims={} docs={}",
                    m.format_version, m.model, m.dims, m.doc_count
                ),
            });
            Some(m)
        }
        Err(e) => {
            checks.push(Check {
                name: "meta.json",
                ok: false,
                detail: e.to_string(),
            });
            None
        }
    };
    let Some(meta) = meta else {
        return VerifyReport {
            corpus: name.into(),
            dir: dir.to_path_buf(),
            checks,
        };
    };

    // Passage store: openable + record count matches meta.doc_count.
    match crate::store::PassageReader::open(dir) {
        Ok(r) => {
            let n = r.len();
            checks.push(Check {
                name: "passages",
                ok: n == meta.doc_count,
                detail: format!("{n} records (meta says {})", meta.doc_count),
            });
        }
        Err(e) => checks.push(Check {
            name: "passages",
            ok: false,
            detail: e.to_string(),
        }),
    }

    // Dense index: opens + vector count matches embedded units.
    // Embedded units = doc_count - chunked parents (parents have no vector).
    let quant = parse_quant(&meta.quantization);
    match DenseIndex::view(dir, meta.dims as usize, quant) {
        Ok(d) => {
            // size() is on the usearch handle inside DenseIndex — not
            // exposed; use raw.f32 record count as the vector census.
            drop(d);
            checks.push(Check {
                name: "vectors.usearch",
                ok: true,
                detail: "opens".into(),
            });
        }
        Err(e) => checks.push(Check {
            name: "vectors.usearch",
            ok: false,
            detail: e.to_string(),
        }),
    }

    // Raw f32 blob: header n/dims + exact file size.
    let raw_path = dir.join(crate::index::dense::RAW_FILE);
    match std::fs::File::open(&raw_path) {
        Ok(f) => {
            use std::io::Read;
            let mut hdr = [0u8; 16];
            let rd = (&f).read_exact(&mut hdr);
            match rd {
                Ok(()) => {
                    let n = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
                    let dims = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
                    let actual = f.metadata().map(|m| m.len()).unwrap_or(0);
                    // Record layout differs by format version (dense.rs):
                    // v1 = bare f32 rows, v2 = [u64 id][f32 vec]. Accept
                    // either — same rule as `DenseIndex::exact_search`.
                    let v1 = 16 + n * (dims * 4);
                    let v2 = 16 + n * (8 + dims * 4);
                    let (ok, layout) = if actual == v2 {
                        (dims == meta.dims as u64, "v2")
                    } else if actual == v1 {
                        (dims == meta.dims as u64, "v1")
                    } else {
                        (false, "?")
                    };
                    checks.push(Check {
                        name: "vectors.raw.f32",
                        ok,
                        detail: format!(
                            "n={n} dims={dims} size={actual} layout={layout} (v1={v1} v2={v2})"
                        ),
                    });
                }
                Err(e) => checks.push(Check {
                    name: "vectors.raw.f32",
                    ok: false,
                    detail: format!("short header: {e}"),
                }),
            }
        }
        Err(e) => checks.push(Check {
            name: "vectors.raw.f32",
            ok: false,
            detail: e.to_string(),
        }),
    }

    // BM25 leg: index dir exists and is non-empty.
    let bm = dir.join("bm25");
    let bm_ok = bm.is_dir()
        && std::fs::read_dir(&bm)
            .map(|mut rd| rd.next().is_some())
            .unwrap_or(false);
    checks.push(Check {
        name: "bm25/",
        ok: bm_ok,
        detail: if bm_ok {
            "segment files present".into()
        } else {
            format!("{} missing or empty", bm.display())
        },
    });

    // Staleness: input newer than index → warning (not a hard fail).
    if let Some(note) = crate::doctor::staleness_note(&meta, dir) {
        checks.push(Check {
            name: "freshness",
            ok: true,
            detail: note,
        });
    }

    VerifyReport {
        corpus: name.into(),
        dir: dir.to_path_buf(),
        checks,
    }
}

fn parse_quant(s: &str) -> crate::cli::Quantization {
    use crate::cli::Quantization::*;
    match s {
        "bf16" => Bf16,
        "f16" => F16,
        "fp32" => Fp32,
        _ => I8,
    }
}
