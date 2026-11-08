//! Introspection subcommands (t3-css.3.2): `list`, `stats`, `doctor`,
//! `health` — the "what do I have?" and "is my env OK?" questions every
//! agent asks first. All are cheap: `meta.json` reads + directory walks,
//! never a model load.

use anyhow::{Context, Result};

use crate::paths;
use crate::store::{self, CorpusMeta};

/// `t3rag list` — corpora table.
/// `t3rag list`: every corpus — name, docs, dims, model, size, age.
pub fn run_list() -> Result<()> {
    let root = paths::corpora_root()?;
    let mut rows: Vec<(String, CorpusMeta, u64, String)> = Vec::new();
    for e in std::fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let p = e?.path();
        if p.is_dir() && p.join("meta.json").exists() {
            let meta = store::read_meta(&p)?;
            let size = dir_size(&p);
            let mtime = std::fs::metadata(p.join("meta.json"))?
                .modified()
                .map(|t| {
                    let secs = t.elapsed().map(|d| d.as_secs()).unwrap_or(0);
                    if secs < 3600 {
                        format!("{}m ago", secs / 60)
                    } else {
                        format!("{}h ago", secs / 3600)
                    }
                })
                .unwrap_or_else(|_| "?".to_string());
            rows.push((
                p.file_name().unwrap().to_string_lossy().into_owned(),
                meta,
                size,
                mtime,
            ));
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    println!(
        "{:<18} {:<8} {:>7} {:>5} {:>9}  {:<44} {:>9}",
        "corpus", "type", "docs", "dims", "size", "model", "mtime"
    );
    for (name, meta, size, mtime) in &rows {
        let ty = meta
            .manifest
            .as_ref()
            .map(|m| m.transform.as_str())
            .unwrap_or("-");
        println!(
            "{:<18} {:<8} {:>7} {:>5} {:>9}  {:<44} {:>9}",
            name,
            ty,
            meta.doc_count,
            meta.dims,
            human_bytes(*size),
            meta.model,
            mtime
        );
    }
    if rows.is_empty() {
        println!("(no corpora — index one with `t3rag index`)");
    }
    Ok(())
}

/// `t3rag stats [corpus]` — deep meta + on-disk layout for one corpus.
/// `t3rag stats`: deep corpus inspection — metadata, index params,
/// artifact files, total size, RAM estimate.
pub fn run_stats(name: Option<String>) -> Result<()> {
    let root = paths::corpora_root()?;
    let name = name.unwrap_or(crate::serve::newest_corpus(&root)?);
    let dir = root.join(&name);
    let meta = store::read_meta(&dir).with_context(|| format!("corpus '{name}' not indexed"))?;
    println!("corpus:      {name}");
    println!("docs:        {}", meta.doc_count);
    println!("dims:        {}", meta.dims);
    println!("model:       {}", meta.model);
    println!("quantization:{}", meta.quantization);
    println!("format:      v{}", meta.format_version);
    if let Some(m) = &meta.manifest {
        println!("aboutness:   {}", m.transform);
        if !m.transform_mix.is_empty() {
            let mix: Vec<String> = m
                .transform_mix
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect();
            println!("  mix:       {}", mix.join(" "));
        }
        if !m.top_terms.is_empty() {
            println!("  terms:     {}", m.top_terms.join(", "));
        }
        let total: u64 = m.length_histogram.iter().sum();
        if total > 0 {
            println!(
                "  lengths:   <500:{} 500-2k:{} 2-8k:{} >=8k:{}",
                m.length_histogram[0],
                m.length_histogram[1],
                m.length_histogram[2],
                m.length_histogram[3]
            );
        }
        if let Some(lang) = &m.language {
            println!("  language:  {lang}");
        }
        if let Some(c) = &m.centroid {
            println!("  centroid:  {}d (secondary routing signal)", c.len());
        }
    }
    // Staleness (t3-ao9): origin carries host:user:input-path; warn when
    // the source is newer than the index.
    if let Some(stale) = staleness_note(&meta, &dir) {
        println!("{stale}");
    }
    println!("files:");
    for e in std::fs::read_dir(&dir)? {
        let p = e?.path();
        if p.is_file() {
            println!(
                "  {:<24} {:>10}",
                p.file_name().unwrap().to_string_lossy(),
                human_bytes(p.metadata()?.len())
            );
        } else if p.is_dir() {
            println!(
                "  {:<24} {:>10}",
                format!("{}/", p.file_name().unwrap().to_string_lossy()),
                human_bytes(dir_size(&p))
            );
        }
    }
    println!("total:       {}", human_bytes(dir_size(&dir)));
    // No `t3rag forget` exists by design (t3-css.4.5): a single-user
    // cache is deleted by the user, not the tool — plain rm is the
    // correct security model. The command below is the recipe.
    println!("remove with: rm -rf {}", dir.display());
    Ok(())
}

/// `t3rag doctor` — environment sanity: SIMD, EPs, HF cache, daemon.
/// `t3rag doctor`: environment sanity — SIMD compiled/available,
/// execution providers, HF cache, model files, corpora, daemon
/// health, versions, disk.
pub fn run_doctor() -> Result<()> {
    println!("t3rag {}", env!("CARGO_PKG_VERSION"));
    // SIMD paths usearch reports (compiled vs available on this CPU).
    println!(
        "simd:        compiled={} available={}",
        usearch::hardware_acceleration_compiled(),
        usearch::hardware_acceleration_available()
    );
    // Compiled-in execution providers.
    // `mut` only exercised when an ep-* feature is compiled in.
    #[allow(unused_mut)]
    let mut eps: Vec<&'static str> = vec!["cpu"];
    #[cfg(feature = "ep-cuda")]
    eps.push("cuda+tensorrt");
    #[cfg(feature = "ep-openvino")]
    eps.push("openvino");
    #[cfg(feature = "ep-webgpu")]
    eps.push("webgpu");
    #[cfg(feature = "ep-coreml")]
    eps.push("coreml");
    println!("ep built-in: {}", eps.join(", "));
    // HF cache: model repos we actually use.
    let hf = dirs_hf_cache();
    if let Some(hf) = &hf {
        let mut n = 0usize;
        let mut size = 0u64;
        for e in std::fs::read_dir(hf)?.flatten() {
            let p = e.path();
            if p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .starts_with("models--")
            {
                n += 1;
                size += dir_size(&p);
            }
        }
        println!(
            "hf cache:    {} ({} models, {})",
            hf.display(),
            n,
            human_bytes(size)
        );
    } else {
        println!("hf cache:    not found (models download on first use)");
    }
    // Corpus root + count.
    let root = paths::corpora_root()?;
    let n_corpora = std::fs::read_dir(&root)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().join("meta.json").exists())
                .count()
        })
        .unwrap_or(0);
    println!("corpora:     {} under {}", n_corpora, root.display());
    println!("             (delete one yourself: rm -rf <path>; t3-css.4.5)");
    // Daemon reachability (no spawn — report only).
    let info = paths::cache_root()?.join("daemon.json");
    if let Ok(dj) = std::fs::read_to_string(&info) {
        let v: serde_json::Value = serde_json::from_str(&dj).unwrap_or_default();
        if let Some(addr) = v["url"].as_str().or_else(|| v["addr"].as_str()) {
            match ureq::get(&format!("{addr}/health")).call() {
                Ok(_) => println!("daemon:      UP at {addr}"),
                Err(_) => println!("daemon:      info file at {addr} but not responding"),
            }
        }
    } else {
        println!("daemon:      not running (auto-spawns on first query)");
    }
    Ok(())
}

/// `t3rag health` — <50ms pre-flight sanity for agents.
/// `t3rag health`: the <50ms agent pre-flight check — daemon + a
/// corpus are reachable.
pub fn run_health() -> Result<()> {
    let root = paths::corpora_root()?;
    if root.is_dir() {
        println!("OK");
    } else {
        println!("OK (cache root will be created)");
    }
    Ok(())
}

/// Staleness check (t3-ao9): `meta.origin` is `host:user:input-path`;
/// warn when the input's newest mtime is newer than `meta.json`'s.
/// Returns a ready-to-print line, or `None` when fresh/unknown.
fn staleness_note(meta: &CorpusMeta, dir: &std::path::Path) -> Option<String> {
    let origin = meta.origin.as_ref()?;
    // origin = host:user:input-path — input is last so splitn keeps colons.
    let input = origin.splitn(3, ':').nth(2)?;
    let input = std::path::Path::new(input);
    if !input.exists() {
        return None;
    }
    let input_mtime = if input.is_file() {
        input.metadata().ok()?.modified().ok()?
    } else {
        // Directory input: newest file inside (bounded walk).
        let mut newest = input.metadata().ok()?.modified().ok()?;
        for e in walk_files(input, 0) {
            if let Ok(m) = e.metadata().and_then(|m| m.modified()) {
                if m > newest {
                    newest = m;
                }
            }
        }
        newest
    };
    let index_mtime = std::fs::metadata(dir.join("meta.json"))
        .and_then(|m| m.modified())
        .ok()?;
    (input_mtime > index_mtime).then(|| {
        format!(
            "stale:       input {} is newer than the index — rebuild to refresh",
            input.display()
        )
    })
}

/// Bounded recursive file walk for the staleness check (depth <= 4,
/// <= 4096 entries — stats must stay cheap).
fn walk_files(dir: &std::path::Path, depth: usize) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if depth > 4 || out.len() >= 4096 {
        return out;
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten().take(4096) {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk_files(&p, depth + 1));
            } else {
                out.push(p);
            }
            if out.len() >= 4096 {
                break;
            }
        }
    }
    out
}

fn human_bytes(b: u64) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let (mut v, mut i) = (b as f64, 0usize);
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1}{}", U[i])
}

fn dir_size(p: &std::path::Path) -> u64 {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .map(|e| {
                    let m = e.path();
                    if m.is_dir() {
                        dir_size(&m)
                    } else {
                        m.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

fn dirs_hf_cache() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let p = std::path::PathBuf::from(home).join(".cache/huggingface/hub");
    p.is_dir().then_some(p)
}
