//! Machine-readable self-description (t3-css.3.3): `capabilities`
//! dumps the whole CLI surface as JSON so agents discover flags/env
//! without reading docs; `models` shows the registry + cache status;
//! `export` round-trips a corpus to JSONL; `completions`/`man` are the
//! standard shell plumbing.

use anyhow::{Context, Result};
use clap::CommandFactory;

use crate::cli::{Cli, ExportArgs};
use crate::embed::ModelSpec;
use crate::paths;

/// `t3rag capabilities` — full CLI surface as JSON.
/// `t3rag capabilities`: the full CLI surface as JSON (commands,
/// flags, env vars, endpoints) — the machine-readable agent contract.
pub fn run_capabilities() -> Result<()> {
    let cmd = Cli::command();
    let out = serde_json::json!({
        "name": "t3rag",
        "version": env!("CARGO_PKG_VERSION"),
        "description": cmd.get_about().map(|s| s.to_string()),
        "commands": cmd.get_subcommands().map(describe_cmd).collect::<Vec<_>>(),
        "daemon": {
            "default_addr": "127.0.0.1:7317",
            "endpoints": ["GET /health", "GET /corpora", "POST /search", "POST /mcp"],
        },
        "mcp_tool": "search_traces",
        "cache_root": paths::cache_root().ok().map(|p| p.display().to_string()),
        "env": {
            "T3RAG_*": "flag env-var plumbing (t3-css.3.1)",
            "RUST_LOG": "tracing filter",
        },
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn describe_cmd(c: &clap::Command) -> serde_json::Value {
    serde_json::json!({
        "name": c.get_name(),
        "about": c.get_about().map(|s| s.to_string()),
        "args": c.get_arguments().map(|a| serde_json::json!({
            "id": a.get_id().to_string(),
            "long": a.get_long().map(|s| s.to_string()),
            "help": a.get_help().map(|s| s.to_string()),
            "required": a.is_required_set(),
        })).collect::<Vec<_>>(),
    })
}

/// `t3rag models`: the embedder registry + rerankers with HF-cache
/// status (`cached`/`dims`/`instruction` columns).
pub fn run_models() -> Result<()> {
    let hf = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".cache/huggingface/hub"));
    println!(
        "{:<48} {:>5} {:>8} {:<18} instruction",
        "repo", "dims", "cached", "role"
    );
    for e in crate::embed::REGISTRY {
        let cached = hf
            .as_ref()
            .map(|h| {
                h.join(format!("models--{}", e.repo.replace('/', "--")))
                    .exists()
            })
            .unwrap_or(false);
        println!(
            "{:<48} {:>5} {:>8} {:<18} {}",
            e.repo,
            e.dims,
            if cached { "yes" } else { "-" },
            "embedder",
            e.instruction
        );
    }
    for name in ["minilm", "bge-base", "bge-m3"] {
        let spec = crate::rerank::spec(name)?;
        let cached = hf
            .as_ref()
            .map(|h| {
                h.join(format!("models--{}", spec.repo_id.replace('/', "--")))
                    .exists()
            })
            .unwrap_or(false);
        println!(
            "{:<48} {:>5} {:>8} {:<18} -",
            spec.repo_id,
            "-",
            if cached { "yes" } else { "-" },
            format!("reranker ({name})"),
        );
    }
    Ok(())
}

/// `t3rag export <corpus>` — round-trip to JSONL.
/// `t3rag export`: dump a corpus's passages back to JSONL (the
/// round-trip that makes `index` inspectable).
///
/// # Errors
/// The corpus is not indexed, or the output cannot be written.
pub fn run_export(args: ExportArgs) -> Result<()> {
    let root = paths::corpora_root()?;
    let name = args.corpus.unwrap_or(crate::serve::newest_corpus(&root)?);
    let dir = root.join(&name);
    let meta =
        crate::store::read_meta(&dir).with_context(|| format!("corpus '{name}' not indexed"))?;
    let reader = crate::store::PassageReader::open(&dir)?;
    let out = std::fs::File::create(&args.out)
        .with_context(|| format!("create {}", args.out.display()))?;
    let mut w = std::io::BufWriter::new(out);
    use std::io::Write;
    for id in 0..meta.doc_count {
        let p = reader.get(id)?;
        writeln!(
            w,
            "{}",
            serde_json::json!({
                "id": p.id, "text": p.text, "title": p.title,
                "question": p.question, "source": p.source,
                "parent_id": p.parent_id, "chunk_idx": p.chunk_idx,
            })
        )?;
    }
    println!(
        "exported {} passages from '{}' -> {}",
        meta.doc_count,
        name,
        args.out.display()
    );
    Ok(())
}

/// `t3rag completions <shell>` — generate shell completions.
/// `t3rag completions`: emit shell completions to stdout.
pub fn run_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let mut out = std::io::stdout();
    clap_complete::generate(shell, &mut cmd, "t3rag", &mut out);
    Ok(())
}

/// `t3rag man <dir>` — generate man pages into a directory.
/// `t3rag man`: write one man page per subcommand into `dir`.
///
/// # Errors
/// `dir` cannot be created, or a page fails to write.
pub fn run_man(dir: &std::path::Path) -> Result<()> {
    let cmd = Cli::command();
    std::fs::create_dir_all(dir)?;
    clap_mangen::generate_to(cmd, dir)?;
    println!("man pages -> {}", dir.display());
    Ok(())
}

/// `t3rag publish` — upload a corpus dir to a HF dataset repo via the
/// `hf` CLI (t3-css.3.9). A README.md repo card is generated alongside
/// carrying model/dims/quant/provenance so `pull` consumers can check
/// compatibility. The corpus dir is self-contained (meta.json +
/// passages + vectors + bm25); tantivy/usearch index internals are NOT
/// covered by format_version — the card records the producing t3rag
/// version.
///
/// # Errors
/// The corpus isn't indexed, `hf` is missing, or the upload fails.
pub fn run_publish(args: crate::cli::PublishArgs) -> Result<()> {
    let root = paths::corpora_root()?;
    let name = args.corpus.unwrap_or(crate::serve::newest_corpus(&root)?);
    let dir = root.join(&name);
    let meta =
        crate::store::read_meta(&dir).with_context(|| format!("corpus '{name}' not indexed"))?;
    let card = format!(
        "---\ntags: [t3rag]\n---\n\
         # t3rag corpus `{name}`\n\n\
         Pre-built retrieval index. Install with `t3rag pull --repo {}`.\n\n\
         - t3rag: {}\n- model: {}\n- dims: {}\n- quantization: {}\n\
         - docs: {}\n- origin: {}\n",
        args.repo,
        env!("CARGO_PKG_VERSION"),
        meta.model,
        meta.dims,
        meta.quantization,
        meta.doc_count,
        meta.origin.as_deref().unwrap_or("unknown"),
    );
    let tmp = std::env::temp_dir().join(format!("t3rag-card-{name}.md"));
    std::fs::write(&tmp, card)?;
    let run = |a: &[&str]| -> Result<()> {
        let st = std::process::Command::new("hf")
            .args(a)
            .status()
            .context("run `hf` — is the Hugging Face CLI installed?")?;
        if !st.success() {
            anyhow::bail!("hf {:?} exited {}", a, st);
        }
        Ok(())
    };
    run(&[
        "upload",
        &args.repo,
        &dir.to_string_lossy(),
        "--repo-type",
        &args.repo_type,
    ])?;
    run(&[
        "upload",
        &args.repo,
        &tmp.to_string_lossy(),
        "README.md",
        "--repo-type",
        &args.repo_type,
    ])?;
    println!("published '{name}' -> {} ({})", dir.display(), args.repo);
    Ok(())
}

/// `t3rag pull` — download a published corpus into the corpora root so
/// `search`/`serve`/`mcp` use it unchanged. Refuses to overwrite an
/// existing corpus dir.
///
/// # Errors
/// The target exists, `hf` fails, or the download lacks `meta.json`.
pub fn run_pull(args: crate::cli::PullArgs) -> Result<()> {
    let root = paths::corpora_root()?;
    let name = args
        .name
        .clone()
        .or_else(|| args.repo.rsplit('/').next().map(String::from))
        .context("--name required when repo has no name component")?;
    let dir = root.join(&name);
    if dir.exists() {
        anyhow::bail!("corpus '{name}' already exists at {}", dir.display());
    }
    let st = std::process::Command::new("hf")
        .args([
            "download",
            &args.repo,
            "--repo-type",
            &args.repo_type,
            "--local-dir",
            &dir.to_string_lossy(),
        ])
        .status()
        .context("run `hf` — is the Hugging Face CLI installed?")?;
    if !st.success() {
        anyhow::bail!("hf download exited {st}");
    }
    crate::store::read_meta(&dir)
        .with_context(|| format!("{} has no valid meta.json — not a t3rag corpus?", args.repo))?;
    println!("pulled '{}' -> {}", name, dir.display());
    Ok(())
}

/// `t3rag show <corpus> <id>` — print one passage in full (t3-vqd):
/// text + question/source/title + chunk linkage. Pure store read — no
/// model, no daemon required.
///
/// # Errors
/// The corpus isn't indexed, or `id` is out of range.
pub fn run_show(corpus: Option<String>, id: u64) -> Result<()> {
    let root = paths::corpora_root()?;
    let name = corpus.unwrap_or(crate::serve::newest_corpus(&root)?);
    let dir = root.join(&name);
    let reader = crate::store::PassageReader::open(&dir)
        .with_context(|| format!("corpus '{name}' not indexed"))?;
    let p = reader.get(id)?;
    println!("id:        {}", p.id);
    if let Some(t) = &p.title {
        println!("title:     {t}");
    }
    if let Some(q) = &p.question {
        println!("question:  {q}");
    }
    if let Some(s) = &p.source {
        println!("source:    {s}");
    }
    if p.parent_id != p.id {
        println!("parent:    {} (chunk {})", p.parent_id, p.chunk_idx);
    }
    println!("text:\n{}", p.text);
    Ok(())
}

/// `t3rag feedback` — append a relevance verdict to
/// `~/.cache/t3rag/feedback.jsonl` (t3-bj5): the accretion substrate that
/// `insights`/`eval --history` aggregate later.
///
/// # Errors
/// The cache root can't be created or the file can't be written.
pub fn run_feedback(args: crate::cli::FeedbackArgs) -> Result<()> {
    if args.ids.is_empty() {
        anyhow::bail!("--ids is required (passage ids the verdict applies to)");
    }
    let root = paths::cache_root()?;
    std::fs::create_dir_all(&root)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("feedback.jsonl"))?;
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    writeln!(
        f,
        "{}",
        serde_json::json!({
            "ts": ts,
            "query": args.query,
            "passage_ids": args.ids,
            "verdict": format!("{:?}", args.verdict).to_lowercase(),
            "corpus": args.corpus,
        })
    )?;
    println!(
        "recorded {:?} verdict on {} passage(s)",
        args.verdict,
        args.ids.len()
    );
    Ok(())
}

/// `t3rag suggest "<query>"` — corpus routing (t3-69w): one embed via the
/// daemon's `/suggest`, per-corpus {transform, docs, top_terms,
/// similarity?}. Type+stats lead — similarity is a secondary hint (T3
/// views share a problem space). Without a daemon, prints the
/// manifest-only table (no embedding, no similarity — honest degradation).
///
/// # Errors
/// The daemon can't be reached and manifest-only output fails too.
pub fn run_suggest(query: &str, json: bool) -> Result<()> {
    if let Some(base) = crate::serve::discover_daemon() {
        let mut resp = ureq::get(&format!("{base}/suggest"))
            .query("q", query)
            .call()
            .context("daemon /suggest")?;
        let v: serde_json::Value = resp.body_mut().read_json().context("parse")?;
        if json {
            println!("{}", serde_json::to_string_pretty(&v)?);
            return Ok(());
        }
        println!("corpus           transform   docs   similarity  top-terms");
        for r in v["corpora"].as_array().iter().flat_map(|a| a.iter()) {
            let sim = r["similarity"]
                .as_f64()
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "n/a".into());
            let terms = r["top_terms"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|t| t.as_str())
                        .take(4)
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            println!(
                "{:<16} {:<10} {:>5}  {:>9}  {}",
                r["corpus"].as_str().unwrap_or("?"),
                r["transform"].as_str().unwrap_or("?"),
                r["docs"].as_u64().unwrap_or(0),
                sim,
                terms
            );
        }
        println!("\nroute: --corpus <name[,name]> on the next search");
        return Ok(());
    }
    // No daemon: manifest-only (no embed) — honest degradation.
    println!("corpus           transform   docs   (no daemon — no similarity)");
    let rd = std::fs::read_dir(paths::corpora_root()?)?;
    for e in rd.flatten() {
        if let Ok(meta) = crate::store::read_meta(&e.path()) {
            let m = meta.manifest.as_ref();
            println!(
                "{:<16} {:<10} {:>5}",
                e.file_name().to_string_lossy(),
                m.map(|m| m.transform.as_str()).unwrap_or("unknown"),
                meta.doc_count,
            );
        }
    }
    Ok(())
}

/// `t3rag insights` — aggregate log.jsonl + feedback.jsonl (t3-qc4):
/// per-corpus query volume, zero-result queries (coverage gaps), verdict
/// distribution, latency. Reports only what the logs actually record —
/// feature-usage telemetry needs the log schema to grow a `flags` field.
///
/// # Errors
/// The cache root can't be read.
pub fn run_insights(since_hours: Option<u64>, json_out: bool) -> Result<()> {
    let root = paths::cache_root()?;
    let cutoff = since_hours.map(|h| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(h * 3600)
    });
    let read_log = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(root.join(name))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    };
    let parse_ts = |v: &serde_json::Value| -> u64 {
        // ts is RFC3339 or epoch secs; cheap parse: digits → epoch, else 0.
        match &v["ts"] {
            serde_json::Value::Number(n) => n.as_u64().unwrap_or(0),
            serde_json::Value::String(s) => s
                .chars()
                .filter(|c| c.is_ascii_digit())
                .take(10)
                .collect::<String>()
                .parse()
                .unwrap_or(0),
            _ => 0,
        }
    };
    let logs: Vec<_> = read_log("log.jsonl")
        .into_iter()
        .filter(|v| cutoff.is_none_or(|c| parse_ts(v) >= c))
        .collect();
    let fb: Vec<_> = read_log("feedback.jsonl");

    let mut per_corpus: std::collections::BTreeMap<String, u64> = Default::default();
    let mut zero_hits = 0u64;
    let mut zero_queries: Vec<String> = Vec::new();
    let mut total_ms = 0f64;
    let mut clients: std::collections::BTreeMap<String, u64> = Default::default();
    for v in &logs {
        for c in v["corpora"].as_str().unwrap_or("").split(',') {
            if !c.is_empty() {
                *per_corpus.entry(c.into()).or_default() += 1;
            }
        }
        *clients
            .entry(v["client"].as_str().unwrap_or("?").into())
            .or_default() += 1;
        total_ms += v["took_ms"].as_f64().unwrap_or(0.0);
        if v["top_ids"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(true)
        {
            zero_hits += 1;
            if let Some(q) = v["query"].as_str() {
                zero_queries.push(q.into());
            }
        }
    }
    let mut verdicts: std::collections::BTreeMap<String, u64> = Default::default();
    for v in &fb {
        *verdicts
            .entry(v["verdict"].as_str().unwrap_or("?").into())
            .or_default() += 1;
    }
    let report = serde_json::json!({
        "queries": logs.len(),
        "feedback": fb.len(),
        "per_corpus": per_corpus,
        "per_client": clients,
        "zero_result_queries": zero_hits,
        "top_zero_hit_queries": zero_queries.iter().take(10).collect::<Vec<_>>(),
        "verdicts": verdicts,
        "avg_took_ms": if logs.is_empty() { 0.0 } else { total_ms / logs.len() as f64 },
    });
    if json_out {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "queries: {}  (zero-result: {})  avg: {:.0}ms",
            report["queries"], report["zero_result_queries"], report["avg_took_ms"]
        );
        for (c, n) in &per_corpus {
            println!("  {c:<16} {n}");
        }
        if !verdicts.is_empty() {
            println!("verdicts: {verdicts:?}");
        }
        if !zero_queries.is_empty() {
            println!("zero-hit queries:");
            for q in zero_queries.iter().take(10) {
                println!("  - {q}");
            }
        }
    }
    Ok(())
}

/// `t3rag selftest` — alias-level sanity (same as `health`, plus a clap
/// parse of the full arg surface so the help tree itself is proven).
pub fn run_selftest() -> Result<()> {
    let _ = ModelSpec::qwen3_embedding_0_6b(); // registry default parses
    Cli::command().debug_assert();
    println!("OK");
    Ok(())
}
