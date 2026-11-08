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

/// `t3rag selftest` — alias-level sanity (same as `health`, plus a clap
/// parse of the full arg surface so the help tree itself is proven).
pub fn run_selftest() -> Result<()> {
    let _ = ModelSpec::qwen3_embedding_0_6b(); // registry default parses
    Cli::command().debug_assert();
    println!("OK");
    Ok(())
}
