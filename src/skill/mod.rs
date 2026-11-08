//! SKILL.md emitter (t3-wkp.15).
//!
//! `t3rag skill` prints a skill wrapper for agent discovery (what t3rag is,
//! the `search_traces` MCP tool + config snippets, CLI equivalents,
//! when-to-use guidance); `--install <dir>` writes it into a skills dir.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::cli::SkillArgs;

/// The SKILL.md content: self-contained agent-facing description of t3rag.
/// Frontmatter satisfies the Agent Skills spec (`name` + `description`
/// are required — without it spec-compliant loaders ignore the file).
const SKILL_MD: &str = r##"---
name: t3rag
description: Retrieve T3 thinking traces — how prior problems were reasoned about — via local hybrid search (dense ANN + BM25 + cross-encoder rerank). Use before tackling non-trivial problems where prior reasoning experience helps.
---

# t3rag — thinking-trace retrieval

Rust-native RAG over T3 thinking traces. Local ONNX embeddings
(Qwen3-Embedding-0.6B) + usearch ANN + tantivy BM25 + RRF fusion +
optional BGE-v2-m3 rerank. No API keys, no Python.

## When to use

Use t3rag BEFORE solving a non-trivial problem when prior reasoning
experience might help: similar bugs, algorithm patterns, debugging
playbooks, structural approaches. Skip it for trivial lookups — it
returns *how problems were reasoned about*, not factual docs.

## Query craft (the highest-leverage skill)

Prototype lesson that still holds: **mega-queries fail; sharp
sub-problems retrieve.** Query for ONE failing symptom, API, or
constraint — "PowerShell ValidateSet + splatting", not "build my whole
app". On a miss, refine with a second sharper query. Corpora are named
per transform — `*_raw` (episode bodies), `*_struct` (cheatsheets),
`*_reflect` (mistakes), `*_semantic` (insights); mid-debug favors raw,
planning favors struct/semantic, "what goes wrong" favors reflect.

## MCP tool

`search_traces` — params:
- `query` (string, required): the problem/question to find traces for
- `top_k` (int, default 3)
- `corpus` (string, optional): corpus name (e.g. `reflect`, `construct`,
  `semantic`); default = most recently indexed
- `rerank` (bool, default true)

### Register (daemon HTTP MCP — shared, warm, recommended)

```json
{ "mcpServers": { "t3rag": { "url": "http://127.0.0.1:7317/mcp" } } }
```

Start the daemon once: `t3rag serve` (auto-spawn also happens on first
stdio-proxy call).

### Register (stdio proxy — for MCP clients without HTTP)

```json
{ "mcpServers": { "t3rag": { "command": "t3rag", "args": ["mcp"] } } }
```

The stdio server NEVER loads the model itself — it proxies to the
daemon, spawning it if absent.

## CLI equivalents

```bash
t3rag index --input corpus.jsonl --name mycorpus   # build a corpus
t3rag search "query text" --corpus mycorpus        # query (daemon if up)
t3rag search "query" --no-daemon                   # cold in-process path
t3rag search "query" --no-rerank                   # fused order, no rerank
t3rag search "query" --exact                       # brute-force (recall ref)
```

Input formats: JSONL (`text`/`output` + optional `question`/`title`/`source`),
directory of .md/.txt, or .parquet (auto-detects text column, `--text-col`).

## Output

Top-k passages with fused/rerank scores, hydrated to whole parents
(chunked long passages collapse back to their source doc). `--json`
emits `{query, corpus, results:[{rank,passage_id,score,text,title,
question,source}]}`.
"##;

/// `t3rag skill` entry point: print, install, or uninstall the
/// SKILL.md wrapper. Spec layout: `<skills-dir>/t3rag/SKILL.md` — the
/// bare `SKILL.md`-at-root form is what spec-loaders reject.
/// skills root is ignored by spec-compliant loaders).
pub fn run_skill(args: SkillArgs) -> Result<()> {
    let target = match (args.install.as_ref(), args.agents) {
        (Some(dir), _) => dir.clone(),
        (None, Some(agents)) => PathBuf::from(agents.skills_dir()),
        (None, None) => {
            print!("{SKILL_MD}");
            return Ok(());
        }
    };
    if args.uninstall {
        return uninstall(&target, args.yes);
    }
    install(&target, args.force)
}

/// References shipped alongside SKILL.md (t3-css.3.8): the query-craft
/// lessons from the prototype that make agents retrieve well.
const QUERY_CRAFT: &str = "# Query craft (prototype lessons)\n\
- Query for ONE failing symptom/API/constraint — `\"PowerShell ValidateSet + splatting\"`, not `\"build my app\"`. Mega-queries fail; sharp sub-problems retrieve.\n\
- Mid-project queries (`this error, this API`) outperform greenfield prompts on these corpora.\n\
- Refine after a miss — a second sharper query beats a bigger first one.\n\
- Query by intent → corpus: debugging → `*_raw`, planning → `*_struct`/`*_semantic`, `what goes wrong?` → `*_reflect`.\n\
- When NOT to use: trivial lookups (single fact, single file) — go read the file.\n";

const CORPORA: &str = "# Corpus naming + diversity\n\
- Per-transform corpora: `fable5_v2_{raw,struct,reflect,semantic}` — search them together with `--corpus all` or a comma list.\n\
- ≤1 hit per source (`--max-per-source 1`): keeps one episode from dominating the top-k.\n\
- Keep `question` fields lean in ingested JSONL — preamble-polluted questions poison their retrieval neighborhood.\n";

fn install(dir: &Path, force: bool) -> Result<()> {
    let skill_dir = dir.join("t3rag");
    std::fs::create_dir_all(&skill_dir)
        .with_context(|| format!("create {}", skill_dir.display()))?;
    let path = skill_dir.join("SKILL.md");
    if path.exists() && !force {
        bail!("{} exists — pass --force to overwrite", path.display());
    }
    std::fs::write(&path, SKILL_MD).with_context(|| format!("write {}", path.display()))?;
    // references/ (t3-css.3.8): query-craft + corpus-strategy notes.
    let refs = skill_dir.join("references");
    std::fs::create_dir_all(&refs)?;
    std::fs::write(refs.join("query-craft.md"), QUERY_CRAFT)?;
    std::fs::write(refs.join("corpora.md"), CORPORA)?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Guardrailed uninstall (t3-css.3.8): prints the exact target,
/// requires `--yes`, removes only `<skills-dir>/t3rag/` — never the
/// skills dir itself, never outside it.
fn uninstall(dir: &Path, yes: bool) -> Result<()> {
    let skill_dir = dir.join("t3rag");
    if !skill_dir.join("SKILL.md").exists() {
        bail!("no t3rag skill installed under {}", skill_dir.display());
    }
    println!("about to remove {}", skill_dir.display());
    if !yes {
        bail!("refusing without --yes");
    }
    std::fs::remove_dir_all(&skill_dir)
        .with_context(|| format!("remove {}", skill_dir.display()))?;
    println!("removed {}", skill_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn skill_md_documents_the_tool() {
        assert!(super::SKILL_MD.contains("search_traces"));
        assert!(super::SKILL_MD.contains("t3rag serve"));
        assert!(super::SKILL_MD.contains("mcpServers"));
    }
}
