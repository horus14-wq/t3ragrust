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
pub const SKILL_MD: &str = r##"---
name: t3rag
description: Retrieve T3 thinking traces — how prior problems were reasoned about — via local hybrid search (dense ANN + BM25 + cross-encoder rerank). Use before tackling non-trivial problems where prior reasoning experience helps.
---

# t3rag — thinking-trace retrieval

Rust-native RAG over T3 thinking traces. Local ONNX embeddings
(Qwen3-Embedding-0.6B) + usearch ANN + tantivy BM25 + RRF fusion +
optional BGE-v2-m3 rerank. No API keys, no Python.

## The loop (do all six steps)

1. **Orient** — `t3rag health` (is anything up?), `t3rag list` (which
   corpora exist, their transform `type`, staleness `!`),
   `t3rag suggest "<problem>"` (ranked corpora for THIS query — type +
   stats lead, centroid similarity is only a hint).
2. **Formulate** — query the *problem domain*, not the solution. For
   pitfalls/failures use failure language ("common mistakes",
   "what goes wrong"); for approach use how-to phrasing; for the core
   idea use conceptual phrasing. See `references/query-craft.md`.
3. **Retrieve** — `t3rag search "<query>" --format toon` (one line per
   hit, agent-token-optimal). Default searches the newest corpus;
   pass `--corpus <name>` or `--corpus all` once `suggest` tells you
   which views matter. MCP: `search_traces`.
4. **Inspect** — `t3rag show <passage_id>` (or MCP `get_passage`)
   before citing: scores rank within a result set, they do NOT measure
   absolute relevance. Never threshold a score; read `meta` +
   `warnings[]` in `--format json` output instead.
5. **Iterate** — miss or thin results? `--expand` (PRF: blends top hits
   into the query), `--like <id>` / `--like-text "..."` (more-like-this
   neighborhood drill-down, dense-only), `--exclude <id,id,...>`
   (drop seen hits and re-fill), `--group-by question` (collapse
   struct/reflect/semantic views of the same problem), or a second,
   sharper query. `t3rag verify --corpus <name>` if results look wrong.
6. **Hydrate + cite** — cite the passage `source` field (file:line or
   `…#src_idx=N` back to the raw corpus). Chunked long passages return
   whole parents already; `--token-budget N` caps injected context.

## Query craft for traces

- Query the PROBLEM, not the answer: "off-by-one in binary search
  bounds" retrieves better than pasting your candidate fix.
- Want failure knowledge? Ask for it: "common mistakes in …",
  "what goes wrong when …" → routes to `*_reflect` corpora.
- Scores are relative within one result set — a 0.03 here and a 0.03
  there are unrelated. Judge by rank + reading `show` output, never by
  absolute thresholds. Check `warnings[]` (`leg_disagreement`,
  `token_budget_exhausted`, truncation) — they carry the real signal.
- One sharp query beats one mega-query; refine after a miss.

## Worked examples

Pitfalls lookup:
```bash
t3rag suggest "rust borrow checker lifetime errors"   # → reflect
t3rag search "common mistakes rust lifetimes borrow checker" \
    --corpus reflect --format toon --top-k 5
t3rag show 4217 --corpus reflect                     # read before citing
```

Neighborhood drill-down after a good hit:
```bash
t3rag search "tokio spawn_blocking vs block_in_place" \
    --corpus semantic --format toon
t3rag search --like 512 --corpus semantic --format toon   # dense-only neighbors
```

Dedup across transform views:
```bash
t3rag search "CRDT OR-set tombstones" --corpus all \
    --group-by question --format toon
```

## MCP tools

`search_traces` (query, corpus, top_k, rerank, like, like_text, exclude,
group_by, expand, token_budget, format…), `suggest`, `list_corpora`,
`batch_search` (queries[] → array of result sets), `get_passage`
(corpus+id → full text, no model needed). MCP also exposes prompts
(`t3rag-playbook`) and per-corpus resources.

### Register (daemon HTTP MCP — shared, warm, recommended)

```json
{ "mcpServers": { "t3rag": { "url": "http://127.0.0.1:7317/mcp" } } }
```

Start the daemon once: `t3rag serve` (the stdio proxy also spawns it in
the background when absent).

Client dialects: VS Code `.vscode/mcp.json` uses
`{"servers": {"t3rag": {"type": "http", "url": "…"}}}`; Codex/Grok TOML
uses `[mcp_servers.t3rag] url = "…"`; `claude mcp add --transport http
t3rag http://127.0.0.1:7317/mcp` for Claude Code.

### Register (stdio proxy — for MCP clients without HTTP)

```json
{ "mcpServers": { "t3rag": { "command": "t3rag", "args": ["mcp"] } } }
```

The stdio server NEVER loads the model itself — it proxies to the
daemon. Handshake (`initialize`, `tools/list`, `ping`) answers
immediately without the daemon; a cold daemon spawns in the background
and only `tools/call` waits on it. Both NDJSON and `Content-Length`
framing are auto-detected; logs go to stderr only.

## CLI quick reference

```bash
t3rag index --input corpus.jsonl --name mycorpus   # build (auto: model+daemon)
t3rag search "query" --corpus mycorpus --format toon
t3rag search "query" --no-daemon                   # cold in-process path
t3rag search "query" --no-rerank                   # fused order, no rerank
t3rag search "query" --exact                       # brute-force (recall ref)
t3rag ask "query"                                   # prompt-injection block
t3rag suggest "query"                               # corpus routing
t3rag show N [--corpus C]                            # full passage text
t3rag list / stats / verify / doctor [--deep]       # introspection
t3rag feedback --ids 1,2 --verdict relevant         # accrete: log verdicts
t3rag insights                                      # usage aggregates
```

Input formats: JSONL (`text`/`output` + optional `question`/`title`/`source`),
directory of .md/.txt, or .parquet (auto-detects text column, `--text-col`).

## Output

`--format json` emits `{query, corpus, meta, warnings[], results:[…]}` —
`meta` carries corpus/version info, `warnings[]` flags truncation,
leg disagreement, budget exhaustion. Hits hydrate to whole parents
(chunked long passages collapse back to their source doc).
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

/// References shipped alongside SKILL.md (t3-css.3.8, expanded t3-w2f):
/// the query-craft lessons that make agents retrieve well.
const QUERY_CRAFT: &str = "# Query craft\n\
- Query the PROBLEM DOMAIN, not the solution — describe the situation,\n\
  symptom, or constraint; never paste your candidate fix as the query.\n\
- Failure/pitfall intent → failure language: \"common mistakes in …\",\n\
  \"what goes wrong when …\", \"why does … fail\". Approach intent → how-to\n\
  phrasing (\"how to …\", \"steps to …\"). Conceptual intent → the idea\n\
  itself (\"what is the core of …\"). `t3rag suggest` uses the same signal.\n\
- Mega-queries fail; sharp sub-problems retrieve. One failing symptom,\n\
  API, or constraint per query — refine after a miss, don't broaden.\n\
- Mid-project queries (`this error, this API`) outperform greenfield\n\
  prompts on these corpora.\n\
- Scores are RELATIVE within one result set — fused/rerank scores are\n\
  not calibrated across corpora or queries. Rank + `t3rag show` beat any\n\
  threshold. `warnings[]` in JSON output carries leg_disagreement,\n\
  truncation, and budget signals — read them before trusting a hit.\n\
- Iterate cheaply: `--expand` (PRF) for recall misses, `--like <id>` for\n\
  neighborhood drill-down, `--exclude <id,...>` to re-fill after reading,\n\
  `--group-by question` to collapse the three transform views of one\n\
  problem into a single row.\n\
- When NOT to use: trivial lookups (single fact, single file) — go read\n\
  the file.\n";

const CORPORA: &str = "# Corpus naming + routing\n\
- Transform views share the same problem space: `*_struct` = procedural\n\
  cheatsheets, `*_reflect` = mistakes/critical-checks, `*_semantic` =\n\
  compressed core insight, `*_raw` = full episode bodies. They are\n\
  complementary lenses, not competitors — `--corpus all` +\n\
  `--group-by question` often beats picking one.\n\
- `t3rag suggest \"<query>\"` ranks corpora before you search; treat its\n\
  ordering as a prior, not a verdict (centroid similarity separates the\n\
  T3 views poorly by design).\n\
- `t3rag list` shows each corpus's transform `type` and flags stale\n\
  indexes with `!` (source input newer than the index — rebuild before\n\
  trusting results). `t3rag verify --corpus <name>` checks integrity.\n\
- ≤1 hit per source (`--max-per-source 1`): keeps one episode from\n\
  dominating the top-k.\n\
- Keep `question` fields lean in ingested JSONL — preamble-polluted\n\
  questions poison their retrieval neighborhood.\n";

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

    /// t3-w2f AC: spec frontmatter (name + description) and the full
    /// orient→cite loop are present in the emitted SKILL.md.
    #[test]
    fn skill_md_has_frontmatter_and_playbook() {
        let md = super::SKILL_MD;
        assert!(md.starts_with("---\nname: t3rag\n"), "frontmatter");
        assert!(md.contains("description:"), "frontmatter description");
        for step in [
            "Orient",
            "Formulate",
            "Retrieve",
            "Inspect",
            "Iterate",
            "Hydrate",
        ] {
            assert!(md.contains(step), "playbook step {step}");
        }
        // Query craft + iteration flags the playbook teaches.
        for needle in [
            "--format toon",
            "--like",
            "--exclude",
            "--expand",
            "--group-by",
            "suggest",
            "show",
            "warnings[]",
        ] {
            assert!(md.contains(needle), "missing {needle}");
        }
    }

    /// t3-w2f AC: --install into a fresh dir produces the spec layout
    /// (<dir>/t3rag/SKILL.md + references/*).
    #[test]
    fn install_writes_spec_layout() {
        let dir = std::env::temp_dir().join(format!("t3rag-skill-test-{}", std::process::id()));
        super::install(&dir, true).expect("install");
        let skill_md = dir.join("t3rag/SKILL.md");
        assert!(skill_md.exists(), "SKILL.md missing");
        let body = std::fs::read_to_string(&skill_md).unwrap();
        assert!(body.starts_with("---\nname: t3rag"));
        for r in ["query-craft.md", "corpora.md"] {
            assert!(dir.join("t3rag/references").join(r).exists(), "{r} missing");
        }
    }
}
