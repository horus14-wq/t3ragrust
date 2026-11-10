# t3rag — Agent & User Guide

A Rust-native retrieval system over T3 thinking/reasoning traces, built for
coding agents and LLMs. Local embeddings, hybrid retrieval, a resident daemon,
HTTP + MCP access — no API keys, no hosted services required.

---

## 1. Mental model

```
query ──► embed ──► [dense ANN] ─┐
                 └─► [BM25]     ─┼─► RRF fusion ──► [optional reranker] ──► hydrate ──► results
                 └─► [SPLADE]*  ─┘                                        (parent passages)
                (*optional, feature-gated)
```

- **Corpora** are immutable indexed artifact dirs under
  `~/.cache/t3rag/corpora/<name>/` (`passages.bin/.idx`, `vectors.usearch`,
  `vectors.raw.f32`, `bm25/`, `sparse.bin`, `meta.json`). `T3RAG_HOME`
  relocates everything.
- **The daemon** (`t3rag serve`) keeps models + indexes resident per user.
  The CLI auto-discovers it via `daemon.json` and auto-spawns it when needed.
  Without a daemon, `search` falls back to a cold in-process path (slow —
  model load every time).
- **Chunked passages**: long passages are stored whole (the parent) and
  indexed as chunks; hits hydrate back to the parent automatically.
- **Response shape**: every search returns `results`, `meta` (corpora,
  model, dims, quantization, preset, index age, elapsed), and `warnings[]`
  (truncation, staleness, missing legs, PRF seeds). Errors are structured:
  `{error: {code, message, hint, retryable}}`.

---

## 2. Quick start

```bash
# 1. Index a corpus (JSONL / markdown dir / parquet)
t3rag index --input traces.jsonl --name mytraces

# 2. Search (auto-spawns the daemon on first use)
t3rag search "binary search on a monotone predicate" --corpus mytraces

# 3. Fetch one passage in full — no model needed
t3rag show --corpus mytraces 42
```

Useful housekeeping: `t3rag list`, `t3rag stats`, `t3rag doctor`,
`t3rag models` (embedding-model registry), `t3rag capabilities`.

JSONL ingest needs any one text field: `text` | `output` | `content` |
`passage` | `document` (`question`/`title`/`source` optional) — generic
document dumps index fine, not just T3 trace records.

Embedding models: `--model` accepts any `owner/name` HF repo. The
registry carries per-family instructions + pooling (last-token for
Qwen3, CLS for BGE, masked-mean for e5/GTE/jina); unregistered repos
get generic defaults, override with `T3RAG_POOLING=last|mean|cls`.
The query-side instruction is `--instruction` / `T3RAG_INSTRUCTION`.

---

## 3. The agent loop (recommended workflow)

The system is designed for an iterative retrieve–inspect–refine loop.
Five actions cover most use:

```bash
# 1. ORIENT — which corpus should this query hit?
t3rag suggest "lock-free queue race condition"

# 2. RETRIEVE — fused multi-corpus search
t3rag search "lock-free queue race condition" \
     --corpus construct,reflect --top-k 5

# 3. INSPECT — hydrate a promising hit fully (no model load)
t3rag show --corpus construct 865

# 4. ITERATE — neighbors of a hit, minus rejects
t3rag search "lock-free queue" --corpus construct \
     --like 865 --exclude 196 --top-k 3

# 5. RECORD — leave a verdict; compounds into eval data
t3rag feedback --query "lock-free queue race" --ids 865 \
     --verdict relevant --corpus construct
t3rag insights          # usage/quality roll-up over log + feedback
```

MCP-connected agents get the same loop as tools: `suggest` →
`search_traces` → `get_passage` → `search_traces(like, exclude)` →
`batch_search`, plus `list_corpora` for orientation. The `t3rag-playbook`
MCP prompt teaches this in-band; `t3rag://corpus/<name>` resources expose
corpus manifests.

---

## 4. Search flags (the iterative surface)

| Flag | What it does |
|---|---|
| `--corpus a,b / all` | Multi-corpus query, merged by RRF (daemon path) |
| `--top-k N`, `--offset N` | Page through the ranking |
| `--candidates N` | Per-leg candidate pool before fusion (default 50) |
| `--preset fast\|balanced\|max` | Quality/latency bundle |
| `--rerank` / `--no-rerank`, `--reranker NAME`, `--rerank-candidates N` | Cross-encoder stage — **off by default** (biggest warm cost). `--balanced`/`--max` presets enable it; `--no-rerank` overrides a preset |
| `--also "phrasing"` (repeatable) | Multi-query fusion — extra phrasings fused into ONE ranking |
| `--like ID`, `--like-text "…"` | More-like-this (dense-only; seed excluded) |
| `--exclude ID,ID` | Subtract hits pre-top-k (iterative refinement) |
| `--expand` | Pseudo-relevance feedback: top-3 vectors blended 0.7q+0.3·hits, dense leg re-run; `expanded_from` in warnings |
| `--group-by question\|source\|corpus` | Bucket hits — reveals the T3 lens structure |
| `--token-budget N` | Stop filling once ~N tokens of text returned (chars/4) |
| `--batch FILE` | JSONL of independent queries → one result object per line |
| `--filter "key~substr"` | Post-filter on source/title/question |
| `--max-per-source N` | Per-source diversity cap |
| `--min-score F`, `--max-chars N` | Score floor / text truncation (warns) |
| `--exact` | Brute-force scan (recall oracle; needs `vectors.raw.f32`) |
| `--explain` | Per-leg scores on every hit + `stages_ms` timings |
| `--ef-search N`, `--rrf-k F` | ANN recall dial / fusion smoothing |
| `--sparse` | Add SPLADE leg (needs `--features sparse` build + `sparse.bin`) |
| `--format toon\|text\|md\|json\|jsonl\|prompt` | `prompt` emits `<retrieved-traces>` blocks for context injection |
| `--no-daemon`, `--daemon auto\|always\|never` | Path control; batch/group/expand/also/like need the daemon |

## 5. Index flags (build-time levers)

`--dims` (MRL truncation), `--quantization i8|bf16|f16|fp32`, `--model`,
`--instruction`, `--embed-batch`, `--embed-token-budget`, `--pool`,
`--threads`, `--ep cpu|cuda|openvino|webgpu|directml|auto`,
`--min-chars`/`--max-chars` (ingest filters), `--dedup` (exact-hash),
`--no-chunk` / `--always-chunk` / `--chunk-tokens` / `--chunk-chars`,
`--batch-size` (parquet), `--bm25-heap-mb`, `--bm25-threads`,
`--connectivity`, `--expansion-add` (HNSW params),
`--no-raw-vectors` (halves disk; `--exact` then fails honestly),
`--sparse`, `--dry-run`, `--progress`/`--no-progress` (indicatif).

## 6. Daemon, HTTP, MCP

- `t3rag serve [--port 7317 --pool N --idle-timeout S]` — resident daemon.
- HTTP: `POST /search`, `POST /ask`, `GET /passages/{corpus}/{id}`,
  `GET /corpora`, `GET /suggest?q=`, `GET /health`, `GET /metrics`,
  `POST /mcp` (JSON-RPC), `GET /models`, `GET /capabilities`.
- MCP stdio: `t3rag mcp` (proxies the daemon — never loads models
  itself). Tools: `search_traces`, `get_passage`, `batch_search`,
  `list_corpora`, `suggest`. Prompts: `t3rag-playbook`. Resources:
  `t3rag://corpus/<name>`.
- Install integration: `t3rag skill --install <dir>` writes a spec-layout
  `SKILL.md`; `t3rag mcp` wires into any MCP client.

### MCP transport guide (GH#3)

Two registrations cover every client — prefer the URL whenever the client
accepts one:

| Shape | Config | Use when |
|-------|--------|----------|
| **Streamable HTTP** (recommended) | `{"mcpServers": {"t3rag": {"url": "http://127.0.0.1:7317/mcp"}}}` | Any client accepting a URL — models stay warm, handshake is instant, many clients share one daemon |
| **stdio** | `{"mcpServers": {"t3rag": {"command": "t3rag", "args": ["mcp"]}}}` | Subprocess-only clients (Claude Desktop, `claude mcp add`, Codex TOML, etc.) |

Stdio notes:

- `t3rag mcp` auto-detects **NDJSON and Content-Length** framing and
  replies in kind — both dialects work.
- `initialize`/`tools/list`/`ping` never touch the daemon — they answer
  immediately even with no daemon running; the daemon is spawned in the
  background at startup and only a `tools/call` (or a corpus-backed
  prompt/resource) waits on it.
- All logging goes to stderr; stdout carries only protocol frames.
- If stdio times out on a cold box, register the HTTP URL instead — the
  daemon only needs to be running once (`t3rag serve`).

Client-specific snippets:

```toml
# Codex CLI / Grok Build — ~/.codex/config.toml (or ~/.grok/config.toml)
[mcp_servers.t3rag]
url = "http://127.0.0.1:7317/mcp"
```

```json
// VS Code — .vscode/mcp.json
{ "servers": { "t3rag": { "type": "http", "url": "http://127.0.0.1:7317/mcp" } } }
```

```bash
# Claude Code
claude mcp add --transport http t3rag http://127.0.0.1:7317/mcp
# or stdio: claude mcp add t3rag -- t3rag mcp
```

## 7. Corpus lifecycle

```bash
t3rag list                 # names, doc counts, loaded state
t3rag stats                # per-corpus details + removal recipe
t3rag publish --name c --repo user/ds   # upload via `hf` CLI (card generated)
t3rag publish --name c --repo user/ds --subdir c   # upload into repo subdir (multi-corpus repos)
t3rag pull --repo user/ds --name c      # download + validate meta.json
t3rag pull --repo user/ds --subdir c    # pull one subdir (local name defaults to subdir basename)
```

Prebuilt corpora are published at
`horus14-wq/fable5-v2-coding-t3-indexes` — four subdirectories
(`struct`, `semantic`, `reflect`, `raw-chunked`) indexing 52K+ coding
reasoning traces each (Qwen3-Embedding-0.6B, i8):

```bash
t3rag pull --repo horus14-wq/fable5-v2-coding-t3-indexes --subdir reflect
t3rag search "async deadlock debugging" --corpus struct,reflect
```

There is deliberately **no `t3rag forget`** — deletion stays a user-run
`rm -rf ~/.cache/t3rag/corpora/<name>` so no agent-reachable primitive
can destroy indexes.

## 8. Measured performance (this system)

| Metric | fixture (25 docs) | construct (2744 docs) |
|---|---|---|
| embed | ~0 ms (query LRU) | ~0 ms |
| dense ANN | 0.1 ms | 0.8 ms |
| bm25 | 1.4 ms | 5.3 ms (p99 29 ms) |
| hydrate | 0.1 ms | 0.1 ms |
| **e2e (warm)** | **p50 14 ms / p95 79 ms** | **p50 12 ms / p95 52 ms** |
| storage | 256 KB | 22 MB |

- **Cold start**: ~40–110 s (daemon spawn + model load) — pay once per
  session; the daemon's whole reason to exist.
- **Reranker** is opt-in (default off) because it's the warm bottleneck:
  ~1–2 s/query over 50 candidates on an idle CPU, and ~80 s+ measured on a
  box at load-average 16. Enable with `--rerank` or the `--balanced`/`--max`
  presets; `--rerank-candidates` trims the pool it scores.
  **EP note:** on NVIDIA, run the daemon with `--ep cuda` — DirectML
  executes the cross-encoder but measured ~3–5 s/pair (slower than CPU;
  GPU "busy" at low watts — soft DML scheduling). **Model note:**
  `minilm` is ~15–20× faster than `bge-m3` but is a 22M-param web-QA
  ranker — expect weaker ordering on T3 traces; `bge-base` is the sane
  middle. Daemon knobs: `--rerank-sessions N` (shard candidate pairs
  across N sessions; memory ×N), `--rerank-tokens` (512 → 256 saves ~2×).
- **Indexing** is offline-scale: a 25-doc index ≈ 4 min (mostly model
  load); a 500-doc slice ≈ 38 min. Queries are the cheap part by design.

## 9. Honest limitations

- `--sparse` needs a `--features sparse` daemon build; otherwise it warns
  `sparse_leg_missing` and degrades (never fakes a leg).
- Corpora indexed before manifests report `transform=unknown` /
  `similarity=n/a` in `suggest`/`list` — re-index to populate.
- `suggest` similarity is a *secondary* hint: T3 views share a problem
  space, so centroids separate weakly. Type + stats are the routing signal.
- `--expand`/PRF showed zero recall delta on the 25-doc fixture (top-3 was
  already the answer); treat it as an opt-in lever, not a default.
- `--exact` requires `vectors.raw.f32` — `--no-raw-vectors` indexes can't
  exact-scan (error says so).
- Chunk trigger uses an est-token heuristic (chars/3); real tokenizer
  counts are not wired yet.
- Verified at ≤ ~2.7K-doc corpora in-session; 52K-scale numbers are from
  earlier indexing runs, not fresh benchmarks.
- `insights` reports only what the logs record — per-feature usage needs a
  `flags` field in the query log (not yet logged).

## 10. Troubleshooting

| Symptom | Check |
|---|---|
| `corpus_not_found` | `t3rag list` — error body includes indexed names |
| Slow single query (~45s+) | Daemon mid-spawn; next query is warm. `--daemon always` to require it |
| `sparse_leg_missing` | Rebuild with `--features sparse` + re-index with `--sparse` |
| `exact scan unavailable` | Corpus built with `--no-raw-vectors` — ANN search still works |
| Daemon stuck/stale | `rm ~/.cache/t3rag/daemon.json`; CLI auto-respawns |
| Health | `t3rag doctor`, `GET /health`, `GET /metrics`, `t3rag log` |
