# t3rag — Improvement Plan, Round 1

**Date:** 2026-11-08 · **Scope:** full-repo deep dive + dependency/ecosystem research
**Method:** read every source file (~3.6K LOC), DESIGN.md §1–9, git log, open/closed beads;
web research on every dependency's features/flags; survey of similar projects
(fastembed, nidus, turbovec, embedvec, foxstash, mem0/Zep/Letta/cognee); review of
Dicklesworthstone's own ecosystem (cass, frankensearch, bv, br, agent-mail, dcg).

Real benchmark corpus already on this machine: `/data/projects/t3data/f5_v2_t3_{struct,
reflect,semantic}.jsonl` (~52.7K passages each, 128–244 MB) and
`f5_v2_filtered_coding.jsonl` (53.8K, 487 MB) — right in the paper's 57–155K range.
Eval fixtures exist but are unwired: `tests/fixtures/{mini.jsonl, queries.jsonl,
md/}` (`queries.jsonl` already carries `relevant_ids` — recall@k-ready).

---

## 0. TL;DR — the ten highest-leverage items

1. **`--max` preset is broken.** It points at `onnx-community/Qwen3-Embedding-4B-ONNX`,
   which has no `onnx/model_*.onnx` candidates — only a 2 MB `model.onnx` stub +
   16.1 GB `model.onnx_data` at repo root. All five `ONNX_FILE_CANDIDATES` 404.
   (`src/search/mod.rs:52`, `src/embed/model.rs:5-11`)
2. **External-data ONNX siblings are never downloaded.** `onnx/model_fp16.onnx`
   and `onnx/model.onnx` (candidates #4/#5) are stubs whose weights live in
   `<name>_data` files ONNX Runtime resolves *adjacent to the .onnx file*. If the
   quantized variants ever fail, the fallback path breaks opaquely.
   (`src/embed/session.rs:305-331`)
3. **`--model` is effectively Qwen3-only.** `for_hf_repo` hardcodes the Qwen3 query
   instruction for *every* repo (BGE/e5/jina want different prefixes — quality
   silently degrades), and `forward_batch` unconditionally feeds `position_ids`,
   which non-Qwen exports don't declare → session.run error. (`src/embed/model.rs:43-51`,
   `src/embed/session.rs:212-263`)
4. **`t3rag search` does not auto-spawn the daemon** despite the DESIGN.md/AGENTS.md
   invariant ("search auto-spawns/proxies to the daemon"). Every CLI search pays the
   ~5–15 s cold model load. `ensure_daemon()` exists but is private to the mcp module.
   (`src/search/mod.rs:77-90`, `src/mcp/mod.rs:173-199`)
5. **Matryoshka dims are free performance.** Qwen3-Embedding supports truncation to
   32–1024 dims with graceful quality loss. A `--dims` flag (stored in `meta.json`,
   which already has a `dims` field) gives ~4× smaller/faster indexes at 256d.
6. **SKILL.md violates the Agent Skills spec.** No YAML frontmatter (`name` +
   `description` required), installs as bare `SKILL.md` instead of `<skill>/SKILL.md`.
   Spec-compliant loaders will ignore it. (`src/skill/mod.rs:14-73`)
7. **MCP speaks protocol `2024-11-05`** — four spec revisions stale (current:
   `2026-07-28`; last with initialize handshake: `2025-11-25`). No
   `Mcp-Session-Id`, no `Mcp-Protocol-Version` validation, no SSE/batch. Decide:
   adopt `rmcp` or extend the hand-rolled layer honestly.
8. **`--exact` is silently ignored when the daemon is up** — `SearchRequest` has no
   `exact` field and `thin_search` never sends it. Same class of bug: MCP `tools/call`
   drops `candidates` even though `/search` accepts it. (`src/serve/mod.rs:97-105`,
   `src/mcp/mod.rs:130-144`)
9. **Benchmarking/eval infra is the open bead (t3-wkp.17) and everything needed exists:**
   `queries.jsonl` with `relevant_ids`, real 52K-doc corpora in `/data/projects/t3data`,
   `--exact` recall reference already implemented. Wire it into `t3rag eval`/`bench`.
10. **Dead deps + missing flags:** `rayon`, `rustc-hash`, `thiserror` are declared but
    unused; ~25 hardcoded constants (RRF_K, HNSW params, chunk thresholds, pool sizes,
    LRU cap, port, tokio workers, rerank token cap) have no CLI/env escape hatch.

---

## 1. Bugs & correctness issues (fix first)

| # | Issue | Where | Fix |
|---|-------|-------|-----|
| B1 | `--max` preset → nonexistent ONNX candidates; 16 GB fp32 external-data repo | `search/mod.rs:52`, `embed/model.rs` | Point at a repo that actually has quantized `onnx/` exports (verify onnx-community catalog; e.g. check for a 4B quant repo or use `Qwen3-Embedding-4B` gguf→onnx path). Alternatively drop `--max` embedder to a realistic CPU model until external-data fetch lands. |
| B2 | `.onnx_data` external files never fetched | `embed/session.rs::fetch_files` | After resolving the `.onnx` path, list repo files (hf-hub `repo.files()`/tree API) and download every `*.onnx_data` / sibling the model references into the same cache dir; ORT resolves them relative to the model path. Also add a post-download existence check. |
| B3 | Wrong/missing query instruction for non-Qwen `--model` | `embed/model.rs:43-51` | Model registry: `{repo → instruction, expected dims, onnx candidates, tokenizer file}`. Add `--instruction` override flag. BGE-family wants `Represent this sentence for searching relevant passages: `; e5 wants `query: `/`passage: `; jina-v2-code wants its own. |
| B4 | `position_ids` fed unconditionally — breaks non-Qwen embedder exports | `embed/session.rs:238-242` | Mirror the reranker's approach: introspect `session.inputs()` once at load, keep an input-name → feed-kind map, only feed declared inputs (token_type_ids/position_ids/kv-cache generically). |
| B5 | `t3rag search` never spawns daemon | `search/mod.rs:77-90` | Share `ensure_daemon()` (move to `serve` or a `daemon_client` module); `search` uses it unless `--no-daemon`. Consider cass's lesson: *robot/JSON searches should not block on a daemon spawn* — spawn only for interactive output, or bound the wait and fall back to cold. (`--daemon auto\|always\|never` tri-state flag.) |
| B6 | `--exact` ignored via daemon; MCP drops `candidates` | `serve/mod.rs:97-105`, `mcp/mod.rs` | Add `exact`, `candidates`, `preset` fields to `SearchRequest`/MCP schema; forward them. |
| B7 | Tantivy `index.reader()` constructed per query | `index/bm25.rs:110` | Store one `IndexReader` in `Bm25Index` at open (it's `Send+Sync`, cheap `.searcher()` per query). Hot-path win in daemon. |
| B8 | SKILL.md missing required frontmatter + wrong install layout | `skill/mod.rs` | Emit agentskills.io-compliant `--- name: t3rag / description: … ---` frontmatter; `--install <dir>` writes `<dir>/t3rag/SKILL.md`; add `--agents claude\|devin\|cursor\|generic` presets targeting `.claude/skills/`, `.devin/skills/`, `.agents/skills/`. |
| B9 | MCP protocol `2024-11-05`, no session/version semantics | `mcp/mod.rs:22` | Either adopt `rmcp` (v1.x supports `2026-07-28` stateless + legacy session modes; DESIGN.md already names it) or implement `Mcp-Session-Id` issuance, `Mcp-Protocol-Version` header validation, and bump `PROTOCOL_VERSION` honestly. The current single-shot JSON POST satisfies neither old-session nor new-stateless spec cleanly. |
| B10 | Corpus name collisions across projects/users | `paths.rs`, `search/mod.rs::resolve_corpus` | Shared-checkout reality (AGENTS.md): two agents doing `index --name traces` clobber each other via rotate-aside. Add `meta.json.origin` (input path hash/host user) + warn on cross-origin reuse, or `--namespace`/project-scoped corpus dirs. Also `resolve_corpus` "newest by mtime" is fragile — record `last_used` in daemon state / a `state.json`. |
| B11 | QueryParser syntax errors degrade silently to zero BM25 leg | `index/bm25.rs:112-118` | Already non-fatal, but user queries containing `:`, `"`, `-`, `&&` get mis-parsed rather than searched literally. Escape user input or fall back to a bag-of-terms `BooleanQuery` on parse failure; log at debug which path ran. |
| B12 | fp16 ONNX exports have known ORT per-thread memory blowup | n/a (research) | Another reason uint8/int8 should stay the preferred candidates — keep them first in `ONNX_FILE_CANDIDATES`. |

---

## 2. Performance improvements

### 2.1 Model / ORT runtime (`src/embed/session.rs`, `src/rerank/mod.rs`)

ort 2.0.0-rc.15 is in the lockfile (manifest floor rc.10 — fine). Unused session
knobs worth wiring:

- **`with_memory_pattern(false)`** — ORT docs: disable memory-pattern optimization
  when input shapes vary. t3rag batches vary in seq len every call → likely a real
  win (or at least avoids pathological re-planning). Make it a flag
  `--memory-pattern on|off|auto` (auto = off when dynamic shapes detected).
- **`with_inter_threads` / `with_parallel_execution`** — ORT parallel mode helps
  branchy graphs; Qwen3 is mostly sequential, but measure. Expose `--inter-threads`.
- **`session.dynamic_block_base` config entry** — ORT's dynamic cost model lowers
  latency *variance* (p99) under preemption; set `4` per ORT troubleshooting docs.
  Relevant on this shared VPS.
- **CPU arena**: `DisableCpuMemArena` equivalent via config entries — ORT team says
  arena "is not very performant in multi-threaded scenarios on CPU". A/B it.
- **`run_async`** exists but still `&mut self` — does NOT remove the need for the
  session pool. Keep pool; document why.
- **`Canceler`** (api-22) — lets `t3rag index` abort a hung model load; nice for
  `--idle-timeout`-style UX. Low priority.
- **EP coverage**: `ort::ep::CoreML` exists behind ort's `coreml` feature — add
  `ep-coreml` cargo feature + `Ep::Coreml` variant for macOS (DESIGN.md promises
  "GPU/Metal auto-detect"; CoreML *is* the Apple path). CUDA/TRT/OpenVINO/WebGPU
  already wired.
- **Query embed is the latency floor** (~80–200 ms CPU for 0.6B). Levers, in order:
  MRL `--dims` (smaller graph output doesn't speed the forward pass, but shrinks
  everything downstream), GPU/Metal EP, smaller embedder preset (`--fast` should
  probably map to a ~100–300M model, not just "no rerank"), ONNX graph opts already
  at Level3 — could try `Level4`/`Level99`? (verify ort exposes; ORT levels go to
  `ORT_ENABLE_ALL`.)
- **Reranker is the biggest variable.** Current default `bge-reranker-v2-m3` is
  ~568M params — heavy for CPU. Add a reranker registry + `--reranker` flag:
  - `bge-reranker-v2-m3` (current, best quality)
  - `bge-reranker-base` (~278M)
  - `ms-marco-MiniLM-L-6-v2` (~33M — 10–20× faster, classic MS MARCO cross-encoder;
    verify an onnx-community export exists)
  - `Qwen3-Reranker-0.6B` — **exists** at `onnx-community/Qwen3-Reranker-0.6B-ONNX`
    (`onnx/model_quantized.onnx` int8, `model_q4.onnx`), but it's CausalLM-style:
    scores via yes/no token logits at last position, NOT a `[batch,1]` logit head.
    Needs a scoring adapter (chat template + two-token softmax). Medium effort,
    big quality potential (instruction-aware, 100+ langs, same family as embedder).
  - Preset mapping: `--fast` → MiniLM or none; `--balanced` → bge-base;
    `--max` → bge-m3 or Qwen3-Reranker.
- **`RERANK_MAX_TOKENS = 512`** hardcoded (DESIGN said ~1024) → `--rerank-tokens`.
- **fp16 per-thread memory blowup** (ORT issue #23484) — keep uint8/int8 preferred;
  document why fp16 stays last-resort.

### 2.2 Index build (`src/index/`, `src/ingest/`, `src/chunk.rs`)

- **`EMBED_BATCH=16`, `EMBED_TOKEN_BUDGET=4096`** hardcoded → flags
  (`--embed-batch`, `--embed-token-budget`). The 4 GB ORT tensor limit comment is
  real; expose but keep safe defaults.
- **Tokenizer estimate `len()/3`** for chunking/budget is crude → use the loaded
  tokenizer for real counts where it matters (chunk trigger), keep estimate for
  batch budgeting. Flags: `--chunk-tokens` (7000), `--chunk-chars` (15000),
  `--no-chunk` / `--always-chunk` for the A/B eval the open bead wants.
- **`flush_batch` clones every text** (`batch.iter().map(|(_,t)| t.clone())`) —
  passages can be ~15 KB each; switch batch to `Vec<(u64, Arc<str>)>` or indices
  into a slab. Moderate index-time win, zero risk.
- **Dir ingest is serial** — `rayon::par_iter` over the sorted file list (dep
  already declared!). JSONL could try `simd-json` (optional dep) — measure first.
- **Parquet `with_batch_size(8192)`** hardcoded → `--batch-size`.
- **Dedup at ingest** — exact-hash dedup is nearly free (rustc-hash, already a
  dep): skip byte-identical passages before embedding. Near-dup (simhash/minhash)
  is a v2 item; corpora like `f5_v2_filtered_coding` likely have dups.
- **HNSW params hardcoded** (`CONNECTIVITY=16`, `EXPANSION_ADD=128`,
  `EXPANSION_SEARCH=64`) → `--connectivity`, `--expansion-add` at index time;
  **`--ef-search` at query time** — usearch 2.26 exposes
  `Index::change_expansion_search(&self, n)` on the safe API *and* it works on
  `view()`ed indexes (it's a plain field). This is the standard recall↔latency
  dial every vector DB exposes; t3rag should too. Also `Index::metadata(path)`
  reads the header without loading — useful for `t3rag stats`/`doctor`.
- **Quantization**: `Quantization` enum has I8/Bf16/Fp32. usearch `ScalarKind`
  also offers `F16`, `U8`, `B1` (binary!), `E5M2/E4M3/E3M2/E2M3` — binary quant is
  a legit experiment for 155K×1024 vectors (huge memory cut; recall hit needs
  measuring via `--exact` + eval). Expose as `--quantization b1` etc. once
  validated. `hardware_acceleration()` reports SIMD availability → surface in
  `t3rag doctor`.
- **`vectors.raw.f32`** duplicates every vector for `--exact`/rebuild — add
  `--no-raw-vectors` to halve disk when recall-reference isn't needed (meta flag).
- **MRL `--dims N`** (see TL;DR #5): truncate embeddings to N dims (Qwen3 supports
  32–1024; min recommended 32). Store in `meta.json.dims` (field exists), truncate
  query + passage vectors symmetrically, renormalize. Enables `--fast` to mean
  "256d + MiniLM rerank" honestly. Requires re-index per dims — that's fine,
  corpora are cheap-ish to rebuild and `meta.json.format_version` governs.
- **Incremental index / `index --update`** — today any corpus change = full
  re-embed (the dominant cost by far). usearch supports `add`+`save`, tantivy
  supports appends, `passages.bin/.idx` is append-friendly *if* ids stay dense —
  feasible v2: delta segments or a generation manifest. Frankensearch's Quill
  engine does "delta-visible indexing" natively — see §8.
- **Tantivy schema**: `TEXT` = default English tokenizer (lowercase + **stemming**
  + stopwords?) — bad for identifiers/trace jargon. Register a custom analyzer
  (`SimpleTokenizer → RemoveLong → LowerCaser`, no stemmer) as e.g. `"t3"`, keep
  `"default"` for prose; flag `--tokenizer default|t3|raw`. Also `TEXT` indexes
  positions — BM25 doesn't need them: `IndexRecordOption::WithFreqs` shrinks the
  index and speeds indexing. `set_merge_policy` + writer heap (50 MB **per
  thread**, max 8 threads) → `--bm25-heap-mb`, `--bm25-threads`.
- **Progress**: `tokenizers` ships `indicatif` behind its `progressbar` feature
  (currently off via `default-features=false`) — or add `indicatif` directly for
  an index progress bar (docs embedded/sec, ETA). High UX value on 52K-doc runs.

### 2.3 Query path (`src/search/`, `src/serve/`)

- **`rrf_fuse` uses `std::collections::HashMap`** → `rustc_hash::FxHashMap`
  (dep declared, unused). Also `seen_parents`/`seen` sets → FxHashSet.
- **`exact_search` is a scalar byte-loop, single-threaded** → parallelize over
  rows with rayon (dep exists!) or `std::thread::scope` chunks; or better, use
  usearch's own `Index::exact_search`/`filtered_search` (SIMD'd inside usearch).
  The custom raw-blob scan exists for the "no ANN" guarantee — keep it but make
  it fast; it's the recall oracle for eval.
- **`QueryLru` is O(cap) `retain` per lookup** — fine at 128, but expose
  `--cache-size N` / `--no-cache`; consider caching *final responses* keyed by
  (query, corpus, top_k, rerank) — agent retries are the common case.
- **Leg weighting**: RRF treats dense and BM25 legs equally. nidus exposes
  `vector_weight`/`text_weight` — add `--dense-weight`/`--bm25-weight` (or a
  `--fusion` preset). Cheap, measurable via eval.
- **`--rrf-k`** flag (const 60 today).
- **Diversity/caps**: nidus has `limit_per`/`diversity` — cheap version: dedupe
  hits by `source`/`title` beyond parent-collapse (`--max-per-source N`).
- **Pagination**: `--offset`/`--limit` beyond top_k; MCP `top_k` already there.
- **Score semantics**: after rerank, `score` is a raw cross-encoder logit — a
  different scale than RRF. JSON output should carry `score` + `score_kind`
  (`rrf`|`rerank_logit`) and ideally per-leg scores (`dense`, `bm25`, `rerank`) —
  the `explain`/`plan` pattern from nidus. Debugging hybrid retrieval without
  this is guesswork.
- **`--min-score`**: threshold on rerank probability (sigmoid) so agents can tell
  "no relevant trace" from "weak match". Pairs with `isError` semantics in MCP.
- **Result size control**: `--max-chars N` / `--snippet` — full passages are
  token-heavy for agents; TOON/compact formats (below) plus truncation flags.
- **`resolve_corpus` fallback**: list available corpus names in the error;
  `--corpus all` or comma-list for cross-corpus search (the three T3 transforms
  are designed to be searched together-ish; today you must pick one).
- **Daemon `do_search`**: `spawn_blocking` per request is right; but reranker
  `Mutex<Option<RerankerPool>>` serializes ALL rerank queries behind one lock —
  fine for single-user, note for concurrency. `tokio worker_threads(4)` →
  `--tokio-threads`/`--workers`.
- **`/health`** could include model repo, dims, EP actually in use, corpus count,
  cache size — agents love cheap health probes (cass `health` is <50 ms).

### 2.4 Storage

- `passages.bin` ~uncompressed. Options: keep as-is (mmap needs raw), or optional
  zstd **block** compression with an index (v2, complex), or store text *also* in
  tantivy stored fields and drop passages.bin for BM25-hit hydration (duplication,
  probably not worth it). Recommend: defer; revisit if corpus > ~1 GB.
- `passages.idx` is u64 offsets — could be u48/relative for size; irrelevant at
  this scale. Skip.
- `meta.json`: add `dims` (have), `rrf_k`, `connectivity`, `tokenizer`,
  `instruction`, `created_at`, `origin`, `index_params` — makes corpora
  self-describing for `t3rag stats`/`doctor`/eval comparisons.

---

## 3. New CLI flags — the "Dickle treatment"

His projects (bv, cass) expose *everything*: `--robot-*` output modes, `--format
json|toon`, `--cpu-profile`, `--profile-startup`, `--stats`, `--recipe`,
`--search-weights`, env-var fallbacks, `capabilities`/`introspect`/`doctor`/
`selftest`/`onboarding`/`completions`/`man` subcommands. t3rag's current surface is
good but ~25 constants are hardcoded. Proposed additions, grouped:

### Global / plumbing
- `--config <path>` + `T3RAG_*` env vars on every flag (clap `env` feature) —
  cass/bv pattern; config file `t3rag.toml` (XDG config dir) for defaults.
- `--cache-dir <path>` / `T3RAG_HOME` — `paths.rs` hardcodes `dirs::cache_dir`;
  tests, CI, and multi-instance dev need the override (fastembed precedent:
  `FASTEMBED_CACHE_DIR`, `HF_HOME` already respected by hf-hub).
- `-v/--verbose`, `--quiet`, `--log-level <lvl>` — tracing EnvFilter exists;
  surface it as flags instead of only `RUST_LOG`.
- `--format json|jsonl|toon|md` on `search` — **TOON** is the signature move
  (bv `-f toon`: token-efficient output for agents). `--json` stays as alias.
- `--yes` on destructive commands; documented exit codes.

### `t3rag index`
`--dims N` (MRL) · `--onnx-file <name>`/`--model-quant uint8|int8|fp16|fp32` ·
`--connectivity` · `--expansion-add` · `--embed-batch` · `--embed-token-budget` ·
`--chunk-tokens` · `--chunk-chars` · `--no-chunk` · `--dedup` · `--tokenizer
default|t3|raw` · `--bm25-heap-mb` · `--bm25-threads` · `--no-raw-vectors` ·
`--batch-size` (parquet) · `--min-chars`/`--max-chars` · `--update` (incremental,
v2) · `--watch` (fsnotify re-index — or leave to `schedule`) · `--dry-run`
(count/embed-preview without writing) · `--progress`/`--no-progress` ·
`--preset fast|balanced|max` (parity with search).

### `t3rag search`
`--dims` implied by corpus meta · `--ef-search N` (usearch runtime!) ·
`--rrf-k` · `--dense-weight`/`--bm25-weight` · `--reranker <model>` ·
`--rerank-tokens` · `--rerank-candidates` (decouple from retrieval `candidates`) ·
`--min-score` · `--max-per-source` · `--offset` · `--max-chars` ·
`--explain`/`--plan` (per-leg scores + stage timings) · `--cache-size`/`--no-cache` ·
`--daemon auto|always|never` · `--corpus` accept list/`all` · `--filter
source=…|title~…` (tantivy fields already indexed) · `--instruction` (Qwen3 is
instruction-aware; +1–5% per model card) · `--timing` (stage breakdown:
embed/ann/bm25/rerank/hydrate ms — the single most useful bench flag).

### `t3rag serve`
`--bind <addr>` (hardcoded 127.0.0.1 today) · `--token <bearer>` (optional auth if
non-loopback) · `--workers N` (tokio threads=4 hardcoded) · `--cache-size` ·
`--preload-reranker` (lazy-load today → first reranked query stalls seconds) ·
`--preload-corpus` exists · `--max-body-mb` · `--request-timeout` · `--metrics`
(Prometheus endpoint) · `--log-file`.

### `t3rag mcp`
Expose full tool schema: `candidates`, `preset`, `ef_search`, `min_score`,
`max_chars`, filters — `/search` already accepts most; the MCP schema is narrower
than the HTTP API (drift bug B6).

### `t3rag skill`
`--agents claude|devin|cursor|generic` · `--uninstall` · spec-compliant output (B8).

### New subcommands (cass-inspired, all cheap)
- `t3rag list` — corpora table (name, docs, dims, model, size, mtime)
- `t3rag stats [corpus]` — deep meta: index params, files, RAM estimate
- `t3rag doctor` — env check: CPU SIMD (`usearch::hardware_acceleration`), EP
  availability, HF cache, model files present, daemon health, versions, disk
- `t3rag eval --queries <jsonl> [--corpus X]` — recall@k / MRR / nDCG vs
  `relevant_ids` (fixture exists!); `--exact` comparison mode = ANN recall audit
- `t3rag bench [--corpus X]` — stage-latency harness: cold vs warm, p50/p95,
  docs/sec index throughput, model load time
- `t3rag capabilities` / `t3rag introspect` — machine-readable command+flag+env
  listing for agents (cass pattern; ~free via clap introspection or a JSON dump)
- `t3rag completions <shell>` (clap_complete) · `t3rag man` (clap_mangen)
- `t3rag selftest`/`health` — <50 ms sanity for agent pre-flight
- `t3rag export <corpus> --format jsonl|parquet` — round-trip corpora
- `t3rag forget <corpus>`/`rm` — **deliberately hard to fire** (Rule-1-safe:
  a user-invoked feature, never agent-triggerable). If it exists at all, it
  must meet ALL of these:
  - **Exact match only** — no globs, no prefix/fuzzy matching; the literal
    corpus name must be spelled out.
  - **Dry-run by default** — prints name, doc count, bytes, and the exact
    path it will remove, then requires `--yes` to proceed.
  - **In-use refusal** — refuses to delete a corpus a live daemon has loaded.
  - **Scope-locked** — can only remove a single artifact dir directly under
    `corpora/`; can never escape the cache root or take a path argument.
  - **No shared delete primitive** — no recursive/rm helper other commands
    can reach into.
  - **Optional**: keep it out of v1 entirely — `rm` by the user is already
    the right security model.
- `t3rag models` — known model registry + download/cache status
- `t3rag schedule`/`index --watch` — freshness automation (cass uses
  launchd/systemd timers)

---

## 4. Feature ideas — user-facing

Ordered roughly by leverage-for-effort:

1. **`t3rag eval` + `t3rag bench`** — unblocks bead t3-wkp.17 and *every* other
   perf/quality claim in this doc becomes measurable. Highest priority.
2. **Multi-corpus search** (`--corpus all` / list) — the three T3 transforms
   (struct/reflect/semantic) are complementary; agents currently must guess which.
   Fuse across corpora at RRF level (they share the embedding space if same model).
3. **Metadata filters** — `source`, `title`, `question` are already indexed in
   tantivy; expose them in CLI + MCP schema. Provenance-aware retrieval
   ("only traces from corpus X / model Y").
4. **`--timing`/`--explain` stage breakdown** — embed/ann/bm25/rerank/hydrate ms
   per query; cheap to add (tracing::debug! calls already exist in `do_search`),
   huge for the latency-budget work.
5. **Score provenance in output** — `score_kind`, per-leg scores, `isError`-worthy
   `min_score` flag. Agents need to know *why* a passage ranked.
6. **Result truncation/summarization controls** — `--max-chars`, `--snippet`,
   TOON format. Token economy is the product for agent consumers.
7. **`t3rag ask` / `--format prompt`** — emit a ready-to-paste context block
   (top-k passages in the paper's injection format) for piping into an LLM/agent.
   Directly serves the T3 use case.
8. **Dedup + near-dedup at ingest** — exact hash now; simhash/minhash LSH later.
9. **Incremental indexing** (`index --update`, delta segments) — corpora are
   append-mostly; full re-embed is the #1 index cost.
10. **Query-side improvements (v2 research)**: HyDE-style expansion, multi-query
    fusion, instruction tuning per task (`--instruction` is the cheap 80% version),
    optional sparse leg (SPLADE via fastembed's `SparseTextEmbedding`, or Qdrant's
    approach) as a third RRF leg.
11. **Temporal/provenance fields** — mem0/Zep lessons: `created_at`, decay,
    `as_of` queries. Probably overkill for static trace corpora; note only.
12. **Graph layer (cognee/Graphiti-style)** — entity/relation extraction over
    traces → graph-augmented retrieval. Research item, not v1.
13. **HTTP extras** (nidus parity): `/metrics`, request IDs, `explain`/`plan`,
    `offset` pagination, bearer `--token`, body limits, timeouts.
14. **MCP tool breadth** (nidus parity): `list_corpora`, `corpus_stats`, `health`,
    `forget` (opt-in), richer `search_traces` schema. Maybe `remember`-style
    `add_passage` for agent self-improvement loops — powerful but needs care
    (corpus integrity, format_version).
15. **Skill ecosystem**: spec-compliant SKILL.md + per-agent install targets;
    `references/` with query-writing guidance; possibly a companion "t3rag-power-use"
    skill for eval/bench workflows.

---

## 5. Developer experience

- **Dead deps**: remove `rayon`, `rustc-hash`, `thiserror` *or use them* —
  rayon+rustc-hash have obvious jobs (§2.3); thiserror unused entirely (anyhow
  everywhere — fine for a bin, just drop the dep).
- **Dep upgrades worth evaluating**: `tokenizers 0.21→0.23` (rayon already a hard
  dep; `encode_batch` is parallel by default — `TOKENIZERS_PARALLELISM` env
  controls it; known 6× overhead issue #1929 on 64+ core boxes — irrelevant at 12
  vCPU but note `par_iter`+`encode` workaround if profiling shows it), `ureq 2→3`
  (3.5.1 already in tree), `hf-hub 0.4→latest` (**big one**: new `HFClient` API has
  Xet chunk-dedup downloads, repo file listing/tree — needed for B2's `.onnx_data`
  siblings anyway, async option, `HF_HUB_CACHE` control), `ort` manifest floor is
  fine (lock has rc.15).
- **`rmcp` decision** — DESIGN.md names it; code hand-rolled JSON-RPC with a
  solid justification comment ("one tool ⇒ SDK is unjustified weight"). With B9's
  protocol gap, either commit to rmcp (dep weight: tower/tokio-stream/etc.) or
  extend hand-rolled to `2025-11-25` semantics + document the deviation in
  DESIGN.md §8 (locked-decision amendment — file a bead, don't silently diverge).
- **Feature-gate heavy deps for compile time**: `parquet`+`arrow` is the biggest
  optional surface — `parquet` cargo feature (default on) lets CLI-only builds skip
  it. `axum`/`tokio`/`ureq` only matter for serve/mcp — single binary keeps them,
  but `cargo build --timings` audit is overdue (AGENTS.md prescribes it).
- **`cargo sweep` is installed** — dep-freshness audits. `cargo nextest` installed
  — adopt for tests (JUnit output for CI).
- **Tests**: add integration tests — index→search roundtrip on `mini.jsonl`,
  daemon spawn + `/search` + `/mcp` e2e, eval recall floor on fixtures,
  chunker proptest. Currently only unit tests; fixtures are dead weight until wired.
- **rust-toolchain**: local cargo is *nightly* 1.101; CI is stable (a `-Z` flag
  already bit once — commit 68d0880). Pin `rust-toolchain.toml` to stable or
  document `cargo +stable` — cheap reproducibility win.
- **README.md missing** — users land on DESIGN.md (internal spec). Add README:
  quickstart, architecture, flag reference, bench methodology pointer.
- **Error UX**: `resolve_corpus`/`resolve_model` failures should list valid
  choices (corpus names, model aliases) — clap `PossibleValuesParser`/`ValueEnum`
  for `--model` aliases would catch typos at parse time.
- **`dirs` vs `etcetera`/`directories`**: dirs 6 is fine; just add the
  `--cache-dir`/env override rather than swapping crates.

---

## 6. Dependency research notes (per-crate)

| Crate (locked) | Findings / unused knobs |
|---|---|
| `ort 2.0.0-rc.15` | Session opts unused: `with_memory_pattern` (disable for dynamic shapes!), `with_inter_threads`, `with_parallel_execution`, `with_inter_op_spinning`, config entries (`session.dynamic_block_base=4` for p99 variance; arena control). `ep::CoreML` exists → add `ep-coreml` feature + `Ep::Coreml`. `run_async` still `&mut self` — pool stays. `Canceler` for load aborts. fp16 exports have per-thread ORT memory blowup (issue #23484) — keep uint8/int8 preferred. |
| `usearch 2.26.6` | Safe API has `change_expansion_search/add` (runtime ef — the `--ef-search` flag), `exact_search`, **`filtered_search(key→bool)`** (predicate filtering at ANN level!), `hardware_acceleration()`, `memory_stats`, `Index::metadata(path)` header-only read, `restore`/`restore_view`, `rename`/`remove`/`compact`/`reset`, `stats_per_level`. `ScalarKind`: F32/F16/BF16/I8/U8/**B1 binary**/E5M2/E4M3/E3M2/E2M3. |
| `tantivy 0.24.2` | Custom tokenizers via `index.tokenizers().register()` — build a no-stemmer "code" analyzer. `IndexRecordOption::WithFreqs` (drop positions) for BM25-only. `set_merge_policy`, writer heap is **per-thread** (50 MB×N, ≤8 threads). `Index::reader()` per query is wasteful — pool it (B7). |
| `tokenizers 0.21.4` | rayon+rayon-cond are hard deps → `encode_batch` parallel by default; `TOKENIZERS_PARALLELISM` env overrides; `set_parallelism()` API exists. Latest is 0.23.x — upgrade path clear, `fancy-regex` feature still supported. |
| `hf-hub 0.4.3` | Old sync/async API. **Latest hf-hub is a rewrite**: `HFClient` async + `blocking` feature, Xet chunk-dedup transfers, repo tree/file listing (required for B2), `HF_HUB_CACHE`, `local_dir` downloads, commit APIs. Upgrade = B2 enabler + faster 1–2 GB model fetches. |
| `rmcp` (not a dep) | v1.x supports protocol `2026-07-28` (stateless, no sessions) + legacy session mode for ≤`2025-11-25`; `legacy_session_mode`/`with_json_response`/`allowed_hosts`/`sse_keep_alive` config. Hand-rolled layer is 2 protocol versions behind. |
| `ureq` | Lock has **both** 2.12.1 (direct dep) and 3.5.1 (transitive). ureq 3 is current — migrate direct dep. |
| `axum 0.8.9` | Only `json` feature — consider `tower-http` (trace/timeout/compression), `DefaultBodyLimit`, `TimeoutLayer`; add `/metrics`. |
| `tokio 1.54` | Features lean already; `worker_threads(4)` hardcoded → flag. |
| `parquet/arrow 55.2` | Heavy dep tree — feature-gate behind cargo `parquet` feature for compile time. |
| `rayon`, `rustc-hash`, `thiserror` | **Declared, unused.** rayon → exact_search/dir-ingest; rustc-hash → all hot HashMaps; thiserror → drop or adopt for error types. |
| `postcard`, `memmap2`, `dirs`, `clap 4.6`, `serde_json`, `tracing` | All appropriate; clap `env` feature + `clap_complete`/`clap_mangen` for the new subcommands. |

---

## 7. Similar projects & lessons learned

### Dicklesworthstone ecosystem (directly relevant — same author as this repo's tooling)

- **cass (`coding_agent_session_search`)** — the closest sibling: indexes *agent
  session histories* (30+ providers) and searches them. Lessons already paid for:
  - "robot/JSON searches **never spawn** the daemon, even with `--daemon`, because
    their bounded budget cannot wait" → informs B5's design (spawn for humans,
    don't block machines).
  - Warm-daemon reuse via `CASS_DAEMON_SOCKET` env — t3rag's `daemon.json`
    discovery is the same idea; add env override for socket/port.
  - `frankensearch` powers its semantic leg: **pure-Rust MiniLM on safetensors —
    no ONNX at all** — plus native Quill BM25 (see below).
  - CLI surface to emulate: `capabilities`, `introspect`, `doctor`, `selftest`,
    `onboarding`, `guide`, `completions`, `man`, `pack`, `export`, `forget`,
    `schedule`, `models`, `health` (<50 ms), `api-version`.
- **frankensearch** — two-tier progressive hybrid search: fast tier
  (potion-128M/Model2Vec static embeddings, ~0.57 ms) → immediate RRF results,
  then MiniLM-L6-v2 quality tier (~128 ms) refinement; `SearchPhase` callbacks.
  Also: **Quill** — his native pure-Rust BM25 engine (FSLX segments,
  delta-visible indexing, block-max/MaxScore pruning, search-while-indexing,
  deterministic ordering, tantivy kept only as conformance oracle). Options for
  t3rag: (a) keep tantivy (works, he proved Quill at cass scale), (b) evaluate
  `frankensearch-quill` as the lexical leg (native, incremental, no tantivy dep
  weight), (c) adopt `frankensearch-*` crates wholesale for fusion/rerank/index.
  Recommend: file a bead to spike Quill-vs-tantivy on the 52K corpus; keep the
  ORT/Qwen3 embedder regardless (frankensearch's tiers are small generic models —
  wrong tool for thinking-trace quality).
- **bv** — `--format json|toon`, `--robot-*` flags, `--stats`, `--cpu-profile`,
  `--profile-startup`, `--recipe`, `--search-weights`, env fallbacks. TOON output
  is the signature agent-token optimization.
- **br / agent-mail / dcg / ubs / ntm / sbh** — process/tooling, less relevant
  technically, but the *coordination* patterns (beads ↔ mail thread_id, file
  reservations) are already adopted here per AGENTS.md.

### External similar projects

- **nidus** (duckedup) — *the* closest architectural sibling: pure-Rust vector
  store + BM25 + RRF + HTTP serve + MCP, typed metadata filters, multi-collection
  scoped search. Steal: 11-tool MCP surface (`remember/recall/text_search/
  hybrid_search/list_collections/stats/forget/get/browse/related/suggest/
  code_search`), `rrf_k`/`candidates`/`vector_weight`/`text_weight`/`limit_per`/
  `diversity`/`offset`/`explain`/`highlight`/`plan` knobs, bearer auth + body
  limits + metrics middleware, append-only crash-safe log, `suggest` prefix
  completion. Its `code_search` (group by file, return symbol spans not source)
  is a great fit for trace corpora.
- **fastembed 7.x** (Qdrant) — validates the ort+ONNX path; now also has
  `SparseTextEmbedding` (SPLADE), `Bgem3Embedding` (dense+sparse+ColBERT in one
  pass), `TextRerank`, `InitOptions` (cache_dir, intra_threads cap), Qwen3 via
  candle. Lesson: model registry + init options pattern; sparse-leg idea for v2.
- **turbovec** — TurboQuant 2–4 bit/dim SIMD search, 7.8–15.5× smaller than fp32,
  incremental `sync()`. Alternative vector backend if usearch quantization
  disappoints; has Python+Rust APIs.
- **embedvec** — HNSW + RaBitQ (1–8 bit) / E8 / H4 lattice quantization,
  metadata filters, Fjall/Sled/RocksDB persistence. Quantization ideas.
- **foxstash** — SIMD + HNSW + SQ8/RaBitQ + WAL incremental persistence +
  zstd/lz4/gzip compression + WASM. The "incremental index + compression" combo
  t3rag lacks.
- **mem0 / Zep(Graphiti) / Letta / cognee / Supermemory** — agent-memory
  features worth borrowing selectively: dedup/merge on ingest (mem0), temporal
  facts + provenance (Graphiti), memory blocks/culling (Letta), KG-augmented
  retrieval (cognee). For *static trace corpora* most are overkill — the
  transferable bits are dedup, provenance (`source` already exists), and
  confidence/score thresholds.
- **T3 paper repo** (`narabzad/t3-*` datasets on HF) — the original eval corpora;
  `t3rag eval` could eventually run the paper's AIME/GPQA-style queries, but the
  near-term eval is recall@k on `queries.jsonl` + downstream-task A/B.

---

## 8. Benchmarking plan

### 8.1 Install on this dev machine

Already present: `perf` (linux-tools), `numactl`, `taskset`, `cargo-nextest`,
`cargo-sweep`, `cargo-miri`, nightly cargo.

Install:
```bash
# wall-time CLI comparisons (warmup, outlier handling)
sudo apt install hyperfine          # or: cargo install hyperfine
# statistical micro-benchmarks (dev-dependency, no install needed)
#   divan (attr macros, alloc measurement, lighter) or criterion (regression detection)
# flame graphs / sampling CPU profiles
cargo install flamegraph            # needs perf (present) — or cargo-flamegraph
cargo install samply                # Firefox-Profiler-format CPU profiles, great for ORT/native frames
# instruction counts / callgrind (valgrind suite)
sudo apt install valgrind           # for iai-callgrind / callgrind_annotate
# heap/allocation profiling
sudo apt install heaptrack heaptrack-dev   # GUI-less: heaptrack_print
# optional: stress/consistency
sudo apt install stress-ng          # thermal/neighbor-noise control during bench runs
```

In-repo (dev-deps, not machine installs): `divan` or `criterion` for
`benches/` (rrf_fuse, exact_search, chunk::split, store roundtrip, tokenizer
throughput); `iai-callgrind` if we want instruction-count stability in CI;
`bencher` if CI regression tracking is wanted later.

### 8.2 In-repo harness (the real deliverable)

- **`t3rag bench`** subcommand: for a corpus, run N warm queries × M iterations,
  report p50/p95/p99 per stage (embed / ann / bm25 / rerank / hydrate) using the
  `--timing` instrumentation; modes: `--cold` (fresh process, model load
  included), `--warm` (daemon), `--no-rerank`, `--exact`, `--dims N` sweeps.
  Emit JSON for tracking over time.
- **`t3rag eval`** subcommand: consume `queries.jsonl`-format files
  (`{query, relevant_ids[]}`), compute recall@1/3/5, MRR, nDCG@10; `--exact` mode
  doubles as ANN-vs-brute recall audit; `--ab` mode runs whole-passage vs
  chunked corpora side-by-side (closes the open chunking question in DESIGN §9).
- **`benches/`** for micro targets; **`tests/` integration** for pipeline e2e.
- **Real corpus**: index `f5_v2_t3_reflect.jsonl` (52,748 docs, 244 MB) once per
  variant — it's the paper-scale corpus. Keep `mini.jsonl` for CI-fast tests.
- **Methodology**: pin with `taskset -c 0-11`, check governor (`cpupower`),
  run hyperfine `--warmup 3 --min-runs 10` for CLI-level comparisons, report
  both wall-time and (where it matters) `perf stat` instructions. Daemon warm
  latency is the product metric; cold start is a one-time cost — measure both,
  optimize warm first (it's 90% of agent interactions).
- **Baseline to beat (DESIGN §6)**: warm ~200–700 ms CPU end-to-end; index
  throughput unknown — `t3rag bench` establishes it. Suspect reranker dominates
  warm latency → MiniLM/bge-base reranker + `--rerank-tokens` tuning is the
  first lever to pull.

---

## 9. Suggested sequencing (bead-sized)

**P0 — correctness (do first, all small):**
1. B1+B2: fix `--max` preset + `.onnx_data` external-file fetch (hf-hub upgrade or
   tree-listing workaround) — *unblocks the flagship quality preset*
2. B3+B4: model registry (per-model instruction/dims/input-signature) +
   `--instruction` flag — *makes `--model` actually usable*
3. B5: `search` auto-spawn daemon (respecting cass's robot-mode lesson) —
   *biggest perceived-perf win, ~30 lines*
4. B6+B7+B11: forward `exact`/`candidates` through daemon+MCP; pool tantivy
   reader; lenient query fallback
5. B8: spec-compliant SKILL.md + `--agents` install targets
6. Dead-dep cleanup (use rayon/rustc-hash or drop; drop thiserror)

**P1 — measurable perf:**
7. `t3rag bench` + `t3rag eval` (t3-wkp.17) + `--timing`/`--explain` output
8. `--dims` MRL + `--ef-search` + `--rrf-k`/leg weights + `--reranker` registry
   (MiniLM first) + `--rerank-tokens`
9. ORT session opts: `memory_pattern`, `dynamic_block_base`, inter-threads flags;
   `ep-coreml` feature
10. Index-time: rayon dir-ingest, real token counts for chunking, dedup,
    custom tantivy tokenizer + position-free BM25, progress bar

**P2 — surface area:**
11. Flag sprawl pass (§3 tables) + config file + env vars + `capabilities`/
    `doctor`/`completions`/`man` subcommands + TOON format
12. Multi-corpus search + metadata filters + `--corpus all`
13. MCP protocol decision (rmcp vs extend hand-rolled) + richer tool schema
14. HTTP hardening: `/metrics`, timeouts, body limits, optional bearer token

**P3 — strategic:**
15. Incremental indexing / `index --update`
16. Quill-vs-tantivy spike (frankensearch) on 52K corpus
17. Sparse third leg (SPLADE/BGE-M3) research; HyDE/multi-query research
18. README + docs/ + rust-toolchain pin + CI bench-regression tracking

---

## Appendix A — hardcoded values inventory (flag candidates)

| Constant | Value | Location | Proposed flag |
|---|---|---|---|
| `RRF_K` | 60.0 | `search/mod.rs:21` | `--rrf-k` |
| `EMBED_BATCH` | 16 | `index/mod.rs:22` | `--embed-batch` |
| `EMBED_TOKEN_BUDGET` | 4096 | `index/mod.rs:26` | `--embed-token-budget` |
| `MAX_EMBED_TOKENS` | 8192 | `embed/mod.rs:28` | `--max-embed-tokens` |
| `CHUNK_TRIGGER_EST_TOKENS` | 7000 | `chunk.rs:12` | `--chunk-tokens` |
| `CHUNK_TARGET_CHARS` | 15000 | `chunk.rs:15` | `--chunk-chars` |
| `RERANK_MAX_TOKENS` | 512 | `rerank/mod.rs:26` | `--rerank-tokens` |
| HNSW `CONNECTIVITY/EXP_ADD/EXP_SEARCH` | 16/128/64 | `index/dense.rs:25-27` | `--connectivity`, `--expansion-add`, `--ef-search` |
| tantivy writer heap | 50 MB | `index/bm25.rs:55` | `--bm25-heap-mb` |
| `QueryLru` cap | 128 | `serve/mod.rs:205` | `--cache-size` |
| daemon port | 7317 | `cli.rs:126` | exists (`--port`) + env |
| tokio workers | 4 | `serve/mod.rs:183` | `--workers` |
| preset candidates | 20/50/100 | `search/mod.rs:48-56` | `--preset` parity on index/serve |
| preset `--max` model | 4B-ONNX | `search/mod.rs:52` | fix repo (B1) + registry |
| default reranker | bge-reranker-v2-m3 | `rerank/mod.rs:31` | `--reranker` |
| est_tokens | len/3 | `chunk.rs:18` | real tokenizer counts |
| cache root | `dirs::cache_dir` | `paths.rs:7` | `--cache-dir`/`T3RAG_HOME` |
| MCP protocol | `2024-11-05` | `mcp/mod.rs:22` | B9 decision |
| ONNX file candidates | 5 names | `embed/model.rs:5-11` | `--onnx-file` + ext-data fix |
| PAD_IDs | 151643 / 1 | `embed/session.rs:97`, `rerank/mod.rs:27` | per-model registry |
| bind addr | 127.0.0.1 | `serve/mod.rs:226` | `--bind` |
| corpus default | newest mtime | `search/mod.rs:278` | `state.json`/`--corpus` |

## Appendix B — sources

- Repo: `DESIGN.md`, `src/**` (all files read), `Cargo.toml`/`Cargo.lock`,
  git log, `br list` (open: t3-wkp epic + t3-wkp.17 eval/bench)
- ort: docs.rs `ort::session::builder` (SessionBuilder methods), `ort::ep::coreml`,
  ONNX Runtime threading/arena/dynamic-cost-model docs, issue #23484 (fp16 mem)
- usearch: docs.rs `usearch::Index` 2.26.6 full method list
- tantivy: docs.rs indexer (IndexWriter heap-per-thread, MAX_NUM_THREAD=8,
  merge policy, custom tokenizer example)
- tokenizers: crates.io features + GitHub issue #1929 (encode_batch parallelism)
- hf-hub: docs.rs latest (`HFClient`, Xet, tree APIs) vs locked 0.4.3
- rmcp: modelcontextprotocol/rust-sdk README (protocol 2026-07-28, SEP-2567)
- Qwen3-Embedding: HF model cards + QwenLM/Qwen3-Embedding repo (MRL dims table,
  instruction-aware note); onnx-community repo trees (0.6B `onnx/` variants,
  4B root-only external data); Qwen3-Reranker-0.6B-ONNX card (yes/no logit scoring)
- fastembed: docs.rs features (7.x: onnx/candle backends, sparse, BGE-M3, rerank)
- GitHub: Dicklesworthstone/{beads_rust, mcp_agent_mail_rust, beads_viewer,
  coding_agent_session_search, frankensearch, destructive_command_guard,
  agentic_coding_flywheel_setup}; local `bv --help`/`cass --help` output;
  duckedup/nidus docs (MCP tools, hybrid RRF opts, HTTP API); RyanCodrai/turbovec;
  embedvec, foxstash crates.io pages; mem0/Zep/Letta/cognee comparison writeups
- Bench tooling: Rust Performance Book (nnethercote), Crate Atlas profiling guide,
  criterion/divan/iai/hyperfine docs
