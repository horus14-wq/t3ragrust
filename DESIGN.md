# t3rag — Design Notes

Rust-native retrieval tool for **thinking traces** (T3), packaged as a CLI / MCP server / agent skill for LLM coding tools.

Source material: `papers/t3-main/` (official repo), paper: *RAG over Thinking Traces Can Improve Reasoning Tasks*.

---

## 1. What the paper does

- **Corpus:** raw + T3-transformed thinking traces from strong reasoners (Gemini-2-thinking, GPT-OSS-120B, QwQ-32B). ~57–155K passages per corpus. Hosted on HF as `narabzad/t3-*` datasets.
- **T3 transformations** (offline, LLM-driven via `data_transform/transform.py`):
  - `t3_struct` — structural normalization into step-by-step scaffolds (1 trace → multiple passages)
  - `t3_reflect` — contrastive form: common mistakes, misleading paths, how to avoid
  - `t3_semantic` — compressed core reasoning idea
- **Retrieval:** `intfloat/e5-base-v2` embeddings + FAISS `IndexFlatIP`, top-3 passages injected into the solver prompt.
- **Result:** consistent gains on AIME 2025–26, GPQA-Diamond, LiveCodeBench across GPT-5, Gemini-2.5-Flash, GPT-OSS-120B.

## 2. Why the paper's stack is dated (Nov 2026)

| Component | Paper (2024) | Problem | 2026 alternative |
|---|---|---|---|
| Embedder | `e5-base-v2` (278M, 2022) | Weak vs. current models; **512-token context** forces chunking of long traces | Qwen3-Embedding-0.6B/4B/8B (32K ctx), BGE-M3, EmbeddingGemma-300m, F2LLM-v2 (code-tuned) |
| Vector index | FAISS `IndexFlatIP` | Stale Rust bindings; overkill infra for ~155K vectors | `usearch` (HNSW, single-header) or LanceDB |
| Pipeline | Python + sentence-transformers | Not embeddable in coding tools | Pure Rust: `fastembed` + `usearch`/`lancedb` |

**The context-window point matters most:** thinking traces are long; e5-base's 512-token limit is why the paper chunked at 512 and lost coherence. Qwen3-Embedding handles 32K tokens natively → embed whole T3 passages or large trace segments whole.

## 3. Embedding model landscape (MTEB, Nov 2026)

| Model | MTEB avg | Size | Local? | Notes |
|---|---|---|---|---|
| Harrier-OSS v1 27B (MS) | ~74.3 | 27B | impractical | SOTA, too heavy |
| Qwen3-Embedding-8B | ~70–75 | 8B | GPU/Metal only | Best quality |
| **Qwen3-Embedding-4B** | ~74.6 (en v2) | 4B | borderline | Strong |
| **Qwen3-Embedding-0.6B** | **70.7** | 0.6B | **CPU-friendly** | Recommended default |
| KaLM-Embedding-Gemma3-12B | ~72 | 12B | no | |
| F2LLM-v2-14B/8B/4B/1.7B (Codefuse) | ~65–69 | varies | heavy | Code-domain tuned — relevant since corpus is reasoning/code traces |
| voyage-code-4 | — | API | API only | Purpose-built code retrieval (Aug 2026) |
| jina-embeddings-v5-omni | ~67 | small | ONNX? | New gen |
| e5-base-v2 (paper) | ~63 | 278M | yes | Baseline to beat |

**Rust path (revised):** [`ort`](https://crates.io/crates/ort) + ONNX Runtime — **not** fastembed/candle. Rationale: candle is a heavy compile dep AND ~7x slower CPU inference than ORT (rust-embedding-bench). `onnx-community/Qwen3-Embedding-0.6B-ONNX` provides official exports (fp32/fp16/int8/uint8/q4); `ort` downloads prebuilt ORT binaries → fast compile AND fast runtime. We hand-roll the standard recipe (~200 lines): instruction prefix → tokenize → ONNX session → last-token pooling → L2 normalize. fastembed remains fallback if the ONNX export misbehaves; its ONNX models (BGE-M3, e5-multilingual, EmbeddingGemma) are also reachable via ort.

## 4. Vector store comparison

| | `usearch` | LanceDB | brute force |
|---|---|---|---|
| Nature | Single-header HNSW-ish ANN | Embedded columnar vector DB (Arrow/Lance) | Exact scan |
| Deps / compile | Tiny (~dozens of crates) | Heavy (hundreds: arrow, object_store, lance) | none |
| Binary size | ~MB | +tens of MB | — |
| Persistence | Single index file | Dataset dir, versioned | recompute |
| Extras | metrics, SIMD bf16/i8 | hybrid FTS+vector, SQL filter, metadata | — |
| Fit for ~155K static passages | **ideal** | overkill (but future-proof) | actually viable (~ms with SIMD) |

**Recommendation:** `usearch`. LanceDB only if we later want hybrid FTS+vector in one engine or per-user corpora with metadata.

**Decision: hybrid search in v1** — `usearch` (dense) + `tantivy` (BM25) fused with RRF. Traces contain exact-match signals (identifiers, theorem names, variable names) that dense vectors underweight.

> **Perf note:** BM25+RRF felt slow in past Python stacks, but that was implementation overhead (`rank_bm25` pure-Python scoring, Elasticsearch network hops, Python fusion loops) — not the algorithm. Tantivy answers BM25 in ~1ms on this corpus size; RRF is a list sort. Query-path cost is dominated by embedding inference (~50–200ms for a 0.6B model on CPU) regardless → hybrid adds ~1–5ms, effectively free.

## 5. Reranking (two-stage retrieval)

- **Stage 1:** hybrid retrieval (dense ANN + BM25 → RRF) → top ~50 candidates (cheap, ~ms).
- **Stage 2:** cross-encoder reranker scores each (query, passage) pair jointly — sees the actual interaction, unlike bi-encoder embeddings which compress passage and query independently.
- **Why it helps:** embeddings optimize recall (broad similarity); cross-encoders optimize precision ("does *this* trace help solve *this* query?"). Only top-3 get injected into the prompt, so precision directly determines context quality and token waste.
- **Cost:** ~50 forward passes ≈ 50–500ms — irrelevant for CLI/MCP latency budget.
- **Implementation:** `fastembed` has a rerank API (BGE reranker family, e.g. bge-reranker-v2-m3).
- **Verdict:** **in v1 scope** — highest-leverage quality upgrade after the embedder itself. `--rerank` flag, on by default if fast enough.

## 6. Proposed architecture

Single Rust binary `t3rag`, clap subcommands:

```
t3rag index  --input <jsonl|dir> --name <corpus-name>
             # read passages → embed → usearch + tantivy + packed passage store

t3rag search "<query>" [--corpus X] [--top-k 3] [--no-rerank] [--json]
             # embed query → usearch ANN + tantivy BM25 → RRF → rerank → print passages

t3rag serve   # daemon: model + indexes resident, HTTP + streamable-HTTP MCP
t3rag mcp     # stdio MCP → proxies to daemon (fallback for non-HTTP clients)
t3rag skill   # emits/installs a SKILL.md wrapper for agent discovery (optional)
```

- **Index artifact:** one dir per corpus (see §7) — packed passages + usearch index + tantivy BM25 + `meta.json`.
- **Models:** default `Qwen3-Embedding-0.6B` via `ort` + ONNX (uint8/fp16 export preferred); `--model` flag allows 4B/8B, BGEM3, multilingual-e5-large, jina-v2-code.
- **MCP:** `rmcp` SDK, tool `search_traces(query, corpus?, top_k?) -> passages`. This is the primary interface for coding agents; CLI for humans/scripts; skill optional.
- **No Python, no API keys required.** Localhost daemon is optional-but-recommended; everything runs in-process otherwise.

### Concurrency & daemon mode

**Multithreading:**
- `index`: embarrassingly parallel — batch passages, fan out embedding batches across cores (rayon + candle `num-cpu-threads`); tantivy indexing is internally parallel already. GPU/Metal via fastembed candle features is a later option.
- `search`: query embedding is a single forward pass (~50–200ms, can't parallelize); ANN+BM25 ~ms; **reranker parallelizes** — 50 independent (query, passage) scoring passes batch/fan out → ~50–100ms.

**Daemon (`t3rag serve`)** — the main latency win is avoiding per-query model load:
- Qwen3-0.6B weights ~1–2GB; load+warmup is seconds. Daemon keeps model resident + usearch index mmap'd (shared pages) + pooled tantivy readers.
- Serves HTTP on `127.0.0.1:PORT`; MCP clients use **streamable HTTP transport** (rmcp supports it) instead of stdio-spawn-per-session.
- Concurrency: tokio + semaphore around inference (ORT sessions aren't safely concurrent — serialize or pool 2–4 sessions; plenty for single-user agent load).
- Cost: ~1–3GB RAM resident. Mitigation: `--idle-timeout` to unload, or accept ~5–10s cold start in pure-CLI mode.

**One daemon, three thin front-ends** — CLI vs MCP is NOT a real either/or; both are ~equally fast clients (~10–50ms spawn vs ~100–300ms inference = negligible difference). What matters is warm vs cold:

```
t3rag serve     # daemon: model + indexes resident, HTTP + streamable-HTTP MCP endpoint
t3rag search    # CLI → thin HTTP client; auto-spawns daemon if absent
t3rag mcp       # stdio MCP → PROXIES to daemon (does not load models itself)
```

- Stdio MCP that self-hosts the model would pay cold-start every session — avoided by proxying.
- HTTP-capable MCP clients connect to the daemon directly, skipping the stdio proxy entirely.
- Cold-start fallback (`--no-daemon` / daemon disabled) exists for minimal-footprint use only.

### Latency budget & speed knobs (warm daemon)

| Stage | CPU | GPU/Metal |
|---|---|---|
| Transport (localhost HTTP keep-alive) | <1ms | <1ms |
| Query embedding (0.6B) | 80–200ms | 10–40ms |
| usearch ANN + tantivy BM25 + RRF | 2–5ms | 2–5ms |
| Rerank (50 pairs) | 100–500ms | 20–60ms |
| **Total** | **~200–700ms** | **~50–150ms** |

Daemon vs cold CLI: ~5–15s → **10–50x faster**. Remaining levers:

- **Reranker is the biggest variable** — truncate passages to ~1024 tokens before scoring (relevance doesn't need the full trace), parallelize pairs (rayon), configurable candidate count; smaller reranker option (MiniLM cross-encoder ~33M vs bge-reranker-v2-m3 ~568M).
- **Query embedding is the floor** — single forward pass can't parallelize; only levers are GPU/Metal or a smaller embedder.
- **i8-quantized vectors** in usearch → ~4x smaller index, better cache locality.
- **Query LRU cache** — agent retries/similar queries return instantly.
- **Startup warmup** — dummy embed+search at daemon boot so query #1 is hot.
- Transport already sub-ms; named pipe/UDS not worth the complexity.

**Quality presets** (speed ↔ quality is a dial, not a fixed point):
- `--fast` — no/tiny reranker + smallest embedder → ~50–150ms
- `--balanced` (default) — Qwen3-0.6B + truncated parallel rerank → ~200–500ms
- `--max` — 4B/8B embedder + full rerank → seconds, best precision

### Crate strategy (compile-time vs runtime)

Chosen for BOTH fast builds and fast binary — the `ort`-not-candle revision is the big one:

| Crate | Role | Compile cost | Runtime notes |
|---|---|---|---|
| `ort` (prebuilt ORT binaries) | ONNX inference — embed + rerank | low-med; **no C++ build** | fastest CPU inference; intra-op threads |
| `hf-hub`, `tokenizers` | model + tokenizer download/encode | low | — |
| `usearch` | vector index | med (single C++ TU via cc) | SIMD, mmap |
| `tantivy` | BM25 | med (pure-Rust tree) | internally parallel |
| `tokio` (minimal features) | HTTP server | med | keep feature list lean |
| `rmcp` | MCP | med-heavy | fallback: hand-rolled JSON-RPC (~200 lines) if dep tree hurts |
| `rayon`, `memmap2` | parallel batches + mmap | low/trivial | — |
| `parquet` + `arrow` | parquet ingest | med | stream row groups, extract string cols |
| `parking_lot`, `rustc-hash` | hot-path sync/maps | trivial | — |
| ~~candle/fastembed~~ | — | **avoided** — heavy compile + ~7x slower CPU inference | — |

**Compile-speed practices:**
- `ort` `download-binaries` feature → prebuilt ONNX Runtime, zero C++ compilation
- GPU backends (cuda/metal) feature-gated, off by default
- minimal tokio/rmcp feature sets; `cargo build --timings` to audit
- `[profile.release] lto="thin", codegen-units=1, strip=true`; dev `debug="line-tables-only"`
- `rust-lld`/`lld-link` linker via `.cargo/config.toml`; sccache optional

**Runtime-speed practices:**
- ORT intra-op thread count tuned; batch embedding calls; rayon for index batches
- inference on dedicated threads, not tokio tasks (avoid executor overhead in hot path)
- i8-quantized vectors + mmap everything (see §7)

## 7. Storage layout ("the database")

Not one file — **three stores + metadata** per corpus, bundled as one artifact dir:

```
~/.cache/t3rag/corpora/<name>/
├── meta.json            # model, dims, quantization, doc count, format version
├── passages.bin + .idx  # packed binary: length-prefixed blobs + offset table (mmap random access)
├── vectors.usearch      # usearch HNSW index (i8/bf16/fp32; mmap-able)
└── bm25/                # tantivy index directory (segments — inherently multi-file)
```

Options considered per store:

- **Passages:** JSONL (simple, load-all ~500MB) vs packed binary+offsets (mmap, ~0 RAM, fastest — **chosen**) vs SQLite (single file/ACID but heavier than needed). JSONL remains the ingest/export format only.
- **Vectors:** usearch single-file (**chosen**) vs raw vectors + brute force (exact, no ANN recall loss; i8 scan 155K×1024 ≈ ~160MB → ~2–5ms SIMD — viable fallback/`--exact` mode) vs sqlite-vec (slower).
- **BM25:** tantivy directory — no real alternative at this quality/speed point.

Everything is mmap-friendly → daemon "warm" state is mostly page cache; cold start gets faster across runs. `meta.json` format-version field allows evolution.

### Corpus strategy — user-provided

- **User supplies the corpus** — no HF dependency. `t3rag index --input <path>` accepts:
  - **Parquet** (`.parquet`) — HF dataset format; reads via `parquet`+`arrow` crates, string columns extracted (`text`/`content`/`passage` column auto-detected or `--text-col` flag)
  - JSONL (one passage per line; `text` required, `question`/`title`/`source` optional metadata)
  - Directory of `.md`/`.txt` files (filename → title)
- Index artifact cached per corpus under `~/.cache/t3rag/corpora/<name>/`; `search` auto-builds if missing.

### Chunking policy — recommendation

- **Default: no chunking** — embed each passage whole. T3 corpora are already distilled (struct/reflect/semantic scaffolds, typically << 32K context); whole-passage embeddings preserve reasoning coherence. The paper's 512-token chunking was forced by e5-base's limit, not optimal.
- **Chunk only when a passage exceeds ~8K tokens** (conservative vs 32K limit; long inputs slow embedding).
- **Split on structural boundaries** (step markers, headers, paragraph breaks) — traces have natural step structure — not fixed token windows. Light/no overlap.
- **Parent hydration:** each chunk stores `parent_id` + `chunk_idx`; retrieval matches chunks but returns full parent passages — fine-grained recall + complete context.
- **Verify empirically** on the real corpus: A/B whole-passage vs chunked on sample queries.

## 8. Locked decisions (Nov 8, 2026)

- [x] **Embedder:** Qwen3-Embedding-0.6B via **`ort` + ONNX** (onnx-community export, uint8/fp16; NOT fastembed-candle — candle is heavy to compile and ~7x slower CPU inference). Fallback: fastembed-candle. `--model` flag leaves room for 4B/8B/BGEM3/jina-v2-code later
- [x] **Vector store:** `usearch` — single-file HNSW, minimal deps
- [x] **Hybrid search:** `tantivy` BM25 + RRF fusion — in v1 scope
- [x] **Reranker:** BGE reranker via ONNX/`ort` — in v1 scope, `--rerank`/`--no-rerank` flag
- [x] **Interface:** single Rust binary, clap subcommands: `index` + `search` + `serve` (daemon, HTTP+MCP) + `mcp` (stdio fallback) + optional `skill`
- [x] **Concurrency:** rayon-parallel index embedding; tokio server; parallel rerank; inference semaphore (2–4 sessions)
- [x] **Speed:** quality presets (`--fast`/`--balanced`/`--max`), i8 vector quantization, rerank passage truncation (~1024 tok) + parallel scoring, query LRU cache, daemon startup warmup, GPU/Metal auto-detect
- [x] **Crates:** `ort` (prebuilt ORT) + `hf-hub`/`tokenizers` + `usearch` + `tantivy` + `tokio` (minimal) + `rmcp` + `rayon`/`memmap2`; candle avoided; release LTO+strip, rust-lld, feature-gated GPU
- [x] **Stack:** pure Rust, no Python, no API keys required

## 9. Open questions

- [x] Corpus source → **user-provided** via `t3rag index --input` (JSONL or dir of .md/.txt)
- [ ] Chunking → recommendation given (whole-passage default, structural chunking >8K tokens, parent hydration); **verify empirically** on real corpus
