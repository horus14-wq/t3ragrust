<div align="center">

# T3RAG-Rust

**Rust-native hybrid retrieval over thinking traces — for coding agents and the LLMs driving them.**

[![CI](https://github.com/horus14-wq/t3ragrust/actions/workflows/ci.yml/badge.svg)](https://github.com/horus14-wq/t3ragrust/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/horus14-wq/t3ragrust)](https://github.com/horus14-wq/t3ragrust/releases/latest)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](rust-toolchain.toml)
[![Corpora on HF](https://img.shields.io/badge/%F0%9F%A4%97%20datasets-prebuilt%20corpora-yellow)](https://huggingface.co/datasets/horus14-wq/fable5-v2-coding-t3-indexes)

</div>

T3RAG-Rust indexes *reasoning traces* — the step-by-step thinking LLM agents
produce while solving problems — and makes them searchable in milliseconds.
One static binary gives you dense embeddings, BM25, hybrid fusion, and
optional reranking, all running on local ONNX models. No Python, no FAISS,
no API keys, no hosted service required.

Point your coding agent at it over MCP, or query it yourself from the CLI.

## Why T3RAG-Rust exists

The [T3 retrieval work](https://huggingface.co/datasets/horus14-wq/fable5-v2-coding-t3)
this builds on showed that searching a model's *reasoning* — not just its
final answers — returns context an agent can actually act on. Its reference
stack was dated Python: e5 + FAISS + glue scripts. T3RAG-Rust reimplements
the idea as a single Rust binary you can run anywhere — no Python env, no
vector-DB service, no GPU unless you want one.

**How it differs from the usual suspects:**

- *vs. the original T3 stack (Python + e5 + FAISS)* — same retrieval idea,
  but self-contained: one binary, ONNX Runtime instead of PyTorch, usearch +
  tantivy instead of FAISS, and a daemon/MCP layer it never had.
- *vs. RAG frameworks (LangChain, txtai, etc.)* — those are libraries you
  wire into an app. T3RAG-Rust is a finished tool: ingest → index → serve →
  MCP, with corpus packaging (`pull`/`publish`) so indexes are shareable
  artifacts, not rebuild-everywhere state.
- *vs. grep* — BM25 is in here too, fused with dense recall. You get both.

## Highlights

- 🧠 **Built for traces, not documents.** Structural chunking with parent
  hydration, question↔text pairing, and episode-aware result grouping —
  shaped for `(question, reasoning)` records, not generic text blobs.
- 🔍 **Real hybrid search.** Qwen3 embeddings → usearch HNSW (dense),
  tantivy BM25 (lexical), weighted RRF fusion, optional SPLADE sparse leg
  and cross-encoder rerank — pick per query or per preset.
- 🔌 **Agent-first.** One daemon keeps models and indexes warm; any number
  of agents hit it over HTTP or MCP (streamable-HTTP *and* stdio).
  `t3rag skill` installs a `SKILL.md` so agents discover it on their own.
- 📦 **Batteries included.** Ingest JSONL / Markdown dirs / Parquet, publish
  and pull corpora to Hugging Face, plus `eval`, `bench`, `verify`,
  `doctor`, and `capabilities` out of the box.
- 🏠 **Actually local.** ONNX Runtime with pluggable execution providers —
  CPU by default; CUDA, TensorRT, DirectML, CoreML, OpenVINO, WebGPU via
  flavored builds. Works on a laptop.

## Demo

```console
$ t3rag search "SQL NULL not equal semantics in WHERE clauses" --corpus raw-chunked
 id      score   title                                              source
 112997  0.0412  approximate_mips_global_probing · ds               frontiercs2:…/run_01M4…#t14
  89102  0.0398  reasoning / reflection                             f5_v2:…#412
  51577  0.0381  reasoning / structure                              f5_v2:…#901

$ t3rag ask "how does redis persistence configuration fail" --max-chars 600
# → returns a prompt-injection-ready context block for your LLM
```

## Contents

- [Why T3RAG-Rust exists](#why-t3rag-rust-exists)
- [Quickstart](#quickstart)
- [Corpora on Hugging Face](#corpora-on-hugging-face)
- [How it works](#how-it-works)
- [Command surface](#command-surface)
- [Flags worth knowing](#flags-worth-knowing)
- [Reranking & performance](#reranking--performance)
- [Measure](#measure)
- [Project status](#project-status)
- [License](#license)

## Quickstart

**Install a prebuilt binary** from
[Releases](https://github.com/horus14-wq/t3ragrust/releases) —
Linux (x86_64/aarch64), macOS, Windows; CPU plus GPU-flavored builds:

```bash
curl -fsSL https://raw.githubusercontent.com/horus14-wq/t3ragrust/main/install.sh | bash
# T3RAG_EP=cuda|coreml|openvino|directml for flavored builds;
# T3RAG_VERSION / T3RAG_DIR to pin or relocate.
# cuda needs NVIDIA driver + CUDA 13 runtime + cuDNN 9.
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/horus14-wq/t3ragrust/main/install.ps1 | iex
# $env:T3RAG_EP='cuda'   # or directml / openvino / cpu
```

**Get a corpus** — pull a prebuilt index (no GPU, no waiting) or build your own:

```bash
t3rag pull --repo horus14-wq/fable5-v2-coding-t3-indexes --subdir struct
#   subdirs: struct | semantic | reflect | raw-chunked

# or index your own (JSONL / dir of .md|.txt / parquet):
t3rag index --input traces.jsonl --name traces
```

**Use it:**

```bash
t3rag search "PowerShell ValidateSet splatting"     # daemon auto-spawns
t3rag ask "how does X fail" --max-chars 600         # prompt-injection block

t3rag serve                                        # http://127.0.0.1:7317
t3rag mcp                                          # stdio MCP (proxies to daemon)
t3rag skill --agents devin                         # install the agent skill
```

MCP clients that accept a URL should register the daemon's streamable-HTTP
endpoint (`{"url": "http://127.0.0.1:7317/mcp"}`) — models stay warm and the
handshake is instant. `t3rag mcp` is the stdio fallback for subprocess-only
clients; it auto-detects NDJSON vs Content-Length framing, answers
`initialize`/`tools/list` immediately, and spawns the daemon in the
background on first tool call.

Uninstalling is deleting the binary plus `~/.cache/t3rag/` — nothing else
is touched.

## Corpora on Hugging Face

| Dataset | Contents |
|---|---|
| [`fable5-v2-coding-t3-indexes`](https://huggingface.co/datasets/horus14-wq/fable5-v2-coding-t3-indexes) | Prebuilt indexes for `t3rag pull`: **struct** / **semantic** / **reflect** (~50K records each) and **raw-chunked** (119,663 passages) — ~50K+ coding reasoning traces per corpus, ready to query |
| [`fable5-v2-coding-t3`](https://huggingface.co/datasets/horus14-wq/fable5-v2-coding-t3) | The source JSONLs those indexes were built from — dedup'd raw traces plus the three T3 transforms, for indexing your own variants |
| [`frontiercs2-t3`](https://huggingface.co/datasets/horus14-wq/frontiercs2-t3) | 36,792 raw-trace records extracted from FrontierCS-2 agent runs: reasoning from 16 models across 48 hard CS tasks (task README as `question`, turn-level `reasoning_content` as `text`) |

## How it works

```
JSONL / .md dir / Parquet
        │
   ingest + structural chunking (parents hydrate children)
        │
   Qwen3-Embedding-0.6B (ONNX, last-token pool, L2 norm)
        │
   ┌────┴─────────────────┐
 usearch HNSW (i8)     tantivy BM25        ──► weighted RRF fusion
        └────┬─────────────────┘                     │
             └──── optional: SPLADE / cross-encoder rerank
                                                     │
        serve: per-user daemon ◄─────────────────────┘
        HTTP :7317  ·  streamable-HTTP MCP  ·  stdio MCP proxy
```

- **Embedder**: local ONNX via `ort` — default
  `onnx-community/Qwen3-Embedding-0.6B-ONNX` (1024-dim, Matryoshka
  truncation to `--dims 32..1024`). `--model` / `T3RAG_MODEL` swaps in any
  registered repo; `t3rag models` lists them.
- **Dense**: usearch HNSW (`--quantization i8|bf16|f16|fp32`) plus
  `vectors.raw.f32` raw fp32 kept for `--exact` recall audits.
- **Lexical**: tantivy BM25 (`title`/`question` TEXT, `source` STRING).
- **Fusion**: weighted RRF (`--rrf-k`, `--dense-weight`, `--bm25-weight`).
- **Daemon**: per-user, `~/.cache/t3rag/daemon.json` for discovery;
  `--token`/`--bind`/`--max-body` hardening knobs.

## Command surface

| Command | What it does |
|---|---|
| `index` | ingest → chunk → embed → write dense + BM25 indexes (`--update` reuses stored vectors) |
| `search` | hybrid query; auto-spawns/proxies the daemon (`--no-daemon` for cold runs) |
| `ask` | same hits shaped as a prompt-injection context block |
| `serve` | per-user daemon: HTTP + streamable-HTTP MCP, resident models |
| `mcp` | stdio MCP proxy for subprocess-only clients |
| `skill` | generate/install the `SKILL.md` agents use to discover t3rag |
| `pull` / `publish` | fetch / upload corpus packages on a HF dataset repo |
| `eval` / `bench` | relevance scoring + latency benchmarking |
| `verify` / `doctor` | corpus integrity gate + environment diagnostics |
| `stats` / `show` | deep corpus stats + print a passage |
| `models` / `capabilities` | registered models + full machine-readable surface |
| `insights` / `feedback` | usage analytics + relevance verdicts |

## Flags worth knowing

- `--corpus a,b` / `all` — multi-corpus fused search.
- `--explain` — per-stage ms + per-leg scores + `score_kind`.
- `--format text|json|jsonl|toon|md|prompt`.
- `--filter key~substr`, `--max-per-source`, `--max-chars`,
  `--min-score`, `--offset` — result shaping.
- `--fast` / `--balanced` / `--max` — latency↔quality presets.
- `--dims N` — Matryoshka vector truncation (Qwen3, 32..1024).
- Index-time: `--embed-batch`, `--embed-token-budget`, `--min-chars`,
  `--max-chars`, `--dedup`, `--no-chunk`, `--dry-run`, `--update`.
- Env: every flag has a `T3RAG_*` mirror; `T3RAG_HOME` moves the cache root.

## Reranking & performance

Reranking is **off by default** — it's the expensive stage (~seconds–minutes
per query on CPU, where hybrid alone is ~ms). Enable per-query with
`--rerank` or via `--balanced`/`--max`:

- `--reranker minilm|bge-base|bge-m3` — minilm is ~15–20× faster than
  bge-m3 but a weaker ranker; bge-base is the middle ground.
- On NVIDIA boxes prefer `--ep cuda` — DirectML runs the forward but
  measured ~3–5 s/pair (soft DML load, high util / low watts).
- `serve` takes `--rerank-sessions N` to shard pairs across N loaded
  sessions (GPU forward overlap; each adds model-size memory),
  `--rerank-tokens` (512 default; 256 is a cheap speedup), and
  `--preload-reranker` to warm the pool at startup.
- Pairs are length-sorted and micro-batched internally, so a batch of 50
  never pays 512-token padding for every pair.

## Measure

```bash
t3rag eval   --corpus fixture --queries tests/fixtures/queries.jsonl
t3rag bench  --corpus fixture --iters 50 [--cold]
t3rag verify --corpus fixture        # structural + id-bounds integrity gate
t3rag stats  --corpus fixture && t3rag doctor
```

Benchmarks and the ANN-vs-exact audit methodology are in `DESIGN.md`;
prototype lessons live in `docs/`.

## Project status

`v0.1.x` — the pipeline, daemon, and MCP surface are working software with
releases on all three OSes; APIs and corpus formats may still move between
minor versions (`meta.json` carries a `format_version`). Issues and PRs
welcome.

## License

Dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
