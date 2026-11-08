# t3rag

Rust-native retrieval over thinking traces (T3): index reasoning corpora,
then hybrid-search them from any agent — dense ANN + BM25 + optional
cross-encoder rerank, all local ONNX (no API keys).

## Quickstart

```bash
# Index a corpus (JSONL / a dir of .md/.txt / parquet)
t3rag index --input traces.jsonl --name traces

# ...or pull a prebuilt one — no indexing or GPU needed:
t3rag pull --repo horus14-wq/fable5-v2-coding-t3-indexes --subdir struct
#   (subdirs: struct | semantic | reflect | raw-chunked — 52K+ coding
#    reasoning traces each, ready to query)

# Query it (daemon auto-spawns on first call)
t3rag search "PowerShell ValidateSet splatting"
t3rag ask "how does X fail" --max-chars 600   # prompt-injection block

# Persistent daemon + MCP for agent clients
t3rag serve                                 # http://127.0.0.1:7317
t3rag mcp                                   # stdio MCP (proxies to daemon)
t3rag skill --agents devin                  # install the agent skill
```

## Architecture

- **Embedder**: local ONNX models via `ort` (default
  `onnx-community/Qwen3-Embedding-0.6B-ONNX`; `--model` / `T3RAG_MODEL`
  for any registered repo — `t3rag models` lists them).
- **Dense**: usearch HNSW (`--quantization i8|bf16|f16|fp32`) +
  `vectors.raw.f32` raw fp32 for `--exact` recall audits.
- **Lexical**: tantivy BM25 (`title`/`question` TEXT, `source` STRING).
- **Fusion**: weighted RRF (`--rrf-k`, `--dense-weight`, `--bm25-weight`).
- **Rerank**: optional cross-encoder, **off by default** (~seconds–minute
  per query on CPU). Enable per-query with `--rerank`, or via the
  `--balanced`/`--max` presets; `--reranker minilm|bge-base|bge-m3`.
- **Daemon**: per-user, `~/.cache/t3rag/daemon.json` for discovery;
  HTTP + streamable-HTTP MCP; `--token`/`--bind`/`--max-body` hardening.

## Flags worth knowing

- `--corpus a,b` / `all` — multi-corpus fused search.
- `--explain` — per-stage ms + per-leg scores + `score_kind`.
- `--format text|json|jsonl|toon|md|prompt`.
- `--filter key~substr`, `--max-per-source`, `--max-chars`,
  `--min-score`, `--offset` — result shaping.
- `--dims N` — Matryoshka truncation (Qwen3, 32..1024).
- Index-time: `--embed-batch`, `--embed-token-budget`, `--min-chars`,
  `--max-chars`, `--dedup`, `--no-chunk`, `--dry-run`.
- Env: `T3RAG_*` mirrors flags; `T3RAG_HOME` overrides the cache root.
- `t3rag capabilities` — full machine-readable surface.

## Measure

```bash
t3rag eval  --corpus fixture --queries tests/fixtures/queries.jsonl
t3rag bench --corpus fixture --iters 50 [--cold]
t3rag stats --corpus fixture && t3rag doctor
```

Benchmarks and the ANN-vs-exact audit methodology are in `DESIGN.md`;
prototype lessons live in `docs/`.
