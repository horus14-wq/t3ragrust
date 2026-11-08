use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

/// Rust-native retrieval over thinking traces (T3).
#[derive(Parser)]
#[command(name = "t3rag", version, about, long_about = None)]
pub struct Cli {
    /// Increase log verbosity (-v = debug).
    #[arg(short, long, global = true)]
    pub verbose: bool,
    /// Errors only.
    #[arg(short, long, global = true)]
    pub quiet: bool,
    /// Tracing filter (overrides -v/-q): e.g. `debug`, `t3rag=trace`.
    #[arg(long, global = true, env = "T3RAG_LOG")]
    pub log_level: Option<String>,
    /// TOML config with flag defaults (flags win over config).
    #[arg(long, global = true, env = "T3RAG_CONFIG")]
    pub config: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// Dispatch the parsed subcommand. `search`/`ask` first merge
    /// `--config` TOML defaults into their unset `Option` args.
    ///
    /// # Errors
    /// Config load/parse fails, or the subcommand's `run_*` fails.
    pub fn run(self) -> Result<()> {
        let cfg = crate::config::load(self.config.as_deref())?;
        match self.command {
            Command::Index(args) => crate::index::run_index(args),
            Command::Search(mut args) => {
                args.apply_config(&cfg);
                crate::search::run_search(args)
            }
            Command::Ask(mut args) => {
                args.apply_config(&cfg);
                args.format = OutFmt::Prompt; // injectable context block
                crate::search::run_search(args)
            }
            Command::Serve(args) => crate::serve::run_serve(args),
            Command::Mcp(args) => crate::mcp::run_mcp(args),
            Command::Eval(args) => crate::eval::run_eval(args),
            Command::Bench(args) => crate::bench::run_bench(args),
            Command::Skill(args) => crate::skill::run_skill(args),
            Command::List => crate::doctor::run_list(),
            Command::Log(args) => crate::log::run_log(&args),
            Command::Stats(args) => crate::doctor::run_stats(args.corpus),
            Command::Doctor { deep } => crate::doctor::run_doctor(deep),
            Command::Verify { corpus } => {
                print!("{}", crate::verify::verify(corpus)?.render());
                Ok(())
            }
            Command::Health => crate::doctor::run_health(),
            Command::Capabilities => crate::capabilities::run_capabilities(),
            Command::Models => crate::capabilities::run_models(),
            Command::Export(args) => crate::capabilities::run_export(args),
            Command::Publish(args) => crate::capabilities::run_publish(args),
            Command::Pull(args) => crate::capabilities::run_pull(args),
            Command::Show { corpus, id } => crate::capabilities::run_show(corpus, id),
            Command::Feedback(args) => crate::capabilities::run_feedback(args),
            Command::Suggest { query, json } => crate::capabilities::run_suggest(&query, json),
            Command::Insights { since, json } => crate::capabilities::run_insights(since, json),
            Command::Completions { shell } => crate::capabilities::run_completions(shell),
            Command::Man { dir } => crate::capabilities::run_man(&dir),
            Command::Selftest => crate::capabilities::run_selftest(),
        }
    }
}

/// Top-level subcommands (see each variant for behavior).
#[derive(Subcommand)]
pub enum Command {
    /// Build a corpus: read passages, embed, write vector+lexical indexes.
    Index(IndexArgs),
    /// Query a corpus: embed query, hybrid retrieve, rerank, print passages.
    Search(SearchArgs),
    /// Query + emit an LLM-injectable context block (search --format prompt).
    Ask(SearchArgs),
    /// Daemon: keep model + indexes resident; HTTP + streamable-HTTP MCP.
    Serve(ServeArgs),
    /// Stdio MCP server that proxies to a running daemon (never loads models).
    Mcp(McpArgs),
    /// Retrieval evaluation: recall@k / MRR / nDCG over a queries file.
    Eval(EvalArgs),
    /// Latency bench: p50/p95/p99 per stage through the daemon.
    Bench(BenchArgs),
    /// Emit or install a SKILL.md wrapper for agent discovery.
    Skill(SkillArgs),
    /// List indexed corpora (name, docs, dims, model, size, age).
    List,
    /// Search history: filter the append-only query log (t3-pbh).
    Log(LogArgs),
    /// Deep stats for one corpus (default: newest).
    Stats(StatsArgs),
    /// Environment sanity: SIMD, EPs, HF cache, corpora, daemon.
    Doctor {
        /// Also run `verify` integrity checks on every corpus.
        #[arg(long)]
        deep: bool,
    },
    /// Corpus integrity: meta/store/vector/BM25 consistency + staleness.
    Verify {
        /// Corpus name (default: newest).
        #[arg(long, env = "T3RAG_CORPUS")]
        corpus: Option<String>,
    },
    /// <50ms pre-flight check (agents ping this).
    Health,
    /// Machine-readable CLI surface (commands/flags/endpoints as JSON).
    Capabilities,
    /// Embedding + reranker model registry with cache status.
    Models,
    /// Export a corpus back to JSONL (round-trip).
    Export(ExportArgs),
    /// Upload a corpus to a Hugging Face dataset repo (t3-css.3.9).
    Publish(PublishArgs),
    /// Download a published corpus into the local corpora root.
    Pull(PullArgs),
    /// Rank corpora for a query before searching (t3-69w).
    Suggest {
        /// The query/problem to route.
        query: String,
        /// JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Aggregate the query log + feedback into usage insights (t3-qc4).
    Insights {
        /// Only include entries from the last N hours.
        #[arg(long)]
        since: Option<u64>,
        /// JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Record a relevance verdict on a query's hits (t3-bj5).
    Feedback(FeedbackArgs),
    /// Print one passage in full (no model needed).
    Show {
        /// Corpus name (default: newest).
        #[arg(long, env = "T3RAG_CORPUS")]
        corpus: Option<String>,
        /// Passage id.
        id: u64,
    },
    /// Generate shell completions.
    Completions {
        #[arg(long, value_enum)]
        shell: clap_complete::Shell,
    },
    /// Generate man pages into a directory.
    Man {
        #[arg(long)]
        dir: PathBuf,
    },
    /// Clap-tree sanity check (<50ms).
    Selftest,
}

/// Arguments for `t3rag index`: ingest -> embed -> write indexes.
#[derive(Args)]
pub struct IndexArgs {
    /// Corpus input: JSONL file, directory of .md/.txt, or .parquet file.
    #[arg(long, env = "T3RAG_INPUT")]
    pub input: PathBuf,
    /// Corpus name (artifact dir under the cache root).
    #[arg(long, env = "T3RAG_NAME")]
    pub name: String,
    /// Matryoshka dims: truncate vectors to N (L2-renormed; model must be
    /// MRL-capable — Qwen3 truncates cleanly to 32..1024).
    #[arg(long, env = "T3RAG_DIMS")]
    pub dims: Option<usize>,
    /// Text column for parquet input (auto-detect: text/content/passage).
    #[arg(long)]
    pub text_col: Option<String>,
    /// Embedding model override (default: Qwen3-Embedding-0.6B ONNX).
    #[arg(long, env = "T3RAG_MODEL")]
    pub model: Option<String>,
    /// Passage instruction prefix override (e5-family `passage: `;
    /// empty for most models).
    #[arg(long, env = "T3RAG_INSTRUCTION")]
    pub instruction: Option<String>,
    /// Vector quantization for the dense index.
    #[arg(long, value_enum, default_value_t = Quantization::I8, env = "T3RAG_QUANTIZATION")]
    pub quantization: Quantization,
    /// Docs per embed batch (default 16).
    #[arg(long, default_value_t = 16)]
    pub embed_batch: usize,
    /// Token budget per embed batch (default 4096 — ORT tensor limit).
    #[arg(long, default_value_t = 4096)]
    pub embed_token_budget: usize,
    /// Drop passages shorter than N chars.
    #[arg(long)]
    pub min_chars: Option<usize>,
    /// Drop passages longer than N chars (pre-chunk).
    #[arg(long)]
    pub max_chars: Option<usize>,
    /// Exact-hash dedup by passage text at ingest.
    #[arg(long)]
    pub dedup: bool,
    /// Disable structural chunking (A/B eval control).
    #[arg(long)]
    pub no_chunk: bool,
    /// Incremental update (t3-css.4.1): rebuild the corpus but reuse
    /// stored vectors for unchanged passage texts — only new/edited
    /// rows get embedded. Requires an existing compatible corpus
    /// (same model/dims/instruction, raw vectors present).
    #[arg(long)]
    pub update: bool,
    /// Count/ingest only — no embeddings or index writes.
    #[arg(long)]
    pub dry_run: bool,
    /// Also build a learned sparse (SPLADE) leg — needs --features sparse.
    #[arg(long)]
    pub sparse: bool,
    /// Parquet read batch size (rows per RecordBatch).
    #[arg(long, default_value_t = 8192)]
    pub batch_size: usize,
    /// Tantivy writer heap in MB.
    #[arg(long, default_value_t = 50)]
    pub bm25_heap_mb: usize,
    /// Tantivy indexing threads (0 = tantivy's own default).
    #[arg(long, default_value_t = 0)]
    pub bm25_threads: usize,
    /// usearch HNSW connectivity (M).
    #[arg(long, default_value_t = 16)]
    pub connectivity: usize,
    /// usearch expansion during index build.
    #[arg(long, default_value_t = 128)]
    pub expansion_add: usize,
    /// Skip vectors.raw.f32 — halves index disk, but --exact search then
    /// fails honestly ("built with --no-raw-vectors").
    #[arg(long)]
    pub no_raw_vectors: bool,
    /// Force structural chunking for every passage (wins over --no-chunk).
    #[arg(long)]
    pub always_chunk: bool,
    /// Chunk trigger in est-tokens (overrides the 7000 default).
    #[arg(long)]
    pub chunk_tokens: Option<usize>,
    /// Chunk target size in chars (overrides 15000).
    #[arg(long)]
    pub chunk_chars: Option<usize>,
    /// Show a progress bar on long indexes (auto-off without a tty).
    #[arg(long, default_value_t = true, overrides_with = "no_progress")]
    pub progress: bool,
    /// Disable the index progress bar.
    #[arg(long)]
    pub no_progress: bool,
    /// ONNX execution provider. `auto` tries every compiled-in EP and
    /// silently falls back to CPU. Non-cpu choices require the matching
    /// cargo feature (ep-cuda / ep-openvino / ep-webgpu).
    #[arg(long, value_enum, default_value_t = Ep::Auto, env = "T3RAG_EP")]
    pub ep: Ep,
    /// Parallel embedding sessions (each loads its own ~600MB model copy).
    #[arg(long, default_value_t = 1, env = "T3RAG_POOL")]
    pub pool: usize,
    /// ORT intra-op threads per session (0 = cores/pool).
    #[arg(long, default_value_t = 0, env = "T3RAG_THREADS")]
    pub threads: usize,
}

/// Arguments for `t3rag search`/`t3rag ask`.
#[derive(Args)]
pub struct SearchArgs {
    /// The query text.
    pub query: String,
    /// Extra query phrasings (repeatable) — each is embedded and its
    /// rankings fused into the same RRF merge (t3-css.4.4 multi-query).
    #[arg(long = "also", value_name = "QUERY")]
    pub also: Vec<String>,
    /// Corpus to search — name, comma-list, or 'all' (default: newest).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Option<String>,
    /// Number of passages to return.
    #[arg(long, default_value_t = 3, env = "T3RAG_TOP_K")]
    pub top_k: usize,
    /// Candidates fetched per retrieval leg before fusion/rerank.
    #[arg(long, default_value_t = 50, env = "T3RAG_CANDIDATES")]
    pub candidates: usize,
    /// Post-filter: 'key~substr' (repeatable; keys: source/title/question).
    #[arg(long)]
    pub filter: Vec<String>,
    /// Cap hits per source (episode diversity).
    #[arg(long)]
    pub max_per_source: Option<usize>,
    /// Truncate each passage's text to N chars.
    #[arg(long)]
    pub max_chars: Option<usize>,
    /// Drop hits below score.
    #[arg(long)]
    pub min_score: Option<f32>,
    /// Skip the first N ranked hits (pagination).
    #[arg(long)]
    pub offset: Option<usize>,
    /// Exclude passage ids from results (comma-list; repeatable) —
    /// applied before top_k so the list still fills (t3-b68).
    #[arg(long, value_delimiter = ',')]
    pub exclude: Vec<u64>,
    /// More-like-this: dense search seeded by a passage id's text — the
    /// seed id is excluded from results (t3-kz0; daemon path).
    #[arg(long)]
    pub like: Option<u64>,
    /// More-like-this seeded by ad-hoc text instead of a passage id.
    #[arg(long)]
    pub like_text: Option<String>,
    /// Group results by question|source|corpus — makes the multi-corpus
    /// lens structure visible (t3-tdu).
    #[arg(long)]
    pub group_by: Option<String>,
    /// Pseudo-relevance feedback: blend top-3 hit vectors into the query
    /// and re-run the dense leg (t3-4vh; daemon path).
    #[arg(long)]
    pub expand: bool,
    /// Stop filling results once passage text exceeds N tokens
    /// (chars/4 heuristic) — keeps injected context inside the window.
    #[arg(long)]
    pub token_budget: Option<usize>,
    /// Batch mode: JSONL file of independent queries ({"query": "..."} or
    /// a plain string per line) — one JSON result object emitted per line
    /// (t3-w4d; daemon path).
    #[arg(long, env = "T3RAG_BATCH")]
    pub batch: Option<PathBuf>,
    /// Enable the cross-encoder reranker (slow on CPU).
    #[arg(long, overrides_with = "no_rerank")]
    pub rerank: bool,
    /// Disable the cross-encoder reranker.
    #[arg(long, overrides_with = "rerank")]
    pub no_rerank: bool,
    /// Add the learned sparse (SPLADE) leg when the corpus has sparse.bin.
    #[arg(long)]
    pub sparse: bool,
    /// Emit structured JSON instead of human-readable output.
    #[arg(long)]
    pub json: bool,
    /// Output format: text, json, jsonl, toon (compact for agents), md.
    #[arg(long, value_enum, default_value_t = OutFmt::Text)]
    pub format: OutFmt,
    /// Fast preset: smallest embedder, tiny/no reranker (~50-150ms).
    #[arg(long, group = "preset")]
    pub fast: bool,
    /// Balanced preset: 0.6B embedder + truncated parallel rerank.
    #[arg(long, group = "preset")]
    pub balanced: bool,
    /// Max preset: 4B/8B embedder + full rerank (seconds, best precision).
    #[arg(long, group = "preset")]
    pub max: bool,
    /// Embedding model override.
    #[arg(long, env = "T3RAG_MODEL")]
    pub model: Option<String>,
    /// Daemon usage: auto=spawn if absent, always=spawn-or-fail, never=cold
    /// in-process path. Machine output (--json) never blocks on a spawn.
    #[arg(long, value_enum, default_value_t = DaemonMode::Auto)]
    pub daemon: DaemonMode,
    /// Bypass the daemon entirely (equivalent to --daemon never).
    #[arg(long)]
    pub no_daemon: bool,
    /// Exact brute-force scan instead of ANN (recall reference).
    #[arg(long)]
    pub exact: bool,
    /// Query instruction prefix override (models are instruction-tuned;
    /// the registry default applies when unset).
    #[arg(long, env = "T3RAG_INSTRUCTION")]
    pub instruction: Option<String>,
    /// Emit per-stage timings + per-leg scores in the response.
    #[arg(long)]
    pub explain: bool,
    /// ANN ef-search override (recall↔latency dial; usearch runtime).
    #[arg(long)]
    pub ef_search: Option<usize>,
    /// RRF constant k (default 60).
    #[arg(long, default_value_t = 60.0)]
    pub rrf_k: f32,
    /// Dense leg weight in fusion (default 1.0).
    #[arg(long, default_value_t = 1.0)]
    pub dense_weight: f32,
    /// BM25 leg weight in fusion (default 1.0).
    #[arg(long, default_value_t = 1.0)]
    pub bm25_weight: f32,
    /// Reranker name (registry: minilm/bge-base/bge-m3 or repo id).
    #[arg(long, env = "T3RAG_RERANKER")]
    pub reranker: Option<String>,
    /// Candidates fed to the reranker (decoupled from --candidates).
    #[arg(long)]
    pub rerank_candidates: Option<usize>,
    /// ONNX execution provider (see `t3rag index --ep`).
    #[arg(long, value_enum, default_value_t = Ep::Auto, env = "T3RAG_EP")]
    pub ep: Ep,
    /// Parallel embedding sessions (query path only uses one; >1 is for
    /// concurrent workloads).
    #[arg(long, default_value_t = 1, env = "T3RAG_POOL")]
    pub pool: usize,
    /// ORT intra-op threads per session (0 = cores/pool).
    #[arg(long, default_value_t = 0, env = "T3RAG_THREADS")]
    pub threads: usize,
}

/// Arguments for `t3rag serve` (daemon lifecycle + hardening).
#[derive(Args)]
pub struct ServeArgs {
    /// Port to bind.
    #[arg(long, default_value_t = 7317, env = "T3RAG_PORT")]
    pub port: u16,
    /// Bind address (default loopback; use 0.0.0.0 only behind a firewall +
    /// --token).
    #[arg(long, default_value = "127.0.0.1", env = "T3RAG_BIND")]
    pub bind: String,
    /// Bearer token for /search and /mcp (unset = open on loopback).
    #[arg(long, env = "T3RAG_TOKEN")]
    pub token: Option<String>,
    /// Max request body bytes (default 1MB).
    #[arg(long, default_value_t = 1_048_576)]
    pub max_body: usize,
    /// Unload model/indexes after N seconds idle.
    #[arg(long, env = "T3RAG_IDLE_TIMEOUT")]
    pub idle_timeout: Option<u64>,
    /// Preload a corpus at startup (repeatable; default: none until first query).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Vec<String>,
    /// Embedding model override (default: Qwen3-Embedding-0.6B ONNX).
    #[arg(long, env = "T3RAG_MODEL")]
    pub model: Option<String>,
    /// Query instruction prefix override (registry default applies when unset).
    #[arg(long, env = "T3RAG_INSTRUCTION")]
    pub instruction: Option<String>,
    /// Daemon's default reranker (registry: minilm/bge-base/bge-m3).
    #[arg(long, default_value = "bge-m3", env = "T3RAG_RERANKER")]
    pub reranker: String,
    /// Reranker pair truncation cap (default 512).
    #[arg(long, default_value_t = 512)]
    pub rerank_tokens: usize,
    /// ONNX execution provider (see `t3rag index --ep`).
    #[arg(long, value_enum, default_value_t = Ep::Auto, env = "T3RAG_EP")]
    pub ep: Ep,
    /// Parallel embedding sessions (inference semaphore size).
    #[arg(long, default_value_t = 2, env = "T3RAG_POOL")]
    pub pool: usize,
    /// ORT intra-op threads per session (0 = cores/pool).
    #[arg(long, default_value_t = 0, env = "T3RAG_THREADS")]
    pub threads: usize,
}

/// Arguments for `t3rag mcp` (stdio MCP proxy).
#[derive(Args)]
pub struct McpArgs {
    /// Daemon base URL override (default: discover/auto-spawn).
    #[arg(long)]
    pub daemon_url: Option<String>,
}

/// Arguments for `t3rag eval` (recall@k/MRR/nDCG over a queries file).
#[derive(Args)]
pub struct EvalArgs {
    /// Queries file: JSONL of {query, relevant_ids[]}.
    /// Not required with --history (which generates one).
    #[arg(long, env = "T3RAG_QUERIES", required_unless_present = "history")]
    pub queries: Option<PathBuf>,
    /// Build the queries file from log.jsonl + feedback.jsonl instead of
    /// reading one (t3-0xi): real usage becomes the regression set.
    #[arg(long)]
    pub history: bool,
    /// Output path for --history (default: stdout).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Corpus to eval (repeatable — multiple = side-by-side A/B).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Vec<String>,
    /// Cross-encoder rerank.
    #[arg(long)]
    pub rerank: bool,
    /// Brute-force exact scan instead of ANN (recall-reference audit).
    #[arg(long)]
    pub exact: bool,
    /// Fast preset (no rerank, fewer candidates).
    #[arg(long)]
    pub fast: bool,
    /// Add the learned sparse (SPLADE) leg when the corpus has sparse.bin.
    #[arg(long)]
    pub sparse: bool,
    /// Pseudo-relevance feedback (t3-4vh) for the A/B eval gate.
    #[arg(long)]
    pub expand: bool,
    /// JSON output.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `t3rag bench` (per-stage latency percentiles).
#[derive(Args)]
pub struct BenchArgs {
    /// Query text (one line; bench repeats it — LRU dedupes query embed,
    /// which is the point: measures the warm resident path).
    #[arg(long)]
    pub query: String,
    /// Corpus (default: newest).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Vec<String>,
    /// Warm iterations.
    #[arg(long, default_value_t = 20)]
    pub iterations: usize,
    /// Also measure cold-start (spawn + first query).
    #[arg(long)]
    pub cold: bool,
    /// Cross-encoder rerank.
    #[arg(long)]
    pub rerank: bool,
    /// Brute-force exact scan.
    #[arg(long)]
    pub exact: bool,
    /// Passages to return.
    #[arg(long, default_value_t = 3, env = "T3RAG_TOP_K")]
    pub top_k: usize,
    /// Fused candidates.
    #[arg(long, default_value_t = 50, env = "T3RAG_CANDIDATES")]
    pub candidates: usize,
    /// JSON output.
    #[arg(long)]
    pub json: bool,
}

/// Output formats for `search` (t3-css.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutFmt {
    /// Human-readable blocks (default).
    Text,
    /// Full JSON response object.
    Json,
    /// One JSON object per hit.
    Jsonl,
    /// Compact one-line-per-hit (agent-token-optimal).
    Toon,
    /// Markdown hits.
    Md,
    /// Prompt-injection block (the `ask` format).
    Prompt,
}

impl SearchArgs {
    /// Merge config defaults into unset Option args (flags win).
    pub fn apply_config(&mut self, cfg: &crate::config::Config) {
        if self.corpus.is_none() {
            self.corpus = cfg.corpus.clone();
        }
        if self.model.is_none() {
            self.model = cfg.model.clone();
        }
        if self.reranker.is_none() {
            self.reranker = cfg.reranker.clone();
        }
        if self.instruction.is_none() {
            self.instruction = cfg.instruction.clone();
        }
    }
    /// Effective format: --format wins; --json aliases to Json.
    pub fn out_format(&self) -> OutFmt {
        if self.json {
            OutFmt::Json
        } else {
            self.format
        }
    }
}

/// Arguments for `t3rag log` (query-history reader, t3-pbh).
#[derive(Args)]
pub struct LogArgs {
    /// Only entries newer than this: 90s, 10m, 1h, 7d (bare number = seconds).
    #[arg(long)]
    pub since: Option<String>,
    /// Only entries that searched this corpus.
    #[arg(long)]
    pub corpus: Option<String>,
    /// Only entries whose query matches this regex.
    #[arg(long)]
    pub query_regex: Option<String>,
    /// Emit raw JSONL lines instead of the table.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `t3rag stats` (deep corpus inspection).
#[derive(Args)]
pub struct StatsArgs {
    /// Corpus name (default: newest).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Option<String>,
}

/// Arguments for `t3rag export` (corpus -> JSONL round-trip).
#[derive(Args)]
pub struct ExportArgs {
    /// Corpus name (default: newest).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Option<String>,
    /// Output JSONL path.
    #[arg(long)]
    pub out: PathBuf,
}

/// Arguments for `t3rag publish` (corpus -> HF dataset repo).
#[derive(Args)]
pub struct PublishArgs {
    /// Corpus name (default: newest).
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Option<String>,
    /// Target repo, e.g. `owner/name` (a HF dataset repo).
    #[arg(long)]
    pub repo: String,
    /// HF repo type.
    #[arg(long, default_value = "dataset")]
    pub repo_type: String,
    /// Subdirectory within the repo to upload into (multi-corpus repos).
    #[arg(long)]
    pub subdir: Option<String>,
}

/// Arguments for `t3rag pull` (HF dataset repo -> local corpus).
#[derive(Args)]
pub struct PullArgs {
    /// Source repo, e.g. `owner/name`.
    #[arg(long)]
    pub repo: String,
    /// Local corpus name (default: repo's last path component).
    #[arg(long, env = "T3RAG_NAME")]
    pub name: Option<String>,
    /// HF repo type.
    #[arg(long, default_value = "dataset")]
    pub repo_type: String,
    /// Subdirectory within the repo to pull (default name: its last path component).
    #[arg(long)]
    pub subdir: Option<String>,
}

/// Verdict on a search hit set (t3-bj5).
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Verdict {
    Relevant,
    Irrelevant,
}

/// Arguments for `t3rag feedback` — appends to ~/.cache/t3rag/feedback.jsonl.
#[derive(Args)]
pub struct FeedbackArgs {
    /// The query that produced the hits.
    #[arg(long)]
    pub query: Option<String>,
    /// Passage ids the verdict applies to (comma-list).
    #[arg(long, value_delimiter = ',')]
    pub ids: Vec<u64>,
    /// Verdict: relevant | irrelevant.
    #[arg(long, value_enum)]
    pub verdict: Verdict,
    /// Corpus the hits came from.
    #[arg(long, env = "T3RAG_CORPUS")]
    pub corpus: Option<String>,
}

/// Arguments for `t3rag skill` (SKILL.md emit/install/uninstall).
#[derive(Args)]
pub struct SkillArgs {
    /// Write SKILL.md into this directory instead of printing.
    /// Produces `<dir>/t3rag/SKILL.md` (Agent Skills spec layout).
    #[arg(long)]
    pub install: Option<PathBuf>,
    /// Install into a known agent skills dir (relative to CWD):
    /// claude -> .claude/skills, devin -> .devin/skills,
    /// cursor/generic -> .agents/skills.
    #[arg(long, value_enum)]
    pub agents: Option<SkillAgents>,
    /// Overwrite an existing SKILL.md.
    #[arg(long)]
    pub force: bool,
    /// Remove the installed t3rag/ skill dir (requires --yes; prints
    /// the exact path first — deletion is deliberate).
    #[arg(long)]
    pub uninstall: bool,
    /// Confirm the uninstall.
    #[arg(long)]
    pub yes: bool,
}

/// Agent-specific skills-dir preset for `t3rag skill --agents`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum SkillAgents {
    /// `.claude/skills` (Claude Code).
    Claude,
    /// `.devin/skills` (Devin).
    Devin,
    /// `.agents/skills` (Cursor-compatible shared dir).
    Cursor,
    /// `.agents/skills` (plain agents dir).
    Generic,
}

impl SkillAgents {
    /// Repo-relative skills directory this preset installs into.
    pub fn skills_dir(self) -> &'static str {
        match self {
            Self::Claude => ".claude/skills",
            Self::Devin => ".devin/skills",
            Self::Cursor | Self::Generic => ".agents/skills",
        }
    }
}

/// Whether `t3rag search` uses the resident daemon (t3-css.1.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DaemonMode {
    /// Use if running; spawn if absent (default).
    Auto,
    /// Spawn-or-fail: error instead of cold fallback.
    Always,
    /// Never touch the daemon — cold in-process path.
    Never,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Quantization {
    /// 8-bit integer — smallest/faster, default.
    I8,
    /// Brain floating point.
    Bf16,
    /// IEEE half precision.
    F16,
    /// Full precision — largest/slower, best recall ceiling.
    Fp32,
}

/// ONNX Runtime execution-provider selection (runtime flag; availability
/// is gated by the ep-* cargo feature the binary was built with).
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Ep {
    /// Try every compiled-in EP (non-fatal failures), fall back to CPU.
    Auto,
    /// Force CPU only.
    Cpu,
    Cuda,
    Tensorrt,
    Openvino,
    Webgpu,
    /// Apple CoreML EP (macOS/arm64; needs --features ep-coreml).
    Coreml,
    /// DirectML EP (Windows/AMD/NVIDIA/Intel GPUs; needs --features ep-directml).
    Directml,
}
