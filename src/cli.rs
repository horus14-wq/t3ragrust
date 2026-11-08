use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

/// Rust-native retrieval over thinking traces (T3).
#[derive(Parser)]
#[command(name = "t3rag", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        match self.command {
            Command::Index(args) => crate::index::run_index(args),
            Command::Search(args) => crate::search::run_search(args),
            Command::Serve(args) => crate::serve::run_serve(args),
            Command::Mcp(args) => crate::mcp::run_mcp(args),
            Command::Eval(args) => crate::eval::run_eval(args),
            Command::Bench(args) => crate::bench::run_bench(args),
            Command::Skill(args) => crate::skill::run_skill(args),
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Build a corpus: read passages, embed, write vector+lexical indexes.
    Index(IndexArgs),
    /// Query a corpus: embed query, hybrid retrieve, rerank, print passages.
    Search(SearchArgs),
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
}

#[derive(Args)]
pub struct IndexArgs {
    /// Corpus input: JSONL file, directory of .md/.txt, or .parquet file.
    #[arg(long)]
    pub input: PathBuf,
    /// Corpus name (artifact dir under the cache root).
    #[arg(long)]
    pub name: String,
    /// Text column for parquet input (auto-detect: text/content/passage).
    #[arg(long)]
    pub text_col: Option<String>,
    /// Embedding model override (default: Qwen3-Embedding-0.6B ONNX).
    #[arg(long)]
    pub model: Option<String>,
    /// Passage instruction prefix override (e5-family `passage: `;
    /// empty for most models).
    #[arg(long)]
    pub instruction: Option<String>,
    /// Vector quantization for the dense index.
    #[arg(long, value_enum, default_value_t = Quantization::I8)]
    pub quantization: Quantization,
    /// ONNX execution provider. `auto` tries every compiled-in EP and
    /// silently falls back to CPU. Non-cpu choices require the matching
    /// cargo feature (ep-cuda / ep-openvino / ep-webgpu).
    #[arg(long, value_enum, default_value_t = Ep::Auto)]
    pub ep: Ep,
    /// Parallel embedding sessions (each loads its own ~600MB model copy).
    #[arg(long, default_value_t = 1)]
    pub pool: usize,
    /// ORT intra-op threads per session (0 = cores/pool).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Args)]
pub struct SearchArgs {
    /// The query text.
    pub query: String,
    /// Corpus to search (default: most recently indexed).
    #[arg(long)]
    pub corpus: Option<String>,
    /// Number of passages to return.
    #[arg(long, default_value_t = 3)]
    pub top_k: usize,
    /// Candidates fetched per retrieval leg before fusion/rerank.
    #[arg(long, default_value_t = 50)]
    pub candidates: usize,
    /// Enable the cross-encoder reranker (default).
    #[arg(long, overrides_with = "no_rerank")]
    pub rerank: bool,
    /// Disable the cross-encoder reranker.
    #[arg(long, overrides_with = "rerank")]
    pub no_rerank: bool,
    /// Emit structured JSON instead of human-readable output.
    #[arg(long)]
    pub json: bool,
    /// Fast preset: smallest embedder, tiny/no reranker (~50-150ms).
    #[arg(long, group = "preset")]
    pub fast: bool,
    /// Balanced preset (default): 0.6B embedder + truncated parallel rerank.
    #[arg(long, group = "preset")]
    pub balanced: bool,
    /// Max preset: 4B/8B embedder + full rerank (seconds, best precision).
    #[arg(long, group = "preset")]
    pub max: bool,
    /// Embedding model override.
    #[arg(long)]
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
    #[arg(long)]
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
    #[arg(long)]
    pub reranker: Option<String>,
    /// Candidates fed to the reranker (decoupled from --candidates).
    #[arg(long)]
    pub rerank_candidates: Option<usize>,
    /// ONNX execution provider (see `t3rag index --ep`).
    #[arg(long, value_enum, default_value_t = Ep::Auto)]
    pub ep: Ep,
    /// Parallel embedding sessions (query path only uses one; >1 is for
    /// concurrent workloads).
    #[arg(long, default_value_t = 1)]
    pub pool: usize,
    /// ORT intra-op threads per session (0 = cores/pool).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Args)]
pub struct ServeArgs {
    /// Port to bind on 127.0.0.1.
    #[arg(long, default_value_t = 7317)]
    pub port: u16,
    /// Unload model/indexes after N seconds idle.
    #[arg(long)]
    pub idle_timeout: Option<u64>,
    /// Preload a corpus at startup (repeatable; default: none until first query).
    #[arg(long)]
    pub corpus: Vec<String>,
    /// Embedding model override (default: Qwen3-Embedding-0.6B ONNX).
    #[arg(long)]
    pub model: Option<String>,
    /// Query instruction prefix override (registry default applies when unset).
    #[arg(long)]
    pub instruction: Option<String>,
    /// Daemon's default reranker (registry: minilm/bge-base/bge-m3).
    #[arg(long, default_value = "bge-m3")]
    pub reranker: String,
    /// Reranker pair truncation cap (default 512).
    #[arg(long, default_value_t = 512)]
    pub rerank_tokens: usize,
    /// ONNX execution provider (see `t3rag index --ep`).
    #[arg(long, value_enum, default_value_t = Ep::Auto)]
    pub ep: Ep,
    /// Parallel embedding sessions (inference semaphore size).
    #[arg(long, default_value_t = 2)]
    pub pool: usize,
    /// ORT intra-op threads per session (0 = cores/pool).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Args)]
pub struct McpArgs {
    /// Daemon base URL override (default: discover/auto-spawn).
    #[arg(long)]
    pub daemon_url: Option<String>,
}

#[derive(Args)]
pub struct EvalArgs {
    /// Queries file: JSONL of {query, relevant_ids[]}.
    #[arg(long)]
    pub queries: PathBuf,
    /// Corpus to eval (repeatable — multiple = side-by-side A/B).
    #[arg(long)]
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
    /// JSON output.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct BenchArgs {
    /// Query text (one line; bench repeats it — LRU dedupes query embed,
    /// which is the point: measures the warm resident path).
    #[arg(long)]
    pub query: String,
    /// Corpus (default: newest).
    #[arg(long)]
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
    #[arg(long, default_value_t = 3)]
    pub top_k: usize,
    /// Fused candidates.
    #[arg(long, default_value_t = 50)]
    pub candidates: usize,
    /// JSON output.
    #[arg(long)]
    pub json: bool,
}

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
}

/// Agent-specific skills-dir preset for `t3rag skill --agents`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum SkillAgents {
    Claude,
    Devin,
    Cursor,
    Generic,
}

impl SkillAgents {
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
    I8,
    Bf16,
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
}
