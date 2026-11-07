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
    /// Vector quantization for the dense index.
    #[arg(long, value_enum, default_value_t = Quantization::I8)]
    pub quantization: Quantization,
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
    /// Bypass the daemon: run the cold in-process path.
    #[arg(long)]
    pub no_daemon: bool,
    /// Exact brute-force scan instead of ANN (recall reference).
    #[arg(long)]
    pub exact: bool,
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
}

#[derive(Args)]
pub struct McpArgs {
    /// Daemon base URL override (default: discover/auto-spawn).
    #[arg(long)]
    pub daemon_url: Option<String>,
}

#[derive(Args)]
pub struct SkillArgs {
    /// Write SKILL.md into this directory instead of printing.
    #[arg(long)]
    pub install: Option<PathBuf>,
    /// Overwrite an existing SKILL.md.
    #[arg(long)]
    pub force: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Quantization {
    I8,
    Bf16,
    Fp32,
}
