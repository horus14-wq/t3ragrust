mod chunk;
mod cli;
mod embed;
mod index;
mod ingest;
mod mcp;
mod passage;
mod paths;
mod rerank;
mod search;
mod serve;
mod skill;
mod store;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    cli::Cli::parse().run()
}
