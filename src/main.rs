mod bench;
mod capabilities;
mod chunk;
mod cli;
mod config;
mod doctor;
mod embed;
mod eval;
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
    let cli = cli::Cli::parse();
    // -v / -q / --log-level resolve to a tracing filter (t3-css.3.1).
    let filter = cli.log_level.clone().or_else(|| {
        if cli.quiet {
            Some("error".into())
        } else if cli.verbose {
            Some("debug".into())
        } else {
            None
        }
    });
    let env_filter = filter
        .map(EnvFilter::new)
        .or_else(|| EnvFilter::try_from_default_env().ok())
        .unwrap_or_else(|| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
    cli.run()
}
