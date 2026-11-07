//! Resident daemon (t3-wkp.12).
//!
//! `t3rag serve`: tokio HTTP on 127.0.0.1 plus streamable-HTTP MCP. Model
//! and indexes stay resident — usearch mmap'd, pooled tantivy readers.
//! ORT sessions are not safely concurrent: pool 2-4 behind a semaphore on
//! dedicated inference threads (not tokio tasks). Startup warmup, per-request
//! stage timing logs, --idle-timeout unload, graceful shutdown.

use anyhow::{bail, Result};

use crate::cli::ServeArgs;

/// `t3rag serve` entry point (implemented in t3-wkp.12).
pub fn run_serve(args: ServeArgs) -> Result<()> {
    bail!(
        "not implemented: t3rag serve --port {} (idle_timeout={:?})",
        args.port,
        args.idle_timeout
    )
}
