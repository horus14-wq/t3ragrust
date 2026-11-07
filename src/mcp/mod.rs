//! MCP surface (t3-wkp.14).
//!
//! Tool `search_traces(query, corpus?, top_k?) -> passages`. Two transports:
//! streamable-HTTP mounted on the daemon, and `t3rag mcp` — a stdio MCP that
//! PROXIES to the daemon and must never load the model itself. SDK is `rmcp`;
//! if its dep tree hurts compile times the sanctioned fallback is hand-rolled
//! JSON-RPC for this one tool.

use anyhow::{bail, Result};

use crate::cli::McpArgs;

/// `t3rag mcp` entry point (implemented in t3-wkp.14).
pub fn run_mcp(args: McpArgs) -> Result<()> {
    bail!(
        "not implemented: t3rag mcp (daemon_url={:?})",
        args.daemon_url
    )
}
