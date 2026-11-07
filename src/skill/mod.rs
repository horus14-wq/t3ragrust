//! SKILL.md emitter (t3-wkp.15).
//!
//! `t3rag skill` prints a skill wrapper for agent discovery (what t3rag is,
//! the `search_traces` MCP tool + config snippets, CLI equivalents,
//! when-to-use guidance); `--install <dir>` writes it into a skills dir.

use anyhow::{bail, Result};

use crate::cli::SkillArgs;

/// `t3rag skill` entry point (implemented in t3-wkp.15).
pub fn run_skill(args: SkillArgs) -> Result<()> {
    bail!(
        "not implemented: t3rag skill (install={:?}, force={})",
        args.install,
        args.force
    )
}
