//! `--config <path>` TOML defaults (t3-css.3.1): flags win, config fills
//! what flags leave unset, built-in defaults fill the rest. Only
//! `Option<>` args merge cleanly — scalar flags with defaults can't
//! distinguish "absent" from "passed", so they stay flag-only.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// `--config` TOML values: flag defaults merged into `Option<>`
/// args only (scalar flags can't distinguish absent-from-default,
/// t3-css.3.1). Flags win over config.
#[derive(Deserialize, Default)]
pub struct Config {
    /// Default `--corpus` for `search`/`ask`.
    pub corpus: Option<String>,
    /// Default `--model` embedder spec.
    pub model: Option<String>,
    /// Default `--reranker` spec.
    pub reranker: Option<String>,
    /// Default `--instruction` query prefix override.
    pub instruction: Option<String>,
}

/// Load `--config` TOML; `None`/`missing` yields the defaults.
///
/// # Errors
/// The file cannot be read or its TOML is invalid.
pub fn load(path: Option<&Path>) -> Result<Config> {
    let Some(p) = path else {
        return Ok(Config::default());
    };
    let text =
        std::fs::read_to_string(p).with_context(|| format!("read config {}", p.display()))?;
    toml::from_str(&text).with_context(|| format!("parse config {}", p.display()))
}
