//! Append-only query log (t3-pbh) — the L5 accretion substrate.
//!
//! The daemon is the single writer: every handled `/search` (and every
//! MCP `search_traces` call, which shares `do_search`) appends one JSON
//! line to `<cache>/log.jsonl`. `t3rag log` reads and filters it back so
//! an agent can review its own search history — "did I already try
//! this?", "which queries keep failing?". Deliberately no DB: two
//! `writeln!` sites and a reader.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Serialize;
use tracing::warn;

use crate::cli::LogArgs;

/// One handled-search record appended to `log.jsonl`.
#[derive(Debug, Serialize)]
pub struct SearchEntry<'a> {
    /// RFC3339 UTC timestamp (second precision).
    pub ts: String,
    /// Origin of the query: `cli`, `mcp`, or `http`.
    pub client: &'a str,
    /// The query text.
    pub query: &'a str,
    /// Comma-joined corpus names the query ran against.
    pub corpora: String,
    /// Ranked parent passage ids returned (top-k order).
    pub top_ids: Vec<u64>,
    /// Scores aligned with `top_ids`.
    pub top_scores: Vec<f32>,
    /// End-to-end search time in milliseconds.
    pub took_ms: f64,
    /// Quality preset in effect, when one was named.
    pub preset: Option<&'a str>,
}

/// Path of the append-only query log (`<cache>/log.jsonl`).
pub fn log_path() -> Result<PathBuf> {
    Ok(crate::paths::cache_root()?.join("log.jsonl"))
}

/// Current UTC time as RFC3339 (second precision) for a log `ts`.
pub fn now_ts() -> String {
    time::OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .ok()
        .and_then(|t| {
            t.format(&time::format_description::well_known::Rfc3339)
                .ok()
        })
        .unwrap_or_default()
}

/// Append one entry; a log failure warns but never fails the search.
pub fn append(entry: &SearchEntry<'_>) {
    let line = match serde_json::to_string(entry) {
        Ok(l) => l,
        Err(e) => {
            warn!("query log serialize failed: {e}");
            return;
        }
    };
    let res = (|| -> Result<()> {
        let path = log_path()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        writeln!(f, "{line}")?;
        Ok(())
    })();
    if let Err(e) = res {
        warn!("query log append failed: {e:#}");
    }
}

/// `t3rag log` — read `log.jsonl`, apply filters, print matches.
/// `--json` passes the raw JSONL lines through for machine consumers.
pub fn run_log(args: &LogArgs) -> Result<()> {
    let path = log_path()?;
    let since_secs = args.since.as_deref().map(parse_since).transpose()?;
    let cutoff = since_secs.map(|s| {
        time::OffsetDateTime::now_utc().unix_timestamp() - i64::try_from(s).unwrap_or(i64::MAX)
    });
    let re = args
        .query_regex
        .as_deref()
        .map(regex::Regex::new)
        .transpose()
        .context("invalid --query-regex")?;
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "no query log yet ({}) — searches append here",
                path.display()
            );
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue; // tolerate a torn tail line
        };
        if let Some(cut) = cutoff {
            let ts = v["ts"]
                .as_str()
                .and_then(|t| {
                    time::OffsetDateTime::parse(t, &time::format_description::well_known::Rfc3339)
                        .ok()
                })
                .map(|t| t.unix_timestamp());
            match ts {
                Some(t) if t >= cut => {}
                _ => continue,
            }
        }
        if let Some(c) = &args.corpus {
            let hit = v["corpora"]
                .as_str()
                .map(|s| s.split(',').any(|n| n.trim() == c))
                .unwrap_or(false);
            if !hit {
                continue;
            }
        }
        if let Some(re) = &re {
            let q = v["query"].as_str().unwrap_or("");
            if !re.is_match(q) {
                continue;
            }
        }
        if args.json {
            println!("{line}");
        } else {
            rows.push(v);
        }
    }
    if args.json {
        return Ok(());
    }
    println!(
        "{:<20} {:<5} {:>8}  {:<24} {:<18} query",
        "ts", "client", "took_ms", "corpora", "top_ids"
    );
    for v in &rows {
        let ts = v["ts"].as_str().unwrap_or("?");
        let client = v["client"].as_str().unwrap_or("?");
        let took = v["took_ms"].as_f64().unwrap_or(0.0);
        let corpora = v["corpora"].as_str().unwrap_or("");
        let ids = v["top_ids"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|i| i.as_u64())
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        let query = v["query"].as_str().unwrap_or("");
        let shown: String = query.chars().take(80).collect();
        let ellipsis = if query.chars().count() > 80 {
            "…"
        } else {
            ""
        };
        println!(
            "{:<20} {:<5} {:>8.0}  {:<24} {:<18} {shown}{ellipsis}",
            ts, client, took, corpora, ids
        );
    }
    if rows.is_empty() {
        println!("(no matching log entries)");
    }
    Ok(())
}

/// Parse a `--since` duration: `<n><unit>` with unit s/m/h/d
/// (bare number = seconds). e.g. `90s`, `10m`, `1h`, `7d`.
fn parse_since(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num, mult) = if let Some(p) = s.strip_suffix('s') {
        (p, 1u64)
    } else if let Some(p) = s.strip_suffix('m') {
        (p, 60)
    } else if let Some(p) = s.strip_suffix('h') {
        (p, 3600)
    } else if let Some(p) = s.strip_suffix('d') {
        (p, 86_400)
    } else {
        (s, 1)
    };
    let n: u64 = num
        .parse()
        .with_context(|| format!("bad --since '{s}' (want e.g. 90s, 10m, 1h, 7d)"))?;
    n.checked_mul(mult).context("--since duration overflows")
}

#[cfg(test)]
mod tests {
    use super::parse_since;

    #[test]
    fn since_units() {
        assert_eq!(parse_since("90s").unwrap(), 90);
        assert_eq!(parse_since("10m").unwrap(), 600);
        assert_eq!(parse_since("1h").unwrap(), 3600);
        assert_eq!(parse_since("7d").unwrap(), 604_800);
        assert_eq!(parse_since("42").unwrap(), 42);
        assert!(parse_since("soon").is_err());
        assert!(parse_since("1w").is_err());
    }
}
