//! MCP layer (t3-wkp.14).
//!
//! Two surfaces, one JSON-RPC dispatcher:
//! - daemon `POST /mcp` — streamable-HTTP MCP (warm, shared; recommended)
//! - `t3rag mcp` — stdio MCP proxy for clients that can't speak HTTP; it
//!   NEVER loads models itself — forwards `tools/call` to the daemon,
//!   auto-spawning `t3rag serve` when absent.
//!
//! Hand-rolled JSON-RPC 2.0: the MCP surface here is one tool
//! (`search_traces`) so pulling an SDK is unjustified weight. Surface:
//! initialize, notifications/initialized, ping, tools/list, tools/call.

use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tracing::info;

use crate::cli::McpArgs;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// The single tool exposed over MCP.
fn tools_list() -> Value {
    json!({
        "tools": [{
            "name": "search_traces",
            "description": "Search T3 thinking traces: how prior problems were reasoned about (reflections, constructs, semantics). Use before tackling non-trivial problems where prior reasoning experience helps.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the problem/question to find traces for"},
                    "top_k": {"type": "integer", "default": 3, "description": "passages to return"},
                    "corpus": {"type": "string", "description": "corpus name (e.g. reflect/construct/semantic); default = newest"},
                    "rerank": {"type": "boolean", "default": true, "description": "cross-encoder rerank of fused candidates"}
                },
                "required": ["query"]
            }
        }]
    })
}

/// Dispatch one JSON-RPC message through `exec` (the search executor).
/// Returns a response message, or None for notifications.
pub fn handle_rpc(msg: &Value, exec: &dyn Fn(&Value) -> Result<Value>) -> Option<Value> {
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let is_notification = msg.get("id").is_none();

    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "t3rag", "version": env!("CARGO_PKG_VERSION")}
        })),
        "notifications/initialized" | "notifications/cancelled" => return None,
        "ping" => Ok(json!({})),
        "tools/list" => Ok(tools_list()),
        "tools/call" => match msg.pointer("/params/name").and_then(Value::as_str) {
            Some("search_traces") => {
                let args = msg
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match exec(&args) {
                    Ok(v) => Ok(v),
                    Err(e) => Ok(json!({
                        "content": [{"type": "text", "text": format!("search failed: {e:#}")}],
                        "isError": true
                    })),
                }
            }
            Some(other) => Err(format!("unknown tool: {other}")),
            None => Err("tools/call missing tool name".into()),
        },
        _ => Err(format!("method not found: {method}")),
    };

    if is_notification {
        return None;
    }
    match result {
        Ok(v) => Some(json!({"jsonrpc": "2.0", "id": id, "result": v})),
        Err(e) => Some(json!({
            "jsonrpc": "2.0", "id": id,
            "error": {"code": -32601, "message": e}
        })),
    }
}

/// Render daemon search results into an MCP tool response.
pub fn tool_response(resp: &Value) -> Value {
    let mut text = String::new();
    if let Some(results) = resp.get("results").and_then(Value::as_array) {
        for r in results {
            text.push_str(&format!(
                "--- [{}] passage {} (score {:.4})\n",
                r["rank"].as_u64().unwrap_or(0),
                r["passage_id"].as_u64().unwrap_or(0),
                r["score"].as_f64().unwrap_or(0.0)
            ));
            if let Some(t) = r["title"].as_str() {
                text.push_str(&format!("title: {t}\n"));
            }
            if let Some(q) = r["question"].as_str() {
                text.push_str(&format!("question: {q}\n"));
            }
            if let Some(t) = r["text"].as_str() {
                text.push_str(t.trim());
                text.push('\n');
            }
        }
    }
    json!({"content": [{"type": "text", "text": text}]})
}

// ---------------- stdio proxy ----------------

/// `t3rag mcp` — newline-delimited JSON-RPC on stdin/stdout.
pub fn run_mcp(args: McpArgs) -> Result<()> {
    let base = match args.daemon_url.as_deref() {
        Some(u) => u.to_string(),
        None => ensure_daemon()?,
    };
    info!(%base, "mcp stdio proxy ready");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let exec = |a: &Value| -> Result<Value> {
        let resp = ureq::post(&format!("{base}/search"))
            .send_json(json!({
                "query": a["query"].as_str().unwrap_or(""),
                "corpus": a["corpus"].as_str(),
                "top_k": a["top_k"].as_u64().unwrap_or(3),
                "rerank": a["rerank"].as_bool().unwrap_or(true),
            }))
            .context("daemon /search")?;
        if resp.status() != 200 {
            anyhow::bail!("daemon returned {}", resp.status());
        }
        let v: Value = resp.into_json().context("daemon response")?;
        Ok(tool_response(&v))
    };
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                writeln!(
                    out,
                    "{}",
                    json!({
                        "jsonrpc":"2.0","id":null,
                        "error":{"code":-32700,"message":format!("parse error: {e}")}
                    })
                )?;
                continue;
            }
        };
        if let Some(resp) = handle_rpc(&msg, &exec) {
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    Ok(())
}

/// Find the daemon, spawning `t3rag serve` if absent; poll for readiness.
fn ensure_daemon() -> Result<String> {
    if let Some(base) = crate::serve::discover_daemon() {
        if ureq::get(&format!("{base}/health")).call().is_ok() {
            return Ok(base);
        }
    }
    let exe = std::env::current_exe().context("current exe")?;
    std::process::Command::new(exe)
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn t3rag serve")?;
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if Instant::now() > deadline {
            anyhow::bail!("daemon did not become ready within 180s");
        }
        if let Some(base) = crate::serve::discover_daemon() {
            if ureq::get(&format!("{base}/health")).call().is_ok() {
                return Ok(base);
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_returns_server_info() {
        let m = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize"});
        let r = handle_rpc(&m, &|_| unreachable!()).unwrap();
        assert_eq!(r["result"]["serverInfo"]["name"], "t3rag");
    }

    #[test]
    fn tools_list_exposes_search_traces() {
        let m = serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
        let r = handle_rpc(&m, &|_| unreachable!()).unwrap();
        let name = r["result"]["tools"][0]["name"].as_str().unwrap();
        assert_eq!(name, "search_traces");
    }

    #[test]
    fn tools_call_uses_executor() {
        let m = serde_json::json!({
            "jsonrpc":"2.0","id":3,"method":"tools/call",
            "params":{"name":"search_traces","arguments":{"query":"q"}}
        });
        let r = handle_rpc(&m, &|_| {
            Ok(serde_json::json!({"content":[{"type":"text","text":"ok"}]}))
        })
        .unwrap();
        assert_eq!(r["result"]["content"][0]["text"], "ok");
    }

    #[test]
    fn notification_gets_no_response() {
        let m = serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        assert!(handle_rpc(&m, &|_| unreachable!()).is_none());
    }

    #[test]
    fn unknown_tool_errors() {
        let m = serde_json::json!({
            "jsonrpc":"2.0","id":4,"method":"tools/call",
            "params":{"name":"nope","arguments":{}}
        });
        let r = handle_rpc(&m, &|_| unreachable!()).unwrap();
        assert!(r["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown tool"));
    }
}
