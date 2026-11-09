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

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tracing::info;

use crate::cli::McpArgs;

// MCP spec revision we implement (t3-css.3.6 — decision: keep the
// hand-rolled tools-only layer for v1, declare it honestly; rmcp is
// deferred — see bead notes). Negotiation: clients compare our
// declared version against the requested one in params.protocolVersion
// and terminate if incompatible — we always state ours.
const PROTOCOL_VERSION: &str = "2025-06-18";

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
                    "candidates": {"type": "integer", "default": 50, "description": "fused candidates before top_k/rerank"},
                    "rerank": {"type": "boolean", "default": false, "description": "cross-encoder rerank of fused candidates (slow on CPU)"},
                    "exact": {"type": "boolean", "default": false, "description": "brute-force exact scan instead of ANN"},
                    "ef_search": {"type": "integer", "description": "ANN ef-search (recall-latency dial)"},
                    "rrf_k": {"type": "number", "default": 60, "description": "RRF fusion constant"},
                    "dense_weight": {"type": "number", "default": 1.0, "description": "dense leg weight"},
                    "bm25_weight": {"type": "number", "default": 1.0, "description": "lexical leg weight"},
                    "explain": {"type": "boolean", "default": false, "description": "per-stage timings + per-leg scores"},
                    "reranker": {"type": "string", "enum": ["minilm", "bge-base", "bge-m3"], "description": "cross-encoder model"},
                    "filter": {"type": "array", "items": {"type": "string"}, "description": "post-filters like 'source~substr'/'title~substr'"},
                    "max_per_source": {"type": "integer", "description": "cap hits per source (episode-diversity)"},
                    "max_chars": {"type": "integer", "description": "truncate passage text"},
                    "min_score": {"type": "number", "description": "drop hits below score"},
                    "sparse": {"type": "boolean", "default": false, "description": "add the learned SPLADE leg (corpus must have sparse.bin)"},
                    "like": {"type": "integer", "description": "more-like-this: dense search seeded by a passage id (seed excluded)"},
                    "like_text": {"type": "string", "description": "more-like-this seeded by ad-hoc text"},
                    "queries": {"type": "array", "items": {"type": "string"}, "description": "extra query phrasings fused into the ranking"},
                    "exclude": {"type": "array", "items": {"type": "integer"}, "description": "passage ids to exclude from results (iterative refinement)"},
                    "group_by": {"type": "string", "enum": ["question", "source", "corpus"], "description": "group hits into buckets by field"},
                    "expand": {"type": "boolean", "default": false, "description": "pseudo-relevance feedback — blend top hits into the query"},
                    "token_budget": {"type": "integer", "description": "cap total passage text at ~N tokens (chars/4)"},
                    "rerank_candidates": {"type": "integer", "description": "candidates fed to reranker"},
                    "preset": {"type": "string", "enum": ["fast", "balanced", "max"], "description": "quality preset (sets candidates/rerank defaults)"},
                    "model": {"type": "string", "description": "embedding model repo override (must match the daemon's)"}
                },
                "required": ["query"]
            }
        }, {
            "name": "list_corpora",
            "description": "List indexed corpora with manifest info (transform type, docs, top terms) — call first to route a query.",
            "inputSchema": {"type": "object", "properties": {}}
        }, {
            "name": "suggest",
            "description": "Rank corpora for a query (type+stats lead; similarity is a hint) — call before choosing --corpus.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the problem/question to route"}
                },
                "required": ["query"]
            }
        }, {
            "name": "batch_search",
            "description": "Run N independent queries in one call — each gets its own result set (distinct from the 'queries' arg on search_traces, which fuses phrasings into ONE ranking).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}, "description": "independent queries to run"},
                    "corpus": {"type": "string", "description": "corpus name"},
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": ["queries"]
            }
        }, {
            "name": "get_passage",
            "description": "Fetch one passage in full (text + metadata + chunk linkage) by corpus and passage id — use after search_traces surfaces a promising hit.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "corpus": {"type": "string", "description": "corpus name"},
                    "passage_id": {"type": "integer", "description": "passage id from a search hit"}
                },
                "required": ["passage_id"]
            }
        }]
    })
}

/// Dispatch one JSON-RPC message through `exec` (the search executor).
/// Returns a response, or `None` for notifications (no `id`).
///
/// Implements `initialize`, `notifications/*`, `ping`, `tools/list`,
/// `tools/call`; everything else gets `-32601` (honest tools-only
/// surface, t3-css.3.6).
/// `exec` receives `(tool_name, arguments)` — one closure branches
/// across the whole tool surface (t3-vqd: get_passage added).
/// `ctx` builds the state prompts/resources answer from on demand
/// (GH#3): `{corpora: [{name, manifest}], playbook: "<SKILL.md text>"}`
/// (t3-ixu). Called only by prompts/resources methods — never for
/// initialize/tools/list, which must stay daemon-free.
pub fn handle_rpc(
    msg: &Value,
    exec: &dyn Fn(&str, &Value) -> Result<Value>,
    ctx: &dyn Fn() -> Value,
) -> Option<Value> {
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
        // t3-ixu: teach the playbook in-band + expose corpus manifests as
        // resources — an agent can orient without a prior SKILL.md read.
        "prompts/list" => Ok(json!({
            "prompts": [{
                "name": "t3rag-playbook",
                "description": "How to use t3rag as an agent: orient → retrieve → inspect → iterate → cite"
            }]
        })),
        "prompts/get" => Ok(json!({
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": ctx()["playbook"].as_str().unwrap_or("playbook unavailable")
                }
            }]
        })),
        "resources/list" => Ok(json!({
            "resources": ctx()["corpora"].as_array().map(|cs| cs.iter().map(|c| json!({
                "uri": format!("t3rag://corpus/{}", c["name"].as_str().unwrap_or("?")),
                "name": format!("t3rag corpus {}", c["name"].as_str().unwrap_or("?")),
                "mimeType": "application/json"
            })).collect::<Vec<_>>()).unwrap_or_default()
        })),
        "resources/read" => {
            let uri = msg
                .pointer("/params/uri")
                .and_then(Value::as_str)
                .unwrap_or("");
            let name = uri.strip_prefix("t3rag://corpus/").unwrap_or("");
            match ctx()["corpora"].as_array().and_then(|cs| {
                cs.iter()
                    .find(|c| c["name"].as_str() == Some(name))
                    .cloned()
            }) {
                Some(c) => Ok(json!({
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": serde_json::to_string_pretty(&c).unwrap_or_default()
                    }]
                })),
                None => Err(format!("unknown resource: {uri}")),
            }
        }
        "tools/call" => match msg.pointer("/params/name").and_then(Value::as_str) {
            Some(
                name @ ("search_traces" | "get_passage" | "batch_search" | "list_corpora"
                | "suggest"),
            ) => {
                let args = msg
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match exec(name, &args) {
                    Ok(v) => Ok(v),
                    Err(e) => Ok(json!({
                        "content": [{"type": "text", "text": format!("{name} failed: {e:#}")}],
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

/// Render daemon search results into the MCP `content` shape
/// (`text` blocks per hit).
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

/// Stdio framing dialect (GH#3): the MCP spec's stdio transport is
/// newline-delimited JSON (NDJSON), but LSP-heritage hosts speak
/// `Content-Length` headers. Auto-detected per message; responses echo
/// the detected framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Framing {
    /// `{...}\n` — the spec's stdio dialect.
    Ndjson,
    /// `Content-Length: N\r\n\r\n{...}` — LSP-style framing.
    ContentLength,
}

/// Read one JSON-RPC message body, sniffing framing on each read.
/// Returns `None` on clean EOF. Header blocks (`Content-Length: N`
/// followed by more headers, a blank line, then N bytes) and bare NDJSON
/// lines are both accepted; blank lines are skipped.
fn read_message<R: BufRead>(r: &mut R, framing: &mut Option<Framing>) -> Result<Option<String>> {
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line).context("read stdin")? == 0 {
            return Ok(None);
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    if line
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("content-length:")
    {
        *framing = Some(Framing::ContentLength);
        let len: usize = line
            .split(':')
            .nth(1)
            .and_then(|v| v.trim().parse().ok())
            .context("bad Content-Length header")?;
        // Consume any remaining headers (Content-Type, …) to the blank line.
        loop {
            line.clear();
            if r.read_line(&mut line)? == 0 || line.trim().is_empty() {
                break;
            }
        }
        let mut body = vec![0u8; len];
        r.read_exact(&mut body).context("read message body")?;
        String::from_utf8(body)
            .map(Some)
            .context("message body is not UTF-8")
    } else {
        *framing = Some(Framing::Ndjson);
        Ok(Some(line.trim_end().to_string()))
    }
}

/// Emit one JSON-RPC response in the negotiated framing.
fn write_message<W: Write>(w: &mut W, framing: Framing, msg: &Value) -> Result<()> {
    let body = msg.to_string();
    match framing {
        Framing::Ndjson => writeln!(w, "{body}")?,
        Framing::ContentLength => write!(w, "Content-Length: {}\r\n\r\n{body}", body.len())?,
    }
    w.flush()?;
    Ok(())
}

/// `t3rag mcp`: stdio MCP server that proxies each `tools/call` to
/// the running daemon via HTTP — never loads models itself.
///
/// The daemon is resolved lazily on first use (GH#3): `ensure_daemon`
/// can block for minutes cold-spawning, and `initialize`/`tools/list`
/// must answer inside the client's handshake budget. A warm thread kicks
/// the spawn at startup so the wait overlaps the handshake instead of
/// sitting inside it. Only `tools/call` (and corpus-backed
/// prompts/resources) ever touch the daemon.
///
/// # Errors
/// A stream is unreadable, or a proxied daemon query fails.
pub fn run_mcp(args: McpArgs) -> Result<()> {
    let daemon_url = args.daemon_url.clone();
    let base = std::sync::Arc::new(std::sync::LazyLock::new(move || match daemon_url {
        Some(u) => Ok(u),
        None => crate::serve::ensure_daemon().map_err(|e| format!("{e:#}")),
    }));
    {
        let warm = std::sync::Arc::clone(&base);
        std::thread::spawn(move || {
            let _ = std::sync::LazyLock::force(&warm);
        });
    }
    let daemon = || -> Result<&str> {
        std::sync::LazyLock::force(&base)
            .as_deref()
            .map_err(|e| anyhow::anyhow!("{e}"))
    };
    info!("mcp stdio proxy ready");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let exec = |name: &str, a: &Value| -> Result<Value> {
        let base = daemon()?;
        // batch_search (t3-w4d): N independent POSTs to the warm daemon.
        if name == "batch_search" {
            let corpus = a["corpus"].as_str().unwrap_or("");
            let top_k = a["top_k"].as_u64().unwrap_or(3);
            let mut out = Vec::new();
            for q in a["queries"]
                .as_array()
                .map(|qs| {
                    qs.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
            {
                let mut resp = ureq::post(&format!("{base}/search"))
                    .send_json(
                        json!({"query": q, "client": "mcp", "corpus": corpus, "top_k": top_k}),
                    )
                    .context("daemon /search")?;
                let v: Value = resp.body_mut().read_json().unwrap_or_default();
                out.push(v);
            }
            return Ok(json!({
                "content": [{"type": "text", "text": serde_json::to_string(&out)?}]
            }));
        }
        // list_corpora / suggest (t3-ixu): proxy the daemon's GETs.
        // Render as a table — NOT tool_response, which formats hit-shaped
        // results and would mangle corpus rows into fake passages.
        if name == "list_corpora" {
            let mut resp = ureq::get(&format!("{base}/corpora"))
                .call()
                .context("daemon /corpora")?;
            let v: Value = resp.body_mut().read_json().context("daemon response")?;
            let mut t = String::from("corpus\tdocs\tloaded\ttransform\ttop_terms\n");
            for c in v.as_array().iter().flat_map(|a| a.iter()) {
                t += &format!(
                    "{}\t{}\t{}\t{}\t{}\n",
                    c["name"].as_str().or(c["corpus"].as_str()).unwrap_or("?"),
                    c["docs"].as_u64().or(c["doc_count"].as_u64()).unwrap_or(0),
                    c["loaded"].as_bool().unwrap_or(false),
                    c["transform"]
                        .as_str()
                        .or(c["manifest"]["transform"].as_str())
                        .unwrap_or("unknown"),
                    c["top_terms"]
                        .as_array()
                        .or(c["manifest"]["top_terms"].as_array())
                        .map(|ts| ts
                            .iter()
                            .filter_map(|t| t.as_str())
                            .take(4)
                            .collect::<Vec<_>>()
                            .join(","))
                        .unwrap_or_default(),
                );
            }
            return Ok(json!({"content": [{"type": "text", "text": t}]}));
        }
        if name == "suggest" {
            let q = a["query"].as_str().unwrap_or("");
            let mut resp = ureq::get(&format!("{base}/suggest"))
                .query("q", q)
                .call()
                .context("daemon /suggest")?;
            let v: Value = resp.body_mut().read_json().context("daemon response")?;
            let mut t = String::from("corpus\ttransform\tdocs\tsimilarity\n");
            for r in v["corpora"].as_array().iter().flat_map(|a| a.iter()) {
                t += &format!(
                    "{}\t{}\t{}\t{}\n",
                    r["corpus"].as_str().unwrap_or("?"),
                    r["transform"].as_str().unwrap_or("?"),
                    r["docs"].as_u64().unwrap_or(0),
                    r["similarity"]
                        .as_f64()
                        .map(|s| format!("{s:.3}"))
                        .unwrap_or_else(|| "n/a".into()),
                );
            }
            return Ok(json!({"content": [{"type": "text", "text": t}]}));
        }
        // get_passage (t3-vqd): straight GET to the daemon's route —
        // corpus defaults to the daemon's newest when omitted.
        if name == "get_passage" {
            let corpus = a["corpus"].as_str().unwrap_or("latest");
            let id = a["passage_id"].as_u64().unwrap_or(0);
            let mut resp = ureq::get(&format!("{base}/passages/{corpus}/{id}"))
                .call()
                .context("daemon /passages")?;
            if resp.status().as_u16() != 200 {
                anyhow::bail!("daemon returned {}", resp.status().as_u16());
            }
            let v: Value = resp.body_mut().read_json().context("daemon response")?;
            let mut t = String::new();
            if let Some(x) = v["title"].as_str() {
                t.push_str(&format!("title: {x}\n"));
            }
            if let Some(x) = v["question"].as_str() {
                t.push_str(&format!("question: {x}\n"));
            }
            if let Some(x) = v["source"].as_str() {
                t.push_str(&format!("source: {x}\n"));
            }
            t.push_str(v["text"].as_str().unwrap_or(""));
            return Ok(json!({"content": [{"type": "text", "text": t}]}));
        }
        let mut resp = ureq::post(&format!("{base}/search"))
            .send_json(json!({
                "query": a["query"].as_str().unwrap_or(""),
                "client": "mcp",
                "corpus": a["corpus"].as_str(),
                "top_k": a["top_k"].as_u64().unwrap_or(3),
                "rerank": a["rerank"].as_bool().unwrap_or(false),
                "exclude": a["exclude"],
            }))
            .context("daemon /search")?;
        if resp.status().as_u16() != 200 {
            anyhow::bail!("daemon returned {}", resp.status().as_u16());
        }
        let v: Value = resp.body_mut().read_json().context("daemon response")?;
        Ok(tool_response(&v))
    };
    // t3-ixu: ctx for prompts/resources — corpora/manifests + the SKILL
    // playbook, built only when a method asks for it (daemon-free
    // initialize/tools/list stay that way).
    let build_ctx = || -> Value {
        let mut corpora = Vec::new();
        if let Ok(b) = daemon() {
            if let Ok(mut r) = ureq::get(&format!("{b}/corpora")).call() {
                if let Ok(v) = r.body_mut().read_json::<Value>() {
                    corpora = v.as_array().cloned().unwrap_or_default();
                }
            }
        }
        json!({"corpora": corpora, "playbook": crate::skill::SKILL_MD})
    };
    let mut reader = std::io::BufReader::new(stdin.lock());
    serve_stream(&mut reader, &mut out, &exec, &build_ctx)
}

/// The stdio read/dispatch/write loop — split from `run_mcp` so tests
/// can drive it over in-memory buffers in both framings.
fn serve_stream<R: BufRead, W: Write>(
    reader: &mut R,
    out: &mut W,
    exec: &dyn Fn(&str, &Value) -> Result<Value>,
    ctx: &dyn Fn() -> Value,
) -> Result<()> {
    let mut framing: Option<Framing> = None;
    while let Some(body) = read_message(reader, &mut framing)? {
        let msg: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => {
                write_message(
                    out,
                    framing.unwrap_or(Framing::Ndjson),
                    &json!({
                        "jsonrpc":"2.0","id":null,
                        "error":{"code":-32700,"message":format!("parse error: {e}")}
                    }),
                )?;
                continue;
            }
        };
        if let Some(resp) = handle_rpc(&msg, exec, ctx) {
            write_message(out, framing.unwrap_or(Framing::Ndjson), &resp)?;
        }
    }
    Ok(())
}

// Daemon spawn/readiness lives in `serve` (shared with `search`).

#[cfg(test)]
mod tests {
    use super::*;

    fn no_ctx() -> Value {
        json!({})
    }

    #[test]
    fn initialize_returns_server_info() {
        let m = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize"});
        let r = handle_rpc(&m, &|_, _| unreachable!(), &no_ctx).unwrap();
        assert_eq!(r["result"]["serverInfo"]["name"], "t3rag");
    }

    #[test]
    fn tools_list_exposes_search_traces() {
        let m = serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
        let r = handle_rpc(&m, &|_, _| unreachable!(), &no_ctx).unwrap();
        let name = r["result"]["tools"][0]["name"].as_str().unwrap();
        assert_eq!(name, "search_traces");
    }

    #[test]
    fn tools_call_uses_executor() {
        let m = serde_json::json!({
            "jsonrpc":"2.0","id":3,"method":"tools/call",
            "params":{"name":"search_traces","arguments":{"query":"q"}}
        });
        let r = handle_rpc(
            &m,
            &|_, _| Ok(serde_json::json!({"content":[{"type":"text","text":"ok"}]})),
            &no_ctx,
        )
        .unwrap();
        assert_eq!(r["result"]["content"][0]["text"], "ok");
    }

    #[test]
    fn notification_gets_no_response() {
        let m = serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        assert!(handle_rpc(&m, &|_, _| unreachable!(), &no_ctx).is_none());
    }

    #[test]
    fn unknown_tool_errors() {
        let m = serde_json::json!({
            "jsonrpc":"2.0","id":4,"method":"tools/call",
            "params":{"name":"nope","arguments":{}}
        });
        let r = handle_rpc(&m, &|_, _| unreachable!(), &no_ctx).unwrap();
        assert!(r["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown tool"));
    }

    // ---- stdio framing (GH#3): NDJSON + Content-Length auto-detect ----

    #[test]
    fn read_ndjson_line() {
        let mut input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n".as_slice();
        let mut f = None;
        let msg = read_message(&mut input, &mut f).unwrap().unwrap();
        assert!(msg.contains("ping"));
        assert_eq!(f, Some(Framing::Ndjson));
    }

    #[test]
    fn read_content_length_framed() {
        let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}";
        let wire = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut input = wire.as_bytes();
        let mut f = None;
        let msg = read_message(&mut input, &mut f).unwrap().unwrap();
        assert_eq!(msg, body);
        assert_eq!(f, Some(Framing::ContentLength));
    }

    #[test]
    fn read_content_length_extra_headers_and_multibyte() {
        // vscode-jsonrpc sends Content-Type; len is BYTES not chars.
        let body = "{\"q\":\"héllo→wörld\"}";
        let wire = format!(
            "Content-Length: {}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n{}",
            body.len(),
            body
        );
        let mut input = wire.as_bytes();
        let mut f = None;
        assert_eq!(read_message(&mut input, &mut f).unwrap().unwrap(), body);
    }

    #[test]
    fn read_framed_messages_back_to_back() {
        let a = "{\"id\":1}";
        let b = "{\"id\":2}";
        let wire = format!(
            "Content-Length: {}\r\n\r\n{}Content-Length: {}\r\n\r\n{}",
            a.len(),
            a,
            b.len(),
            b
        );
        let mut input = wire.as_bytes();
        let mut f = None;
        assert_eq!(read_message(&mut input, &mut f).unwrap().unwrap(), a);
        assert_eq!(read_message(&mut input, &mut f).unwrap().unwrap(), b);
        assert!(read_message(&mut input, &mut f).unwrap().is_none());
    }

    #[test]
    fn read_skips_blank_lines_then_eof() {
        let mut input = b"\n\r\n{\"id\":1}\n".as_slice();
        let mut f = None;
        assert_eq!(
            read_message(&mut input, &mut f).unwrap().unwrap(),
            "{\"id\":1}"
        );
        assert!(read_message(&mut input, &mut f).unwrap().is_none());
    }

    #[test]
    fn write_echoes_detected_framing() {
        let mut buf = Vec::new();
        write_message(&mut buf, Framing::Ndjson, &json!({"id":1})).unwrap();
        assert_eq!(buf, b"{\"id\":1}\n");
        buf.clear();
        write_message(&mut buf, Framing::ContentLength, &json!({"id":1})).unwrap();
        assert_eq!(buf, b"Content-Length: 8\r\n\r\n{\"id\":1}");
    }

    /// GH#3 acceptance: an NDJSON client gets initialize + tools/list
    /// back without a daemon ever being touched.
    #[test]
    fn stdio_ndjson_handshake_needs_no_daemon() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n"
        );
        let mut r = std::io::BufReader::new(input.as_bytes());
        let mut out = Vec::new();
        serve_stream(&mut r, &mut out, &|_, _| panic!("daemon hit"), &no_ctx).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let init: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(init["result"]["serverInfo"]["name"], "t3rag");
        let tools: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(tools["result"]["tools"].as_array().unwrap().len(), 5);
    }

    /// GH#3 acceptance: a Content-Length (LSP-heritage) client gets the
    /// same handshake in its own framing.
    #[test]
    fn stdio_content_length_handshake() {
        let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}";
        let wire = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut r = std::io::BufReader::new(wire.as_bytes());
        let mut out = Vec::new();
        serve_stream(&mut r, &mut out, &|_, _| panic!("daemon hit"), &no_ctx).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("Content-Length: "));
        let body_start = text.find("\r\n\r\n").unwrap() + 4;
        let v: Value = serde_json::from_str(&text[body_start..]).unwrap();
        assert_eq!(v["result"]["serverInfo"]["name"], "t3rag");
    }
}
