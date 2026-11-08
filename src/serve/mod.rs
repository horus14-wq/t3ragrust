//! `t3rag serve` daemon (t3-wkp.12): keeps the model session pool + mmap'd
//! indexes resident; clients (t3rag search thin-client, agents via MCP, curl)
//! share it over loopback HTTP. One daemon per user.
//!
//! Endpoints:
//!   GET  /health           -> {ok, uptime_s, corpora_loaded, sessions}
//!   GET  /corpora          -> [{name, doc_count}]
//!   POST /search           -> SearchRequest -> SearchResponse
//!
//! Discovery: writes `~/.cache/t3rag/daemon.json` {pid, port, base_url} on
//! bind so `t3rag search`/`t3rag mcp` can auto-find (and auto-spawn) it.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::ServeArgs;
use crate::embed::{self, EmbedderPool};
use crate::index::{Bm25Index, DenseIndex};
use crate::paths;
use crate::rerank::{self, RerankerPool};
use crate::search::rrf_fuse;
use crate::store::{self, CorpusMeta, PassageReader};

/// A resident, queryable corpus.
struct Corpus {
    meta: CorpusMeta,
    dense: DenseIndex,
    lex: Bm25Index,
    reader: PassageReader,
}

/// Query-embedding cache (t3-wkp.16): identical queries skip the ~1-2s
/// embed call entirely. Tiny FIFO-LRU — no dep needed at this size.
struct QueryLru {
    map: HashMap<String, Vec<f32>>,
    order: std::collections::VecDeque<String>,
    cap: usize,
}

impl QueryLru {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            cap,
        }
    }
    fn get(&mut self, q: &str) -> Option<Vec<f32>> {
        if let Some(v) = self.map.get(q) {
            let v = v.clone();
            self.order.retain(|k| k != q);
            self.order.push_back(q.to_string());
            Some(v)
        } else {
            None
        }
    }
    fn put(&mut self, q: String, v: Vec<f32>) {
        if self.map.contains_key(&q) {
            self.order.retain(|k| k != &q);
        } else if self.map.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(q.clone());
        self.map.insert(q, v);
    }
}

struct DaemonState {
    corpora_root: std::path::PathBuf,
    embedder: EmbedderPool,
    embedder_repo: String,
    reranker: Mutex<Option<RerankerPool>>,
    corpora: RwLock<HashMap<String, Arc<Corpus>>>,
    query_cache: Mutex<QueryLru>,
    ep: crate::cli::Ep,
    threads: usize,
    started: Instant,
    last_used: Mutex<Instant>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    corpus: Option<String>,
    top_k: Option<usize>,
    candidates: Option<usize>,
    rerank: Option<bool>,
    model: Option<String>,
}

#[derive(Serialize)]
struct HitOut {
    rank: usize,
    passage_id: u64,
    score: f32,
    text: String,
    title: Option<String>,
    question: Option<String>,
    source: Option<String>,
}

#[derive(Serialize)]
struct SearchResponse<'a> {
    query: &'a str,
    corpus: String,
    results: Vec<HitOut>,
}

impl DaemonState {
    fn touch(&self) {
        *self.last_used.lock().unwrap() = Instant::now();
    }

    /// Fetch (or lazy-load) a corpus by name; falls back to the newest.
    fn corpus(&self, name: Option<&str>) -> Result<(String, Arc<Corpus>)> {
        let name = match name {
            Some(n) => n.to_string(),
            None => newest_corpus(&self.corpora_root)?,
        };
        if let Some(c) = self.corpora.read().unwrap().get(&name) {
            return Ok((name, c.clone()));
        }
        let dir = self.corpora_root.join(&name);
        let meta =
            store::read_meta(&dir).with_context(|| format!("corpus '{name}' not indexed"))?;
        let quant = match meta.quantization.as_str() {
            "i8" => crate::cli::Quantization::I8,
            "bf16" => crate::cli::Quantization::Bf16,
            _ => crate::cli::Quantization::Fp32,
        };
        let c = Arc::new(Corpus {
            dense: DenseIndex::view(&dir, meta.dims as usize, quant)?,
            lex: Bm25Index::open(&dir)?,
            reader: PassageReader::open(&dir)?,
            meta,
        });
        self.corpora
            .write()
            .unwrap()
            .insert(name.clone(), c.clone());
        info!(corpus = %name, "corpus loaded");
        Ok((name, c))
    }
}

fn newest_corpus(root: &std::path::Path) -> Result<String> {
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for e in std::fs::read_dir(root)
        .with_context(|| format!("read {} — no corpora indexed yet", root.display()))?
    {
        let p = e?.path();
        if p.is_dir() && p.join("meta.json").exists() {
            let m = p.metadata()?.modified()?;
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            if best.as_ref().map(|(t, _)| m > *t).unwrap_or(true) {
                best = Some((m, n));
            }
        }
    }
    best.map(|(_, n)| n).context("no corpora indexed yet")
}

/// `t3rag serve` — blocks until shutdown/idle-timeout.
pub fn run_serve(args: ServeArgs) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .build()
        .context("tokio runtime")?;
    rt.block_on(async move { serve(args).await })
}

async fn serve(args: ServeArgs) -> Result<()> {
    let spec = embed::resolve_model(args.model.as_deref());
    info!(model = %spec.repo_id, "daemon warmup: loading embedder");
    let embedder = EmbedderPool::load(&spec, args.threads, args.ep, args.pool)?;
    info!(
        dims = embedder.dims(),
        sessions = args.pool,
        "embedder ready"
    );

    let state = Arc::new(DaemonState {
        corpora_root: paths::corpora_root()?,
        embedder,
        embedder_repo: spec.repo_id.clone(),
        reranker: Mutex::new(None),
        corpora: RwLock::new(HashMap::new()),
        query_cache: Mutex::new(QueryLru::new(128)),
        ep: args.ep,
        threads: args.threads,
        started: Instant::now(),
        last_used: Mutex::new(Instant::now()),
    });

    // Preload corpora requested at startup.
    for name in &args.corpus {
        if let Err(e) = state.corpus(Some(name)) {
            tracing::warn!(corpus = %name, "preload failed: {e:#}");
        }
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/corpora", get(list_corpora))
        .route("/search", post(search))
        .route("/mcp", post(mcp_http))
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    write_daemon_info(args.port)?;
    info!(%addr, "daemon listening");

    // Idle-timeout watcher (optional).
    let idle_secs = args.idle_timeout.map(Duration::from_secs);
    let shutdown = async move {
        tokio::signal::ctrl_c().await.ok();
        info!("shutting down");
    };
    let idle_watch = {
        let st = state.clone();
        async move {
            if let Some(limit) = idle_secs {
                loop {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    if st.last_used.lock().unwrap().elapsed() > limit {
                        info!("idle timeout reached; shutting down");
                        return;
                    }
                }
            } else {
                std::future::pending::<()>().await;
            }
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = shutdown => {},
                _ = idle_watch => {},
            }
        })
        .await
        .context("axum serve")?;
    Ok(())
}

async fn health(State(st): State<Arc<DaemonState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "ok": true,
        "uptime_s": st.started.elapsed().as_secs(),
        "corpora_loaded": st.corpora.read().unwrap().len(),
    }))
}

async fn list_corpora(State(st): State<Arc<DaemonState>>) -> impl IntoResponse {
    let mut out = Vec::new();
    let mut names: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&st.corpora_root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && p.join("meta.json").exists() {
                names.push(p.file_name().unwrap().to_string_lossy().into_owned());
            }
        }
    }
    names.sort();
    for n in names {
        let (loaded, docs) = match st.corpora.read().unwrap().get(&n) {
            Some(c) => (true, c.meta.doc_count),
            None => (false, 0),
        };
        out.push(serde_json::json!({"name": n, "loaded": loaded, "doc_count": docs}));
    }
    Json(serde_json::json!(out))
}

async fn search(
    State(st): State<Arc<DaemonState>>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    st.touch();
    // do_search is synchronous ORT/index work — it must not run on a tokio
    // worker (blocks the reactor AND ORT intra-op scheduling can stall on
    // worker threads). spawn_blocking is the correct pool for it.
    match tokio::task::spawn_blocking(move || {
        do_search(&st, &req).and_then(|r| serde_json::to_value(&r).context("serialize"))
    })
    .await
    {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err(e)) => err_response(format!("{e:#}")),
        Err(e) => err_response(format!("search task aborted: {e}")),
    }
}

fn err_response(msg: String) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": msg})),
    )
        .into_response()
}

/// Shared query path: embed -> ANN + BM25 -> RRF -> optional rerank ->
/// hydrate+dedupe by parent.
fn do_search<'a>(st: &'a DaemonState, req: &'a SearchRequest) -> Result<SearchResponse<'a>> {
    // A model override only works if the daemon loaded that model —
    // otherwise the client must take the cold path.
    if let Some(m) = &req.model {
        if *m != st.embedder_repo {
            anyhow::bail!(
                "daemon runs model {}; restart with --model {m} or use --no-daemon",
                st.embedder_repo
            );
        }
    }
    let (name, corpus) = st.corpus(req.corpus.as_deref())?;
    tracing::debug!(corpus = %name, "stage: corpus ready");
    let top_k = req.top_k.unwrap_or(3);
    let cand = req.candidates.unwrap_or(50);

    let cached = st.query_cache.lock().unwrap().get(&req.query);
    let qvec = match cached {
        Some(v) => v,
        None => {
            let v = {
                tracing::debug!("stage: embed start");
                st.embedder.embed_query(&req.query)?
            };
            st.query_cache
                .lock()
                .unwrap()
                .put(req.query.clone(), v.clone());
            v
        }
    };
    tracing::debug!("stage: embed done");
    let ann = corpus.dense.search(&qvec, cand)?;
    let bm = corpus.lex.search(&req.query, cand).unwrap_or_default();
    let mut ranked = rrf_fuse(&[ann, bm]);
    tracing::debug!("stage: fused");

    if req.rerank.unwrap_or(true) {
        let cand_rows: Vec<(u64, f32)> = ranked.drain(..).take(cand).collect();
        let mut pairs = Vec::with_capacity(cand_rows.len());
        let mut ids = Vec::with_capacity(cand_rows.len());
        for (id, _) in &cand_rows {
            let p = corpus.reader.get(*id)?;
            pairs.push((req.query.clone(), p.text));
            ids.push(*id);
        }
        // Lazy single-flight reranker load; a load failure degrades to the
        // fused ranking rather than failing the query.
        let mut guard = st.reranker.lock().unwrap();
        if guard.is_none() {
            match RerankerPool::load(&rerank::default_spec(), st.threads, st.ep, 1) {
                Ok(p) => *guard = Some(p),
                Err(e) => tracing::warn!("reranker unavailable: {e:#}"),
            }
        }
        if let Some(pool) = guard.as_mut() {
            let scores = pool.score_pairs(&pairs)?;
            let mut rr: Vec<(u64, f32)> = ids.into_iter().zip(scores).collect();
            rr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked = rr;
        }
    }

    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (id, score) in ranked {
        let p = corpus.reader.get(id)?;
        let parent = if p.parent_id != p.id {
            corpus.reader.get(p.parent_id)?
        } else {
            p
        };
        if !seen.insert(parent.id) {
            continue;
        }
        results.push(HitOut {
            rank: results.len() + 1,
            passage_id: parent.id,
            score,
            text: parent.text,
            title: parent.title,
            question: parent.question,
            source: parent.source,
        });
        if results.len() >= top_k {
            break;
        }
    }
    Ok(SearchResponse {
        query: &req.query,
        corpus: name,
        results,
    })
}

/// `POST /mcp` — streamable-HTTP MCP endpoint (single-shot JSON-RPC;
/// session/batch semantics deferred to rmcp if we ever need them).
async fn mcp_http(
    State(st): State<Arc<DaemonState>>,
    Json(msg): Json<serde_json::Value>,
) -> impl IntoResponse {
    st.touch();
    let msg2 = msg.clone();
    let result = tokio::task::spawn_blocking(move || {
        let exec = |args: &serde_json::Value| -> Result<serde_json::Value> {
            let req = SearchRequest {
                query: args["query"].as_str().unwrap_or("").to_string(),
                corpus: args["corpus"].as_str().map(String::from),
                top_k: args["top_k"].as_u64().map(|v| v as usize),
                candidates: args["candidates"].as_u64().map(|v| v as usize),
                rerank: args["rerank"].as_bool(),
                model: args["model"].as_str().map(String::from),
            };
            let resp = do_search(&st, &req)?;
            Ok(crate::mcp::tool_response(&serde_json::to_value(resp)?))
        };
        crate::mcp::handle_rpc(&msg2, &exec)
    })
    .await;
    match result {
        Ok(Some(resp)) => Json(resp).into_response(),
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Err(e) => err_response(format!("mcp task aborted: {e}")),
    }
}

fn write_daemon_info(port: u16) -> Result<()> {
    let path = paths::corpora_root()?
        .parent()
        .context("corpora root has no parent")?
        .join("daemon.json");
    let info = serde_json::json!({
        "pid": std::process::id(),
        "port": port,
        "base_url": format!("http://127.0.0.1:{port}"),
    });
    std::fs::write(&path, info.to_string()).with_context(|| format!("write {}", path.display()))
}

/// Path of the daemon discovery file (`~/.cache/t3rag/daemon.json`).
pub fn daemon_info_path() -> Result<std::path::PathBuf> {
    Ok(paths::corpora_root()?
        .parent()
        .context("corpora root has no parent")?
        .join("daemon.json"))
}

/// Discovery: read the live daemon's base URL if the info file exists and the
/// pid inside is still running. Returns None to signal "use cold path".
pub fn discover_daemon() -> Option<String> {
    let path = daemon_info_path().ok()?;
    let s = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    let pid = v.get("pid")?.as_u64()?;
    // Cheap liveness: /proc/<pid> exists on linux.
    if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
        return None;
    }
    v.get("base_url")?.as_str().map(|s| s.to_string())
}

/// Find the daemon, spawning `t3rag serve` if absent; poll for readiness
/// (model warmup can take ~10-60s on a cold box, longer under load).
pub fn ensure_daemon() -> Result<String> {
    if let Some(base) = discover_daemon() {
        if ureq::get(&format!("{base}/health")).call().is_ok() {
            return Ok(base);
        }
    }
    spawn_daemon()?;
    wait_for_daemon(std::time::Duration::from_secs(180))
}

/// Spawn `t3rag serve` detached (null stdio — the daemon logs to its own file).
fn spawn_daemon() -> Result<()> {
    let exe = std::env::current_exe().context("current exe")?;
    std::process::Command::new(exe)
        .arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn t3rag serve")?;
    Ok(())
}

/// Poll `discover_daemon`+`/health` until ready or timeout.
fn wait_for_daemon(timeout: std::time::Duration) -> Result<String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(base) = discover_daemon() {
            if ureq::get(&format!("{base}/health")).call().is_ok() {
                return Ok(base);
            }
        }
        if std::time::Instant::now() > deadline {
            anyhow::bail!("daemon did not become ready within {timeout:?}");
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}
