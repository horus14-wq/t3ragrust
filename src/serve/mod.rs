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
    /// Corpus dir — read only by the sparse leg (t3-css.4.3).
    #[allow(dead_code)]
    dir: std::path::PathBuf,
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
    /// Lazy per-name reranker pools (t3-css.2.6): a daemon may serve
    /// minilm AND bge-m3 queries — each loads on first use.
    rerankers: Mutex<rustc_hash::FxHashMap<String, RerankerPool>>,
    reranker_default: String,
    rerank_max_tokens: usize,
    corpora: RwLock<HashMap<String, Arc<Corpus>>>,
    query_cache: Mutex<QueryLru>,
    ep: crate::cli::Ep,
    threads: usize,
    /// Bearer token for authed routes (t3-css.3.7).
    token: Option<String>,
    /// Total handled requests (metrics).
    requests: std::sync::atomic::AtomicU64,
    /// Lazy SPLADE engine for the sparse leg (t3-css.4.3).
    #[cfg(feature = "sparse")]
    sparse_engine: Mutex<Option<crate::sparse::SparseEngine>>,
    started: Instant,
    last_used: Mutex<Instant>,
}

#[derive(Deserialize, Default)]
struct SearchRequest {
    query: String,
    /// Client self-identifies for the query log (t3-pbh):
    /// `cli` | `mcp` | `http` (absent = http).
    client: Option<String>,
    corpus: Option<String>,
    top_k: Option<usize>,
    candidates: Option<usize>,
    rerank: Option<bool>,
    model: Option<String>,
    exact: Option<bool>,
    preset: Option<String>,
    /// Emit per-stage latencies + score provenance (t3-css.2.1).
    explain: Option<bool>,
    /// Runtime fusion/ANN knobs (t3-css.2.5).
    ef_search: Option<usize>,
    rrf_k: Option<f32>,
    dense_weight: Option<f32>,
    bm25_weight: Option<f32>,
    /// Reranker name (registry: minilm/bge-base/bge-m3) + its own
    /// candidate cap — decoupled from retrieval `candidates`.
    reranker: Option<String>,
    rerank_candidates: Option<usize>,
    /// Search ergonomics (t3-css.3.4): post-filter `key~substr`,
    /// per-source diversity cap, text truncation, score floor, pagination.
    /// Third RRF leg: learned sparse (SPLADE) if the corpus has sparse.bin.
    #[allow(dead_code)]
    sparse: Option<bool>,
    /// Extra query phrasings (t3-css.4.4): each is embedded + ranked,
    /// all rankings fused into one RRF merge.
    queries: Option<Vec<String>>,
    /// Passage ids to subtract from results (t3-b68) — applied before
    /// top_k slicing so the list still fills.
    exclude: Option<Vec<u64>>,
    filter: Option<Vec<String>>,
    max_per_source: Option<usize>,
    max_chars: Option<usize>,
    min_score: Option<f32>,
    offset: Option<usize>,
    /// t3-kz0: more-like-this — dense search against a seed passage id
    /// (or ad-hoc text via `like_text`); the seed id is excluded.
    like: Option<u64>,
    like_text: Option<String>,
    /// t3-tdu: group results by `question` | `source` | `corpus` — makes
    /// the T3 multi-corpus triple structure visible.
    group_by: Option<String>,
    /// t3-4vh: pseudo-relevance feedback — blend top-3 hit vectors into
    /// the query and re-run the dense leg (the no-LLM recall lever).
    expand: Option<bool>,
    /// t3-6vv: cap total returned passage text at ~N tokens (chars/4).
    token_budget: Option<usize>,
}

#[derive(Serialize, Clone)]
struct HitOut {
    rank: usize,
    passage_id: u64,
    /// Which corpus produced this hit (multi-corpus queries, t3-css.3.4).
    corpus: String,
    /// Ranked score — the kind is named by `score_kind` on the response
    /// (RRF fraction pre-rerank, cross-encoder logit after).
    score: f32,
    /// Per-leg scores for `--explain` (absent legs are null).
    dense_score: Option<f32>,
    bm25_score: Option<f32>,
    rerank_score: Option<f32>,
    text: String,
    title: Option<String>,
    question: Option<String>,
    source: Option<String>,
}

/// Per-stage wall times in milliseconds (t3-css.2.1 `--explain`).
#[derive(Serialize)]
struct StageMs {
    embed_ms: f64,
    dense_ms: f64,
    bm25_ms: f64,
    rerank_ms: f64,
    hydrate_ms: f64,
}

/// Always-on provenance block (t3-ny0): the agent shouldn't need a
/// second call to know what answered the query or whether to trust it.
#[derive(Serialize)]
struct RespMeta<'a> {
    corpora: Vec<String>,
    model: String,
    dims: u32,
    quantization: String,
    preset: Option<&'a str>,
    took_ms: f64,
    /// Hours since the oldest queried corpus was built (`created_at`).
    index_age_h: Option<f64>,
}

/// One `--group-by` bucket (t3-tdu): ordered by best in-group score.
#[derive(Serialize)]
struct GroupOut {
    key: String,
    hits: Vec<HitOut>,
}

#[derive(Serialize)]
struct SearchResponse<'a> {
    query: &'a str,
    corpus: String,
    /// Which scale `score` uses: `rrf` (fused) or `rerank_logit`.
    score_kind: &'static str,
    results: Vec<HitOut>,
    /// Present iff `group_by` was requested; `results` stays the flat
    /// ranked list for compatibility.
    groups: Option<Vec<GroupOut>>,
    stages_ms: Option<StageMs>,
    /// Honesty surface: truncation, staleness, missing legs, fallbacks.
    warnings: Vec<String>,
    meta: RespMeta<'a>,
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
            dir,
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

/// Resolve a `--corpus` spec to concrete corpora (t3-css.3.4):
/// `None`/empty -> newest, `all` -> every indexed corpus, comma-list ->
/// each named corpus (each `Arc`-cached by [`DaemonState::corpus`]).
fn resolve_corpora(st: &DaemonState, spec: Option<&str>) -> Result<Vec<(String, Arc<Corpus>)>> {
    let names: Vec<String> = match spec.map(str::trim) {
        None | Some("") => vec![newest_corpus(&st.corpora_root)?],
        Some("all") => all_corpora(&st.corpora_root)?,
        Some(s) => s.split(',').map(|n| n.trim().to_string()).collect(),
    };
    names.iter().map(|n| st.corpus(Some(n))).collect()
}

/// Every indexed corpus under `root` (sorted, deterministic): a dir
/// counts iff it has `meta.json`. Used by `--corpus all`.
fn all_corpora(root: &std::path::Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .with_context(|| format!("read {}", root.display()))?
        .flatten()
        .filter(|e| e.path().is_dir() && e.path().join("meta.json").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

/// Name of the most recently indexed corpus (the `--corpus` default);
/// uses `meta.json` mtime — `state.json` tracking is the 4.7 follow-up.
///
/// # Errors
/// The corpora root cannot be read, or no corpus exists.
pub fn newest_corpus(root: &std::path::Path) -> Result<String> {
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
/// `t3rag serve` entry: build the tokio runtime and run [`serve`].
///
/// # Errors
/// The runtime fails to build or [`serve`] errors on startup.
pub fn run_serve(args: ServeArgs) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .build()
        .context("tokio runtime")?;
    rt.block_on(async move { serve(args).await })
}

/// Daemon body: load the embedder (plus any `--corpus` preloads),
/// bind the router, write `daemon.json` for discovery, and serve until
/// ctrl-c or `--idle-timeout`.
async fn serve(args: ServeArgs) -> Result<()> {
    let mut spec = embed::resolve_model(args.model.as_deref());
    if let Some(i) = &args.instruction {
        spec.instruction = i.clone();
    }
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
        rerankers: Mutex::new(rustc_hash::FxHashMap::default()),
        reranker_default: args.reranker.clone(),
        rerank_max_tokens: args.rerank_tokens,
        corpora: RwLock::new(HashMap::new()),
        query_cache: Mutex::new(QueryLru::new(128)),
        ep: args.ep,
        threads: args.threads,
        token: args.token.clone(),
        requests: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "sparse")]
        sparse_engine: Mutex::new(None),
        started: Instant::now(),
        last_used: Mutex::new(Instant::now()),
    });

    // Preload corpora requested at startup.
    for name in &args.corpus {
        if let Err(e) = state.corpus(Some(name)) {
            tracing::warn!(corpus = %name, "preload failed: {e:#}");
        }
    }

    // Bearer-token middleware when --token is set (t3-css.3.7): /health
    // stays open for liveness; /search and /mcp require the token.
    let authed = Router::new()
        .route("/search", post(search))
        .route("/mcp", post(mcp_http))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            |State(st): State<Arc<DaemonState>>,
             req: axum::http::Request<axum::body::Body>,
             next: axum::middleware::Next| async move {
                if let Some(t) = &st.token {
                    let ok = req
                        .headers()
                        .get("authorization")
                        .and_then(|h| h.to_str().ok())
                        .map(|h| h == format!("Bearer {t}"))
                        .unwrap_or(false);
                    if !ok {
                        return StatusCode::UNAUTHORIZED.into_response();
                    }
                }
                next.run(req).await
            },
        ));
    let app = Router::new()
        .route("/health", get(health))
        .route("/corpora", get(list_corpora))
        .route("/metrics", get(metrics))
        .route("/passages/{corpus}/{id}", get(passage))
        .route("/suggest", get(suggest))
        .merge(authed)
        .layer(axum::extract::DefaultBodyLimit::max(args.max_body))
        .with_state(state.clone());

    let addr: SocketAddr = format!("{}:{}", args.bind, args.port)
        .parse()
        .with_context(|| format!("bad --bind {}:{}", args.bind, args.port))?;
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

/// `GET /health` — liveness for spawn/discovery probes; stays
/// unauthenticated even under `--token`.
async fn health(State(st): State<Arc<DaemonState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "ok": true,
        "uptime_s": st.started.elapsed().as_secs(),
        "corpora_loaded": st.corpora.read().unwrap().len(),
    }))
}

/// `GET /corpora` — names of corpora loaded in memory (not a disk scan).
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

/// 500 + structured error (t3-v7m): agents get `{error:{code,message,
/// hint,retryable}}` instead of prose to parse. Codes are the contract —
/// classify on the message chain; keep them stable.
fn err_response(msg: String) -> axum::response::Response {
    let (code, hint) = classify_error(&msg);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "code": code,
                "message": msg,
                "hint": hint,
                "retryable": code == "daemon_busy" || code == "model_loading",
            }
        })),
    )
        .into_response()
}

/// Map a failure's message chain to a stable error code + actionable
/// hint (t3-v7m).
fn classify_error(msg: &str) -> (&'static str, String) {
    if msg.contains("not indexed") || msg.contains("no corpora found") {
        let mut hint = String::from("list available corpora: GET /corpora or `t3rag list`");
        if let Ok(root) = paths::corpora_root() {
            let names: Vec<String> = std::fs::read_dir(&root)
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.path().join("meta.json").exists())
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect()
                })
                .unwrap_or_default();
            if !names.is_empty() {
                hint.push_str(&format!("; indexed: {}", names.join(", ")));
            }
        }
        ("corpus_not_found", hint)
    } else if msg.contains("out of range") {
        (
            "passage_not_found",
            "check passage id bounds via `t3rag stats`".into(),
        )
    } else if msg.contains("sparse.bin") || msg.contains("sparse feature") {
        (
            "sparse_unavailable",
            "index with --sparse or build --features sparse".into(),
        )
    } else if msg.contains("model") && (msg.contains("download") || msg.contains("hf")) {
        (
            "model_unavailable",
            "model not cached — run `t3rag selftest` or index once".into(),
        )
    } else if msg.contains("aborted") {
        ("internal", "daemon-side failure — check daemon log".into())
    } else {
        (
            "search_failed",
            "inspect message; `t3rag doctor` for environment sanity".into(),
        )
    }
}

/// Shared query path: embed -> ANN + BM25 -> RRF -> optional rerank ->
/// hydrate+dedupe by parent.
/// The shared search pipeline used by `/search` and the `/mcp` tool:
/// embed (LRU-cached) -> per-corpus ANN + BM25 + optional sparse legs ->
/// RRF merge -> optional cross-encoder -> parent hydration + dedupe ->
/// filters/diversity/truncation/pagination (t3-css.3.4).
fn do_search<'a>(st: &'a DaemonState, req: &'a SearchRequest) -> Result<SearchResponse<'a>> {
    let t0 = Instant::now();
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
    // Multi-corpus (t3-css.3.4): comma-list or "all" → per-corpus legs,
    // fused globally; hits carry their corpus name.
    let corpora = resolve_corpora(st, req.corpus.as_deref())?;
    tracing::debug!(n = corpora.len(), "stage: corpora ready");
    // Preset resolves to candidate/rerank defaults before explicit fields
    // (parity with the cold path's --fast/--balanced/--max).
    st.requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (pc, pr) = match req.preset.as_deref() {
        Some("fast") => (20, false),
        Some("max") => (100, true),
        _ => (50, true),
    };
    let top_k = req.top_k.unwrap_or(3);
    let cand = req.candidates.unwrap_or(pc);
    let want_rerank = req.rerank.unwrap_or(pr);

    // Multi-query (t3-css.4.4): the primary query plus `queries`
    // extras — each embedded, its rankings fused into the same merge.
    let mut queries = Vec::with_capacity(1 + req.queries.as_ref().map_or(0, |q| q.len()));
    queries.push(req.query.clone());
    if let Some(extra) = &req.queries {
        queries.extend(extra.iter().cloned());
    }
    // t3-kz0: --like replaces the query with a passage-side embedding of
    // a seed (corpus passage by id, or ad-hoc text) — dense leg only.
    let like_seed = if let Some(lt) = &req.like_text {
        Some(lt.clone())
    } else {
        match req.like {
            Some(id) => Some(corpora[0].1.reader.get(id)?.text),
            None => None,
        }
    };
    let t_embed = Instant::now();
    let mut qvecs = Vec::with_capacity(queries.len());
    if let Some(seed) = &like_seed {
        let mut v = st.embedder.embed_passages(&[seed.as_str()])?;
        qvecs.push(std::mem::take(&mut v[0]));
    }
    for q in &queries {
        if like_seed.is_some() {
            break; // seed vector replaces query embeddings entirely
        }
        let cached = st.query_cache.lock().unwrap().get(q);
        let v = match cached {
            Some(v) => v,
            None => {
                let v = {
                    tracing::debug!("stage: embed start");
                    st.embedder.embed_query(q)?
                };
                st.query_cache.lock().unwrap().put(q.clone(), v.clone());
                v
            }
        };
        qvecs.push(v);
    }
    let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!("stage: embed done");
    // Multi-corpus legs (t3-css.3.4): same qvec against every corpus,
    // per-corpus rrf, then a global merge on fused score — ids are
    // corpus-local so hits are tagged (corpus_idx, id).
    let t_dense = Instant::now();
    let t_bm = Instant::now();
    let mut dense_ms = 0f64;
    let mut bm25_ms = 0f64;
    /// One corpus's legs + fused ranking (t3-css.3.4).
    struct CorpusLegs {
        ci: usize,
        dense_map: rustc_hash::FxHashMap<u64, f32>,
        bm_map: rustc_hash::FxHashMap<u64, f32>,
        fused: Vec<(u64, f32)>,
    }
    let mut per_corpus: Vec<CorpusLegs> = Vec::new();
    // t3-ny0: warn when --sparse was requested but no queried corpus
    // contributed a sparse leg (feature off or sparse.bin absent).
    let mut sparse_missing = req.sparse == Some(true);
    for (ci, (_name, corpus)) in corpora.iter().enumerate() {
        // Multi-query (t3-css.4.4): every query phrasing contributes its
        // own legs; they're all fused into one ranking below.
        let mut all_legs: Vec<Vec<(u64, f32)>> = Vec::new();
        let mut all_weights: Vec<f32> = Vec::new();
        let mut ann_map: rustc_hash::FxHashMap<u64, f32> = Default::default();
        let mut bm_map: rustc_hash::FxHashMap<u64, f32> = Default::default();
        for (q, qvec) in queries.iter().zip(&qvecs) {
            // MRL dims (t3-css.2.4): match each corpus's stored dims.
            let qv = crate::search::truncate_unit(qvec, corpus.meta.dims as usize);
            let ann = if req.exact == Some(true) {
                corpus.dense.exact_search(&qv, cand)?
            } else {
                corpus.dense.search(&qv, cand, req.ef_search)?
            };
            dense_ms += t_dense.elapsed().as_secs_f64() * 1000.0;
            // --like is dense-only (t3-kz0): BM25 on a seed passage is
            // noise, and there may be no query text at all (like_text).
            let bm = if like_seed.is_none() {
                let bm = corpus.lex.search(q, cand).unwrap_or_default();
                bm25_ms += t_bm.elapsed().as_secs_f64() * 1000.0;
                bm
            } else {
                Vec::new()
            };
            for (id, s) in &ann {
                let e = ann_map.entry(*id).or_insert(f32::NEG_INFINITY);
                *e = e.max(*s);
            }
            for (id, s) in &bm {
                let e = bm_map.entry(*id).or_insert(f32::NEG_INFINITY);
                *e = e.max(*s);
            }
            all_legs.push(ann);
            all_weights.push(req.dense_weight.unwrap_or(1.0));
            if !bm.is_empty() {
                all_legs.push(bm);
                all_weights.push(req.bm25_weight.unwrap_or(1.0));
            }
        }
        // Third leg (t3-css.4.3): SPLADE when requested + sparse.bin exists.
        #[cfg(feature = "sparse")]
        let sparse_leg: Vec<(u64, f32)> = if req.sparse == Some(true) {
            let sp_file = crate::sparse::sparse_file(&corpus.dir);
            if sp_file.exists() {
                let mut g = st.sparse_engine.lock().unwrap();
                if g.is_none() {
                    *g = crate::sparse::SparseEngine::load().ok();
                }
                match g.as_mut() {
                    Some(e) => {
                        let docs = crate::sparse::read_sparse(&sp_file).unwrap_or_default();
                        let qv = e
                            .embed(&[req.query.as_str()])
                            .ok()
                            .and_then(|v| v.into_iter().next());
                        match qv {
                            Some(q) => crate::sparse::score(&docs, &q)
                                .into_iter()
                                .take(cand)
                                .collect(),
                            None => Vec::new(),
                        }
                    }
                    None => Vec::new(),
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        #[cfg(not(feature = "sparse"))]
        let sparse_leg: Vec<(u64, f32)> = Vec::new();
        if !sparse_leg.is_empty() {
            all_legs.push(sparse_leg);
            all_weights.push(1.0);
            sparse_missing = false;
        }
        let fused = rrf_fuse(
            &all_legs,
            req.rrf_k.unwrap_or(crate::search::RRF_K),
            &all_weights,
        );
        per_corpus.push(CorpusLegs {
            ci,
            dense_map: ann_map,
            bm_map,
            fused,
        });
    }
    // Global merge on fused score across corpora.
    let mut ranked: Vec<(usize, u64, f32)> = per_corpus
        .iter()
        .flat_map(|c| c.fused.iter().map(move |(id, s)| (c.ci, *id, *s)))
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    tracing::debug!(corpora = corpora.len(), "stage: fused");

    // t3-4vh: pseudo-relevance feedback. Top-3 first-pass passages are
    // embedded passage-side and blended into the query vector
    // (0.7*q + 0.3*mean(passage vecs)); the expanded vector's dense leg is
    // fused back in per corpus via RRF. `expanded_from` lands in warnings
    // so agents see what the system thinks the query meant.
    let mut expanded_from: Vec<u64> = Vec::new();
    if req.expand == Some(true) && !ranked.is_empty() && !qvecs.is_empty() {
        let seeds: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|(ci, id, _)| corpora[*ci].1.reader.get(*id).map(|p| p.text))
            .collect::<Result<_>>()?;
        expanded_from = ranked.iter().take(3).map(|(_, id, _)| *id).collect();
        let pvs = st
            .embedder
            .embed_passages(&seeds.iter().map(String::as_str).collect::<Vec<_>>())?;
        // 0.7*q + 0.3 * mean(passage vectors), re-normalized to unit length.
        let mut eq = qvecs[0].clone();
        for a in &mut eq {
            *a *= 0.7;
        }
        let share = 0.3 / pvs.len().max(1) as f32;
        for pv in &pvs {
            for (a, b) in eq.iter_mut().zip(pv.iter()) {
                *a += b * share;
            }
        }
        let n = eq.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for a in &mut eq {
            *a /= n;
        }
        for (ci, (_, corpus)) in corpora.iter().enumerate() {
            let eqv = crate::search::truncate_unit(&eq, corpus.meta.dims as usize);
            let ann = corpus.dense.search(&eqv, cand, req.ef_search)?;
            per_corpus[ci].fused = rrf_fuse(
                &[per_corpus[ci].fused.clone(), ann],
                req.rrf_k.unwrap_or(crate::search::RRF_K),
                &[1.0, 0.7],
            );
        }
        ranked = per_corpus
            .iter()
            .flat_map(|c| c.fused.iter().map(move |(id, s)| (c.ci, *id, *s)))
            .collect();
        ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    }
    let ranked_total = ranked.len();

    let mut score_kind = "rrf";
    let mut rerank_map = rustc_hash::FxHashMap::default();
    let mut rerank_ms = 0.0;
    if want_rerank {
        let t_rr = Instant::now();
        // Preset → reranker mapping (measured: m3 ~1.2s/pair on CPU — the
        // balanced preset uses minilm; max keeps m3).
        let rr_name = req
            .reranker
            .clone()
            .or_else(|| match req.preset.as_deref() {
                Some("balanced") => Some("minilm".to_string()),
                Some("max") => Some("bge-m3".to_string()),
                _ => None,
            })
            .unwrap_or_else(|| st.reranker_default.clone());
        let rr_cand = req.rerank_candidates.unwrap_or(cand).min(cand);
        let cand_rows: Vec<(usize, u64, f32)> = ranked.drain(..).take(rr_cand).collect();
        let mut pairs = Vec::with_capacity(cand_rows.len());
        let mut ids = Vec::with_capacity(cand_rows.len());
        for (ci, id, _) in &cand_rows {
            let p = corpora[*ci].1.reader.get(*id)?;
            pairs.push((req.query.clone(), p.text));
            ids.push((*ci, *id));
        }
        // Lazy per-name reranker pools; a load failure degrades to the
        // fused ranking rather than failing the query.
        let mut guard = st.rerankers.lock().unwrap();
        if !guard.contains_key(&rr_name) {
            match rerank::spec(&rr_name)
                .and_then(|s| RerankerPool::load(&s, st.threads, st.ep, 1, st.rerank_max_tokens))
            {
                Ok(p) => {
                    guard.insert(rr_name.clone(), p);
                }
                Err(e) => tracing::warn!(reranker = %rr_name, "reranker unavailable: {e:#}"),
            }
        }
        if let Some(pool) = guard.get_mut(&rr_name) {
            let scores = pool.score_pairs(&pairs)?;
            let mut rr: Vec<(usize, u64, f32)> = ids
                .into_iter()
                .zip(scores)
                .map(|((ci, id), s)| (ci, id, s))
                .collect();
            rr.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            rerank_map = rr.iter().map(|(ci, id, s)| ((*ci, *id), *s)).collect();
            ranked = rr;
            score_kind = "rerank_logit";
        }
        rerank_ms = t_rr.elapsed().as_secs_f64() * 1000.0;
    }
    let t_hyd = Instant::now();

    // Ergonomics (t3-css.3.4): filters/diversity/truncation/pagination.
    let filters: Vec<(String, String)> = req
        .filter
        .iter()
        .flatten()
        .filter_map(|f| {
            f.split_once('~')
                .map(|(k, v)| (k.to_string(), v.to_lowercase()))
        })
        .collect();
    let max_ps = req.max_per_source.unwrap_or(usize::MAX);
    // --exclude (t3-b68): parent passage ids subtracted post-fusion,
    // before top_k slicing — iterative refinement never re-shows a hit.
    let exclude: rustc_hash::FxHashSet<u64> = req.exclude.iter().flatten().copied().collect();
    let mut per_source: rustc_hash::FxHashMap<String, usize> = Default::default();
    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();
    let mut text_truncated = false;
    let mut budget_used = 0usize;
    let mut budget_exhausted = false;
    for (ci, id, score) in ranked.into_iter().skip(req.offset.unwrap_or(0)) {
        if let Some(ms) = req.min_score {
            if score < ms {
                continue;
            }
        }
        let (_name, corpus) = &corpora[ci];
        let p = corpus.reader.get(id)?;
        let parent = if p.parent_id != p.id {
            corpus.reader.get(p.parent_id)?
        } else {
            p
        };
        // t3-kz0: never return the --like seed itself — compare the
        // hydrated parent id (chunk ids differ from the passage id).
        if req.like.is_some_and(|s| s == parent.id) {
            continue;
        }
        if exclude.contains(&parent.id) {
            continue;
        }
        if !seen.insert(parent.id) {
            continue;
        }
        // Post-filters: substring match on source/title/question.
        if !filters.is_empty() {
            let ok = filters.iter().all(|(k, v)| {
                let f = match k.as_str() {
                    "source" => &parent.source,
                    "title" => &parent.title,
                    "question" => &parent.question,
                    _ => &None,
                };
                f.as_deref()
                    .map(|s| s.to_lowercase().contains(v.as_str()))
                    .unwrap_or(false)
            });
            if !ok {
                continue;
            }
        }
        // Diversity: ≤ max_per_source hits per source string.
        let src_key = parent.source.clone().unwrap_or_default();
        if *per_source.entry(src_key).or_insert(0) >= max_ps {
            continue;
        }
        *per_source
            .get_mut(&parent.source.clone().unwrap_or_default())
            .unwrap() += 1;
        // Explain mode (t3-css.2.1): per-leg scores by *chunk* id.
        let text = match req.max_chars {
            Some(mc) if parent.text.len() > mc => {
                text_truncated = true;
                parent.text.chars().take(mc).collect()
            }
            _ => parent.text,
        };
        // --token-budget (t3-6vv): fill in rank order until ~N tokens of
        // text are returned; the remainder reports as truncated.
        if let Some(tb) = req.token_budget {
            budget_used += text.len() / 4;
            if budget_used > tb {
                budget_exhausted = true;
                break;
            }
        }
        results.push(HitOut {
            rank: results.len() + 1,
            passage_id: parent.id,
            corpus: corpora[ci].0.clone(),
            score,
            dense_score: per_corpus[ci].dense_map.get(&id).copied(),
            bm25_score: per_corpus[ci].bm_map.get(&id).copied(),
            rerank_score: rerank_map.get(&(ci, id)).copied(),
            text,
            title: parent.title,
            question: parent.question,
            source: parent.source,
        });
        if results.len() >= top_k {
            break;
        }
    }
    let hydrate_ms = t_hyd.elapsed().as_secs_f64() * 1000.0;
    let stages_ms = if req.explain == Some(true) {
        Some(StageMs {
            embed_ms,
            dense_ms,
            bm25_ms,
            rerank_ms,
            hydrate_ms,
        })
    } else {
        None
    };
    // Query log (t3-pbh): the daemon is the single writer — every
    // handled search appends one JSONL line for `t3rag log`/insights.
    crate::log::append(&crate::log::SearchEntry {
        ts: crate::log::now_ts(),
        client: req.client.as_deref().unwrap_or("http"),
        query: &req.query,
        corpora: corpora
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(","),
        top_ids: results.iter().map(|h| h.passage_id).collect(),
        top_scores: results.iter().map(|h| h.score).collect(),
        took_ms: t0.elapsed().as_secs_f64() * 1000.0,
        preset: req.preset.as_deref(),
    });
    // t3-ny0: warnings + always-on meta block.
    let mut warnings = Vec::new();
    if results.len() == top_k && ranked_total.saturating_sub(req.offset.unwrap_or(0)) > top_k {
        warnings.push(format!(
            "results_truncated: {} of {} ranked hits returned (raise top_k or use --offset)",
            results.len(),
            ranked_total
        ));
    }
    if sparse_missing {
        warnings.push("sparse_leg_missing: --sparse requested but no corpus has sparse.bin (or built without the sparse feature)".into());
    }
    if text_truncated {
        warnings.push("passage_text_truncated: --max-chars applied".into());
    }
    if budget_exhausted {
        warnings.push(format!(
            "token_budget_exhausted: stopped after ~{} tokens of passage text",
            budget_used
        ));
    }
    if !expanded_from.is_empty() {
        warnings.push(format!(
            "expanded_from: PRF seeded by passages {expanded_from:?}"
        ));
    }
    // t3-c3j: leg-agreement + rank-gap honesty hints. These are real
    // computed signals, NOT calibrated confidence bands — RRF/rerank
    // scores aren't calibrated scales, so a scalar `confidence` would
    // be fabricated precision. Bands land once eval --history has
    // enough labeled data to calibrate thresholds per corpus.
    // Skipped for --like (dense-only by design — no BM25 leg ran).
    if like_seed.is_none() && !results.is_empty() {
        let top = &results[0];
        if top.dense_score.is_some() != top.bm25_score.is_some() {
            warnings.push("leg_disagreement: top hit surfaced by only one retrieval leg".into());
        }
        if results.len() > 1 {
            let (s1, s2) = (results[0].score, results[1].score);
            if s1 > 0.0 && (s1 - s2).abs() / s1 < 0.10 {
                warnings.push(format!(
                    "narrow_rank_gap: top-2 scores within 10% ({s1:.4} vs {s2:.4})"
                ));
            }
        }
        let both = results
            .iter()
            .filter(|h| h.dense_score.is_some() && h.bm25_score.is_some())
            .count();
        if results.len() > 1 && both * 2 < results.len() {
            warnings.push(format!(
                "weak_leg_agreement: {both}/{} top hits appear in both dense+BM25 legs",
                results.len()
            ));
        }
    }
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let index_age_h = corpora
        .iter()
        .filter_map(|(_, c)| {
            c.meta
                .created_at
                .as_deref()
                .and_then(|s| s.parse::<u64>().ok())
                .map(|ts| now_s.saturating_sub(ts) as f64 / 3600.0)
        })
        .fold(None, |acc: Option<f64>, h| {
            Some(acc.map_or(h, |a| a.max(h)))
        });
    if index_age_h.is_some_and(|h| h > 168.0) {
        warnings.push(format!(
            "stale_corpus: oldest queried index is {:.0}h old",
            index_age_h.unwrap_or(0.0)
        ));
    }
    // t3-tdu: group-by over the flat results — same question across
    // corpora collapses into one bucket ordered by its best score.
    let groups = req.group_by.as_deref().map(|key| {
        let key_of = |h: &HitOut| -> String {
            match key {
                "question" => h.question.clone().unwrap_or_else(|| "(none)".into()),
                "source" => h.source.clone().unwrap_or_else(|| "(none)".into()),
                "corpus" => h.corpus.clone(),
                _ => "(all)".into(),
            }
        };
        let mut order: Vec<String> = Vec::new();
        let mut map: std::collections::HashMap<String, Vec<HitOut>> =
            std::collections::HashMap::new();
        for h in &results {
            let k = key_of(h);
            if !map.contains_key(&k) {
                order.push(k.clone());
            }
            map.entry(k).or_default().push(h.clone());
        }
        order.sort_by(|a, b| {
            let bs = map[b].iter().map(|h| h.score).fold(f32::MIN, f32::max);
            let as_ = map[a].iter().map(|h| h.score).fold(f32::MIN, f32::max);
            bs.partial_cmp(&as_).unwrap_or(std::cmp::Ordering::Equal)
        });
        order
            .into_iter()
            .map(|k| GroupOut {
                hits: map.remove(&k).unwrap_or_default(),
                key: k,
            })
            .collect::<Vec<_>>()
    });
    let first = corpora.first().map(|(_, c)| c);
    Ok(SearchResponse {
        query: &req.query,
        corpus: corpora
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(","),
        score_kind,
        results,
        groups,
        stages_ms,
        warnings,
        meta: RespMeta {
            corpora: corpora.iter().map(|(n, _)| n.clone()).collect(),
            model: first.map_or(String::new(), |c| c.meta.model.clone()),
            dims: first.map_or(0, |c| c.meta.dims),
            quantization: first.map_or(String::new(), |c| c.meta.quantization.clone()),
            preset: req.preset.as_deref(),
            took_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_age_h,
        },
    })
}

/// `GET /metrics` — uptime + request counter for monitoring (t3-css.3.7).
/// `GET /passages/{corpus}/{id}` — one full passage incl. chunk linkage
/// (t3-vqd). Unauthenticated like /health: it's a read of a local store.
async fn passage(
    State(st): State<Arc<DaemonState>>,
    axum::extract::Path((name, id)): axum::extract::Path<(String, u64)>,
) -> impl IntoResponse {
    match st.corpus(if name == "latest" { None } else { Some(&name) }) {
        Ok((_, c)) => match c.reader.get(id) {
            Ok(p) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "id": p.id, "text": p.text, "title": p.title,
                    "question": p.question, "source": p.source,
                    "parent_id": p.parent_id, "chunk_idx": p.chunk_idx,
                })),
            )
                .into_response(),
            Err(e) => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("{e:#}")})),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("{e:#}")})),
        )
            .into_response(),
    }
}

/// Guess the T3 transform view a query's phrasing calls for (t3-x1z).
/// Heuristic keyword scoring — failure language → reflect, how-to →
/// struct, conceptual → semantic; pasted dumps (newlines/fences/very
/// long) → raw as a fallback when no keyword class fires. Returns the
/// transform label or `None` when phrasing is neutral.
fn intent_transform(q: &str) -> Option<&'static str> {
    const FAIL: &[&str] = &[
        "common mistake",
        "mistake",
        "what goes wrong",
        "goes wrong",
        "why does",
        "why do",
        "fail",
        "fails",
        "failure",
        "bug",
        "pitfall",
        "gotcha",
        "broken",
        "incorrect",
        "debug",
        "crash",
        "panic",
        "undefined",
        "leak",
        "race condition",
        "deadlock",
        "subtle",
        "wrong",
        "error",
        "exception",
        "hang",
        "slow",
    ];
    const HOW: &[&str] = &[
        "how to",
        "how do",
        "how can",
        "steps to",
        "step by step",
        "implement",
        "write a",
        "create",
        "build",
        "design",
        "approach",
        "solve",
        "algorithm",
        "pattern",
        "cheatsheet",
        "tutorial",
        "guide",
        "walkthrough",
        "fix",
    ];
    const CONCEPT: &[&str] = &[
        "what is",
        "what are",
        "explain",
        "why is",
        "why are",
        "core idea",
        "intuition",
        "concept",
        "mental model",
        "tradeoff",
        "trade-off",
        "difference between",
        "understand",
        "insight",
        "theory",
    ];
    // Strong dump signal wins outright: code fences, ≥3 newlines, or
    // very long pastes are raw episodes regardless of keywords.
    if q.contains("```") || q.len() > 800 || q.matches('\n').count() >= 3 {
        return Some("raw");
    }
    let l = q.to_lowercase();
    let score = |words: &[&str]| words.iter().filter(|w| l.contains(*w)).count();
    let (f, h, c) = (score(FAIL), score(HOW), score(CONCEPT));
    let best = f.max(h).max(c);
    if best == 0 {
        // Weak dump fallback: single newline or longish paste → raw.
        return (q.contains('\n') || q.len() > 500).then_some("raw");
    }
    // Ties resolve toward the more specific signal: failure > approach >
    // concept (a "how to fix error" still wants the pitfalls view most).
    if f >= h && f >= c && f > 0 {
        Some("reflect")
    } else if h >= c {
        Some("struct")
    } else {
        Some("semantic")
    }
}

/// `GET /suggest?q=...` — corpus routing (t3-69w): embed once, cosine vs
/// each corpus's manifest centroid where present. Transform type +
/// manifest stats lead the output; similarity is a secondary hint (T3
/// views share a problem space, so centroids separate poorly).
async fn suggest(
    State(st): State<Arc<DaemonState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let Some(q) = params.get("q").cloned().filter(|q| !q.is_empty()) else {
        return err_response("suggest requires ?q=<query>".into());
    };
    let result = tokio::task::spawn_blocking(move || {
        let qv = st.embedder.embed_query(&q)?;
        // t3-x1z: phrasing intent → preferred transform view; applied
        // as a multiplicative prior on similarity (a hint, not a route).
        let intent = intent_transform(&q);
        let mut rows = Vec::new();
        let rd = std::fs::read_dir(&st.corpora_root).context("corpora root")?;
        for e in rd.flatten() {
            let dir = e.path();
            let Ok(meta) = store::read_meta(&dir) else {
                continue;
            };
            let m = meta.manifest.as_ref();
            let transform = m.map(|m| m.transform.as_str()).unwrap_or("unknown");
            let boosted = intent.is_some_and(|i| i == transform);
            let sim = m.and_then(|m| m.centroid.as_ref()).map(|c| {
                let n = qv.len().min(c.len());
                let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
                for i in 0..n {
                    dot += qv[i] * c[i];
                    na += qv[i] * qv[i];
                    nb += c[i] * c[i];
                }
                let s = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                // Intent prior: 1.25x for the matching transform view.
                if boosted {
                    s * 1.25
                } else {
                    s
                }
            });
            rows.push(serde_json::json!({
                "corpus": e.file_name().to_string_lossy(),
                "transform": transform,
                "docs": meta.doc_count,
                "top_terms": m.map(|m| m.top_terms.clone()).unwrap_or_default(),
                "similarity": sim,
                "intent_boost": boosted,
            }));
        }
        rows.sort_by(|a, b| {
            let bs = b["similarity"].as_f64().unwrap_or(f64::MIN);
            let as_ = a["similarity"].as_f64().unwrap_or(f64::MIN);
            bs.partial_cmp(&as_).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok::<_, anyhow::Error>(serde_json::json!({
            "query": q,
            "intent": intent,
            "corpora": rows,
            "note": "type+stats are the routing signal; similarity is a secondary hint (intent-boosted ×1.25)",
        }))
    })
    .await;
    match result {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => err_response(format!("{e:#}")),
        Err(e) => err_response(format!("suggest aborted: {e}")),
    }
}

/// `GET /metrics` — uptime + handled-request counter (t3-css.3.7).
async fn metrics(State(st): State<Arc<DaemonState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "uptime_s": st.started.elapsed().as_secs(),
        "requests_total": st.requests.load(std::sync::atomic::Ordering::Relaxed),
        "corpora_loaded": st.corpora.read().unwrap().len(),
    }))
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
        let exec = |name: &str, args: &serde_json::Value| -> Result<serde_json::Value> {
            // batch_search (t3-w4d): N independent queries → array of
            // result sets (do NOT fuse — that's the `queries` arg).
            if name == "batch_search" {
                let corpus = args["corpus"].as_str().map(String::from);
                let top_k = args["top_k"].as_u64().map(|v| v as usize);
                let qs: Vec<String> = args["queries"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .context("batch_search requires queries[]")?;
                let mut out = Vec::with_capacity(qs.len());
                for q in &qs {
                    let req = SearchRequest {
                        query: q.clone(),
                        // MCP-originated — keeps the query log's
                        // client field honest (t3-pbh).
                        client: Some("mcp".to_string()),
                        corpus: corpus.clone(),
                        top_k,
                        ..SearchRequest::default()
                    };
                    match do_search(&st, &req) {
                        Ok(r) => out.push(serde_json::to_value(r)?),
                        Err(e) => out.push(serde_json::json!({
                            "query": q, "error": format!("{e:#}")
                        })),
                    }
                }
                return Ok(serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string(&out)?
                    }]
                }));
            }
            // list_corpora / suggest (t3-ixu): orientation tools —
            // manifests from disk; routing via the /suggest logic inline.
            if name == "list_corpora" {
                let mut out = Vec::new();
                if let Ok(rd) = std::fs::read_dir(&st.corpora_root) {
                    for e in rd.flatten() {
                        if let Ok(m) = store::read_meta(&e.path()) {
                            out.push(serde_json::json!({
                                "corpus": e.file_name().to_string_lossy(),
                                "docs": m.doc_count,
                                "transform": m.manifest.as_ref()
                                    .map(|m| m.transform.clone()).unwrap_or_else(|| "unknown".into()),
                                "top_terms": m.manifest.as_ref()
                                    .map(|m| m.top_terms.clone()).unwrap_or_default(),
                            }));
                        }
                    }
                }
                return Ok(crate::mcp::tool_response(
                    &serde_json::json!({"corpora": out}),
                ));
            }
            if name == "suggest" {
                let q = args["query"].as_str().context("suggest requires query")?;
                let qv = st.embedder.embed_query(q)?;
                let mut rows = Vec::new();
                if let Ok(rd) = std::fs::read_dir(&st.corpora_root) {
                    for e in rd.flatten() {
                        if let Ok(meta) = store::read_meta(&e.path()) {
                            let m = meta.manifest.as_ref();
                            let sim = m.and_then(|m| m.centroid.as_ref()).map(|c| {
                                let n = qv.len().min(c.len());
                                let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
                                for i in 0..n {
                                    dot += qv[i] * c[i];
                                    na += qv[i] * qv[i];
                                    nb += c[i] * c[i];
                                }
                                (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f64
                            });
                            rows.push(serde_json::json!({
                                "corpus": e.file_name().to_string_lossy(),
                                "transform": m.map(|m| m.transform.as_str()).unwrap_or("unknown"),
                                "docs": meta.doc_count,
                                "similarity": sim,
                            }));
                        }
                    }
                }
                return Ok(crate::mcp::tool_response(&serde_json::json!({"corpora": rows})));
            }
            // get_passage (t3-vqd): pure store read via the loaded corpus —
            // no embed, no fusion; mirrors GET /passages/{corpus}/{id}.
            if name == "get_passage" {
                let corpus = args["corpus"].as_str();
                let id = args["passage_id"]
                    .as_u64()
                    .context("get_passage requires passage_id")?;
                let (_, c) = st.corpus(corpus)?;
                let p = c.reader.get(id)?;
                let mut t = String::new();
                if let Some(x) = &p.title {
                    t.push_str(&format!("title: {x}\n"));
                }
                if let Some(x) = &p.question {
                    t.push_str(&format!("question: {x}\n"));
                }
                if let Some(x) = &p.source {
                    t.push_str(&format!("source: {x}\n"));
                }
                t.push_str(&p.text);
                return Ok(serde_json::json!({"content": [{"type": "text", "text": t}]}));
            }
            let req = SearchRequest {
                query: args["query"].as_str().unwrap_or("").to_string(),
                client: Some("mcp".to_string()),
                corpus: args["corpus"].as_str().map(String::from),
                top_k: args["top_k"].as_u64().map(|v| v as usize),
                candidates: args["candidates"].as_u64().map(|v| v as usize),
                rerank: args["rerank"].as_bool(),
                model: args["model"].as_str().map(String::from),
                exact: args["exact"].as_bool(),
                preset: args["preset"].as_str().map(String::from),
                explain: args["explain"].as_bool(),
                ef_search: args["ef_search"].as_u64().map(|v| v as usize),
                rrf_k: args["rrf_k"].as_f64().map(|v| v as f32),
                dense_weight: args["dense_weight"].as_f64().map(|v| v as f32),
                bm25_weight: args["bm25_weight"].as_f64().map(|v| v as f32),
                reranker: args["reranker"].as_str().map(String::from),
                rerank_candidates: args["rerank_candidates"].as_u64().map(|v| v as usize),
                filter: args["filter"].as_array().map(|fs| {
                    fs.iter()
                        .filter_map(|f| f.as_str().map(String::from))
                        .collect()
                }),
                max_per_source: args["max_per_source"].as_u64().map(|v| v as usize),
                max_chars: args["max_chars"].as_u64().map(|v| v as usize),
                min_score: args["min_score"].as_f64().map(|v| v as f32),
                offset: args["offset"].as_u64().map(|v| v as usize),
                sparse: args["sparse"].as_bool(),
                queries: args["queries"].as_array().map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
                exclude: args["exclude"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()),
                like: args["like"].as_u64(),
                like_text: args["like_text"].as_str().map(String::from),
                group_by: args["group_by"].as_str().map(String::from),
                expand: args["expand"].as_bool(),
                token_budget: args["token_budget"].as_u64().map(|v| v as usize),
            };
            let resp = do_search(&st, &req)?;
            Ok(crate::mcp::tool_response(&serde_json::to_value(resp)?))
        };
        // t3-ixu: ctx for prompts/resources — corpus manifests from disk
        // (loaded or not) + the SKILL playbook.
        let ctx = {
            let mut corpora = Vec::new();
            if let Ok(rd) = std::fs::read_dir(&st.corpora_root) {
                for e in rd.flatten() {
                    if let Ok(m) = store::read_meta(&e.path()) {
                        corpora.push(serde_json::json!({
                            "name": e.file_name().to_string_lossy(),
                            "docs": m.doc_count,
                            "model": m.model,
                            "dims": m.dims,
                            "manifest": m.manifest,
                        }));
                    }
                }
            }
            serde_json::json!({"corpora": corpora, "playbook": crate::skill::SKILL_MD})
        };
        crate::mcp::handle_rpc(&msg2, &exec, &ctx)
    })
    .await;
    match result {
        Ok(Some(resp)) => Json(resp).into_response(),
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Err(e) => err_response(format!("mcp task aborted: {e}")),
    }
}

/// Write `daemon.json` {pid, port, base_url} so `t3rag search`/`mcp`
/// can auto-discover (and auto-spawn) this daemon.
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
/// Path to `daemon.json` under the cache root.
pub fn daemon_info_path() -> Result<std::path::PathBuf> {
    Ok(paths::corpora_root()?
        .parent()
        .context("corpora root has no parent")?
        .join("daemon.json"))
}

/// Discovery: read the live daemon's base URL if the info file exists and the
/// pid inside is still running. Returns None to signal "use cold path".
/// Base URL of a live daemon (from `daemon.json` + `/health`), or
/// `None` when absent/stale. Never spawns — callers choose whether to
/// pay the spawn cost.
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
/// [`discover_daemon`] + spawn-on-miss (t3-css.1.3): returns a live
/// base URL, spawning a detached `t3rag serve` when none is up.
///
/// # Errors
/// The spawn or the 180s readiness wait failed.
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
/// Detach a `t3rag serve` child (current_exe); its log lands in the
/// cache dir. `ensure_daemon` only calls this after a miss.
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

#[cfg(test)]
mod tests {
    use super::intent_transform;

    /// t3-x1z AC: intent-typed queries map to the expected transform.
    #[test]
    fn intent_routes_by_phrasing() {
        assert_eq!(
            intent_transform("what are common mistakes in rust lifetimes"),
            Some("reflect")
        );
        assert_eq!(
            intent_transform("what goes wrong with tokio spawn"),
            Some("reflect")
        );
        assert_eq!(
            intent_transform("how to implement a binary search tree"),
            Some("struct")
        );
        assert_eq!(
            intent_transform("steps to build a rate limiter"),
            Some("struct")
        );
        assert_eq!(
            intent_transform("what is the core idea behind CRDTs"),
            Some("semantic")
        );
        assert_eq!(
            intent_transform("explain the intuition for consistent hashing"),
            Some("semantic")
        );
        // Pasted multi-line dump → raw (strong signal beats keywords).
        assert_eq!(
            intent_transform("panic! at src/main.rs:42\nthread 'main' panicked\nstack backtrace:\n  0: core::ops::function\n  1: app::main"),
            Some("raw")
        );
        // Short panic mention is failure language, not a dump → reflect.
        assert_eq!(
            intent_transform("why does my code panic on empty input"),
            Some("reflect")
        );
        // Neutral phrasing → no prior.
        assert_eq!(intent_transform("tokio semaphore"), None);
    }
}
