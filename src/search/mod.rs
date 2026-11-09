//! Query pipeline (t3-wkp.9; rerank .10 / thin-client .13 / presets .16 next).
//!
//! embed_query -> usearch ANN (top ~N) + tantivy BM25 (top ~N) -> RRF fusion
//! -> rerank -> hydrate top-k passages from the store. `--no-daemon` is
//! the only mode for now — this IS the cold in-process path the daemon and
//! thin client will share.

use rustc_hash::{FxHashMap, FxHashSet};

use anyhow::{bail, Context, Result};

use tracing::warn;

use crate::cli::SearchArgs;
use crate::embed::{self, EmbedderPool};
use crate::index::{Bm25Index, DenseIndex};
use crate::paths;
use crate::store::{self, PassageReader};

/// Default RRF fusion constant `k` — the Cormack-RRF standard.
/// `--rrf-k` overrides it per query (t3-css.2.5).
pub const RRF_K: f32 = 60.0;

/// One cold-path result: the parent-hydrated passage plus its fused
/// (or reranked) score. Mirrors the daemon's `HitOut`.
pub struct Hit {
    /// Passage id within the corpus.
    pub passage_id: u64,
    /// Fused RRF score, or the cross-encoder logit when reranked.
    pub score: f32,
    /// Passage text (truncated by `--max-chars` when set).
    pub text: String,
    /// Optional title/source/question metadata from ingest.
    pub title: Option<String>,
    /// The ingested `question` field, when present.
    pub question: Option<String>,
    /// The ingested `source` path/id, when present.
    pub source: Option<String>,
}

/// Truncate a unit vector to MRL `dims` and re-normalize (t3-css.2.4).
///
/// Matryoshka embeddings keep quality in leading dims; `dims >= len`
/// returns a plain clone. This is what makes `--dims` honest: both index
/// and query vectors truncate identically.
///
/// # Panics
/// Never — but an all-zero slice returns unnormalized (a valid degenerate).
pub fn truncate_unit(v: &[f32], dims: usize) -> Vec<f32> {
    if v.len() <= dims {
        return v.to_vec();
    }
    let t = &v[..dims];
    let norm = t.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return t.to_vec();
    }
    t.iter().map(|x| x / norm).collect()
}

/// Fuse ranked legs by weighted reciprocal rank (t3-css.2.5).
///
/// `score(d) = Σ_legs w_leg / (rrf_k + 1 + rank_in_leg)` — ranks are
/// 0-based so `k+1` is the usual formulation. Legs may be disjoint;
/// empty legs contribute nothing (sparse/absent legs just don't add).
///
/// # Panics
/// `weights` shorter than `legs` indexes out of bounds — always pass
/// one weight per leg.
pub fn rrf_fuse(legs: &[Vec<(u64, f32)>], rrf_k: f32, weights: &[f32]) -> Vec<(u64, f32)> {
    let mut acc: FxHashMap<u64, f32> = FxHashMap::default();
    for (li, leg) in legs.iter().enumerate() {
        let w = weights.get(li).copied().unwrap_or(1.0);
        for (rank, (id, _)) in leg.iter().enumerate() {
            *acc.entry(*id).or_default() += w / (rrf_k + rank as f32 + 1.0);
        }
    }
    let mut v: Vec<(u64, f32)> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Quality presets (t3-wkp.16): map --fast/--balanced/--max to
/// (model override, rerank default, candidates). Explicit --model,
/// --rerank/--no-rerank, and --candidates always win over the preset.
fn resolve_preset(args: &SearchArgs) -> (Option<String>, bool, usize) {
    let (model, rerank, cand) = if args.fast {
        (None::<&str>, false, 20)
    } else if args.max {
        // t3-css.1.1: NO quantized >0.6B Qwen3 ONNX exists on HF — every
        // 4B/8B export is fp32 external-data (~15GB, unusable on CPU), so
        // --max keeps the corpus's own embedder; its quality dials are
        // rerank + candidates (a real bigger-model --max awaits a quantized
        // 4B+ export or the external-data path).
        (None::<&str>, true, 100)
    } else if args.balanced {
        // Explicit --balanced opts into the truncated parallel rerank.
        (None::<&str>, true, 50)
    } else {
        // No preset: rerank off (CPU cross-encoder is expensive).
        (None::<&str>, false, 50)
    };
    (
        args.model.clone().or_else(|| model.map(String::from)),
        if args.rerank {
            true
        } else if args.no_rerank {
            false
        } else {
            rerank
        },
        if args.candidates != 50 {
            args.candidates
        } else {
            cand
        },
    )
}

/// `t3rag search`/`ask` entry: thin-client daemon path first (t3-wkp.13),
/// cold in-process path as fallback. `--no-daemon`/`--daemon` control
/// which runs; `--format` picks the renderer (text/json/jsonl/toon/md/
/// prompt — `ask` forces `prompt`).
///
/// # Errors
/// Model/index load fails on the cold path, or the daemon errors on
/// the thin path (when a daemon exists).
pub fn run_search(args: SearchArgs) -> Result<()> {
    let mode = if args.no_daemon {
        crate::cli::DaemonMode::Never
    } else {
        args.daemon
    };
    if mode != crate::cli::DaemonMode::Never {
        // Cass lesson: machine-format callers must not block on a daemon
        // spawn — they only use an already-running one. Interactive output
        // may wait through a spawn.
        let base = crate::serve::discover_daemon().or_else(|| {
            if args.json {
                None
            } else {
                match crate::serve::ensure_daemon() {
                    Ok(b) => Some(b),
                    Err(e) => {
                        if mode == crate::cli::DaemonMode::Always {
                            warn!("daemon required but failed to start: {e:#}");
                        } else {
                            warn!("daemon spawn failed ({e:#}); cold path");
                        }
                        None
                    }
                }
            }
        });
        if let Some(base) = base {
            // --batch (t3-w4d): N independent queries through the warm
            // daemon; one JSON result object per input line.
            if let Some(file) = &args.batch {
                let content = std::fs::read_to_string(file)
                    .with_context(|| format!("read {}", file.display()))?;
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let q = serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .and_then(|v| v["query"].as_str().map(String::from))
                        .unwrap_or_else(|| line.to_string());
                    match thin_search(&base, &args, Some(&q)) {
                        Ok(v) => println!("{}", serde_json::to_string(&v)?),
                        Err(e) => println!(
                            "{}",
                            serde_json::json!({"query": q, "error": format!("{e:#}")})
                        ),
                    }
                }
                return Ok(());
            }
            match thin_search(&base, &args, None) {
                Ok(resp) => {
                    print_response(&resp, args.out_format())?;
                    return Ok(());
                }
                Err(e) => warn!("daemon query failed ({e:#}); falling back to cold path"),
            }
        } else if mode == crate::cli::DaemonMode::Always {
            anyhow::bail!("--daemon always: no daemon available and spawn failed");
        }
    }
    run_cold_search(args)
}

/// Thin client: POST the query to the resident daemon (t3-wkp.13).
/// Builds the `SearchRequest` JSON from `SearchArgs`, forwarding every
/// flag (exact/preset/explain/ef/rrf/reranker/ergonomics/sparse).
///
/// # Errors
/// The daemon returns non-200 or the body is unparseable.
fn thin_search(
    base: &str,
    args: &SearchArgs,
    query_override: Option<&str>,
) -> Result<serde_json::Value> {
    let (model, rerank, candidates) = resolve_preset(args);
    let preset = if args.fast {
        Some("fast")
    } else if args.max {
        Some("max")
    } else if args.balanced {
        Some("balanced")
    } else {
        None
    };
    // http_status_as_error(false): non-2xx stays a readable Response so
    // the daemon's structured error body reaches us (t3-v7m).
    let agent = ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build(),
    );
    let mut resp = agent
        .post(&format!("{base}/search"))
        .send_json(serde_json::json!({
            "query": query_override.unwrap_or(&args.query),
            "client": "cli",
            "corpus": args.corpus,
            "top_k": args.top_k,
            "candidates": candidates,
            "rerank": rerank,
            "model": model,
            "exact": args.exact,
            "preset": preset,
            "explain": args.explain,
            "ef_search": args.ef_search,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
            "reranker": args.reranker,
            "rerank_candidates": args.rerank_candidates,
            "filter": args.filter,
            "max_per_source": args.max_per_source,
            "max_chars": args.max_chars,
            "min_score": args.min_score,
            "offset": args.offset,
            "exclude": args.exclude,
            "like": args.like,
            "like_text": args.like_text,
            "group_by": args.group_by,
            "expand": args.expand,
            "token_budget": args.token_budget,
            "sparse": args.sparse,
            "queries": args.also,
        }))
        .context("daemon POST /search")?;
    if resp.status().as_u16() != 200 {
        // Structured errors (t3-v7m): surface code+hint, not just status.
        let v: serde_json::Value = resp.body_mut().read_json().unwrap_or_default();
        let e = &v["error"];
        if e.is_object() {
            anyhow::bail!(
                "[{}] {}{}{}",
                e["code"].as_str().unwrap_or("daemon_error"),
                e["message"].as_str().unwrap_or("daemon query failed"),
                if e["hint"].as_str().is_some() {
                    " — "
                } else {
                    ""
                },
                e["hint"].as_str().unwrap_or(""),
            );
        }
        anyhow::bail!("daemon returned {}", resp.status().as_u16());
    }
    resp.body_mut().read_json().context("parse daemon response")
}

/// Shared renderer for daemon (thin-client) + cold results (t3-css.3.1):
/// Text/Json/Jsonl/Toon/Md — `toon` is the agent-token-optimal compact form.
fn print_response(v: &serde_json::Value, fmt: crate::cli::OutFmt) -> Result<()> {
    use crate::cli::OutFmt::*;
    match fmt {
        Json => println!("{}", serde_json::to_string_pretty(v)?),
        // Prompt-injection format (t3-css.3.5): titled context blocks ready
        // to paste into an agent's context, plus a one-line preamble.
        Prompt => {
            let q = v["query"].as_str().unwrap_or("");
            println!("<retrieved-traces query=\"{q}\">");
            if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
                for r in results {
                    let title = r["title"].as_str().unwrap_or("trace");
                    let text = r["text"].as_str().unwrap_or("").trim();
                    println!("<trace title=\"{title}\">\n{text}\n</trace>");
                }
            }
            println!("</retrieved-traces>");
        }
        _ => {
            // --group-by (t3-tdu): when the daemon grouped the hits,
            // render per-bucket headings instead of the flat list.
            if let Some(groups) = v.get("groups").and_then(|g| g.as_array()) {
                for g in groups {
                    println!("## {}", g["key"].as_str().unwrap_or("?"));
                    if let Some(hits) = g["hits"].as_array() {
                        for r in hits {
                            let id = r["passage_id"].as_u64().unwrap_or(0);
                            let score = r["score"].as_f64().unwrap_or(0.0);
                            let title = r["title"].as_str().unwrap_or("");
                            let corpus = r["corpus"].as_str().unwrap_or("");
                            let text = r["text"].as_str().unwrap_or("").trim();
                            match fmt {
                                Toon => println!(
                                    "  [{}] id={id} s={score:.3} {title} ~{}c",
                                    corpus,
                                    text.len()
                                ),
                                Md => println!("- [{corpus}] **{title}** ({score:.3}) — {text}"),
                                _ => println!(
                                    "  [{corpus}] id={id} score={score:.3} {title}\n    {text}"
                                ),
                            }
                        }
                    }
                    println!();
                }
                return Ok(());
            }
            let Some(results) = v.get("results").and_then(|r| r.as_array()) else {
                return Ok(());
            };
            for r in results {
                let rank = r["rank"].as_u64().unwrap_or(0);
                let id = r["passage_id"].as_u64().unwrap_or(0);
                let score = r["score"].as_f64().unwrap_or(0.0);
                let title = r["title"].as_str().unwrap_or("");
                let text = r["text"].as_str().unwrap_or("").trim();
                match fmt {
                    Jsonl => println!("{}", serde_json::to_string(r)?),
                    Toon => println!("[{rank}] id={id} s={score:.3} {title} ~{}c", text.len()),
                    Md => {
                        println!("### [{rank}] {title} (id={id}, s={score:.3})\n\n{text}\n");
                    }
                    _ => {
                        println!("--- [{rank}] id={id} score={score:.4}");
                        if !title.is_empty() {
                            println!("title: {title}");
                        }
                        println!("{text}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Cold in-process path (t3-wkp.9): load model + indexes locally.
fn run_cold_search(args: SearchArgs) -> Result<()> {
    // Multi-query fusion (t3-css.4.4) lives in the daemon's do_search.
    if !args.also.is_empty() {
        anyhow::bail!("--also requires the daemon path — drop --no-daemon");
    }
    if args.group_by.is_some() {
        anyhow::bail!("--group-by requires the daemon path — drop --no-daemon");
    }
    if args.expand {
        anyhow::bail!("--expand requires the daemon path — drop --no-daemon");
    }
    if args.batch.is_some() {
        anyhow::bail!("--batch requires the daemon path — drop --no-daemon");
    }
    let corpus = resolve_corpus(args.corpus.as_deref())?;
    let meta = store::read_meta(&corpus)?;

    let (preset_model, rerank, candidates) = resolve_preset(&args);
    let mut spec = match preset_model.as_deref() {
        Some(m) => embed::resolve_model(Some(m)),
        None => embed::ModelSpec::for_hf_repo(&meta.model),
    };
    if let Some(i) = &args.instruction {
        spec.instruction = i.clone();
    }
    if spec.repo_id != meta.model {
        anyhow::bail!(
            "preset/--model selects {} but corpus was indexed with {}; \
             re-index or drop the override",
            spec.repo_id,
            meta.model
        );
    }
    let emb = EmbedderPool::load(&spec, args.threads, args.ep, args.pool)?;
    let quant = parse_quant(&meta.quantization)?;
    let dense = DenseIndex::view(&corpus, meta.dims as usize, quant)?;
    let lex = Bm25Index::open(&corpus)?;
    let reader = PassageReader::open(&corpus)?;

    // --like (t3-kz0): seed the dense leg with a passage's own text
    // (id or ad-hoc text) — no BM25, seed excluded from results.
    let like_seed = if let Some(lt) = &args.like_text {
        Some(lt.clone())
    } else {
        args.like
            .map(|id| reader.get(id).map(|p| p.text))
            .transpose()?
    };
    let qvec = match &like_seed {
        Some(seed) => emb.embed_passages(&[seed.as_str()])?.remove(0),
        None => emb.embed_query(&args.query)?,
    };
    // MRL dims (t3-css.2.4): query vectors truncate to the corpus's stored
    // dims so --dims N indexes serve without a query-side flag.
    let qvec = truncate_unit(&qvec, meta.dims as usize);
    let ann = if args.exact {
        dense.exact_search(&qvec, candidates)?
    } else {
        dense.search(&qvec, candidates, args.ef_search)?
    };
    let (fused, legs) = if like_seed.is_none() {
        let bm = lex.search(&args.query, candidates).unwrap_or_else(|e| {
            warn!("bm25 leg failed: {e:#}");
            Vec::new()
        });
        (
            rrf_fuse(
                &[ann, bm],
                args.rrf_k,
                &[args.dense_weight, args.bm25_weight],
            ),
            "dense+bm25",
        )
    } else {
        (
            rrf_fuse(&[ann], args.rrf_k, &[args.dense_weight]),
            "dense(like)",
        )
    };
    tracing::debug!(legs, "cold-path fusion");

    let mut ranked: Vec<(u64, f32)> = fused;
    if rerank {
        // Cross-encoder rerank of the fused candidates (t3-wkp.10): hydrate
        // candidate texts, score (query, passage) pairs, reorder by logit.
        // t3-css.2.6: reranker registry + decoupled candidate cap.
        let rr_name = args
            .reranker
            .clone()
            .or_else(|| {
                if args.balanced {
                    Some("minilm".to_string())
                } else if args.max {
                    Some("bge-m3".to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "bge-m3".to_string());
        let rr_cand = args.rerank_candidates.unwrap_or(candidates).min(candidates);
        let cand: Vec<(u64, f32)> = ranked.into_iter().take(rr_cand).collect();
        let pool = crate::rerank::RerankerPool::load(
            &crate::rerank::spec(&rr_name)?,
            args.threads,
            args.ep,
            args.pool,
            crate::rerank::RERANK_MAX_TOKENS,
        )?;
        let mut pairs = Vec::with_capacity(cand.len());
        let mut ids = Vec::with_capacity(cand.len());
        for (id, _) in &cand {
            let p = reader.get(*id)?;
            // Same pre-slice as the daemon path (GH#4.6): the pair
            // tokenizer truncates to rerank_tokens anyway.
            let text: String = p.text.chars().take(2048).collect();
            pairs.push((args.query.clone(), text));
            ids.push(*id);
        }
        let scores = pool.score_pairs(&pairs)?;
        let mut reranked: Vec<(u64, f32)> = ids.into_iter().zip(scores).collect();
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked = reranked;
    }

    // Hydrate + dedupe by parent (t3-wkp.11): a chunk hit returns its whole
    // parent passage; multiple chunks of one parent collapse to the best.
    // --exclude (t3-b68): same parent-id subtraction as the daemon path.
    let exclude: FxHashSet<u64> = args.exclude.iter().copied().collect();
    let mut hits = Vec::new();
    let mut budget_used = 0usize;
    let mut seen_parents = FxHashSet::default();
    for (id, score) in ranked {
        let p = reader.get(id)?;
        let parent = if p.parent_id != p.id {
            reader.get(p.parent_id)?
        } else {
            p
        };
        if exclude.contains(&parent.id) {
            continue;
        }
        if args.like.is_some_and(|s| s == parent.id) {
            continue;
        }
        // --token-budget (t3-6vv): fill in rank order until ~N tokens.
        if let Some(tb) = args.token_budget {
            budget_used += parent.text.len() / 4;
            if budget_used > tb {
                eprintln!("token_budget_exhausted: stopped after ~{budget_used} tokens");
                break;
            }
        }
        if !seen_parents.insert(parent.id) {
            continue;
        }
        hits.push(Hit {
            passage_id: parent.id,
            score,
            text: parent.text,
            title: parent.title,
            question: parent.question,
            source: parent.source,
        });
        if hits.len() >= args.top_k {
            break;
        }
    }

    // Cold path renders through the same JSON shape the daemon returns
    // (t3-css.3.1 — one renderer for every format).
    let v = serde_json::json!({
        "query": args.query,
        "corpus": args.corpus.as_deref().unwrap_or("<auto>"),
        "results": hits.iter().enumerate().map(|(i, h)| serde_json::json!({
            "rank": i + 1,
            "passage_id": h.passage_id,
            "score": h.score,
            "text": h.text,
            "title": h.title,
            "question": h.question,
            "source": h.source,
        })).collect::<Vec<_>>(),
    });
    print_response(&v, args.out_format())
}

fn resolve_corpus(name: Option<&str>) -> Result<std::path::PathBuf> {
    if let Some(n) = name {
        if n == "all" || n.contains(',') {
            anyhow::bail!(
                "multi-corpus ('{n}') requires the daemon path — it auto-spawns; \
                 use a single corpus name or drop --no-daemon"
            );
        }
        let d = paths::corpus_dir(n)?;
        if !d.exists() {
            bail!("corpus '{n}' not indexed — run: t3rag index --input <path> --name {n}");
        }
        return Ok(d);
    }
    // Default: most recently modified corpus dir.
    let root = paths::corpora_root()?;
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for e in std::fs::read_dir(&root)
        .with_context(|| format!("read {} — no corpora indexed yet", root.display()))?
    {
        let p = e?.path();
        if p.is_dir() && p.join("meta.json").exists() {
            let m = p.metadata()?.modified()?;
            if best.as_ref().map(|(t, _)| m > *t).unwrap_or(true) {
                best = Some((m, p));
            }
        }
    }
    best.map(|(_, p)| p)
        .context("no corpora found — run: t3rag index --input <path> --name <name>")
}

fn parse_quant(s: &str) -> Result<crate::cli::Quantization> {
    use crate::cli::Quantization::*;
    match s {
        "i8" => Ok(I8),
        "bf16" => Ok(Bf16),
        "fp32" => Ok(Fp32),
        other => bail!("unknown quantization '{other}' in meta.json"),
    }
}

#[cfg(test)]
mod tests {
    use super::{rrf_fuse, RRF_K};

    #[test]
    fn rrf_prefers_docs_in_both_legs() {
        let dense = vec![(1u64, 0.9f32), (2, 0.8), (3, 0.7)];
        let bm25 = vec![(2u64, 5.0f32), (4, 4.0)];
        let fused = rrf_fuse(&[dense, bm25], RRF_K, &[1.0, 1.0]);
        assert_eq!(fused[0].0, 2, "doc in both legs should rank first");
        assert!(fused.iter().any(|(id, _)| *id == 4));
    }
}
