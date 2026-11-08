//! Query pipeline (t3-wkp.9; rerank .10 / thin-client .13 / presets .16 next).
//!
//! embed_query -> usearch ANN (top ~N) + tantivy BM25 (top ~N) -> RRF fusion
//! -> [rerank] -> hydrate top-k passages from the store. `--no-daemon` is
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

/// RRF constant (standard k=60).
pub const RRF_K: f32 = 60.0;

pub struct Hit {
    pub passage_id: u64,
    pub score: f32,
    pub text: String,
    pub title: Option<String>,
    pub question: Option<String>,
    pub source: Option<String>,
}

/// Truncate a unit vector to MRL `dims` and re-normalize (t3-css.2.4).
/// Full-length input is returned as a clone; over-length is sliced+renormed.
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

/// Fuse ranked legs by weighted reciprocal rank (t3-css.2.5):
/// score(d) = Σ w_leg/(rrf_k + rank_i(d)). `weights[i]` scales leg i's
/// contribution (1.0 = equal); `rrf_k` is the standard k (60 default).
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
    } else {
        // --balanced is the default and behaves like "no preset".
        (None::<&str>, true, 50)
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

/// `t3rag search` entry point — thin-client daemon path first (t3-wkp.13),
/// falling back to the cold in-process path when no daemon is discoverable
/// (or it errors), unless `--no-daemon` forces cold.
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
            match thin_search(&base, &args) {
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
fn thin_search(base: &str, args: &SearchArgs) -> Result<serde_json::Value> {
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
    let mut resp = ureq::post(&format!("{base}/search"))
        .send_json(serde_json::json!({
            "query": args.query,
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
        }))
        .context("daemon POST /search")?;
    if resp.status().as_u16() != 200 {
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

    let qvec = emb.embed_query(&args.query)?;
    // MRL dims (t3-css.2.4): query vectors truncate to the corpus's stored
    // dims so --dims N indexes serve without a query-side flag.
    let qvec = truncate_unit(&qvec, meta.dims as usize);
    let ann = if args.exact {
        dense.exact_search(&qvec, candidates)?
    } else {
        dense.search(&qvec, candidates, args.ef_search)?
    };
    let bm = lex.search(&args.query, candidates).unwrap_or_else(|e| {
        warn!("bm25 leg failed: {e:#}");
        Vec::new()
    });
    let fused = rrf_fuse(
        &[ann, bm],
        args.rrf_k,
        &[args.dense_weight, args.bm25_weight],
    );

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
            pairs.push((args.query.clone(), p.text));
            ids.push(*id);
        }
        let scores = pool.score_pairs(&pairs)?;
        let mut reranked: Vec<(u64, f32)> = ids.into_iter().zip(scores).collect();
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked = reranked;
    }

    // Hydrate + dedupe by parent (t3-wkp.11): a chunk hit returns its whole
    // parent passage; multiple chunks of one parent collapse to the best.
    let mut hits = Vec::new();
    let mut seen_parents = FxHashSet::default();
    for (id, score) in ranked {
        let p = reader.get(id)?;
        let parent = if p.parent_id != p.id {
            reader.get(p.parent_id)?
        } else {
            p
        };
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
