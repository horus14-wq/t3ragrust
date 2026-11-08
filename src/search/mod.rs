//! Query pipeline (t3-wkp.9; rerank .10 / thin-client .13 / presets .16 next).
//!
//! embed_query -> usearch ANN (top ~N) + tantivy BM25 (top ~N) -> RRF fusion
//! -> [rerank] -> hydrate top-k passages from the store. `--no-daemon` is
//! the only mode for now — this IS the cold in-process path the daemon and
//! thin client will share.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tracing::warn;

use crate::cli::SearchArgs;
use crate::embed::{self, EmbedderPool};
use crate::index::{Bm25Index, DenseIndex};
use crate::paths;
use crate::store::{self, PassageReader};

/// RRF constant (standard k=60).
const RRF_K: f32 = 60.0;

pub struct Hit {
    pub passage_id: u64,
    pub score: f32,
    pub text: String,
    pub title: Option<String>,
    pub question: Option<String>,
    pub source: Option<String>,
}

/// Fuse ranked legs by reciprocal rank: score(d) = Σ 1/(K + rank_i(d)).
pub fn rrf_fuse(legs: &[Vec<(u64, f32)>]) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for leg in legs {
        for (rank, (id, _)) in leg.iter().enumerate() {
            *acc.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
    }
    let mut v: Vec<(u64, f32)> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// `t3rag search` entry point — cold in-process path (t3-wkp.9).
pub fn run_search(args: SearchArgs) -> Result<()> {
    let corpus = resolve_corpus(args.corpus.as_deref())?;
    let meta = store::read_meta(&corpus)?;

    let spec = match &args.model {
        Some(m) => embed::resolve_model(Some(m)),
        None => embed::ModelSpec::for_hf_repo(&meta.model),
    };
    let emb = EmbedderPool::load(&spec, args.threads, args.ep, args.pool)?;
    let quant = parse_quant(&meta.quantization)?;
    let dense = DenseIndex::view(&corpus, meta.dims as usize, quant)?;
    let lex = Bm25Index::open(&corpus)?;
    let reader = PassageReader::open(&corpus)?;

    let qvec = emb.embed_query(&args.query)?;
    let ann = if args.exact {
        dense.exact_search(&qvec, args.candidates)?
    } else {
        dense.search(&qvec, args.candidates)?
    };
    let bm = lex
        .search(&args.query, args.candidates)
        .unwrap_or_else(|e| {
            warn!("bm25 leg failed: {e:#}");
            Vec::new()
        });
    let fused = rrf_fuse(&[ann, bm]);

    let mut ranked: Vec<(u64, f32)> = fused;
    if args.rerank {
        // Cross-encoder rerank of the fused candidates (t3-wkp.10): hydrate
        // candidate texts, score (query, passage) pairs, reorder by logit.
        let cand: Vec<(u64, f32)> = ranked.into_iter().take(args.candidates).collect();
        let pool = crate::rerank::RerankerPool::load(
            &crate::rerank::default_spec(),
            args.threads,
            args.ep,
            args.pool,
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

    let mut hits = Vec::new();
    for (id, score) in ranked.into_iter().take(args.top_k) {
        let p = reader.get(id)?;
        hits.push(Hit {
            passage_id: id,
            score,
            text: p.text,
            title: p.title,
            question: p.question,
            source: p.source,
        });
    }

    if args.json {
        #[derive(Serialize)]
        struct JHit<'a> {
            rank: usize,
            passage_id: u64,
            score: f32,
            text: &'a str,
            title: Option<&'a str>,
            question: Option<&'a str>,
            source: Option<&'a str>,
        }
        #[derive(Serialize)]
        struct Out<'a> {
            query: &'a str,
            corpus: &'a str,
            results: Vec<JHit<'a>>,
        }
        let out = Out {
            query: &args.query,
            corpus: args.corpus.as_deref().unwrap_or("<auto>"),
            results: hits
                .iter()
                .enumerate()
                .map(|(i, h)| JHit {
                    rank: i + 1,
                    passage_id: h.passage_id,
                    score: h.score,
                    text: &h.text,
                    title: h.title.as_deref(),
                    question: h.question.as_deref(),
                    source: h.source.as_deref(),
                })
                .collect(),
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        for (i, h) in hits.iter().enumerate() {
            println!("--- [{}] id={} score={:.4}", i + 1, h.passage_id, h.score);
            if let Some(t) = &h.title {
                println!("title: {t}");
            }
            println!("{}", h.text.trim());
        }
    }
    Ok(())
}

fn resolve_corpus(name: Option<&str>) -> Result<std::path::PathBuf> {
    if let Some(n) = name {
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
    use super::rrf_fuse;

    #[test]
    fn rrf_prefers_docs_in_both_legs() {
        let dense = vec![(1u64, 0.9f32), (2, 0.8), (3, 0.7)];
        let bm25 = vec![(2u64, 5.0f32), (4, 4.0)];
        let fused = rrf_fuse(&[dense, bm25]);
        assert_eq!(fused[0].0, 2, "doc in both legs should rank first");
        assert!(fused.iter().any(|(id, _)| *id == 4));
    }
}
