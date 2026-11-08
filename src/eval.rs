//! `t3rag eval` (t3-css.2.3): recall@k / MRR / nDCG harness over
//! `queries.jsonl`-format files — `{query, relevant_ids[]}` lines where
//! ids are passage ids in the target corpus. Runs every query through the
//! daemon (auto-spawned, same path agents use) and reports per-query hits
//! plus aggregate metrics. `--exact` doubles as the ANN-vs-brute-force
//! recall audit; multiple `--corpus` values produce a side-by-side table
//! (the whole-vs-chunked A/B from DESIGN §9).

use std::collections::HashSet;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::cli::EvalArgs;

#[derive(Deserialize)]
struct EvalQuery {
    query: String,
    #[serde(default)]
    relevant_ids: Vec<u64>,
}

/// One corpus's aggregated metrics over all queries.
struct Metrics {
    corpus: String,
    n: usize,
    recall_at: [f64; 4], // k = 1, 3, 5, 10
    mrr: f64,
    ndcg10: f64,
}

pub fn run_eval(args: EvalArgs) -> Result<()> {
    let queries = read_queries(&args.queries)?;
    if queries.is_empty() {
        anyhow::bail!("no queries in {}", args.queries.display());
    }
    let base = crate::serve::ensure_daemon()?;
    let corpora: Vec<String> = if args.corpus.is_empty() {
        vec![crate::serve::newest_corpus(&crate::paths::corpora_root()?)?]
    } else {
        args.corpus.clone()
    };

    let mut all = Vec::new();
    for corpus in &corpora {
        all.push(eval_corpus(&base, corpus, &queries, &args)?);
    }
    if args.json {
        let out: Vec<serde_json::Value> = all
            .iter()
            .map(|m| {
                serde_json::json!({
                    "corpus": m.corpus,
                    "queries": m.n,
                    "recall@1": m.recall_at[0],
                    "recall@3": m.recall_at[1],
                    "recall@5": m.recall_at[2],
                    "recall@10": m.recall_at[3],
                    "mrr": m.mrr,
                    "ndcg@10": m.ndcg10,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "{:<16} {:>6} {:>8} {:>8} {:>8} {:>9} {:>7} {:>8}",
            "corpus", "n", "r@1", "r@3", "r@5", "r@10", "mrr", "ndcg10"
        );
        for m in &all {
            println!(
                "{:<16} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>9.3} {:>7.3} {:>8.3}",
                m.corpus,
                m.n,
                m.recall_at[0],
                m.recall_at[1],
                m.recall_at[2],
                m.recall_at[3],
                m.mrr,
                m.ndcg10
            );
        }
    }
    Ok(())
}

fn read_queries(path: &std::path::Path) -> Result<Vec<EvalQuery>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let rd = std::io::BufReader::new(f);
    use std::io::BufRead;
    rd.lines()
        .enumerate()
        .map(|(i, l)| {
            let l = l?;
            serde_json::from_str(&l).with_context(|| format!("{} line {}", path.display(), i + 1))
        })
        .collect()
}

fn eval_corpus(
    base: &str,
    corpus: &str,
    queries: &[EvalQuery],
    args: &EvalArgs,
) -> Result<Metrics> {
    let ks = [1usize, 3, 5, 10];
    let mut recall = [0f64; 4];
    let mut mrr_sum = 0f64;
    let mut ndcg_sum = 0f64;
    let mut n = 0usize;
    for q in queries {
        let ids = search_ids(base, corpus, &q.query, args)?;
        let rel: HashSet<u64> = q.relevant_ids.iter().cloned().collect();
        let first_hit = ids.iter().position(|id| rel.contains(id));
        for (i, k) in ks.iter().enumerate() {
            if first_hit.is_some_and(|pos| pos < *k) {
                recall[i] += 1.0;
            }
        }
        mrr_sum += first_hit.map_or(0.0, |pos| 1.0 / (pos + 1) as f64);
        // Binary-relevance nDCG@10.
        let dcg: f64 = ids
            .iter()
            .take(10)
            .enumerate()
            .map(|(i, id)| {
                if rel.contains(id) {
                    1.0 / (i as f64 + 2.0).log2()
                } else {
                    0.0
                }
            })
            .sum();
        let idcg: f64 = (0..rel.len().min(10))
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum();
        ndcg_sum += if idcg > 0.0 { dcg / idcg } else { 0.0 };
        n += 1;
    }
    let d = n.max(1) as f64;
    Ok(Metrics {
        corpus: corpus.to_string(),
        n,
        recall_at: [recall[0] / d, recall[1] / d, recall[2] / d, recall[3] / d],
        mrr: mrr_sum / d,
        ndcg10: ndcg_sum / d,
    })
}

/// Query the daemon; return ranked passage ids (top-10 for nDCG@10).
fn search_ids(base: &str, corpus: &str, query: &str, args: &EvalArgs) -> Result<Vec<u64>> {
    let mut resp = ureq::post(&format!("{base}/search"))
        .send_json(serde_json::json!({
            "query": query,
            "corpus": corpus,
            "top_k": 10,
            "rerank": args.rerank,
            "exact": args.exact,
            "preset": if args.fast { Some("fast") } else { None },
        }))
        .context("daemon POST /search")?;
    if resp.status().as_u16() != 200 {
        let body: serde_json::Value = resp.body_mut().read_json().unwrap_or_default();
        anyhow::bail!("daemon query failed: {body}");
    }
    let v: serde_json::Value = resp
        .body_mut()
        .read_json()
        .context("parse daemon response")?;
    Ok(v["results"]
        .as_array()
        .map(|rs| rs.iter().filter_map(|r| r["passage_id"].as_u64()).collect())
        .unwrap_or_default())
}
