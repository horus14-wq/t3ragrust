//! `t3rag bench` (t3-css.2.2): stage-latency harness — runs N warm queries
//! through the daemon's `explain` instrumentation and reports p50/p95/p99
//! per stage (embed/dense/bm25/rerank/hydrate) plus end-to-end. `--cold`
//! measures the one-time daemon spawn+model-warmup cost separately —
//! warm latency is the product metric (90% of agent interactions).

use anyhow::{Context, Result};

use crate::cli::BenchArgs;

/// `t3rag bench`: p50/p95/p99 + mean per stage over `--iters`
/// queries through the daemon (`--explain` instrumentation), plus
/// `--cold` spawn-time measurement (t3-css.2.2).
///
/// # Errors
/// No daemon reachable, or a query fails.
pub fn run_bench(args: BenchArgs) -> Result<()> {
    let base = crate::serve::ensure_daemon()?;
    let corpus = match args.corpus.first() {
        Some(c) => c.clone(),
        None => crate::serve::newest_corpus(&crate::paths::corpora_root()?)?,
    };
    let n = args.iterations.max(1);

    // Cold-start cost: fresh daemon spawn -> first query answered.
    let cold_ms = if args.cold {
        let t = std::time::Instant::now();
        one_query(&base, &corpus, &args.query, &args)?;
        Some(t.elapsed().as_secs_f64() * 1000.0)
    } else {
        // Warmup outside measurement either way (first query may lazy-load
        // the corpus + reranker).
        one_query(&base, &corpus, &args.query, &args)?;
        None
    };

    let mut embed = Vec::with_capacity(n);
    let mut dense = Vec::with_capacity(n);
    let mut bm25 = Vec::with_capacity(n);
    let mut rerank = Vec::with_capacity(n);
    let mut hydrate = Vec::with_capacity(n);
    let mut total = Vec::with_capacity(n);
    for _ in 0..n {
        let (stages, tot_ms) = one_query(&base, &corpus, &args.query, &args)?;
        embed.push(stages["embed_ms"].as_f64().unwrap_or(0.0));
        dense.push(stages["dense_ms"].as_f64().unwrap_or(0.0));
        bm25.push(stages["bm25_ms"].as_f64().unwrap_or(0.0));
        rerank.push(stages["rerank_ms"].as_f64().unwrap_or(0.0));
        hydrate.push(stages["hydrate_ms"].as_f64().unwrap_or(0.0));
        total.push(tot_ms);
    }

    let report = |name: &str, v: &mut Vec<f64>| -> serde_json::Value {
        serde_json::json!({
            "stage": name,
            "p50": pct(v, 0.50),
            "p95": pct(v, 0.95),
            "p99": pct(v, 0.99),
            "mean": v.iter().sum::<f64>() / v.len() as f64,
        })
    };
    let out = serde_json::json!({
        "corpus": corpus,
        "iterations": n,
        "cold_start_ms": cold_ms,
        "stages": [
            report("embed", &mut embed),
            report("dense", &mut dense),
            report("bm25", &mut bm25),
            report("rerank", &mut rerank),
            report("hydrate", &mut hydrate),
            report("total_e2e", &mut total),
        ],
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("corpus={corpus} n={n} cold={:?}ms", cold_ms);
        for row in out["stages"].as_array().unwrap() {
            println!(
                "{:<12} p50={:>7.1}ms p95={:>7.1}ms p99={:>7.1}ms mean={:>7.1}ms",
                row["stage"].as_str().unwrap(),
                row["p50"].as_f64().unwrap(),
                row["p95"].as_f64().unwrap(),
                row["p99"].as_f64().unwrap(),
                row["mean"].as_f64().unwrap(),
            );
        }
    }
    Ok(())
}

/// One explained query -> (stages_ms object, wall e2e ms).
fn one_query(
    base: &str,
    corpus: &str,
    query: &str,
    args: &BenchArgs,
) -> Result<(serde_json::Value, f64)> {
    let t = std::time::Instant::now();
    let mut resp = ureq::post(&format!("{base}/search"))
        .send_json(serde_json::json!({
            "query": query,
            "client": "cli",
            "corpus": corpus,
            "top_k": args.top_k,
            "candidates": args.candidates,
            "rerank": args.rerank,
            "exact": args.exact,
            "explain": true,
        }))
        .context("daemon POST /search")?;
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    if resp.status().as_u16() != 200 {
        let body: serde_json::Value = resp.body_mut().read_json().unwrap_or_default();
        anyhow::bail!("daemon query failed: {body}");
    }
    let v: serde_json::Value = resp
        .body_mut()
        .read_json()
        .context("parse daemon response")?;
    let stages = v["stages_ms"].clone();
    if stages.is_null() {
        anyhow::bail!("daemon response missing stages_ms (explain off?)");
    }
    Ok((stages, wall_ms))
}

/// Nearest-rank percentile on an unsorted sample.
fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}
