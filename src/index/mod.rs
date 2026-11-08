//! Dense + lexical index implementations and the `t3rag index` pipeline
//! (t3-wkp.6, t3-wkp.7, t3-wkp.8).

pub mod bm25;
pub mod dense;

use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracing::info;

use crate::cli::IndexArgs;
use crate::embed::{self, EmbedderPool};
use crate::ingest;
use crate::paths;
use crate::store::{self, CorpusMeta, PassageWriter};

pub use bm25::{Bm25Builder, Bm25Index};
pub use dense::{DenseBuilder, DenseIndex};

/// Max passages per embedding batch.
const EMBED_BATCH: usize = 16;
/// Token budget per forward pass. The merged-KV ONNX export tiles a
/// batch x seq^2 attention mask — unbounded batches exceed ORT's 4GB
/// tensor limit when a long passage is present.
const EMBED_TOKEN_BUDGET: usize = 4096;

/// `t3rag index` — ingest -> pack -> embed -> usearch+tantivy -> meta.json.
///
/// Builds into `<name>.build-<ts>` then renames into place; a previous corpus
/// is rotated to `<name>.replaced-<ts>` (nothing is deleted).
pub fn run_index(args: IndexArgs) -> Result<()> {
    let t0 = Instant::now();
    let (iter, kind, stats) = ingest::open(&args.input, args.text_col.as_deref())?;
    info!(?kind, input = %args.input.display(), "ingest opened");

    let mut spec = embed::resolve_model(args.model.as_deref());
    if let Some(i) = &args.instruction {
        spec.passage_instruction = i.clone();
    }
    info!(model = %spec.repo_id, "loading embedder (first run downloads ~1-2GB)");
    let emb = EmbedderPool::load(&spec, args.threads, args.ep, args.pool)?;
    info!(dims = emb.dims(), "embedder ready");

    let dir = paths::corpus_dir(&args.name)?;
    let build_dir = dir.with_file_name(format!("{}.build-{}", args.name, now_ts()));
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("create {}", build_dir.display()))?;

    let mut writer = PassageWriter::create(&build_dir)?;
    let mut lex = Bm25Builder::create(&build_dir)?;
    let mut dense = DenseBuilder::create(&build_dir, emb.dims(), args.quantization)?;

    let mut next_id = 0u64;
    let mut batch: Vec<(u64, String)> = Vec::with_capacity(EMBED_BATCH);
    let mut batch_tokens = 0usize;
    let mut chunked = 0u64;
    for raw in iter {
        let raw = raw?;
        let Some(mut p) = ingest::to_passage(raw, next_id, &stats) else {
            continue;
        };

        if crate::chunk::needs_chunk(&p.text) {
            // Structural chunking (t3-wkp.11): store the parent for hydration
            // (no vector, no BM25 doc); index the chunks as retrievable units.
            chunked += 1;
            let parent_id = p.id;
            writer.push(&p)?;
            next_id += 1;
            for (ci, text) in crate::chunk::split(&p.text).into_iter().enumerate() {
                let mut c = crate::passage::Passage::new(next_id, text);
                c.title = p.title.clone();
                c.question = p.question.clone();
                c.source = p.source.clone();
                c.parent_id = parent_id;
                c.chunk_idx = ci as u32;
                next_id += 1;
                p = c;
                writer.push(&p)?;
                lex.add(&p)?;
                let est = crate::chunk::est_tokens(&p.text).max(1);
                if !batch.is_empty()
                    && (batch.len() == EMBED_BATCH || batch_tokens + est > EMBED_TOKEN_BUDGET)
                {
                    flush_batch(&emb, &mut dense, &mut batch)?;
                    batch_tokens = 0;
                }
                batch_tokens += est;
                batch.push((p.id, p.text));
            }
            continue;
        }

        next_id += 1;
        writer.push(&p)?;
        lex.add(&p)?;
        // Cheap token estimate (~3 chars/token) for the budget check.
        let est = (p.text.len() / 3).max(1);
        if !batch.is_empty()
            && (batch.len() == EMBED_BATCH || batch_tokens + est > EMBED_TOKEN_BUDGET)
        {
            flush_batch(&emb, &mut dense, &mut batch)?;
            batch_tokens = 0;
            if next_id % 1024 < EMBED_BATCH as u64 {
                info!(docs = next_id, stats = %stats, "indexing");
            }
        }
        batch_tokens += est;
        batch.push((p.id, p.text));
    }
    flush_batch(&emb, &mut dense, &mut batch)?;
    if chunked > 0 {
        info!(chunked_parents = chunked, "structural chunking applied");
    }

    let n = writer.finish()?;
    let nd = dense.finish()?;
    lex.commit()?;
    // Parents are stored for hydration but never embedded — dense count is
    // store count minus un-embedded parents.
    if nd as u64 != n - chunked {
        bail!("dense index {nd} != embedded passages {}", n - chunked);
    }

    let meta = CorpusMeta {
        format_version: store::FORMAT_VERSION,
        model: spec.repo_id.clone(),
        dims: emb.dims() as u32,
        quantization: format!("{:?}", args.quantization).to_lowercase(),
        doc_count: n,
    };
    store::write_meta(&build_dir, &meta)?;

    // Atomic-ish install: rotate any existing corpus aside, move build in.
    if dir.exists() {
        let replaced = dir.with_file_name(format!("{}.replaced-{}", args.name, now_ts()));
        std::fs::rename(&dir, &replaced)
            .with_context(|| format!("rotate {} -> {}", dir.display(), replaced.display()))?;
    }
    std::fs::rename(&build_dir, &dir)
        .with_context(|| format!("install {} -> {}", build_dir.display(), dir.display()))?;

    info!(
        docs = n,
        %stats,
        elapsed = ?t0.elapsed(),
        dir = %dir.display(),
        "index complete"
    );
    println!(
        "indexed {} passages into {} ({:.1?})",
        n,
        dir.display(),
        t0.elapsed()
    );
    Ok(())
}

fn flush_batch(
    emb: &EmbedderPool,
    dense: &mut DenseBuilder,
    batch: &mut Vec<(u64, String)>,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    // Borrowed slices (t3-css.2.9) — the batch used to clone every doc's
    // text for the embed call; `&str` references cost nothing.
    let texts: Vec<&str> = batch.iter().map(|(_, t)| t.as_str()).collect();
    let vecs = emb.embed_passages(&texts)?;
    for ((id, _), v) in batch.drain(..).zip(vecs) {
        dense.add(id, &v)?;
    }
    Ok(())
}

fn now_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
