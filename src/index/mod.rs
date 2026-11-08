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

// Batch knobs live on `IndexArgs` (t3-css.2.8): --embed-batch (16) and
// --embed-token-budget (4096 — merged-KV exports tile batch x seq^2
// attention masks; unbounded batches exceed ORT's 4GB tensor limit).

/// Hostname for provenance records (`meta.json.origin`).
fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// `t3rag index` — ingest -> pack -> embed -> usearch+tantivy -> meta.json.
///
/// Builds into `<name>.build-<ts>` then renames into place; a previous corpus
/// is rotated to `<name>.replaced-<ts>` (nothing is deleted).
///
/// # Errors
/// The input cannot be opened, the embedder fails to load/download, or
/// index construction fails partway (the build dir is left in place for
/// inspection — it is not installed).
pub fn run_index(args: IndexArgs) -> Result<()> {
    let t0 = Instant::now();
    let (iter, kind, stats) = ingest::open(&args.input, args.text_col.as_deref())?;
    info!(?kind, input = %args.input.display(), "ingest opened");

    // Dry run (t3-css.2.8): ingest+count only — placed BEFORE the embedder
    // load so a dry run never downloads a model.
    if args.dry_run {
        let mut n = 0u64;
        for raw in iter {
            let _ = raw?;
            n += 1;
        }
        info!(docs = n, stats = %stats, "dry run complete");
        println!("dry run: {n} passages would index ({stats})");
        return Ok(());
    }

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
    // Matryoshka truncation (t3-css.2.4): --dims stores only the leading N
    // dims (re-normalized); query vectors truncate the same way.
    let store_dims = args.dims.unwrap_or_else(|| emb.dims());
    let mut dense = DenseBuilder::create(&build_dir, store_dims, args.quantization)?;
    // Aboutness manifest (t3-iv2): accumulates transform type, question
    // terms, length histogram, language, and the passage-vector centroid
    // for meta.json — corpus routing for `suggest`/`list`.
    let mut manifest = crate::manifest::ManifestAcc::new(store_dims);

    // Sparse leg (t3-css.4.3): SPLADE alongside dense when --sparse.
    let mut sparse_sink: Option<SparseSink> = None;
    #[cfg(feature = "sparse")]
    if args.sparse {
        sparse_sink = Some(SparseSink::new(&build_dir)?);
    }
    #[cfg(not(feature = "sparse"))]
    if args.sparse {
        bail!("--sparse requires building with --features sparse");
    }

    let mut next_id = 0u64;
    let mut batch: Vec<(u64, String)> = Vec::with_capacity(args.embed_batch);
    let mut batch_tokens = 0usize;
    let mut chunked = 0u64;
    let mut seen_hashes: rustc_hash::FxHashSet<u64> = Default::default();
    for raw in iter {
        let raw = raw?;
        let Some(mut p) = ingest::to_passage(raw, next_id, &stats) else {
            continue;
        };

        // Ingest filters (t3-css.2.8): char bounds + exact-hash dedup.
        if let Some(mn) = args.min_chars {
            if p.text.len() < mn {
                stats.skip("too_short");
                continue;
            }
        }
        if let Some(mx) = args.max_chars {
            if p.text.len() > mx {
                stats.skip("too_long");
                continue;
            }
        }
        if args.dedup && !seen_hashes.insert(crate::ingest::hash_text(&p.text)) {
            stats.skip("duplicate");
            continue;
        }

        if crate::chunk::needs_chunk(&p.text) && !args.no_chunk {
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
                manifest.observe(&p);
                let est = crate::chunk::est_tokens(&p.text).max(1);
                if !batch.is_empty()
                    && (batch.len() == args.embed_batch
                        || batch_tokens + est > args.embed_token_budget)
                {
                    flush_batch(
                        &emb,
                        &mut dense,
                        &mut batch,
                        store_dims,
                        &mut sparse_sink,
                        &mut manifest,
                    )?;
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
        manifest.observe(&p);
        // Cheap token estimate (~3 chars/token) for the budget check.
        let est = (p.text.len() / 3).max(1);
        if !batch.is_empty()
            && (batch.len() == args.embed_batch || batch_tokens + est > args.embed_token_budget)
        {
            flush_batch(
                &emb,
                &mut dense,
                &mut batch,
                store_dims,
                &mut sparse_sink,
                &mut manifest,
            )?;
            batch_tokens = 0;
            if next_id % 1024 < args.embed_batch as u64 {
                info!(docs = next_id, stats = %stats, "indexing");
            }
        }
        batch_tokens += est;
        batch.push((p.id, p.text));
    }
    flush_batch(
        &emb,
        &mut dense,
        &mut batch,
        store_dims,
        &mut sparse_sink,
        &mut manifest,
    )?;
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
        dims: store_dims as u32,
        quantization: format!("{:?}", args.quantization).to_lowercase(),
        doc_count: n,
        // Provenance (t3-css.4.7): created_at + host:user:input origin +
        // the index-time flags that produced this build.
        created_at: Some(now_ts().to_string()),
        origin: Some(format!(
            "{}:{}:{}",
            hostname(),
            std::env::var("USER").unwrap_or_default(),
            args.input.display()
        )),
        instruction: Some(spec.passage_instruction.clone()),
        manifest: Some(manifest.finish()),
        index_params: Some(serde_json::json!({
            "embed_batch": args.embed_batch,
            "embed_token_budget": args.embed_token_budget,
            "dims": args.dims,
            "dedup": args.dedup,
            "no_chunk": args.no_chunk,
            "chunked_parents": chunked,
        })),
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

#[cfg(feature = "sparse")]
use crate::sparse::SparseSink;
#[cfg(not(feature = "sparse"))]
struct SparseSink; // never instantiated (feature-gated use only)

fn flush_batch(
    emb: &EmbedderPool,
    dense: &mut DenseBuilder,
    batch: &mut Vec<(u64, String)>,
    store_dims: usize,
    #[allow(unused_variables)] sparse_sink: &mut Option<SparseSink>,
    manifest: &mut crate::manifest::ManifestAcc,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    // Borrowed slices (t3-css.2.9) — the batch used to clone every doc's
    // text for the embed call; `&str` references cost nothing.
    let texts: Vec<&str> = batch.iter().map(|(_, t)| t.as_str()).collect();
    let vecs = emb.embed_passages(&texts)?;
    #[cfg(feature = "sparse")]
    if let Some(sink) = sparse_sink {
        sink.write_batch(batch.iter().map(|(i, t)| (*i, t.as_str())))?;
    }
    for ((id, _), v) in batch.drain(..).zip(vecs) {
        let v = crate::search::truncate_unit(&v, store_dims);
        manifest.observe_vector(&v);
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
