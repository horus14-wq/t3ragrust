//! Dense + lexical index implementations and the `t3rag index` pipeline
//! (t3-wkp.6, t3-wkp.7, t3-wkp.8).

pub mod bm25;
mod cache;
pub mod dense;

use std::io::IsTerminal;
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
/// Falls back to `"unknown"` outside Linux/container hosts.
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
    let (iter, kind, stats) = ingest::open(&args.input, args.text_col.as_deref(), args.batch_size)?;
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

    // --update (t3-css.4.1): verify the existing corpus is compatible
    // and build the text-hash -> vector reuse cache. Every store is
    // still rebuilt fresh through the normal pipeline — only embedding
    // is skipped for unchanged texts, which is the dominant cost.
    let mut vec_cache = None;
    let mut cache_hits = 0u64;
    let mut cache_misses = 0u64;
    let mut update_dims = None;
    if args.update {
        let dir = paths::corpus_dir(&args.name)?;
        let meta =
            store::read_meta(&dir).context("--update requires an existing indexed corpus")?;
        let dims = args.dims.unwrap_or(meta.dims as usize);
        if dims as u32 != meta.dims {
            bail!("--update --dims {dims} != corpus dims {}", meta.dims);
        }
        if meta.model != spec.repo_id {
            bail!(
                "--update model {} != corpus model {} — re-index fully",
                spec.repo_id,
                meta.model
            );
        }
        if meta.instruction.as_deref().unwrap_or("") != spec.passage_instruction.as_str() {
            bail!("--update instruction differs from corpus — re-index fully");
        }
        if args.sparse || dir.join("sparse.bin").exists() {
            bail!("--update does not support sparse corpora — re-index fully");
        }
        let built = cache::VectorCache::build(&dir, dims)
            .context("read existing corpus vectors")?
            .context("--update needs vectors.raw.f32 — corpus was built with --no-raw-vectors")?;
        info!(entries = built.entries, "vector cache loaded");
        vec_cache = Some(built);
        update_dims = Some(dims);
    }

    let dir = paths::corpus_dir(&args.name)?;
    let build_dir = dir.with_file_name(format!("{}.build-{}", args.name, now_ts()));
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("create {}", build_dir.display()))?;

    let mut writer = PassageWriter::create(&build_dir)?;
    let mut lex = Bm25Builder::create(&build_dir, args.bm25_heap_mb, args.bm25_threads)?;
    // Matryoshka truncation (t3-css.2.4): --dims stores only the leading N
    // dims (re-normalized); query vectors truncate the same way.
    let store_dims = update_dims.or(args.dims).unwrap_or_else(|| emb.dims());
    let mut dense = DenseBuilder::create(
        &build_dir,
        store_dims,
        args.quantization,
        args.connectivity,
        args.expansion_add,
        !args.no_raw_vectors,
    )?;
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
    // indicatif progress (t3-css.2.8): spinner + rate on tty, hidden
    // otherwise — no progress noise in logs/pipes or with --no-progress.
    let pb = if args.progress && !args.no_progress && std::io::stderr().is_terminal() {
        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(indicatif::ProgressStyle::with_template(
            "{spinner} {elapsed_precise} {pos} passages {msg}",
        )?);
        pb.enable_steady_tick(std::time::Duration::from_millis(120));
        pb
    } else {
        indicatif::ProgressBar::hidden()
    };
    for raw in iter {
        pb.inc(1);
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

        if (args.always_chunk
            || crate::chunk::needs_chunk_thresh(
                &p.text,
                args.chunk_tokens
                    .unwrap_or(crate::chunk::CHUNK_TRIGGER_EST_TOKENS),
            ))
            && !args.no_chunk
        {
            // Structural chunking (t3-wkp.11): store the parent for hydration
            // (no vector, no BM25 doc); index the chunks as retrievable units.
            chunked += 1;
            let parent_id = p.id;
            writer.push(&p)?;
            next_id += 1;
            for (ci, text) in crate::chunk::split_with_target(
                &p.text,
                args.chunk_chars.unwrap_or(crate::chunk::CHUNK_TARGET_CHARS),
            )
            .into_iter()
            .enumerate()
            {
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
                        vec_cache.as_ref(),
                        &mut cache_hits,
                        &mut cache_misses,
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
                vec_cache.as_ref(),
                &mut cache_hits,
                &mut cache_misses,
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
        vec_cache.as_ref(),
        &mut cache_hits,
        &mut cache_misses,
    )?;
    pb.finish_and_clear();
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
            "update": args.update,
            "vector_cache_hits": cache_hits,
            "vector_cache_misses": cache_misses,
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
    if args.update {
        println!(
            "update: {} vectors reused, {} embedded",
            cache_hits, cache_misses
        );
    }
    Ok(())
}

#[cfg(feature = "sparse")]
use crate::sparse::SparseSink;
/// Stub for non-`sparse` builds: `--sparse` bails before this is ever
/// constructed (feature-gated use only).
#[cfg(not(feature = "sparse"))]
struct SparseSink;

/// Embed one drained batch and fan results out to the dense builder,
/// sparse sink, and manifest centroid accumulator. No-op on empty
/// batches. With `vec_cache` (--update, t3-css.4.1) texts whose hash
/// hits the previous corpus's stored vectors skip the embedder.
#[allow(clippy::too_many_arguments)]
fn flush_batch(
    emb: &EmbedderPool,
    dense: &mut DenseBuilder,
    batch: &mut Vec<(u64, String)>,
    store_dims: usize,
    #[allow(unused_variables)] sparse_sink: &mut Option<SparseSink>,
    manifest: &mut crate::manifest::ManifestAcc,
    vec_cache: Option<&cache::VectorCache>,
    hits: &mut u64,
    misses: &mut u64,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    // --update: partition into cache hits (vector reused verbatim —
    // already truncated + unit-normed at original index time) and
    // misses (the only texts that reach the embedder).
    let mut to_embed: Vec<(u64, String)> = Vec::with_capacity(batch.len());
    for (id, text) in batch.drain(..) {
        match vec_cache.and_then(|c| c.get(&text)) {
            Some(v) => {
                *hits += 1;
                manifest.observe_vector(v);
                dense.add(id, v)?;
            }
            None => {
                *misses += 1;
                to_embed.push((id, text));
            }
        }
    }
    if to_embed.is_empty() {
        return Ok(());
    }
    // Borrowed slices (t3-css.2.9) — the batch used to clone every doc's
    // text for the embed call; `&str` references cost nothing.
    let texts: Vec<&str> = to_embed.iter().map(|(_, t)| t.as_str()).collect();
    let vecs = emb.embed_passages(&texts)?;
    #[cfg(feature = "sparse")]
    if let Some(sink) = sparse_sink {
        sink.write_batch(to_embed.iter().map(|(i, t)| (*i, t.as_str())))?;
    }
    for ((id, _), v) in to_embed.drain(..).zip(vecs) {
        let v = crate::search::truncate_unit(&v, store_dims);
        manifest.observe_vector(&v);
        dense.add(id, &v)?;
    }
    Ok(())
}

#[cfg(test)]
mod rewrite_tests {
    /// Surgical `question → null` rewrite for MISSING ITEMS catalog
    /// specs (GH#5 data fix — the strip pass missed this shape). Keeps
    /// ids/positions/vectors byte-identical; rebuilds only passages,
    /// bm25 and the manifest — no chunking, no embedding, no HNSW.
    /// Run: `CORPUS_DIR=~/.cache/t3rag/corpora/raw-chunked cargo test
    ///   --release rewrite_missing_items -- --ignored --nocapture`
    #[test]
    #[ignore = "one-off data rewrite — run manually with CORPUS_DIR"]
    fn rewrite_missing_items_questions() {
        use std::path::PathBuf;
        let dir = PathBuf::from(std::env::var("CORPUS_DIR").expect("CORPUS_DIR"));
        let stage =
            dir.with_file_name(format!("{}.rw", dir.file_name().unwrap().to_string_lossy()));
        let _ = std::fs::remove_dir_all(&stage);
        let rd = crate::store::PassageReader::open(&dir).unwrap();
        let mut wr = crate::store::PassageWriter::create(&stage).unwrap();
        let mut bm = crate::index::Bm25Builder::create(&stage, 256, 0).unwrap();
        let mut acc = crate::manifest::ManifestAcc::new(1024);
        let mut nulled = 0u64;
        for id in 0..rd.len() {
            let mut p = rd.get(id).unwrap();
            if p.question
                .as_deref()
                .is_some_and(|q| q.starts_with("MISSING ITEMS"))
            {
                p.question = None;
                nulled += 1;
            }
            wr.push(&p).unwrap();
            bm.add(&p).unwrap();
            acc.observe(&p);
        }
        wr.finish().unwrap();
        bm.commit().unwrap();
        // Centroid + vector stats come from the untouched raw vectors.
        let raw = std::fs::read(dir.join("vectors.raw.f32")).unwrap();
        for v in raw.chunks_exact(4 * 1024) {
            let f: Vec<f32> = v
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            acc.observe_vector(&f);
        }
        let mut meta = crate::store::read_meta(&dir).unwrap();
        meta.manifest = Some(acc.finish());
        crate::store::write_meta(&stage, &meta).unwrap();
        for f in ["vectors.usearch", "vectors.raw.f32"] {
            std::fs::copy(dir.join(f), stage.join(f)).unwrap();
        }
        eprintln!(
            "rewrote {nulled}/{} questions → {}",
            rd.len(),
            stage.display()
        );
    }

    /// Dedupe by exact text-hash (GH#5 follow-up: a 303× "senior red
    /// teamer" cluster slipped past the upstream dedup — ~11 copies of
    /// each of 40 bodies). Drops later occurrences, renumbers ids, and
    /// remaps positional vectors (raw.f32 slices) into fresh
    /// usearch/raw files — no re-embedding. Run:
    ///   `CORPUS_DIR=~/.cache/t3rag/corpora/struct cargo test --release
    ///     dedupe_corpus -- --ignored --nocapture`
    #[test]
    #[ignore = "one-off data dedup — run manually with CORPUS_DIR"]
    fn dedupe_corpus() {
        use std::path::PathBuf;
        let dir = PathBuf::from(std::env::var("CORPUS_DIR").expect("CORPUS_DIR"));
        let stage =
            dir.with_file_name(format!("{}.rw", dir.file_name().unwrap().to_string_lossy()));
        let _ = std::fs::remove_dir_all(&stage);
        let rd = crate::store::PassageReader::open(&dir).unwrap();
        let raw = std::fs::read(dir.join("vectors.raw.f32")).unwrap();
        let mut wr = crate::store::PassageWriter::create(&stage).unwrap();
        let mut bm = crate::index::Bm25Builder::create(&stage, 256, 0).unwrap();
        let mut dense = crate::index::DenseBuilder::create(
            &stage,
            1024,
            crate::cli::Quantization::I8,
            16,
            128,
            true,
        )
        .unwrap();
        let mut acc = crate::manifest::ManifestAcc::new(1024);
        let mut seen = std::collections::HashMap::new();
        let mut id_map = std::collections::HashMap::new();
        let (mut kept, mut dropped) = (0u64, 0u64);
        for id in 0..rd.len() {
            let mut p = rd.get(id).unwrap();
            let h = crate::ingest::hash_text(&p.text);
            if let Some(&twin) = seen.get(&h) {
                id_map.insert(id, twin);
                dropped += 1;
                continue;
            }
            let v: Vec<f32> = raw[id as usize * 4096..(id as usize + 1) * 4096]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            seen.insert(h, kept);
            id_map.insert(id, kept);
            // Chunk records hydrate through parent_id — remap through
            // the id translation (parents precede their chunks; a
            // dropped parent's children adopt the kept twin).
            if p.parent_id != id {
                p.parent_id = *id_map.get(&p.parent_id).unwrap_or(&kept);
            }
            p.id = kept;
            dense.add(kept, &v).unwrap();
            wr.push(&p).unwrap();
            bm.add(&p).unwrap();
            acc.observe(&p);
            acc.observe_vector(&v);
            kept += 1;
        }
        wr.finish().unwrap();
        bm.commit().unwrap();
        dense.finish().unwrap();
        let mut meta = crate::store::read_meta(&dir).unwrap();
        meta.doc_count = kept;
        meta.manifest = Some(acc.finish());
        crate::store::write_meta(&stage, &meta).unwrap();
        eprintln!(
            "dedupe {}: kept {kept}, dropped {dropped} → {}",
            dir.display(),
            stage.display()
        );
    }
}

/// Unix-seconds timestamp for build-dir rotation suffixes.
fn now_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
