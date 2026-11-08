//! Learned sparse leg (t3-css.4.3): SPLADE via fastembed — a third RRF
//! leg that catches rare-term matches dense+BM25 miss. `sparse.bin`
//! stores `[u64 id][u32 len][u32 idx..][f32 val..]` per embedded
//! passage; query scoring is a linear scan (52K scale is fine — a real
//! postings index is a later optimization).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use fastembed::{SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding};

/// Per-passage sparse (SPLADE) vector: `indices` are vocab dims,
/// `values` the learned weights — sorted-merge `dot` against a query.
#[derive(Debug)]
pub struct SparseVec {
    /// Vocabulary dimensions with nonzero weight (ascending).
    pub indices: Vec<u32>,
    /// The learned weight per index (same order as `indices`).
    pub values: Vec<f32>,
}

/// The SPLADE embedder wrapper (`fastembed` SPLADE-PP-v1, ~100MB on
/// first fetch). Engine calls are `&mut` (its tokenizer/ORT are not
/// Sync) — the daemon guards it behind a `Mutex`.
pub struct SparseEngine {
    inner: SparseTextEmbedding,
}

impl SparseEngine {
    /// Initialize the engine; first call downloads the model.
    ///
    /// # Errors
    /// The model cannot be fetched or its session fails to build.
    pub fn load() -> Result<Self> {
        let inner = SparseTextEmbedding::try_new(SparseInitOptions::new(SparseModel::SPLADEPPV1))
            .context("load SPLADE sparse embedder")?;
        Ok(Self { inner })
    }

    /// Embed a batch of texts into [`SparseVec`]s (one per input, in
    /// order).
    ///
    /// # Errors
    /// The tokenization/forward pass fails.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<SparseVec>> {
        let outs: Vec<SparseEmbedding> = self
            .inner
            .embed(texts.to_vec(), None)
            .context("sparse embed")?;
        Ok(outs
            .into_iter()
            .map(|e| SparseVec {
                indices: e.indices.into_iter().map(|i| i as u32).collect(),
                values: e.values,
            })
            .collect())
    }
}

/// Buffered writer for `sparse.bin` (t3-css.4.3 index path): wraps
/// the engine + a file, one `write_batch` per embed flush.
pub struct SparseSink {
    engine: SparseEngine,
    out: std::fs::File,
}

impl SparseSink {
    /// Load the engine + create `<dir>/sparse.bin`.
    ///
    /// # Errors
    /// The model fails to load, or the file cannot be created.
    pub fn new(dir: &Path) -> Result<Self> {
        let engine = SparseEngine::load()?;
        let out = std::fs::File::create(sparse_file(dir))?;
        Ok(Self { engine, out })
    }

    /// Embed `batch` texts + append their sparse records.
    pub fn write_batch<'a>(&mut self, batch: impl Iterator<Item = (u64, &'a str)>) -> Result<()> {
        use std::io::Write;
        let rows: Vec<(u64, &str)> = batch.collect();
        let texts: Vec<&str> = rows.iter().map(|(_, t)| *t).collect();
        let vecs = self.engine.embed(&texts)?;
        let mut buf = Vec::new();
        for ((id, _), v) in rows.iter().zip(vecs.iter()) {
            append_sparse(&mut buf, *id, v);
        }
        self.out.write_all(&buf)?;
        Ok(())
    }
}

/// Append one `[u64 id][u32 len][u32 idx..][f32 val..]` record to
/// `buf` (the `sparse.bin` wire format).
pub fn append_sparse(buf: &mut Vec<u8>, id: u64, v: &SparseVec) {
    buf.extend_from_slice(&id.to_le_bytes());
    buf.extend_from_slice(&(v.indices.len() as u32).to_le_bytes());
    for &i in &v.indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    for &f in &v.values {
        buf.extend_from_slice(&f.to_le_bytes());
    }
}

/// Parse `sparse.bin` into `(id, SparseVec)` records in file order.
///
/// # Errors
/// The file cannot be read. Truncated trailing bytes are silently
/// dropped (matches the read-ahead writer).
pub fn read_sparse(path: &Path) -> Result<Vec<(u64, SparseVec)>> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 12 <= data.len() {
        let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        let n = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()) as usize;
        off += 12;
        let ib = 4 * n;
        let mut indices = Vec::with_capacity(n);
        for i in 0..n {
            indices.push(u32::from_le_bytes(
                data[off + 4 * i..off + 4 * i + 4].try_into().unwrap(),
            ));
        }
        off += ib;
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            values.push(f32::from_le_bytes(
                data[off + 4 * i..off + 4 * i + 4].try_into().unwrap(),
            ));
        }
        off += ib;
        out.push((id, SparseVec { indices, values }));
    }
    Ok(out)
}

/// Score every stored doc against `q` by sorted-merge dot product,
/// sorted by score desc, zeros dropped. O(docs x |q|) — the linear
/// scan is fine at the 52K scale; a postings index is a later opt.
pub fn score(docs: &[(u64, SparseVec)], q: &SparseVec) -> Vec<(u64, f32)> {
    let mut out = Vec::with_capacity(docs.len());
    for (id, d) in docs {
        let s = dot(&d.indices, &d.values, &q.indices, &q.values);
        if s > 0.0 {
            out.push((*id, s));
        }
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn dot(ai: &[u32], av: &[f32], bi: &[u32], bv: &[f32]) -> f32 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut s = 0f32;
    while i < ai.len() && j < bi.len() {
        match ai[i].cmp(&bi[j]) {
            std::cmp::Ordering::Equal => {
                s += av[i] * bv[j];
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    s
}

/// The `sparse.bin` path inside a corpus artifact dir.
pub fn sparse_file(dir: &Path) -> PathBuf {
    dir.join("sparse.bin")
}
