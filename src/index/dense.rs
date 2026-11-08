//! Dense ANN index: `usearch` HNSW, single file, i8 default, mmap reads.
//!
//! Two artifacts are written per corpus:
//! - `vectors.usearch` — the HNSW graph (quantized)
//! - `vectors.raw.f32` — `[u64 n][u64 dims]` header then n records of
//!   `[u64 passage_id][f32 x dims]`: brute-force exact-scan reference
//!   (t3-wkp.6 recall checks + `--exact` mode) and lets us rebuild the ANN
//!   with different params/quantization without re-embedding. Ids are stored
//!   explicitly because chunking embeds a subset of stored passages.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::cli::Quantization;

/// HNSW graph artifact filename inside a corpus dir.
pub const DENSE_FILE: &str = "vectors.usearch";
/// Raw fp32 vector blob filename (exact-scan reference + rebuild source).
pub const RAW_FILE: &str = "vectors.raw.f32";

/// HNSW params (tuneable via IndexOptions fields if needed later).
const CONNECTIVITY: usize = 16;
const EXPANSION_ADD: usize = 128;
const EXPANSION_SEARCH: usize = 64;

/// Map a [`Quantization`] choice onto usearch's scalar kind.
fn scalar_kind(q: Quantization) -> ScalarKind {
    match q {
        Quantization::I8 => ScalarKind::I8,
        Quantization::Bf16 => ScalarKind::BF16,
        Quantization::F16 => ScalarKind::F16,
        Quantization::Fp32 => ScalarKind::F32,
    }
}

/// usearch options for cosine-IP search: vectors are L2-normalized, so
/// inner product == cosine; `multi: false` (one label per key).
fn options(
    dims: usize,
    q: Quantization,
    connectivity: usize,
    expansion_add: usize,
) -> IndexOptions {
    IndexOptions {
        dimensions: dims,
        metric: MetricKind::IP,
        quantization: scalar_kind(q),
        connectivity,
        expansion_add,
        expansion_search: EXPANSION_SEARCH,
        multi: false,
    }
}

/// Build-time writer: ANN adds + raw-vector blob (optional: t3-css.2.8).
pub struct DenseBuilder {
    index: Index,
    raw: Option<BufWriter<File>>,
    raw_path: PathBuf,
    index_path: PathBuf,
    count: usize,
    capacity: usize,
    dims: usize,
}

impl DenseBuilder {
    /// `write_raw=false` (t3-css.2.8 `--no-raw-vectors`): skip the raw
    /// vector blob — halves index disk; exact-scan search then fails
    /// honestly ("built with --no-raw-vectors").
    pub fn create(
        dir: &Path,
        dims: usize,
        quant: Quantization,
        connectivity: usize,
        expansion_add: usize,
        write_raw: bool,
    ) -> Result<Self> {
        let index = Index::new(&options(dims, quant, connectivity, expansion_add))
            .map_err(|e| anyhow::anyhow!("create usearch index: {e}"))?;
        let capacity = 1024usize;
        index
            .reserve(capacity)
            .map_err(|e| anyhow::anyhow!("usearch reserve: {e}"))?;
        let raw_path = dir.join(RAW_FILE);
        let raw = write_raw
            .then(|| {
                let mut r = BufWriter::new(
                    File::create(&raw_path)
                        .with_context(|| format!("create {}", raw_path.display()))?,
                );
                r.write_all(&0u64.to_le_bytes())?; // n placeholder
                r.write_all(&(dims as u64).to_le_bytes())?;
                anyhow::Ok(r)
            })
            .transpose()?;
        // Each record: [u64 passage_id][dims f32] — ids explicit since
        // chunking may skip embedding stored (parent) passages.
        Ok(Self {
            index,
            raw,
            raw_path,
            index_path: dir.join(DENSE_FILE),
            count: 0,
            capacity,
            dims,
        })
    }

    /// Add one `(passage_id, vector)` pair to the ANN graph and (when
    /// enabled) the raw blob. Grows the index reserve geometrically.
    ///
    /// # Errors
    /// `vector` length != index dims, or a usearch/file write fails.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dims {
            bail!("vector dims {} != index dims {}", vector.len(), self.dims);
        }
        if self.count == self.capacity {
            self.capacity *= 2;
            self.index
                .reserve(self.capacity)
                .map_err(|e| anyhow::anyhow!("usearch grow: {e}"))?;
        }
        self.index
            .add(id, vector)
            .map_err(|e| anyhow::anyhow!("usearch add: {e}"))?;
        if let Some(raw) = &mut self.raw {
            raw.write_all(&id.to_le_bytes())?;
            for v in vector {
                raw.write_all(&v.to_le_bytes())?;
            }
        }
        self.count += 1;
        Ok(())
    }

    /// Persist index + patch raw header + fsync.
    pub fn finish(self) -> Result<usize> {
        self.index
            .save(self.index_path.to_str().context("non-utf8 index path")?)
            .map_err(|e| anyhow::anyhow!("usearch save {}: {e}", self.index_path.display()))?;
        if let Some(mut raw) = self.raw {
            raw.flush()?;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&self.raw_path)?;
            f.seek(SeekFrom::Start(0))?;
            f.write_all(&(self.count as u64).to_le_bytes())?;
            f.sync_all()?;
        }
        Ok(self.count)
    }
}

/// Query-time view: mmap'd HNSW + mmap'd raw blob for exact scans
/// (`None` when indexed with `--no-raw-vectors`, t3-css.2.8).
pub struct DenseIndex {
    index: Index,
    raw: Option<Mmap>,
    dims: usize,
}

impl DenseIndex {
    /// Open a query-time view over `dir`'s dense artifacts: mmap the
    /// HNSW graph plus the raw blob when present (`--no-raw-vectors`
    /// corpora still serve ANN search).
    ///
    /// # Errors
    /// The index file is missing/unreadable, or `dims`/`quant` don't
    /// match the build.
    pub fn view(dir: &Path, dims: usize, quant: Quantization) -> Result<Self> {
        let index_path = dir.join(DENSE_FILE);
        let index = Index::new(&options(dims, quant, CONNECTIVITY, EXPANSION_ADD))
            .map_err(|e| anyhow::anyhow!("create usearch handle: {e}"))?;
        index
            .view(index_path.to_str().context("non-utf8 index path")?)
            .map_err(|e| anyhow::anyhow!("usearch view {}: {e}", index_path.display()))?;
        // --no-raw-vectors corpora (t3-css.2.8): raw blob may not exist —
        // ANN search still works; exact_search reports it honestly.
        let raw_path = dir.join(RAW_FILE);
        let raw = if raw_path.exists() {
            let raw_file =
                File::open(&raw_path).with_context(|| format!("open {}", raw_path.display()))?;
            // Safety: read-only map of an immutable artifact.
            Some(
                unsafe { Mmap::map(&raw_file) }
                    .with_context(|| format!("mmap {}", raw_path.display()))?,
            )
        } else {
            None
        };
        Ok(Self { index, raw, dims })
    }

    /// ANN search; returns (passage_id, cosine_score) sorted best-first.
    /// usearch IP distance is `1 - dot`; for unit vectors that is `1 - cosine`.
    /// `ef` overrides the index's expansion_search (recall↔latency dial,
    /// t3-css.2.5) — usearch allows changing it on viewed indexes.
    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Result<Vec<(u64, f32)>> {
        if let Some(ef) = ef {
            self.index.change_expansion_search(ef);
        }
        let m = self
            .index
            .search(query, k)
            .map_err(|e| anyhow::anyhow!("usearch search: {e}"))?;
        Ok(m.keys
            .into_iter()
            .zip(m.distances)
            .map(|(k, d)| (k, 1.0 - d))
            .collect())
    }

    /// Exact brute-force scan over the raw fp32 blob (dot product == cosine
    /// on unit vectors). ~ms on this corpus scale; the recall reference.
    pub fn exact_search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if query.len() != self.dims {
            bail!("query dims {} != index dims {}", query.len(), self.dims);
        }
        let Some(raw) = &self.raw else {
            bail!("corpus indexed with --no-raw-vectors — exact scan unavailable");
        };
        let buf = &raw[..];
        let (n, stride, has_id) = raw_layout(buf, self.dims)?;
        // Parallel row scan (t3-css.1.8/2.9): each row is an independent
        // dims-length dot product — the recall oracle stays fast on 155K-doc
        // corpora.
        use rayon::prelude::*;
        let mut scored: Vec<(u64, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let rec = &buf[16 + i * stride..16 + (i + 1) * stride];
                let (id, row) = if has_id {
                    (u64::from_le_bytes(rec[0..8].try_into().unwrap()), &rec[8..])
                } else {
                    (i as u64, rec)
                };
                let mut dot = 0f32;
                for (j, &q) in query.iter().enumerate() {
                    dot += q * f32::from_le_bytes(row[j * 4..j * 4 + 4].try_into().unwrap());
                }
                (id, dot)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}

/// Parse a `vectors.raw.f32` blob's header + detect its record layout.
/// Returns `(n_records, stride_bytes, has_explicit_id)`.
///
/// Record layout differs by format version: v1 wrote bare f32 rows
/// (id == row index); v2 writes `[u64 id][f32 vec]` (chunked corpora
/// store non-embedded parents). The blob has no version field, but the
/// two sizes can never coincide (they differ by 8·n bytes).
///
/// # Errors
/// The blob is shorter than the 16-byte header, declares dims that
/// don't match `dims`, or its size fits neither layout.
pub(crate) fn raw_layout(buf: &[u8], dims: usize) -> Result<(usize, usize, bool)> {
    if buf.len() < 16 {
        bail!("corrupt {}: too small", RAW_FILE);
    }
    let n = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let rdims = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    if rdims != dims {
        bail!("raw blob dims {rdims} != index dims {dims}");
    }
    let v2_stride = 8 + dims * 4;
    let v1_stride = dims * 4;
    if buf.len() == 16 + n * v2_stride {
        Ok((n, v2_stride, true))
    } else if buf.len() == 16 + n * v1_stride {
        Ok((n, v1_stride, false))
    } else {
        bail!(
            "corrupt {}: size {} fits neither v1 ({}) nor v2 ({}) records",
            RAW_FILE,
            buf.len(),
            16 + n * v1_stride,
            16 + n * v2_stride
        )
    }
}
