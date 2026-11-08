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

pub const DENSE_FILE: &str = "vectors.usearch";
pub const RAW_FILE: &str = "vectors.raw.f32";

/// HNSW params (tuneable via IndexOptions fields if needed later).
const CONNECTIVITY: usize = 16;
const EXPANSION_ADD: usize = 128;
const EXPANSION_SEARCH: usize = 64;

fn scalar_kind(q: Quantization) -> ScalarKind {
    match q {
        Quantization::I8 => ScalarKind::I8,
        Quantization::Bf16 => ScalarKind::BF16,
        Quantization::Fp32 => ScalarKind::F32,
    }
}

fn options(dims: usize, q: Quantization) -> IndexOptions {
    IndexOptions {
        dimensions: dims,
        metric: MetricKind::IP,
        quantization: scalar_kind(q),
        connectivity: CONNECTIVITY,
        expansion_add: EXPANSION_ADD,
        expansion_search: EXPANSION_SEARCH,
        multi: false,
    }
}

/// Build-time writer: ANN adds + raw-vector blob.
pub struct DenseBuilder {
    index: Index,
    raw: BufWriter<File>,
    raw_path: PathBuf,
    index_path: PathBuf,
    count: usize,
    capacity: usize,
    dims: usize,
}

impl DenseBuilder {
    pub fn create(dir: &Path, dims: usize, quant: Quantization) -> Result<Self> {
        let index = Index::new(&options(dims, quant))
            .map_err(|e| anyhow::anyhow!("create usearch index: {e}"))?;
        let capacity = 1024usize;
        index
            .reserve(capacity)
            .map_err(|e| anyhow::anyhow!("usearch reserve: {e}"))?;
        let raw_path = dir.join(RAW_FILE);
        let mut raw = BufWriter::new(
            File::create(&raw_path).with_context(|| format!("create {}", raw_path.display()))?,
        );
        raw.write_all(&0u64.to_le_bytes())?; // n placeholder, patched at finish
        raw.write_all(&(dims as u64).to_le_bytes())?;
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
        self.raw.write_all(&id.to_le_bytes())?;
        for v in vector {
            self.raw.write_all(&v.to_le_bytes())?;
        }
        self.count += 1;
        Ok(())
    }

    /// Persist index + patch raw header + fsync.
    pub fn finish(mut self) -> Result<usize> {
        self.index
            .save(self.index_path.to_str().context("non-utf8 index path")?)
            .map_err(|e| anyhow::anyhow!("usearch save {}: {e}", self.index_path.display()))?;
        self.raw.flush()?;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.raw_path)?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&(self.count as u64).to_le_bytes())?;
        f.sync_all()?;
        Ok(self.count)
    }
}

/// Query-time view: mmap'd HNSW + mmap'd raw blob for exact scans.
pub struct DenseIndex {
    index: Index,
    raw: Mmap,
    dims: usize,
}

impl DenseIndex {
    pub fn view(dir: &Path, dims: usize, quant: Quantization) -> Result<Self> {
        let index_path = dir.join(DENSE_FILE);
        let index = Index::new(&options(dims, quant))
            .map_err(|e| anyhow::anyhow!("create usearch handle: {e}"))?;
        index
            .view(index_path.to_str().context("non-utf8 index path")?)
            .map_err(|e| anyhow::anyhow!("usearch view {}: {e}", index_path.display()))?;
        let raw_path = dir.join(RAW_FILE);
        let raw_file =
            File::open(&raw_path).with_context(|| format!("open {}", raw_path.display()))?;
        // Safety: read-only map of an immutable artifact.
        let raw = unsafe { Mmap::map(&raw_file) }
            .with_context(|| format!("mmap {}", raw_path.display()))?;
        Ok(Self { index, raw, dims })
    }

    /// ANN search; returns (passage_id, cosine_score) sorted best-first.
    /// usearch IP distance is `1 - dot`; for unit vectors that is `1 - cosine`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
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
        let buf = &self.raw[..];
        if buf.len() < 16 {
            bail!("corrupt {}: too small", RAW_FILE);
        }
        let n = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
        let dims = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        if dims != self.dims {
            bail!("raw blob dims {dims} != index dims {}", self.dims);
        }
        // Record layout differs by format version: v1 wrote bare f32 rows
        // (id == row index); v2 writes [u64 id][f32 vec] (chunked corpora
        // store non-embedded parents). The blob has no version field, but
        // the two sizes can never coincide (they differ by 8·n bytes).
        let v2_stride = 8 + dims * 4;
        let v1_stride = dims * 4;
        let (stride, has_id) = if buf.len() == 16 + n * v2_stride {
            (v2_stride, true)
        } else if buf.len() == 16 + n * v1_stride {
            (v1_stride, false)
        } else {
            bail!(
                "corrupt {}: size {} fits neither v1 ({}) nor v2 ({}) records",
                RAW_FILE,
                buf.len(),
                16 + n * v1_stride,
                16 + n * v2_stride
            );
        };
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(n);
        for i in 0..n {
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
            scored.push((id, dot));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}
