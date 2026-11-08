//! Text-hash -> vector reuse cache for `index --update` (t3-css.4.1).
//!
//! Rather than delta-appending to the stores (passages/usearch/tantivy
//! have no cheap merge path), `--update` rebuilds the corpus through the
//! normal pipeline but sources vectors from the *existing* corpus's
//! `vectors.raw.f32` + `passages.bin` whenever a passage text is
//! unchanged. Embedding is the dominant index cost, so skipping it for
//! unchanged rows is where the win is; HNSW/BM25/passage rewrites are
//! minutes, not hours. This also self-heals deletes, reorders, and
//! edited rows for free — no row-id manifest needed.
//!
//! Cache validity is the caller's job: `run_index` only builds this
//! after verifying model/dims/instruction match `meta.json`.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::Mmap;
use rustc_hash::FxHashMap;

use super::dense;
use crate::ingest;
use crate::store::PassageReader;

/// `passage text hash -> stored (already-truncated, unit-norm) vector`.
pub struct VectorCache {
    map: FxHashMap<u64, Vec<f32>>,
    /// Records consumed building the map (observability).
    pub entries: usize,
}

impl VectorCache {
    /// Build the cache from an existing corpus dir: iterate
    /// `vectors.raw.f32` records, resolve each record's passage id to
    /// its stored text, and key the vector by `hash_text(text)`.
    ///
    /// Returns `Ok(None)` when the corpus has no raw blob
    /// (`--no-raw-vectors` builds can't be cheaply updated — the caller
    /// bails with guidance).
    ///
    /// # Errors
    /// The raw blob or passage store is missing/corrupt, or a record's
    /// passage id doesn't resolve.
    pub fn build(dir: &Path, dims: usize) -> Result<Option<Self>> {
        let raw_path = dir.join(dense::RAW_FILE);
        if !raw_path.exists() {
            return Ok(None);
        }
        let file = File::open(&raw_path).with_context(|| format!("open {}", raw_path.display()))?;
        // Safety: read-only map of an immutable build artifact.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", raw_path.display()))?;
        let (n, stride, has_id) = dense::raw_layout(&mmap[..], dims)?;
        let reader = PassageReader::open(dir)?;
        let mut map = FxHashMap::default();
        map.reserve(n);
        for i in 0..n {
            let rec = &mmap[16 + i * stride..16 + (i + 1) * stride];
            let (pid, row) = if has_id {
                (u64::from_le_bytes(rec[0..8].try_into().unwrap()), &rec[8..])
            } else {
                (i as u64, rec)
            };
            let (chunks, _) = row.as_chunks::<4>();
            let vec: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let p = reader
                .get(pid)
                .with_context(|| format!("raw record {i} -> passage {pid}"))?;
            map.insert(ingest::hash_text(&p.text), vec);
        }
        Ok(Some(Self { map, entries: n }))
    }

    /// Cached vector for `text`, if this exact string was embedded
    /// before. The stored vector is already truncated to `dims` and
    /// unit-normalized — feed it to `dense.add`/`observe_vector` as-is.
    #[inline]
    pub fn get(&self, text: &str) -> Option<&[f32]> {
        self.map.get(&ingest::hash_text(text)).map(Vec::as_slice)
    }

    /// Number of cached texts (test/inspection hook).
    #[cfg(test)]
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_hits_and_misses_by_exact_text() {
        let mut map = FxHashMap::default();
        map.insert(ingest::hash_text("hello"), vec![1.0, 0.0]);
        let c = VectorCache { map, entries: 1 };
        assert_eq!(c.get("hello"), Some(&[1.0, 0.0][..]));
        assert_eq!(c.get("hello!"), None);
        assert_eq!(c.get(""), None);
    }

    #[test]
    fn duplicate_texts_share_one_entry() {
        let mut map = FxHashMap::default();
        let h = ingest::hash_text("same");
        map.insert(h, vec![0.5]);
        map.insert(h, vec![0.5]);
        let c = VectorCache { map, entries: 2 };
        assert_eq!(c.len(), 1);
        assert_eq!(c.get("same"), Some(&[0.5][..]));
    }
}
