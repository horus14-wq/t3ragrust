//! mmap-backed reader for the packed passage store.

use std::path::Path;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

use crate::passage::Passage;

/// Read-side handle over `passages.bin`/`passages.idx`: `get(id)`
/// deserializes the postcard record at the indexed offset.
///
/// Both files are mapped whole — warm state is page cache, no heap copies.
pub struct PassageReader {
    bin: Mmap,
    idx: Mmap,
}

impl PassageReader {
    /// Open the passage store under `dir`, mapping both files.
    ///
    /// Fails cleanly on missing/corrupt files: the idx must be a whole
    /// number of u64 entries and the last recorded offset must land
    /// inside bin (catches truncated builds at open).
    ///
    /// # Errors
    /// `passages.bin`/`passages.idx` are missing or unparseable.
    pub fn open(dir: &Path) -> Result<Self> {
        let bin_path = dir.join("passages.bin");
        let idx_path = dir.join("passages.idx");
        let open = |p: &Path| -> Result<Mmap> {
            let f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
            // Safety: read-only map of a file treated as immutable post-build.
            unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", p.display()))
        };
        let bin = open(&bin_path)?;
        let idx = open(&idx_path)?;
        if idx.len() % 8 != 0 {
            bail!(
                "corrupt {}: length {} is not a multiple of 8",
                idx_path.display(),
                idx.len()
            );
        }
        let reader = Self { bin, idx };
        if let Some(last) = reader.last_offset() {
            if last as usize + 4 > reader.bin.len() {
                bail!(
                    "corrupt {}: last offset {} out of bounds for {} ({} bytes)",
                    idx_path.display(),
                    last,
                    bin_path.display(),
                    reader.bin.len()
                );
            }
        }
        Ok(reader)
    }

    fn offset_at(&self, pos: usize) -> u64 {
        let start = pos * 8;
        u64::from_le_bytes(self.idx[start..start + 8].try_into().unwrap())
    }

    fn last_offset(&self) -> Option<u64> {
        (self.idx.len() >= 8).then(|| self.offset_at(self.len() as usize - 1))
    }

    /// Number of stored passages.
    pub fn len(&self) -> u64 {
        (self.idx.len() / 8) as u64
    }

    /// Fetch passage `id` (the dense ordinal — chunks map to their
    /// parent on hydrate) in O(1).
    ///
    /// # Errors
    /// `id` out of range, or the record fails to deserialize.
    pub fn get(&self, id: u64) -> Result<Passage> {
        if id >= self.len() {
            bail!("passage id {id} out of range ({} stored)", self.len());
        }
        let off = self.offset_at(id as usize) as usize;
        let buf = &self.bin[off..];
        let len = u32::from_le_bytes(
            buf.get(0..4)
                .context("corrupt store: truncated len prefix")?
                .try_into()
                .unwrap(),
        ) as usize;
        let payload = buf
            .get(4..4 + len)
            .context("corrupt store: record overruns file")?;
        let passage: Passage = postcard::from_bytes(payload).context("deserialize passage")?;
        if passage.id != id {
            bail!(
                "store/index mismatch: idx slot {id} holds passage {}",
                passage.id
            );
        }
        Ok(passage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::PassageWriter;

    #[test]
    fn write_read_roundtrip() {
        let dir = std::env::temp_dir().join(format!("t3store-{}", std::process::id()));
        {
            let mut w = PassageWriter::create(&dir).unwrap();
            for i in 0..5u64 {
                let mut p = Passage::new(i, format!("passage {i}"));
                p.title = Some(format!("t{i}"));
                w.push(&p).unwrap();
            }
            assert_eq!(w.finish().unwrap(), 5);
        }
        let r = PassageReader::open(&dir).unwrap();
        assert_eq!(r.len(), 5);
        let p = r.get(3).unwrap();
        assert_eq!(p.id, 3);
        assert_eq!(p.text, "passage 3");
        assert_eq!(p.title.as_deref(), Some("t3"));
        assert!(r.get(9).is_err());
        let all: Vec<_> = (0..r.len()).map(|i| r.get(i).unwrap()).collect();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn rejects_non_dense_id() {
        let dir = std::env::temp_dir().join(format!("t3store2-{}", std::process::id()));
        let mut w = PassageWriter::create(&dir).unwrap();
        assert!(w.push(&Passage::new(7, "x".into())).is_err());
    }
}
