//! Streaming writer for the packed passage store.
//!
//! Format — `passages.bin`: repeated records of
//!   `[u32 LE len][postcard(Passage) payload]`
//! `passages.idx`: array of `u64` LE byte offsets into passages.bin, one per
//! record in write order. Ids must be dense ordinals 0..n (idx position ==
//! passage id); the writer enforces this so readers can index directly.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::passage::Passage;

/// Build-time writer for `passages.bin`/`passages.idx`.
///
/// Streams passages into both files inside `dir`, holding no corpus
/// data in memory — each record is serialized and flushed.
pub struct PassageWriter {
    bin: BufWriter<File>,
    idx: BufWriter<File>,
    offset: u64,
    next_id: u64,
}

impl PassageWriter {
    /// Create fresh store files under `dir`.
    ///
    /// # Errors
    /// Any of the three artifact files cannot be created.
    pub fn create(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create corpus dir {}", dir.display()))?;
        let bin_path = dir.join("passages.bin");
        let idx_path = dir.join("passages.idx");
        Ok(Self {
            bin: BufWriter::new(
                File::create(&bin_path)
                    .with_context(|| format!("create {}", bin_path.display()))?,
            ),
            idx: BufWriter::new(
                File::create(&idx_path)
                    .with_context(|| format!("create {}", idx_path.display()))?,
            ),
            offset: 0,
            next_id: 0,
        })
    }

    /// Append one passage (postcard + index entry). `passage.id` must
    /// equal the ordinal write position (0,1,2,...) so the idx stays a
    /// plain offset table.
    ///
    /// # Errors
    /// `passage.id` is not the next dense ordinal, or serialization or
    /// write fails.
    pub fn push(&mut self, passage: &Passage) -> Result<()> {
        if passage.id != self.next_id {
            bail!(
                "passage ids must be dense ordinals: got id {}, expected {}",
                passage.id,
                self.next_id
            );
        }
        let payload = postcard::to_allocvec(passage).context("serialize passage record")?;
        let len = u32::try_from(payload.len()).context("passage record >4GiB")?;
        self.idx.write_all(&self.offset.to_le_bytes())?;
        self.bin.write_all(&len.to_le_bytes())?;
        self.bin.write_all(&payload)?;
        self.offset += 4 + payload.len() as u64;
        self.next_id += 1;
        Ok(())
    }

    /// Flush + fsync both files and return the stored doc count.
    /// Dropping without `finish` leaves the corpus incomplete (the
    /// truncated tail is detectable on open).
    ///
    /// # Errors
    /// A flush or sync fails.
    pub fn finish(mut self) -> Result<u64> {
        self.bin.flush()?;
        self.idx.flush()?;
        self.bin.get_ref().sync_all()?;
        self.idx.get_ref().sync_all()?;
        Ok(self.next_id)
    }
}
