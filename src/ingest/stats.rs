use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Thread-safe skip/read counters shared between a reader iterator and the
/// pipeline that consumes it. Cheap to clone (Arc internals).
#[derive(Clone, Default)]
pub struct SkipStats {
    read: Arc<AtomicU64>,
    skipped: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
    reasons: Arc<Mutex<BTreeMap<String, u64>>>,
}

impl SkipStats {
    pub fn record_read(&self, bytes: u64) {
        self.read.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn skip(&self, reason: &str) {
        self.skipped.fetch_add(1, Ordering::Relaxed);
        *self
            .reasons
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(reason.to_string())
            .or_insert(0) += 1;
    }

    pub fn read(&self) -> u64 {
        self.read.load(Ordering::Relaxed)
    }

    pub fn skipped(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn reasons(&self) -> BTreeMap<String, u64> {
        self.reasons
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl fmt::Display for SkipStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "read={}, skipped={}, bytes={}",
            self.read(),
            self.skipped(),
            self.bytes()
        )?;
        let reasons = self.reasons();
        if !reasons.is_empty() {
            write!(f, " (")?;
            for (i, (r, n)) in reasons.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{r}: {n}")?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}
