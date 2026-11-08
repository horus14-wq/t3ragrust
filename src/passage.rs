use serde::{Deserialize, Serialize};

/// One retrievable unit: dense ordinal `id`, `text`, optional
/// metadata, and the chunk hierarchy fields (`parent_id`/`chunk_idx` —
/// unchunked passages point at themselves).
///
/// `parent_id`/`chunk_idx` support structural chunking (t3-wkp.11): an
/// unchunked passage carries `parent_id == id` and `chunk_idx == 0`;
/// a chunk points at its source passage and hydrates back to it at
/// query time, so results always surface whole passages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Passage {
    /// Ordinal id within the corpus (dense, aligned with index keys).
    pub id: u64,
    /// Passage body text — the unit embedded at index time.
    pub text: String,
    /// Optional display title (dir ingest uses the filename).
    pub title: Option<String>,
    /// Optional originating question, when the corpus records one.
    pub question: Option<String>,
    /// Optional provenance tag (file path, upstream id, ...).
    pub source: Option<String>,
    /// Passage this chunk belongs to (self for unchunked passages).
    pub parent_id: u64,
    /// 0-based position within the parent's chunk list.
    pub chunk_idx: u32,
}

impl Passage {
    /// An unchunked passage: `parent_id == id`, `chunk_idx == 0`, all
    /// metadata `None`. Use struct-update syntax to attach metadata.
    pub fn new(id: u64, text: String) -> Self {
        Self {
            id,
            text,
            title: None,
            question: None,
            source: None,
            parent_id: id,
            chunk_idx: 0,
        }
    }
}
