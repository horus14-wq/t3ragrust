use serde::{Deserialize, Serialize};

/// One retrievable unit of thinking-trace content.
///
/// `parent_id`/`chunk_idx` support structural chunking (t3-wkp.11): unchunked
/// passages carry `parent_id == passage_id` and `chunk_idx == 0`; chunks point
/// at their source passage and hydrate back to it at query time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Passage {
    /// Ordinal id within the corpus (dense, aligned with index keys).
    pub id: u64,
    pub text: String,
    pub title: Option<String>,
    pub question: Option<String>,
    pub source: Option<String>,
    /// Passage this chunk belongs to (self for unchunked passages).
    pub parent_id: u64,
    pub chunk_idx: u32,
}

impl Passage {
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
