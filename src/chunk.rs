//! Structural chunking (t3-wkp.11): split only passages over the ~8K-token
//! embed limit; whole passages stay the retrieval unit by default.
//!
//! Split order is structural-first: blank-line paragraph boundaries and
//! markdown headings; oversized single blocks fall back to line, then char
//! splitting. Chunks get sequential passage ids, `parent_id` = source
//! passage, `chunk_idx` = ordinal — parents are stored for hydration but
//! NOT embedded/indexed (retrieval units are the chunks).

/// Estimated tokens (≈3 chars/token) above which a passage is chunked.
/// ~7K est ≈ safely under the 8,192-token embed truncation.
pub const CHUNK_TRIGGER_EST_TOKENS: usize = 7000;
/// Target chunk size in chars (~5K est tokens — under both the trigger and
/// the 4K-token embed batch budget interplay).
pub const CHUNK_TARGET_CHARS: usize = 15_000;

/// Cheap token estimate (~3 chars/token) for budget checks — real
/// token counts need the embedder's tokenizer (a 2.8 follow-up).
pub fn est_tokens(text: &str) -> usize {
    text.len() / 3
}

/// Whether `text` should be structurally chunked for retrieval
/// (rough token estimate over the chunk threshold).
pub fn needs_chunk(text: &str) -> bool {
    est_tokens(text) > CHUNK_TRIGGER_EST_TOKENS
}

/// Split `text` into structurally-bounded chunks of roughly
/// CHUNK_TARGET_CHARS each. Always returns ≥1 chunk.
/// Split `text` into retrievable chunks on structural boundaries
/// (paragraphs/sections), each under the token budget. Parents are
/// stored un-embedded and hydrated at query time (t3-wkp.11).
pub fn split(text: &str) -> Vec<String> {
    if !needs_chunk(text) {
        return vec![text.to_string()];
    }
    let blocks = structural_blocks(text);
    pack_blocks(&blocks)
}

/// Split into semantic blocks: blank-line groups, with markdown headings
/// starting their own block. Blocks retain their trailing separators where
/// possible so re-joined chunks read naturally.
fn structural_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    for para in text.split("\n\n") {
        // Further split before markdown headings inside a paragraph.
        let mut cur = String::new();
        for line in para.split_inclusive('\n') {
            if line.trim_start().starts_with('#') && !cur.is_empty() {
                blocks.push(std::mem::take(&mut cur));
            }
            cur.push_str(line);
        }
        if !cur.trim().is_empty() {
            cur.push_str("\n\n");
            blocks.push(std::mem::take(&mut cur));
        }
    }
    blocks
}

/// Greedy pack of blocks into <=target chunks; an oversized single block is
/// hard-split at line boundaries (char boundary as last resort).
fn pack_blocks(blocks: &[String]) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for b in blocks {
        if b.len() > CHUNK_TARGET_CHARS {
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            chunks.extend(hard_split(b));
            continue;
        }
        if !cur.is_empty() && cur.len() + b.len() > CHUNK_TARGET_CHARS {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(b);
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

/// Last-resort split: break on newline boundaries near the target; when a
/// single line exceeds the target, cut at a char boundary.
fn hard_split(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in text.split_inclusive('\n') {
        while line.len() > CHUNK_TARGET_CHARS.saturating_sub(cur.len()) && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if line.len() > CHUNK_TARGET_CHARS {
            // Giant single line: emit the pending chunk, then char-cut.
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            for piece in line.as_bytes().chunks(CHUNK_TARGET_CHARS) {
                out.push(String::from_utf8_lossy(piece).into_owned());
            }
            continue;
        }
        cur.push_str(line);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_passage_not_chunked() {
        let t = "short passage";
        assert!(!needs_chunk(t));
        assert_eq!(split(t), vec![t.to_string()]);
    }

    #[test]
    fn long_passage_splits_on_paragraphs() {
        let para = "x".repeat(4000);
        let text = (0..12)
            .map(|_| para.clone())
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks = split(&text);
        assert!(
            chunks.len() >= 3,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.len() <= CHUNK_TARGET_CHARS + 4096,
                "chunk oversize {}",
                c.len()
            );
        }
        // Content preserved (modulo separators).
        let joined: usize = chunks.iter().map(|c| c.matches('x').count()).sum();
        assert_eq!(joined, 12 * 4000);
    }

    #[test]
    fn giant_single_block_hard_splits() {
        let text = "y".repeat(60_000);
        let chunks = split(&text);
        assert!(chunks.len() >= 4);
        for c in &chunks {
            assert!(c.len() <= CHUNK_TARGET_CHARS + 1024);
        }
    }
}
