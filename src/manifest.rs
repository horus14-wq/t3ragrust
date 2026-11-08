//! Corpus aboutness manifest (t3-iv2): index-time accumulation of what a
//! corpus is *about* — detected transform family, top question terms,
//! length histogram, language guess, and a passage-vector centroid.
//!
//! Persisted as `meta.json.manifest` (`Option` — corpora built before
//! this existed simply lack it). The transform label is the PRIMARY
//! routing signal for the T3 corpus family (struct/reflect/semantic/raw
//! are recognizable from output format); the centroid is SECONDARY —
//! T3 views cover the same problem space, so centroid similarity across
//! them is near-uniform and must never rank alone (round-2 review).

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::passage::Passage;

/// Routing manifest stored in `meta.json` (t3-iv2).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Dominant transform label: `struct` | `reflect` | `semantic` |
    /// `raw` | `other` | `mixed` (no label >= 60% of classified passages).
    pub transform: String,
    /// Per-label passage counts behind `transform`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub transform_mix: BTreeMap<String, u64>,
    /// Top question unigrams by frequency (stopwords removed), ~20 max.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_terms: Vec<String>,
    /// Passage char-length buckets: `<500`, `500-2k`, `2k-8k`, `>=8k`.
    #[serde(default)]
    pub length_histogram: [u64; 4],
    /// Crude language guess: `en` | `unknown` (honest — no langdetect dep).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// L2-normalized mean of the stored (possibly MRL-truncated)
    /// passage vectors. Secondary signal only — see module docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centroid: Option<Vec<f32>>,
}

/// Function words + ultra-generic task words excluded from `top_terms`.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
    "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old",
    "see", "two", "way", "who", "did", "get", "let", "put", "say", "she", "too", "use", "what",
    "when", "which", "will", "with", "would", "there", "their", "them", "then", "than", "that",
    "this", "these", "those", "from", "have", "been", "were", "they", "each", "into", "over",
    "such", "only", "also", "more", "most", "other", "some", "any", "about", "between", "through",
    "during", "before", "after", "above", "below", "under", "again", "further", "once", "here",
    "why", "does", "doing", "done", "make", "made", "take", "took", "give", "given", "find",
    "found", "write", "written", "create", "created", "implement", "implemented", "solve", "solved",
    "problem", "problems", "question", "questions", "answer", "code", "coding", "program",
    "function", "functions", "step", "steps", "example", "following", "below", "using", "used",
    "based", "include", "including", "ensure", "returns", "return", "value", "values", "number",
    "numbers", "list", "data", "file", "files", "input", "output", "result", "results", "user",
    "users", "please", "need", "needs", "want", "provide", "complete", "production", "ready",
];

/// English function words counted for the crude language guess.
const EN_FUNCTION_WORDS: &[&str] = &[
    "the", "and", "of", "to", "in", "is", "for", "on", "with", "as", "by", "at", "an", "be",
    "this", "that", "it", "or", "from", "are", "was", "you", "your",
];

/// Accumulates manifest stats during the index pass; `finish()` yields the
/// persisted [`CorpusManifest`]. Cheap: per-passage work is a few `contains`
/// scans + question tokenization; language stats sample every 8th passage.
pub struct ManifestAcc {
    transform: FxHashMap<&'static str, u64>,
    terms: FxHashMap<String, u64>,
    len_hist: [u64; 4],
    centroid_sum: Vec<f32>,
    centroid_n: u64,
    en_hits: u64,
    lang_words: u64,
    sampled: u64,
}

impl ManifestAcc {
    pub fn new(dims: usize) -> Self {
        Self {
            transform: FxHashMap::default(),
            terms: FxHashMap::default(),
            len_hist: [0; 4],
            centroid_sum: vec![0.0; dims],
            centroid_n: 0,
            en_hits: 0,
            lang_words: 0,
            sampled: 0,
        }
    }

    /// Observe one retrievable unit (passage or chunk — parents are stored
    /// for hydration but never retrieved, so they don't count).
    pub fn observe(&mut self, p: &Passage) {
        *self.transform.entry(classify_transform(&p.text)).or_default() += 1;
        let bucket = match p.text.len() {
            n if n < 500 => 0,
            n if n < 2_000 => 1,
            n if n < 8_000 => 2,
            _ => 3,
        };
        self.len_hist[bucket] += 1;
        if let Some(q) = &p.question {
            for tok in tokenize(q) {
                *self.terms.entry(tok).or_default() += 1;
            }
        }
        // Language guess on a sample (every 8th unit, first 2KB).
        self.sampled += 1;
        if self.sampled % 8 == 1 {
            let head: String = p.text.chars().take(2000).collect();
            for w in tokenize(&head) {
                self.lang_words += 1;
                if EN_FUNCTION_WORDS.contains(&w.as_str()) {
                    self.en_hits += 1;
                }
            }
        }
    }

    /// Accumulate one stored (post-truncation) passage vector into the
    /// centroid sum. Called from `flush_batch` on the same vectors the
    /// dense index receives.
    pub fn observe_vector(&mut self, v: &[f32]) {
        if v.len() != self.centroid_sum.len() {
            return;
        }
        for (s, x) in self.centroid_sum.iter_mut().zip(v) {
            *s += x;
        }
        self.centroid_n += 1;
    }

    /// Reduce the accumulator into the persisted manifest.
    pub fn finish(self) -> CorpusManifest {
        let total: u64 = self.transform.values().sum();
        let mix: BTreeMap<String, u64> = self
            .transform
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        // Dominant label needs >=60% to claim the corpus; otherwise "mixed".
        let transform = self
            .transform
            .iter()
            .max_by_key(|(_, n)| *n)
            .filter(|(_, n)| total > 0 && **n * 5 >= total * 3)
            .map(|(k, _)| k.to_string())
            .unwrap_or_else(|| "mixed".to_string());
        let mut terms: Vec<(String, u64)> = self.terms.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let top_terms = terms
            .into_iter()
            .take(20)
            .map(|(t, _)| t)
            .collect::<Vec<_>>();
        let language = if self.lang_words >= 20 && self.en_hits * 50 >= self.lang_words {
            Some("en".to_string())
        } else if self.lang_words >= 20 {
            Some("unknown".to_string())
        } else {
            None
        };
        let centroid = if self.centroid_n > 0 {
            let n = self.centroid_n as f32;
            let mut c: Vec<f32> = self.centroid_sum.iter().map(|s| s / n).collect();
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut c {
                    *x /= norm;
                }
                Some(c)
            } else {
                None
            }
        } else {
            None
        };
        CorpusManifest {
            transform,
            transform_mix: mix,
            top_terms,
            length_histogram: self.len_hist,
            language,
            centroid,
        }
    }
}

/// Classify one passage's transform family from output-format markers.
/// Order matters: reflect/semantic headers are the most specific, struct's
/// "Step 1 + Deliverable" next, role-prefixed dialogue is `raw`.
fn classify_transform(text: &str) -> &'static str {
    if text.contains("Common Mistakes")
        || text.contains("Misleading Intuitions")
        || text.contains("Critical Checks")
    {
        return "reflect";
    }
    if text.contains("Pass 1") && text.contains("Pass 2") && text.contains("Pass 3") {
        return "semantic";
    }
    if (text.contains("Step 1") || text.contains("**Step 1"))
        && (text.contains("Deliverable") || text.contains("Answer:"))
    {
        return "struct";
    }
    let head: String = text.chars().take(64).collect();
    let head = head.trim_start();
    if head.starts_with("ASSISTANT") || head.starts_with("USER") || head.starts_with("HUMAN") {
        return "raw";
    }
    "other"
}

/// Lowercase alphanumeric tokens, len >= 3, stopwords removed.
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3 && !STOPWORDS.contains(t))
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc() -> ManifestAcc {
        ManifestAcc::new(4)
    }

    fn passage(text: &str, question: Option<&str>) -> Passage {
        let mut p = Passage::new(0, text.to_string());
        p.question = question.map(str::to_string);
        p
    }

    #[test]
    fn detects_struct_reflect_semantic_raw() {
        assert_eq!(
            classify_transform("Problem: x\n\nStep 1: do a\nStep 2: do b\n\nDeliverable: f()"),
            "struct"
        );
        assert_eq!(
            classify_transform("**Common Mistakes:** N/A\n**Critical Checks:** - x"),
            "reflect"
        );
        assert_eq!(
            classify_transform("**Pass 1:** outline\n**Pass 2:** moves\n**Pass 3:** insight"),
            "semantic"
        );
        assert_eq!(
            classify_transform("ASSISTANT: ```python\nprint(1)\n```"),
            "raw"
        );
        assert_eq!(classify_transform("just some prose"), "other");
    }

    #[test]
    fn manifest_aggregates() {
        let mut a = acc();
        for i in 0..10 {
            a.observe(&passage(
                "Step 1: x\nDeliverable: y",
                Some("implement or-set tombstones"),
            ));
            a.observe_vector(&[1.0, 0.0, 0.0, 0.0]);
            let _ = i;
        }
        let m = a.finish();
        assert_eq!(m.transform, "struct");
        assert_eq!(m.transform_mix.get("struct"), Some(&10));
        assert!(m.top_terms.contains(&"or-set".to_string()) || m.top_terms.iter().any(|t| t.contains("set")));
        assert_eq!(m.length_histogram.iter().sum::<u64>(), 10);
        let c = m.centroid.unwrap();
        assert!((c[0] - 1.0).abs() < 1e-6);
    }
}
