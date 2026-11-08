//! Lexical index: `tantivy` BM25 under `bm25/` in the corpus dir.
//!
//! Docs store `passage_id` only (text lives in the packed store); `text`,
//! `title`, `question` are indexed TEXT fields for BM25, `source` is STRING
//! metadata. Scores across retrieval legs are not comparable — the RRF
//! layer (search bead) consumes ranks, not these scores.

use std::path::Path;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexWriter};

use crate::passage::Passage;

pub const BM25_DIR: &str = "bm25";

pub struct SchemaFields {
    pub passage_id: Field,
    pub text: Field,
    pub title: Field,
    pub question: Field,
    pub source: Field,
}

fn build_schema() -> (Schema, SchemaFields) {
    let mut b = Schema::builder();
    let fields = SchemaFields {
        passage_id: b.add_u64_field("passage_id", STORED),
        text: b.add_text_field("text", TEXT),
        title: b.add_text_field("title", TEXT),
        question: b.add_text_field("question", TEXT),
        source: b.add_text_field("source", STRING | STORED),
    };
    (b.build(), fields)
}

/// Build-time writer for a fresh `bm25/` index.
pub struct Bm25Builder {
    writer: IndexWriter,
    fields: SchemaFields,
}

impl Bm25Builder {
    pub fn create(corpus_dir: &Path) -> Result<Self> {
        let dir = corpus_dir.join(BM25_DIR);
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let (schema, fields) = build_schema();
        let index = Index::create_in_dir(&dir, schema)
            .with_context(|| format!("create tantivy index in {}", dir.display()))?;
        // 50MB writer heap; tantivy parallelizes segment build internally.
        let writer = index
            .writer(50_000_000)
            .context("create tantivy index writer")?;
        Ok(Self { writer, fields })
    }

    pub fn add(&mut self, p: &Passage) -> Result<()> {
        let f = &self.fields;
        let mut d = doc!(f.passage_id => p.id, f.text => p.text.clone());
        if let Some(t) = &p.title {
            d.add_text(f.title, t);
        }
        if let Some(q) = &p.question {
            d.add_text(f.question, q);
        }
        if let Some(s) = &p.source {
            d.add_text(f.source, s);
        }
        self.writer
            .add_document(d)
            .context("tantivy add_document")?;
        Ok(())
    }

    pub fn commit(mut self) -> Result<()> {
        self.writer.commit().context("tantivy commit")?;
        Ok(())
    }
}

/// Query-time reader — one pooled `IndexReader` per corpus (t3-css.1.5):
/// `IndexReader` is Send+Sync and `.searcher()` is the per-query operation;
/// constructing the reader per query is wasted work on the daemon hot path.
pub struct Bm25Index {
    index: Index,
    fields: SchemaFields,
    reader: tantivy::IndexReader,
}

impl Bm25Index {
    pub fn open(corpus_dir: &Path) -> Result<Self> {
        let dir = corpus_dir.join(BM25_DIR);
        let index = Index::open_in_dir(&dir)
            .with_context(|| format!("open tantivy index {}", dir.display()))?;
        let fields = {
            let schema = index.schema();
            SchemaFields {
                passage_id: schema.get_field("passage_id")?,
                text: schema.get_field("text")?,
                title: schema.get_field("title")?,
                question: schema.get_field("question")?,
                source: schema.get_field("source")?,
            }
        };
        let reader = index.reader().context("tantivy reader")?;
        Ok(Self {
            index,
            fields,
            reader,
        })
    }

    /// BM25 top-k; returns (passage_id, bm25_score). Callers consume ranks.
    ///
    /// Natural-language queries often contain QueryParser metacharacters
    /// (`:`, `"`, `-`, `&&`) — t3-css.1.7: a parse failure falls back to a
    /// literal bag-of-terms instead of dying (or silently zeroing the leg).
    pub fn search(&self, query_str: &str, k: usize) -> Result<Vec<(u64, f32)>> {
        let searcher = self.reader.searcher();
        let qp = QueryParser::for_index(
            &self.index,
            vec![self.fields.text, self.fields.title, self.fields.question],
        );
        let query = match qp.parse_query(query_str) {
            Ok(q) => {
                tracing::debug!("bm25: parsed query");
                q
            }
            Err(e) => {
                tracing::debug!(error = %e, "bm25: parse failed; literal-term fallback");
                self.literal_query(query_str)
            }
        };
        let top = searcher
            .search(&query, &TopDocs::with_limit(k))
            .context("tantivy search")?;
        let mut out = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc = searcher.doc::<tantivy::TantivyDocument>(addr)?;
            let id = doc
                .get_first(self.fields.passage_id)
                .and_then(|v| v.as_u64())
                .context("tantivy doc missing passage_id")?;
            out.push((id, score));
        }
        Ok(out)
    }

    /// Literal bag-of-terms fallback: lowercase whitespace tokens as
    /// per-field TermQueries in a Should BooleanQuery — no metacharacters,
    /// so any string retrieves *something* when the parser chokes.
    fn literal_query(&self, query_str: &str) -> Box<dyn tantivy::query::Query> {
        use tantivy::query::{BooleanQuery, Occur, TermQuery};
        use tantivy::schema::IndexRecordOption;
        use tantivy::Term;
        let mut clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
        for tok in query_str.split_whitespace() {
            let tok = tok.to_lowercase();
            if tok.is_empty() {
                continue;
            }
            for field in [self.fields.text, self.fields.title, self.fields.question] {
                clauses.push((
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(field, &tok),
                        IndexRecordOption::WithFreqs,
                    )),
                ));
            }
        }
        Box::new(BooleanQuery::new(clauses))
    }
}
