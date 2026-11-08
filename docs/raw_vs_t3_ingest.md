# Raw vs T3 ingest

For corpus: `fable5_v2_filtered_coding.jsonl` (and the three T3 JSONLs derived from it). Strategy context: `docs/lessonsfromprototype.md`.

## Raw traces

**What:** the filtered episodes themselves — lean `question` + strategy-B / ASSISTANT `text`. No LLM rewrite.

**How to process:**
1. Use the filtered JSONL as-is (do not re-run T3). Extra fields (`n_thoughts`, etc.) are ignored by ingest.
2. Shape is already valid for `t3rag` if each line has:
   ```json
   {"question": "<lean ask>", "text": "<body>", "source": "<stable id>", "title": "raw"}
   ```
   (`title` is optional; `source` should match the id used on T3 rows for the same episode.)
3. `t3rag index --input <that.jsonl> --name fable5_v2_raw`
4. Chunking: **whole passage by default**. Structural split only if over ~7–8K est tokens (`chunk.rs`). Fixed 512-token windows are for paper e5-base — not needed here (Qwen3-Embedding long context).

**Dense embeds `text` only.** `question` is stored and searched by BM25 (with `text`/`title`). Do not bake `Question:…` into `text` unless you explicitly want dense to see the ask.

## T3 traces

**What:** LLM offline transforms of the **same** episodes — `t3_struct` / `t3_reflect` / `t3_semantic`. Fields are usually `question` + `output` (or `text`); ingest accepts either (`text` wins if both are present).

**How to process:**
1. Point `t3rag index` at each transform file.
2. Prefer **separate corpora**: `fable5_v2_struct`, `_reflect`, `_semantic`.
3. Same chunk policy (almost never triggers — scaffolds are short).

## Comparison

| | Raw | T3 |
| --- | --- | --- |
| Source step | Filter/convert only (Python) | + LLM rewrite per prompt |
| Passage content | Full episode body (tools/prose) | Cheatsheet / mistakes / insight |
| Typical length | Longer | Shorter, denser |
| Best for | Errors, APIs, exact mid-project debug | Approach, pitfalls, compressed procedure |
| Chunking | Rare structural splits | Almost never |
| Index | One named corpus | One corpus per transform type |

**Do not** dump raw+all T3 into one undifferentiated top‑k without diversity (same `source` will dominate). Query by corpus/type, or fuse later with ≤1 hit per source.
