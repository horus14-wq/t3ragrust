# t3rag — Improvement Plan, Round 2: the Agent-Experience Architecture

> Round 1 was about making the *engine* fast and the *surface* broad.
> Round 2 asks a different question: **when an AI agent — not a human —
> is the primary user, what does the system owe that agent?**
>
> The answer turned out to be a tower of abstractions, and this document
> is organized as that tower. Every layer below states what exists
> (verified against the code, not assumed), what is missing, and what
> to build — in the order an agent actually needs it.

---

## 0. The thesis

An agent using t3rag is not "a user running a CLI." It is a mind with a
finite context window, a clock, and no prior knowledge of this machine,
reaching into a memory of other minds' reasoning. Every interaction
should be evaluated against three tests:

1. **Orientation** — can the agent figure out what exists and what is
   relevant *before* spending a retrieval call?
2. **Legibility** — does every response tell the agent what happened,
   how confident the system is, and what to do next — without a second
   round-trip?
3. **Accretion** — does each interaction make the *next* interaction
   (by this agent, another agent, or a future session) cheaper and
   better?

The current system is ~80% built for tests 1–2 at the protocol level
(`--format toon/json`, `capabilities`, `health`, `--explain`) but has
almost nothing for test 3, and its middle layers — query formulation,
result hydration, corpus routing — assume a human who already knows
what the corpora contain. That assumption is the thing to remove.

**The tower** (bottom = foundation, top = compounding value):

```
L6  Distribution & onboarding   install · skill · MCP protocol · docs
L5  Memory & accretion          query log · feedback · insights · eval history
L4  Result legibility           formats · explain · confidence · grouping · hydration
L3  Query craft                 suggest · expand · like · exclude · batch · intent
L2  Models & runtime            registry · EPs · cache · presets          [BUILT]
L1  Corpus resources            index · list · stats · verify · export    [MOSTLY BUILT]
L0  Existence & trust           health · selftest · doctor · capabilities [BUILT]
```

L0–L2 are solid. L1 is missing `verify`/`show`/`pull`/`publish`. L3–L5
are where the agent experience is won or lost. L6 is beaded but
unbuilt. Everything below is organized by layer.

---

## 1. The agent's journey — where the friction actually lives

Walk through a realistic task: *"I'm an agent implementing a CRDT
OR-set. Before writing code, check whether prior reasoning traces can
help."*

| Step | What the agent does today | Friction |
|---|---|---|
| 1. Orient | `t3rag health` → `list` | `list` shows name/docs/dims/size — **not what the corpus is about**. With `struct`, `reflect`, `semantic`, `raw` corpora present, which do I query for "how do I implement X" vs "what pitfalls exist"? I guess, or I search all and hope. |
| 2. Formulate | Write a query from my task context | My context is a paragraph, not a query. `--also` lets me add phrasings but *I* must invent them. No pseudo-relevance feedback, no query assist. |
| 3. Retrieve | `search --format toon` | Fast, compact, good. Scores shown but **scale unlabeled** — is `s=0.034` good? RRF-fused vs rerank-logit are different scales; the response doesn't say which produced the number. |
| 4. Inspect | See `[3] id=87 s=0.041 … ~2400c` | Promising hit is truncated. **There is no way to fetch passage 87 in full.** `export` dumps 52K passages; `--format json` may still truncate; `ask` truncates too. Dead end. |
| 5. Iterate | "That hit was good — more like it" / "not that one" | Impossible. I must reformulate a query and hope the ANN lands nearby. No `--like`, no `--exclude`, no session memory — I can't even see what I already searched. |
| 6. Multi-corpus | `--corpus all` | Works, hits tagged by corpus. But struct+reflect+semantic describe **the same problems** — I get 3 near-duplicate hits for one problem in different views, presented as unrelated results. The system's best property (three complementary views of one trace) is invisible. |
| 7. Trust | Decide whether to keep reading | No confidence signal. No "results are weak — reformulate" guidance. No staleness/corruption check on the index I just searched. |
| 8. Learn | Finish task, move on | Nothing recorded. Next session (mine or another agent's) starts from zero: same orientation cost, same dead ends, same guesses. The system does not accrete. |

Every friction point above is fixable, most of them cheaply. The rest
of this document is the fix list, organized so each layer makes the
layers above it more useful.

---

## 2. L3 — Query craft (the highest-leverage layer)

This is where "least expenditure of resources" is won: a better query
costs one embed (~5ms warm) instead of ten bad searches.

### 2.1 `t3rag suggest "<problem>"` — corpus routing before searching

**The problem:** corpora are opaque. An agent cannot know which corpus
holds relevant reasoning without sampling or wasting searches.

**The fix:** at index time, compute and store a **corpus aboutness
manifest**: detected transform type (`struct`/`reflect`/`semantic`/`raw`
patterns are trivially recognizable from output format — and this is
the primary routing signal for the T3 corpus family), top question
terms, doc-length histogram, language guess. Store in `meta.json` v2.

A **corpus centroid** (mean of passage embeddings — free, vectors are
already computed) is a *secondary* signal at best: for the T3 corpora
the views cover the *same* problem space, so centroid similarities
across corpora will be near-uniform and uninformative — the transform
type and term stats carry the routing value. Keep the similarity
number, but never rank by it alone; surface type+stats first.

Then (type + stats lead; similarity is a hint, not the ordering key):

```
$ t3rag suggest "implementing a CRDT OR-set with tombstone pruning"
corpus           similarity  notes
semantic         0.81        concept-level traces, 52.7k docs
struct           0.74        step-by-step cheatsheets
reflect          0.38        failure patterns — try for pitfalls
raw              0.29        full untransformed traces
→ suggest: --corpus semantic,struct  (or all for max recall)
```

One embed + N dot-products = ~10ms. The agent goes from "guess or
search all" to a routed first query. For the T3 corpora specifically,
`suggest` doubles as **intent detection**: "what could go wrong with X"
routes to `reflect`; "how do I approach X" routes to `struct`;
"what's the core idea of X" routes to `semantic`.

- Files: `src/index/mod.rs` (centroid accumulation during embed pass —
  nearly free), `src/store` (meta.json v2 schema), new `src/suggest.rs`,
  `paths::corpus_dir` unchanged.
- Also expose `GET /corpora` enriched with centroid+aboutness so MCP/HTTP
  clients get the same routing data.
- `--format json` returns machine-readable `{corpus, score, reason}` rows.

### 2.2 `--like <passage-id>` — more-like-this drill-down

**The problem:** a good hit is a beachhead; agents want to explore its
neighborhood (sibling approaches, related pitfalls) without inventing a
new query.

**The fix:** `search --like 87` loads passage 87's text, embeds it
(passage-side, ~10ms), runs dense-only search, returns neighbors with
the seed excluded. No index changes — the passage text is already in
`passages.bin`. Optionally `--like-text "..."` for ad-hoc seeds.

This compounds beautifully with `--exclude`: iterative neighborhood
exploration ("show me more like 87, but not 88 or 91") is exactly how
an agent systematically covers a topic.

### 2.3 `--exclude <id,id,...>` — iterative refinement

Suppress already-seen passage ids from results. Cheap set-subtraction
post-fusion. Combined with `--offset` (exists) and `--like`, gives
agents a complete "walk the result space" toolkit: search → inspect →
like/exclude → repeat, never seeing the same hit twice.

### 2.4 Pseudo-relevance feedback — `--expand` (or `--prf`)

**The problem:** agent queries are often vague ("OR-set tombstones").
Classic IR answer: assume top hits are relevant, expand the query with
their content, re-search. No generative model needed — pure embedding
math with tools already in the binary.

Implementation: after first-pass fusion, take top-3 hit texts, embed
each (passage-side), average with the query vector (weighted ~0.7/0.3),
re-run dense leg, merge. ~30ms extra warm. Return `expanded_terms`
(top distinctive terms from feedback passages) in `--explain` output so
the agent sees *what the system thinks the query meant* — legibility
even when expansion doesn't fire.

This is the single biggest recall lever available without an LLM.
Make it opt-in (`--expand`) with an `auto` option later once eval shows
it helps on the T3 corpora.

### 2.5 `--group-by question|source|corpus` — make T3's structure visible

**The problem (§1.6):** multi-corpus T3 search returns the same problem
in 3 variants as unrelated hits. The corpus set's whole value —
complementary views of one trace — is invisible in flat results.

**The fix:** hits already carry `question` (the passage field) and
`corpus`. Group results by question (normalized): one group per problem,
variants nested inside showing corpus+score each. Output:

```
PROBLEM: Implement Hierarchical OR-Set with Recursive Stability
  [struct]   s=0.91  step-by-step approach
  [semantic] s=0.87  core idea: guarded wrapper around propagation
  [reflect]  s=0.62  pitfalls: lock-ordering deadlock, gc pauses
```

This is arguably *the* signature agent feature for this corpus family:
it converts "3 duplicate hits" into "one problem, three lenses." Works
for any corpus with a `question` field; degrades gracefully to flat
list when absent. `--format toon` gets a compact grouped variant;
`--format prompt` (ask) emits groups as labeled sections — an agent
injecting context sees structure, not a wall of passages.

### 2.6 `--token-budget N` — context-budgeted output

Agent context is the scarcest resource in the system. `--max-chars`
truncates per-passage but doesn't budget the *response*. Add
`--token-budget 2000` (chars≈tokens/4 heuristic or real tokenizer —
tokenizers crate already present): the renderer fills hits in rank
order until budget exhausted, truncating smartly — keep `Problem:` and
`Deliverable:`/`Answer:` sections intact, elide middles. For `ask`,
this guarantees the injected block fits the agent's window.

### 2.7 `--batch <file.jsonl>` — amortized multi-query

Agents routinely have N sub-questions. Today that's N process/HTTP
round-trips. `search --batch queries.jsonl --format jsonl` runs N
*independent* searches (distinct from `--also`, which fuses phrasings
into ONE ranking — keep that distinction crisp in help text) through
the warm daemon, emitting one JSON object per query. MCP gets a
`batch_search` sibling tool or a `queries[]` mode that returns separate
result sets (current `queries` field fuses — do not overload it).

### 2.8 `t3rag feedback` + query log — the accretion substrate

Two small append-only JSONL stores under the cache root:

- `~/.cache/t3rag/log.jsonl` — every search: `{ts, client(cli|mcp|http),
  query, corpora, top_ids, top_scores, took_ms, preset}`. Written by the
  daemon (single writer, no contention).
- `~/.cache/t3rag/feedback.jsonl` — explicit judgments:
  `t3rag feedback --query "..." --passage 87 --verdict relevant`
  (or `--ids 87,88,91 --verdict irrelevant`).

Then `t3rag log [--since 1h] [--corpus X] [--query-regex R]` lets an
agent review *its own search history* — "did I already try this?",
"which queries keep failing?" — and `t3rag insights` aggregates:
per-corpus usage, zero-result rate, feedback verdicts, coverage gaps
("queries matching /or-set|crdt/ returned <3 hits in 80% of cases").

This is the foundation everything in L5 builds on. Cheap: two
`writeln!` calls and a reader.

---

## 3. L4 — Result legibility

### 3.1 `t3rag show <corpus> <passage-id>` + `GET /passages/{corpus}/{id}`

**The §1.4 dead end.** Single-passage hydration: full text, all
metadata (question, source, title, parent linkage for chunks), score
context optional. CLI + HTTP + MCP tool (`get_passage`). ~50 lines —
`PassageReader::get` already exists for hydration; this just exposes it.

### 3.2 Structured, self-describing responses

Every `--format json` response gains a `meta` block (always on, not
opt-in like `--explain`):

```json
"meta": {
  "corpora": ["semantic","struct"], "model": "Qwen3-Embedding-0.6B-ONNX",
  "dims": 1024, "quant": "i8", "preset": "balanced",
  "path": "daemon", "took_ms": {"embed": 4.2, "dense": 1.1, "bm25": 2.8,
                               "rerank": 31.0, "hydrate": 0.4},
  "index_age_h": 36.2, "warnings": ["results_truncated", "corpus_stale?"]
}
```

The agent never needs a second call to know *what answered* and *how*.
`warnings[]` is where honesty lives: truncation, staleness, low
confidence, dedup collapses, EP fallback to CPU.

### 3.3 Confidence bands — honest uncertainty

Fused scores are unlabeled scales. Add `confidence: high|medium|low|none`
per response, derived from signals already computed: top-score absolute
band + rank-1↔rank-2 gap + dense/BM25 leg agreement (both legs surfacing
a hit is stronger evidence than one). Calibrate the thresholds per
corpus via `t3rag eval` on a reference query set; store in meta.json.
An agent that sees `confidence: low` knows to reformulate or `--expand`
rather than hydrate useless text — direct resource savings.

> **Caveat (round-2 review):** this is the shakiest item in the plan —
> RRF and rerank-logit scores are not calibrated scales and may never
> carry a meaningful absolute band across heterogeneous corpora. Leg
> agreement and rank-gap are the more honest signals; if eval
> calibration shows the bands don't separate, ship `warnings[]` hints
> (leg-disagreement, narrow rank-gap) instead of a scalar confidence.

### 3.4 Structured errors everywhere

`--format json` failures currently emit anyhow prose. Emit instead:

```json
{"error": {"code": "corpus_not_found", "message": "...",
           "hint": "available corpora: semantic, struct, reflect, raw",
           "retryable": false}}
```

with exit code 2. Include `did_you_mean` for near-miss corpus names.
Agents self-correct from structured errors dramatically better than
from prose — this is cheap, high-yield ergonomics.

### 3.5 Provenance chain closure

`source_idx` exists in the T3 JSONLs but the ingest reader doesn't map
it — passage `source` degrades to `file:line`. Map `source_idx` →
`source` (or a dedicated field) at ingest; `show` then displays the full
chain: `passage 4812 ← f5_v2_t3_struct.jsonl ← source trace #12345`.
Traceability is what makes retrieved reasoning *citable* in an agent's
own output.

---

## 4. L5 — Memory & accretion (the missing layer)

The system currently has perfect amnesia. The proposals in §2.8
(`log`, `feedback`) are the substrate; this layer is what compounding
looks like on top:

- **`t3rag insights`** — aggregate the log+feedback: per-corpus query
  volume, zero-result queries (coverage gaps), verdict distributions,
  which `--also`/`--expand`/`--like` features get used (feature
  telemetry → tells us what to invest in). *Value is speculative until
  real query volume exists — build `log`+`feedback` first, let the data
  justify the aggregator.*
- **`t3rag eval --history`** — auto-promote logged queries with
  `feedback=rel` verdicts into eval query sets (`{query, relevant_ids}`
  format already supported). Real usage becomes the regression suite —
  the eval set *is* what agents actually ask, not what we imagined.
  *(Same caveat: gated on feedback volume.)*
- **Corpus health score** — `stats` gains: index age vs newest source
  mtime (staleness), dedup ratio, median length, zero-result rate from
  the log. `list` shows a ⚠ on stale corpora.
- **Query-pattern memory for the skill** — `insights` output can be
  pasted into SKILL.md periodically: "agents searching X usually mean Y"
  — the tool's documentation accretes from real usage.

None of this requires new infrastructure — two JSONL files and an
aggregator. It is the difference between a tool and a *practice*.

---

## 5. L1 — Corpus resources (finish the lifecycle)

Mostly built (`index`, `list`, `stats`, `export`, `doctor`). Missing:

- **`t3rag verify <corpus>`** — integrity check: doc_count vs store
  records, dims vs meta.json, format_version vs binary, mmap sanity,
  optional vector spot-check vs re-embed. `doctor --deep` runs it on
  all corpora. An agent should never silently search a corrupt index.
- **`t3rag show`** (§3.1) doubles as the corpus-content inspector.
- **`pull`/`publish`** — already beaded (`t3-css.3.9`).
- **`forget`** — *resolved as a documented skip* (`t3-css.4.5` closed):
  a single-user cache's correct security model is user-run `rm`; a
  tool-level delete would only add a new deletion primitive. `stats`
  prints the exact `rm -rf <path>` recipe and `doctor` hints at it.
- **Corpus manifests v2** — the §2.1 aboutness block (centroid, top
  terms, detected transform type, stats) lives in meta.json; `list
  --json` and `GET /corpora` expose it. This is what makes `suggest`
  and `list` genuinely routable.

---

## 6. L6 — Distribution & onboarding

Beaded (`t3-css.4.12` release pipeline, `t3-css.4.11` CI matrix,
`t3-css.3.10` DirectML). Additional agent-specific items:

- **SKILL.md as a playbook, not a flag list.** The installed skill
  should teach the *workflow*: orient (`health`→`list`→`suggest`) →
  formulate → retrieve (`toon`) → inspect (`show`) → iterate
  (`like`/`exclude`/`expand`) → hydrate → cite. Plus query craft for
  trace retrieval: "query the problem domain, not your solution";
  "for pitfalls, query failure language ('common mistakes', 'what goes
  wrong')"; "scores are relative within a result set — never
  threshold absolutely; check `confidence`". Include 2–3 worked
  examples. The skill is how an agent learns to drive well — invest
  disproportionately here.
- **MCP `prompts` + `resources`.** The hand-rolled server currently
  does tools-only. MCP `prompts/list`+`prompts/get` can serve the same
  playbook *through the protocol* — the calling agent gets taught
  in-band. `resources/list` can expose corpus manifests as
  `t3rag://corpus/{name}`. Bump the declared protocol version while
  touching it (round-1 finding: stale `2024-11-05`).
- **MCP tools parity.** Add `list_corpora`, `get_passage`, `suggest`
  alongside `search_traces` — mirror the CLI gaps closed above.
- **`t3rag doctor --fix`** — where safe (rebuild stale index hint,
  clear corrupt lock files), offer remediation, not just diagnosis.

---

## 7. What NOT to build (honest scoping)

- **No generative query rewriting.** Would need an LLM dep — violates
  the pure-Rust/no-API-key invariant. PRF (§2.4) captures most of the
  benefit mechanically.
- **No vector DB / graph store.** usearch+tantivy+JSONL logs suffice;
  accretion data is kilobytes, not a database.
- **No automatic corpus downloads on search.** `suggest` may *name* a
  corpus it saw on HF metadata, but fetching is always explicit
  (`pull`) — agents shouldn't trigger multi-GB downloads implicitly.
- **No result caching across queries** beyond the existing embed LRU —
  correctness risk for near-zero gain at this scale.
- **No per-user cloud anything.** Local-first is the product.

---

## 8. Priorities

| Pri | Item | Layer | Effort | Why first |
|---|---|---|---|---|
| P0 | `show`/`get_passage` (CLI+HTTP+MCP) | L4 | S | Dead-end fix; unblocks all iteration |
| P0 | Structured JSON errors + `meta` block + warnings | L4 | S | Every response self-describing |
| P0 | `--like` + `--exclude` | L3 | S | Iteration toolkit; pure index reuse |
| P0 | `--group-by question` | L4 | S-M | T3's killer feature made visible |
| P0 | Query log + `t3rag log` | L5 | S | Accretion substrate; everything else builds on it |
| P1 | `suggest` + corpus manifests (type+stats lead, centroid secondary) | L3/L1 | M | Corpus routing — kills the guess-work |
| P1 | `--expand` PRF | L3 | M | Biggest recall lever, no LLM needed |
| P1 | `feedback` + `insights` | L5 | M | Closes the learning loop; `insights` gated on real volume |
| P1 | `--token-budget` | L4 | S-M | Context is the scarce resource |
| P1 | `--batch` | L3 | S-M | Amortizes daemon warmth |
| P1 | SKILL.md playbook + MCP prompts/resources | L6 | M | Teaching beats documenting |
| P2 | Confidence bands (eval-calibrated) | L4 | M | Weakest item — may collapse into `warnings[]` hints if bands don't calibrate |
| P2 | `verify` + staleness in stats | L1 | S-M | Trust layer |
| P2 | Provenance `source_idx`→`source` | L1 | XS | Traceability closure |
| P2 | `eval --history` | L5 | S | Real-usage regression suite |
| P3 | Intent-aware corpus priors | L3 | M-L | On top of suggest |
| — | pull/publish, release matrix, DirectML | L1/L6 | beaded | t3-css.3.9/4.11/4.12/3.10 (4.5 closed: skip, rm recipe) |

(S ≈ <1 day, M ≈ 1–3 days, L ≈ week+; all estimates assume the current
daemon+CLI architecture holds.)

---

## 9. The north star

The end state this plan builds toward: an agent drops into any
project, runs `t3rag health && t3rag suggest "<its problem>"`, gets a
routed corpus recommendation in ~10ms, searches with `--format toon`,
sees results *grouped by problem* with honest confidence labels,
drills into a hit with `show`, explores the neighborhood with
`--like`/`--exclude`, expands a vague query with `--expand`, and every
step lands in a log that makes the *next* agent's orientation cheaper.
The corpora stop being three files and become what they were always
meant to be: **a shared institutional memory of how hard problems get
solved — queryable, citable, and compounding.**

That is the synthetic system: not CLI + daemon + indexes, but a
*legible memory organ* for a fleet of agents — where every layer
exists to make the layer above it more truthful, cheaper, and smarter.
