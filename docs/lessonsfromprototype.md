# Lessons from the Python prototype (still apply to `t3rag`)

Source of these lessons: the swarm-memory **Think-RAG V1** prototype (`BM25` shards, optional `e5-base` + FAISS, Fable/MoreThought corpus work, query experiments, T3 prompt side-by-sides). That stack’s **mechanics** (pickle shards, e5-512 chunking, `python V1/cli.py`) are superseded by this Rust tool. The **corpus judgment, query craft, and retrieval strategy** below still apply.

Companion in this repo: `docs/raw_vs_t3_ingest.md` (ingest recipe). Stock T3 prompts and full side-by-side examples are included below (§12–§13).

---

## 1. What the prototype was trying to do

On-demand retrieval of **past reasoning / agent trajectories** to help an LLM mid-task — not a 24/7 swarm memory, not “search every turn.” Same product intent as `t3rag`: inject a few useful passages, then continue coding.

The paper (*RAG over Thinking Traces*) showed gains on contest-style math/science/code with continuous CoT traces, and that **raw chunked traces** often helped almost as much as T3 transforms. That result **does not automatically transfer** to Fable-style agent dumps or to mid-project product engineering without care about **what** you index and **how** you query.

---

## 2. Domain fit beats format fidelity

### Seed authoring vs general coding

Several MoreThought/Fable dumps are rich as **agent logs** (tools, thoughts) but weak as **general SE mentors**:

| File (prototype ranking) | Domain reality |
| --- | --- |
| `train.jsonl` / `Fablev3` | Almost entirely **seed authoring** (Wave-/SEED_SPEC / fable-code catalog) |
| `fable5-v1` | Seed-heavy, some real SE |
| `Fable5-v2` | **Mixed** — much less seed; coding, science, vague prompts, long ASSISTANT prose |
| Omni Frontier coding parquets | Coding Q&A / prose solutions at huge scale — **wrong shape** for “trajectory RAG,” better as a different index |

**“Best structured dump” ≠ “best for your Reddit-clone / product task.”**  
`train.jsonl` won on tool_calls + `<thought>` tags; it lost on transfer when the agent was authoring tiny training seeds.

### What’s “bad” about seed examples

Not that they’re corrupt — that retrieval neighbors share **authoring boilerplate and seed craft**, not your product architecture. Stripping “I am authoring seeds” helps; it does **not** invent large-system design taste that was never in the trace.

### When seed craft *is* useful

After stripping meta, many episodes still teach normal micro-skills: pin an API contract, fail-closed harness, plant a bug, patch, smoke gate, small HTTP/CSV clients. Useful **mid-feature** when the live problem looks like that; weak for blank-page product design.

### Mid-project vs greenfield

- **Greenfield** (“design a Reddit clone”): these corpora underperform; hits drift to vaguely related SE or seed workflows.
- **Mid-project** (this error, this API, this harness failure): **much better match** to what chunked traces and Omni-style prose actually contain. Prefer sharp local queries over mega “build the whole app” queries.

---

## 3. Prefer a filtered coding subset (Fable5-v2)

Prototype script: `scripts/convert_fable5_v2_filtered.py` →  
`fable5_v2_filtered_coding.jsonl` (~54k kept from 85k).

What it did (Python-only heuristics — still the right *kind* of preprocess for Rust ingest):

- Keep coding / SE-ish episodes; drop obvious off-domain fluff
- Drop or heavily strip seed-authoring
- **Lean questions** (strip Wave-/SEED preambles; keep catalog/task ask)
- Strategy-B style body (`text`): thoughts / actions / truncated OBS / ASSISTANT prose
- Scrub remaining seed self-talk lines from the body
- Never modify the original parquet

**For `t3rag`:** treat that JSONL (or an equivalent) as the canonical **raw** corpus input. Re-running T3 should be on this filtered set, not on raw Wave-4 dumps.

Keyword filters are imperfect (some physics-with-`function` rows slip through). Tune with samples; don’t expect purity without an LLM judge.

---

## 4. Chunk fields: what “good” looks like

Prototype pain: every chunk’s retrieval string was often  
`Question: {4k Wave preamble}\n\n{trajectory}`  
so BM25/e5 matched **SEED_SPEC / Wave-N neighbors** instead of the real task.

**Good (still the bar for Rust JSONL):**

| Field | Role |
| --- | --- |
| `question` | Short task label (hundreds of chars, not thousands of preamble) |
| `text` | Episode body worth retrieving (actions, errors, ASSISTANT solution) |
| `source` | Stable id shared across raw + T3 rows for the same episode |
| `title` | Optional type tag (`raw`, or leave empty and use corpus name) |

Display can keep a fat original prompt elsewhere; **retrieval keys must stay lean.**

### Rust-specific note (does not undo the lesson)

In `t3rag`, **dense embeds `text` only**; **BM25** searches `text` + `title` + `question`. So:

- Still put the lean ask in `question` and the body in `text`.
- Do **not** assume Python V1’s `indexed_text = Question + chunk` is required.
- If you want dense to see the topic, you may prepend into `text` deliberately — optional.

See `docs/raw_vs_t3_ingest.md`.

---

## 5. Raw vs T3 (strategy, not just formats)

### Stock T3 prompts

From the paper repo: `t3_struct`, `t3_reflect`, `t3_semantic` — fill `{trace}` only. Outputs end with `Answer: $\boxed{...}$` (math-contest residue).

Usable on coding JSONL **as-is**; better with light edits for coding:

- Inject `{question}` into the prompt
- Replace `\boxed{final answer}` with a one-line **Deliverable** / approach summary
- Let `reflect` say `N/A — solution-only trace` when there were no failed attempts (common on ASSISTANT prose)

### Thinking-tag vs prose solutions

| Kind | Raw form | T3 effect |
| --- | --- | --- |
| `<thought>` diaries (e.g. seed authoring) | Process, order, false starts | **Large reshape** into procedure / pitfalls |
| Omni / v2 **prose solutions** | Already a tutorial | **Smaller** — mostly compresses |

T3 is not magic content creation. It polishes **reasoning search**; on polished answers it mostly shortens.

### Is T3 worth it for real coding LLM work?

- **Full paper-T3 over unfiltered Fable/Omni:** low ROI for product SE.
- **Selective T3** on filtered coding rows (especially traces with struggle → fix): plausible upside.
- **Higher ROI than blind T3** in the prototype era: lean fields, filtered corpus, sharp queries, hybrid lexical+dense — all of which `t3rag` already targets in-engine.

Paper “raw ≈ T3” held on clean contest CoT. On agent/prose coding data, **unit of retrieval and domain** matter more than transform polish.

### Using all types together

Index raw + struct + reflect + semantic if you want — **storage is fine**.

**Undifferentiated top‑k across all types causes problems:** near-duplicates from the same episode fill the list; mixed advice (steps + mistakes + insight) blurs the agent; scores aren’t calibrated across passage styles.

Prefer:

- **Named corpora** per type (`fable5_v2_raw`, `_struct`, …), or
- One pool with type in metadata/`title`, then **diversify** (≤1 hit per `source`, or enforce a mix)

Query by intent: mid-debug → raw (+ BM25-heavy hybrid); planning → struct/semantic; “what goes wrong?” → reflect.

---

## 6. Query writing (still the highest-leverage skill)

Prototype Reddit-clone tests: **mega-queries fail**; **sharp subproblems** retrieve better.

Write queries like:

- one failing symptom, API, or constraint
- “PowerShell ValidateSet + splatting,” not “build my whole app”
- refine after a miss (second sharper query)

This remains true with Qwen3 embeddings + tantivy + rerank. The embedder and hybrid stack help; they do not fix a vague query.

Cold vs warm latency differs in Rust (daemon), but **query quality** still dominates hit usefulness.

---

## 7. Lexical vs dense (intent still applies)

Prototype:

- **BM25** stronger on exact jargon, flags, error strings
- **e5** more semantic; could drift to “nearby SE essay” content, especially with fat shared questions

`t3rag` fuses dense (usearch) + BM25 (tantivy) with RRF and optional rerank — so you rarely pick one backend manually. The **lesson** remains: exact identifiers still need lexical signal; semantic similarity alone will happily match preamble-ish or tutorial-adjacent neighbors if the corpus is noisy. Clean fields + sharp queries reduce that; rerank helps precision on the final top‑k.

---

## 8. Preprocess: Python vs LLM

| Goal | Tool |
| --- | --- |
| Filter, strip preamble, truncate OBS, emit `{question,text}` | **Python (or any deterministic script)** |
| T3 struct/reflect/semantic | **LLM** (offline batch) |
| Recover deleted `tool_calls` from Fablev3 | Don’t — use a better source dump |

Fablev3 is a poor T3 *and* raw candidate (empty assistants, no tool_calls, no thought tags). Prefer filtered v2 (+ optional thoughts-only extracts elsewhere) over “fix v3 with an LLM.”

---

## 9. What from the prototype to **ignore** for Rust

These were real lessons for V1 but are **not** how `t3rag` should be operated:

- Rebuild mental model around `tokenized.pkl` / 20k-chunk BM25 shards / ~40s shard merge
- Default dense model `e5-base-v2` + FAISS FlatIP + Hub auth warnings
- Mandatory 512-token window + 64 overlap for every raw trace
- Skill instructions that only mention `python V1/cli.py`
- Assuming dense always sees `Question:` + body concatenated

Runtime truth: this repo’s `DESIGN.md`, `AGENTS.md`, `docs/raw_vs_t3_ingest.md`, and the `t3rag` CLI/MCP.

---

## 10. Practical checklist (carry into Rust corpora)

1. Start from **filtered coding** JSONL (lean `question`, clean `text`, stable `source`).
2. Index **raw** as its own corpus (whole-passage; structural chunk only if huge).
3. If you have T3 outputs, index **each transform as its own corpus** (or typed + diversified retrieval).
4. Agents: **sharp mid-project queries**; call on demand, not every turn.
5. Treat T3 as optional distillation — especially valuable on struggle-heavy traces; don’t expect it to fix a bad domain mix.
6. When hits look like Wave-/seed authoring, fix **fields and filters**, not the embedder first.

---

## 11. Artifact map (prototype → meaning here)

| Prototype artifact | Use in Rust world |
| --- | --- |
| `fable5_v2_filtered_coding.jsonl` | Primary **raw** ingest |
| T3 JSONLs from that file (`output`/`text`) | Per-type corpora; ingest already accepts `output` |
| This file (§1–§11) | Strategy lessons that survive the rewrite |
| This file (§12) | Stock T3 prompts + coding tweaks |
| This file (§13) | Thinking-tag vs prose side-by-side + illustrative T3 outputs |
| `docs/raw_vs_t3_ingest.md` | Concrete ingest recipe for raw vs T3 |

---

## 12. Stock T3 prompts (from the paper pipeline)

Copied from `t3-main/data_transform/prompts/`.  
`transform.py` substitutes `{trace}` with the JSONL `text` / `output` field only (question is not injected unless you change the runner).

### 12.1 t3_struct

```text
Convert the reasoning trace into a concise step-by-step cheatsheet.

Guidelines:
- Use at most 7 steps.
- Each step should represent a meaningful action.
- Keep explanations short and clear.
- Focus on reusable reasoning patterns.
- Remove failed or irrelevant attempts.
- If the trace explores multiple distinct approaches, produce one cheatsheet per approach separated by [PASSAGE].

Output format:

Problem: ...

Step 1: ...
...
Step N: ...

Answer: $\boxed{[final answer]}$

Given trace: {trace}
```

### 12.2 t3_reflect

```text
Extract failure patterns and negative knowledge from the reasoning trace.

Guidelines:
- Focus on common mistakes and misleading reasoning paths.
- Explain why these mistakes are tempting.
- Highlight how to detect and avoid them.
- Provide contrast with the correct approach.
- Do not reproduce the full solution.

Output format:

Problem: ...

Common Mistakes: ...

Misleading Intuitions: ...

Critical Checks: ...

Correct Approach (brief): ...

Answer: $\boxed{[final answer]}$

Given trace: {trace}
```

### 12.3 t3_semantic

```text
Compress the reasoning trace into progressively more abstract representations.

Guidelines:
- Provide three levels of abstraction.
- Pass 1: structured outline of reasoning steps.
- Pass 2: only key decisions or moves.
- Pass 3: a single core insight.
- Ensure consistency across all passes.
- Avoid redundancy.

Output format:

Problem: ...

Pass 1: ...

Pass 2: ...

Pass 3: ...

Answer: $\boxed{[final answer]}$

Given trace: {trace}
```

### 12.4 Coding-oriented tweaks

These prompts are usable as-is on `fable5_v2_filtered_coding.jsonl`. For coding-oriented data, consider:

1. Inject the lean question: `Problem (given): {question}` before `Given trace: {trace}`.
2. Replace `Answer: $\boxed{[final answer]}$` with something like `Deliverable: <one-line approach / API contract>`.
3. For `t3_reflect`, allow `N/A — solution-only trace` when there are no failed attempts (common on prose ASSISTANT answers).

---

## 13. Side-by-side examples (thinking-tag vs prose)

Real rows from the prototype databases. T3 outputs follow the official prompts above. Only `{trace}` was substituted (same as `transform.py`). Transforms below were produced manually with those prompts for illustration (not a GPT API batch run).

### 13.A — Thinking-tag style

| | |
| --- | --- |
| Source | `MoreThought-…/train.jsonl` |
| Row | `1790` |
| Kind | Concatenated `<thought>` / `<think>` bodies only (`n_thoughts=13`) |
| Full user prompt length | 8225 chars (Wave-8 seed-authoring preamble); search/display slice below |

#### A.input — question (catalog slice)

```text
Catalog item 1: project: java-salesforce-graphql-accountsts** — Java client for the current stable Salesforce Graphql endpoint: versioned URL, operation/variables body, UI API connection pagination, node/edge decoding, Graphql errors alongside partial values, and field-level-security nulls. Building on this, Catalog item 2: project: py-salesforce-bulk-ingestst** — Python Bulk API 2.0 ingest workflow: produce CSV job, upload bytes with correct content type/line ending, close job, poll documented states, download failed/unprocessed result Csvs, and provide idempotent recovery after ambiguous close responses. Catalog item 3: attribute: ts-salesforce-composite-treeee** — Typescript Composite API relationship writer: construct reference IDs and cross-record references, enforce petition collecti
```

#### A.input — text (full thought concatenation used as T3 `{trace}`)

```text
I'm the Wave-8 chunk 8F author. With that established, Let me start by reading the spec, the Wave-8 catalog, and checking the current state of the seeds path.

The four IDs are free. Now let me study existing Wave-8 seeds in the same languages as templates layoutt only).

Let me look at the docs fixtures and test-harness patterns from these template seeds.

Solid starting points. Now let me fetch the exact operation pages. Starting with Graphql overview/pagination and Bulk API job pages.

Bulk API contract pinned. Next: job states, Rest query resource, Composite API, and the Graphql endpoint/errors pages.

Research complete — marking task 1 done and starting with`py-salesforce-bulk-ingest`.

Now the acceptance assess harness for the Bulk API seed.

Now the task.json with the fail-closed prompt.

Now run the smoke gate baselinee must fail), after that materialize the workspace and craft the reference implementation.

Now the existing working library`tree_client.ts`baselinee code that must keep working).

Now the acceptance validate artifact with regression + new-feature suites.

Baseline fails as expected. Then we'll Now the reference implementation of the new attribute.

All 9 tests pass. Saving the reference patch and re-running smoke.

Now the big protected harness`TestMain.java`.

Now the starter `Json.java`utility the implementer can use unprotectedd).

Baseline fails with compile bug as intended. Writing the reference implementation.

All 6 tests pass with 87 assertions. On that note, Saving the reference patch.
```

#### A.t3_struct

```text
Problem: Author Wave-8 Salesforce API seeds (GraphQL / Bulk ingest / Composite) with failing baselines, harnesses, and reference patches.

Step 1: Confirm seed IDs are free; read SEED_SPEC and same-language Wave-8 templates.
Step 2: Pin external API contracts from docs (endpoints, job states, pagination, errors).
Step 3: Scaffold each seed: task.json, starter code, protected acceptance harness.
Step 4: Run smoke/baseline and verify it fails for the planted reason.
Step 5: Implement the reference solution in a workspace copy.
Step 6: Re-run tests until green; save reference_fix.patch.
Step 7: Repeat per catalog item; keep domains/IDs from colliding.

Answer: $\boxed{spec→docs→failing seed→reference→patch}$
```

#### A.t3_reflect

```text
Problem: Multi-seed Salesforce API authoring under Wave-8 constraints.

Common Mistakes:
- Starting implementation before IDs/templates/docs are pinned.
- Writing a green harness before the baseline fails for the intended bug.
- Re-skinning an existing seed domain/ID instead of a new catalog item.

Misleading Intuitions:
- “I know Bulk/GraphQL well enough” → silent contract drift (content-type, job states, FLS nulls).
- “Tests passing early is good” → you may have weakened the planted failure.

Critical Checks:
- Baseline fails for the documented reason before reference work.
- Docs match the exact API version in task.json.
- Seed ids unused; template layout copied, not business logic.

Correct Approach (brief): Lock contracts and failing harness first; only then write the reference and patch.

Answer: $\boxed{fail-closed harness before reference}$
```

#### A.t3_semantic

```text
Problem: Produce verified Salesforce-API training seeds with patches.

Pass 1: Free IDs → study templates → pin API docs → build failing seed → reference → green tests → save patch; repeat.
Pass 2: Pin contracts; enforce fail-then-fix; emit patch per seed.
Pass 3: Dataset seeds are only trustworthy if the baseline fails for a real planted bug.

Answer: $\boxed{planted failure makes the seed}$
```

### 13.B — Prose solution style

| | |
| --- | --- |
| Source | `MoreThought-…/Omni Frontier - full_clean__v2__coding.parquet` |
| Row | `957` |
| Kind | Finished tutorial-style answer (no `<think>` block) |

#### B.input — question

```text
I want to know how to generate an integer value from a user-input string in Python, which can then be used as a seed for random number generation in code. How can this be implemented?
```

#### B.input — text (full response used as T3 `{trace}`)

~~~~text
In computational logic, seeding a Pseudo-Random Number Generator (PRNG) with a string requires a process called **Deterministic Mapping**. Since PRNG algorithms (like the Mersenne Twister used in Python) require an integer or a byte-sequence as an initial state, we must transform the variable-length string into a fixed-width numeric representation.

### 1. The Native Python Approach
Python's built-in `random` module is highly abstracted. If you pass a string directly to `random.seed()`, Python internally hashes the string to generate an integer for you.

```python
import random

user_input = "StellarExplorer_99"
random.seed(user_input)

print(random.randint(1, 100))  # This will always produce the same result for this string
```

### 2. Manual Implementation: Hashing
If you are building your own PRNG or need to ensure the seed remains identical across different Python environments (where Python's built-in `hash()` might vary due to hash randomization), you should use a cryptographic hash like **SHA-256**.

#### The Logic
1.  **Encode**: Convert the string to bytes (UTF-8).
2.  **Hash**: Pass the bytes through a hashing algorithm to get a hexadecimal digest.
3.  **Convert**: Transform the hex digest (base-16) into a large integer (base-10).

#### Implementation

```python
import hashlib
import random

def string_to_seed(input_string):
    # 1. Create a SHA-256 hash object
    hash_object = hashlib.sha256(input_string.encode('utf-8'))
    
    # 2. Get the hexadecimal representation
    hex_dig = hash_object.hexdigest()
    
    # 3. Convert hex to a large integer
    seed_int = int(hex_dig, 16)
    return seed_int

# Usage
user_str = input("Enter a string to use as a seed: ")
seed = string_to_seed(user_str)

random.seed(seed)
print(f"Generated Seed Integer: {seed % 10**8}...") # Showing a truncated version
print(f"First Random Number: {random.random()}")
```

### 3. Comparison of Methods

| Method | Reproducibility | Complexity | Use Case |
| :--- | :--- | :--- | :--- |
| **Direct `random.seed(str)`** | High (within same session) | Low | General scripting and games. |
| **`hash(str)`** | Low (changes per restart) | Medium | Temporary session-based entropy. |
| **`hashlib` (SHA-256)** | Absolute (cross-platform) | High | Scientific simulations or cross-language tools. |

### Why use an Integer Seed?
PRNGs are not actually random; they are mathematical sequences. The **seed** determines the starting point in that sequence. By converting a string to an integer, you ensure that as long as the input string is the same, the "random" output will be perfectly reproducible, which is essential for debugging, procedural generation in gaming, and scientific verification.
~~~~

#### B.t3_struct

```text
Problem: Map a user string to a stable integer PRNG seed in Python.

Step 1: Decide if you need cross-run / cross-platform stability.
Step 2: For casual use, call random.seed(user_string) and let Python hash internally.
Step 3: For stable seeds, UTF-8-encode the string.
Step 4: SHA-256 the bytes; take hex digest.
Step 5: Convert hex to int (base 16); pass that to random.seed.
Step 6: Avoid built-in hash() for seeds — it is randomized per process.
Step 7: Verify reproducibility by checking the first randint/random() matches across runs.

Answer: $\boxed{SHA-256(string)→int→random.seed}$
```

#### B.t3_reflect

```text
Problem: String → integer seed for Python random.

Common Mistakes:
- Using hash(s) as a seed (changes every process).
- Assuming random.seed(s) is identical across languages/runtimes.
- Truncating the hash carelessly and colliding seeds.

Misleading Intuitions:
- “Any int from the string is fine” → weak/unstable mapping breaks debugging and procedural gen.
- “Crypto hash is overkill” → SHA-256 is the simple portable answer.

Critical Checks:
- Same string ⇒ same first random value after restarts and on another machine.
- Encoding fixed (UTF-8).

Correct Approach (brief): Prefer hashlib.sha256 → int(hex, 16) → random.seed for portable reproducibility.

Answer: $\boxed{don't use hash(); use hashlib}$
```

#### B.t3_semantic

```text
Problem: Deterministic string seeding for PRNGs.

Pass 1: Need int seed → either trust random.seed(str) or hash bytes to int via SHA-256 → seed PRNG → confirm repeatability.
Pass 2: Stable seed ⇒ cryptographic hash to int; skip per-process hash().
Pass 3: Reproducible “randomness” is a fixed map from string → PRNG state.

Answer: $\boxed{stable hash makes a stable seed}$
```

### 13.C — What to notice

| | A thinking-tag | B prose solution |
| --- | --- | --- |
| Raw form | Short internal monologue / next-action diary | Polished how-to with code and table |
| Domain here | Fable seed-authoring workflow | Actual coding Q&A |
| T3 effect | Large reshape into procedure / pitfalls / insight | Mostly compresses an already-structured answer |
| Best use of T3 | When the trace has process and false starts | When you mainly want shorter retrieval cards |

---

**Bottom line:** The prototype proved that **corpus hygiene, lean questions, sharp queries, and typed raw/T3 usage** matter more than which language builds the index. Keep those. Let `t3rag` own embedding, hybrid search, chunking policy, and serving.
