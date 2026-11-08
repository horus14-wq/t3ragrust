# Stock T3 Transform Prompts

The prompts used to generate T3 (thinking-trace) corpora from raw reasoning
traces — copied verbatim from the paper pipeline
(`t3-main/data_transform/prompts/`) via `docs/lessonsfromprototype.md` §12.

`transform.py` substitutes `{trace}` with the JSONL `text` / `output` field
only — the question is **not** injected unless you change the runner.

Three transform families; a typical corpus build produces one output per
transform per trace.

---

## t3_struct — step-by-step cheatsheet

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

## t3_reflect — failure patterns / negative knowledge

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

## t3_semantic — progressive abstraction

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

---

## Coding-oriented tweaks

The stock prompts work as-is on `fable5_v2_filtered_coding.jsonl`-style
rows. For coding-oriented data, consider:

1. Inject the lean question: `Problem (given): {question}` before
   `Given trace: {trace}`.
2. Replace `Answer: $\boxed{[final answer]}$` with
   `Deliverable: <one-line approach / API contract>`.
3. For `t3_reflect`, allow `N/A — solution-only trace` when there are no
   failed attempts (common on prose ASSISTANT answers).

---

## Notes for t3rag ingestion

- `[PASSAGE]` in `t3_struct` separates approaches — each `[PASSAGE]` block
  becomes a retrievable unit when the corpus is indexed.
- Retrieval keys stay lean: the `question` field is what queries match
  against; keep it short and topical (see `docs/lessonsfromprototype.md`
  §9–§11 for query craft).
- After generating traces, ingest per transform family so the lenses stay
  separable: `t3rag index --input t3_struct.jsonl --name struct` etc.
  `--group-by question` then shows the same problem across lenses.
- Side-by-side worked examples (raw vs each transform) are in
  `docs/lessonsfromprototype.md` §13.
