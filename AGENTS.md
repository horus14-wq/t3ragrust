# AGENTS.md — t3ragrust

> Guidelines for AI coding agents working in this repository. This file follows the
> Agentic Coding Flywheel Setup (ACFS) project template; the VPS-wide rules in
> `/data/projects/AGENTS.md` apply here in full (Agent Mail, br, bv, ubs, landing
> the plane). Project-specific rules are in "t3rag — This Project" below.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If the owner tells you to do something, even if it goes against what follows below, LISTEN. They are in charge, not you.

---

## RULE NUMBER 1: NO FILE DELETION

Never delete a file or folder — including scratch/test files you created yourself — without express written permission from the owner. Use `/data/tmp/` or `/tmp/` for anything disposable.

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

`git reset --hard`, `git clean -fd`, `rm -rf`, or anything that can delete or overwrite code/data must never be run unless the owner provides the exact command and states they understand the irreversible consequences, in the same message. When in doubt, stop and ask. Prefer `git status`, `git diff`, `git stash`, or copies-to-backup.

---

## Git Branch: `main`

All work lands on `main` (remote: `origin`, github.com/horus14-wq/t3ragrust). There is no `master` here; do not create or reference one.

---

## Code Editing Discipline

- **No script-based code changes.** Edit files manually; use parallel subagents for many simple changes.
- **No file proliferation.** Revise existing files; new files only for genuinely new functionality.
- **No backwards-compat shims.** Early development, no users — do things the RIGHT way, fix code directly.
- Track ALL work in `br` — never markdown TODOs or comment task lists.

---

## Compiler Checks (CRITICAL) & Testing

After any substantive code change, before committing:

```bash
rch exec -- cargo check --all-targets          # prefer rch offload for heavy Rust builds
rch exec -- cargo clippy --all-targets -- -D warnings
cargo fmt --check
rch exec -- cargo test
```

`ubs <changed-files>` before every commit (exit 0 = safe). If `rch` workers are unhealthy, fall back to local `cargo` and note it.

---

## t3rag — This Project

### What it is

`t3rag` is a Rust-native retrieval tool for **thinking traces** (T3): one binary with
clap subcommands `index` / `search` / `serve` / `mcp` / `skill`, packaged as CLI + MCP
server + optional agent skill. **DESIGN.md is the spec** — including the §8 locked
decisions. Do not reopen a locked decision in passing; if one looks wrong, file a bead.

- **Embedder:** Qwen3-Embedding-0.6B via `ort` + ONNX (`onnx-community` export,
  uint8/fp16 preferred), last-token pooling + L2 norm. **No candle** — it is a heavy
  compile dep and ~7x slower on CPU. fastembed is the named fallback only.
- **Retrieval:** `usearch` (dense ANN, single-file, i8 quant, mmap) + `tantivy` (BM25)
  fused with RRF; BGE cross-encoder reranker via `ort` (top ~50, ~1024-tok truncation,
  parallel scoring).
- **Daemon:** `t3rag serve` keeps model + indexes resident (HTTP + streamable-HTTP MCP
  via `rmcp`); `t3rag mcp` is a stdio proxy to the daemon and must not self-host the
  model. ORT sessions are not safely concurrent — pool 2-4 behind a semaphore on
  dedicated threads.
- **Corpus:** user-provided only (`--input` JSONL / parquet / dir of `.md`|`.txt`).
  No HF dependency, no Python, no API keys.
- **Latency budget (warm, CPU):** ~200-700ms end-to-end; presets
  `--fast` / `--balanced` / `--max` trade quality for speed.

### Toolchain

- **Rust stable** (edition 2021+; pin a toolchain in `rust-toolchain.toml` only if a
  real need appears). Cargo only.
- Deps per DESIGN.md §6: `ort` (download-binaries), `hf-hub`, `tokenizers`,
  `usearch`, `tantivy`, `tokio` (minimal), `rmcp`, `rayon`, `memmap2`, `parquet`+`arrow`,
  `parking_lot`, `rustc-hash`. Keep feature sets lean; audit with `cargo build --timings`.
- New dependencies: check maintenance + licence; prefer versions ≥7 days old.

### Commands

```bash
cargo run -- index --input <path> --name <corpus>   # build a corpus
cargo run -- search "<query>" --corpus <name>       # query it
cargo run -- serve                                  # daemon: HTTP + streamable-HTTP MCP
cargo run -- mcp                                    # stdio MCP proxy to the daemon
bv --robot-triage                                   # what to work on next (never bare bv)
```

### Product invariants you must not break

- Whole-passage embedding by default; structural chunking only above ~8K tokens with
  parent hydration (DESIGN.md §7; being verified empirically — see the eval bead).
- Pure Rust runtime path: no Python, no required API keys, everything local-first.
- Daemon and CLI share one engine — `search` auto-spawns/proxies to the daemon;
  `--no-daemon` is the cold minimal-footprint fallback.
- `meta.json` carries a `format_version`; bump it on any corpus-format change.

### Shared checkout and Git hygiene

- One shared checkout (`/data/projects/t3ragrust`); agents are fungible generalists.
- Reserve the files you edit through Agent Mail (`reason` = bead id) before editing;
  release on push. The `am` pre-commit guard is installed and fails closed when it
  cannot tell who is committing while reservations are active — export
  `AGENT_NAME=<your Agent Mail name>` and never use `AGENT_MAIL_BYPASS=1`/`--no-verify`.
- Commit only your own files: `git add <paths>`, never `git add .` / `git add -A`.
  Other agents' uncommitted changes are normal — never stash/revert/disturb them.

### Beads in this repo

Issue ids use the `t3-` prefix (epic children read `t3-<epic>.<n>`). When the DB is
shared by many agents, wrap `br` calls:

```bash
flock -w 120 .beads/.br-agent.lock br --no-auto-flush --no-auto-import <command> ...
```

Attribution is append-only: claim with `--claim --actor "<agent>" --model "<model>"`,
append `worked: <agent> …` notes per contributor (never overwrite `--notes`), and name
the contributor chain in the close reason. Never clear another agent's claim; when
stopping early, append a stop note and release (`--status open --assignee ""`).

---

## MCP Agent Mail — Multi-Agent Coordination

- **project_key:** `/data/projects/t3ragrust` (absolute path). Register once per
  session: `am agents register --project /data/projects/t3ragrust --program <you> --model <model>`.
- **Shared identifiers:** bead id (`t3-…`) as mail `thread_id`; subjects prefixed
  `[t3-…]`; file-reservation `reason` = bead id.
- Prefer `am` CLI (the MCP transport flaps on sends): `am mail send`, `am inbox`,
  `am file_reservations reserve|release`, `am macros start-session`.
- Typical flow: `bv --robot-next`/`br ready` → reserve paths → `[t3-…] Start` message →
  work → `[t3-…] Completed` reply + release reservations + `br close`.

---

## bv — Graph-Aware Triage

`bv --robot-triage` is the single entry point for "what should I work on" (ranked
recommendations, quick wins, blockers-to-clear). `--robot-plan` for parallel tracks,
`--robot-insights` for graph metrics/cycles. **Never run bare `bv`** — it launches a
blocking TUI.

---

## Landing the Plane (Session Completion)

Before ending a session: file beads for remaining work, run the quality gates, update
bead status, then:

```bash
git pull --rebase
br sync --flush-only
git add .beads/ <changed files>
git commit -m "..."
git push            # work is not complete until push succeeds
```

<!-- br-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`/`bd`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View ready issues (open, unblocked, not deferred)
br ready              # or: bd ready

# List and search
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br search "keyword"   # Full-text search

# Create and update
br create --title="..." --description="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once

# Sync with git
br sync --flush-only  # Export DB to JSONL
br sync --status      # Check sync status
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run `br sync --flush-only` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only open, unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git commit -m "..."     # Commit everything
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always sync before ending session

<!-- end-br-agent-instructions -->

<!-- am:blurb -->
## MCP Agent Mail: coordination for multi-agent workflows

What it is
- A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources.
- Provides identities, inbox/outbox, searchable threads, and advisory file reservations, with human-auditable artifacts in Git.

Why it's useful
- Prevents agents from stepping on each other with explicit file reservations (leases) for files/globs.
- Keeps communication out of your token budget by storing messages in a per-project archive.
- Offers quick reads (`resource://inbox/...`, `resource://thread/...`) and macros that bundle common flows.

How to use effectively
1) Same repository
   - Register an identity: call `ensure_project`, then `register_agent` using this repo's absolute path as `project_key`.
   - Reserve files before you edit: `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)` to signal intent and avoid conflict.
   - Communicate with threads: use `send_message(..., thread_id="FEAT-123")`; check inbox with `fetch_inbox` and acknowledge with `acknowledge_message`.
   - Read fast: `resource://inbox/{Agent}?project=<abs-path>&limit=20&agent_token=<registration_token>` or `resource://thread/{id}?project=<abs-path>&agent=<Agent>&agent_token=<registration_token>&include_bodies=true` unless the current MCP session already authenticated as that agent.
   - Tip: set `AGENT_NAME` in your environment so the pre-commit guard can block commits that conflict with others' active exclusive file reservations.

2) Across different repos in one project (e.g., Next.js frontend + FastAPI backend)
   - Option A (single project bus): register both sides under the same `project_key` (shared key/path). Keep reservation patterns specific (e.g., `frontend/**` vs `backend/**`).
   - Option B (separate projects): each repo has its own `project_key`; use `macro_contact_handshake` or `request_contact`/`respond_contact` to link agents, then message directly. Keep a shared `thread_id` (e.g., ticket key) across repos for clean summaries/audits.

Macros vs granular tools
- Prefer macros when you want speed or are on a smaller model: `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`.
- Use granular tools when you need control: `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`.

Common pitfalls
- "from_agent not registered": always `register_agent` in the correct `project_key` first.
- "FILE_RESERVATION_CONFLICT": adjust patterns, wait for expiry, or use a non-exclusive reservation when appropriate.
- Auth errors: if JWT+JWKS is enabled, include a bearer token with a `kid` that matches server JWKS; static bearer is used only when JWT is disabled.

## Integrating with Beads (dependency-aware task planning)

Beads provides a lightweight, dependency-aware issue database and a CLI (`br`, beads_rust) for selecting "ready work," setting priorities, and tracking status. It complements MCP Agent Mail's messaging, audit trail, and file-reservation signals. Project: [beads_rust](https://github.com/Dicklesworthstone/beads_rust)

**Note:** `br` is non-invasive and never executes git commands. After `br sync --flush-only`, manually `git add .beads/ && git commit`.

Recommended conventions
- **Single source of truth**: Use **Beads** for task status/priority/dependencies; use **Agent Mail** for conversation, decisions, and attachments (audit).
- **Shared identifiers**: Use the Beads issue id (e.g., `t3-wkp.4`) as the Mail `thread_id` and prefix message subjects with `[t3-wkp.4]`.
- **Reservations**: When starting a `t3-###` task, call `file_reservation_paths(...)` for the affected paths; include the issue id in the `reason` and release on completion.

Typical flow (agents)
1) **Pick ready work** (Beads)
   - `br ready --json` -> choose one item (highest priority, no blockers)
2) **Reserve edit surface** (Mail)
   - `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="t3-123")`
3) **Announce start** (Mail)
   - `send_message(..., thread_id="t3-123", subject="[t3-123] Start: <short title>", ack_required=true)`
4) **Work and update**
   - Reply in-thread with progress and attach artifacts/images; keep the discussion in one thread per issue id
5) **Complete and release**
   - `br close t3-123 --reason "Completed"` (Beads is status authority)
   - `br sync --flush-only` then `git add .beads/ && git commit` (br never runs git)
   - `release_file_reservations(project_key, agent_name, paths=["src/**"])`
   - Final Mail reply: `[t3-123] Completed` with summary and links

Mapping cheat-sheet
- **Mail `thread_id`** <-> `t3-###`
- **Mail subject**: `[t3-###] ...`
- **File reservation `reason`**: `t3-###`
- **Commit messages (optional)**: include `t3-###` for traceability

Pitfalls to avoid
- Don't create or manage tasks in Mail; treat Beads as the single task queue.
- Always include `t3-###` in message `thread_id` to avoid ID drift across tools.
<!-- am:blurb:end -->
