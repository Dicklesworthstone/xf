# AGENTS.md — xf (X Archive Finder)

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

---

## Toolchain: Rust & Cargo

We only use **Cargo** in this project, NEVER any other package manager.

- **Edition:** Rust 2024 (nightly required — see `rust-toolchain.toml`)
- **Dependency versions:** Explicit versions for stability
- **Configuration:** Cargo.toml only
- **Unsafe code:** Forbidden (`#![forbid(unsafe_code)]` via crate lints)

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `tantivy` | Full-text search engine and query parser |
| `rusqlite` | SQLite storage for metadata and stats |
| `serde` + `serde_json` | Archive parsing and output serialization |
| `clap` + `clap_complete` | CLI parsing and shell completions |
| `rayon` | Parallel parsing of archive files |
| `chrono` | Timestamp parsing and formatting |
| `tracing` | Structured logging |

### Release Profile

The release build optimizes for binary size:

```toml
[profile.release]
opt-level = "z"     # Optimize for size (lean binary for distribution)
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit for better optimization
panic = "abort"     # Smaller binary, no unwinding overhead
strip = true        # Remove debug symbols
```

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Brittle regex-based transformations create far more problems than they solve.

- **Always make code changes manually**, even when there are many instances
- For many simple changes: use parallel subagents
- For subtle/complex changes: do them methodically yourself

### No File Proliferation

If you want to change something or add a feature, **revise existing code files in place**.

**NEVER** create variations like:
- `mainV2.rs`
- `main_improved.rs`
- `main_enhanced.rs`

New files are reserved for **genuinely new functionality** that makes zero sense to include in any existing file. The bar for creating new files is **incredibly high**.

---

## Project Semantics (xf)

This tool indexes and searches X (Twitter) data archives locally. Keep these invariants intact:

- **Privacy-first:** No network access from core runtime paths; keep data local.
- **Archive format:** Inputs are JavaScript-wrapped JSON (`window.YTD.*`). Parsing must tolerate whitespace/format variations.
- **Search:** Tantivy is the primary engine; SQLite FTS5 is a fallback/secondary store.
- **Metadata correctness:** Preserve IDs, timestamps, and counts exactly; avoid lossy conversions.
- **CLI truthfulness:** If a CLI flag exists, either implement it or remove it. Do not leave options that silently do nothing.

---

## Output Style

- **Text output** is user-facing and may include color. Avoid verbose debug spew unless `--verbose` is set.
- **JSON output** must be stable and machine-parseable. Do not change JSON shapes without explicit intent and tests.

---

## Compiler Checks (CRITICAL)

**After any substantive code changes, you MUST verify no errors were introduced:**

```bash
# Check for compiler errors and warnings
cargo check --all-targets

# Check for clippy lints (pedantic + nursery are enabled)
cargo clippy --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

If you see errors, **carefully understand and resolve each issue**. Read sufficient context to fix them the RIGHT way.

---

## Testing

### Unit Tests

```bash
cargo test
cargo test -- --nocapture
```

### Focused Tests

```bash
cargo test parser
cargo test search
cargo test storage
```

---

## Third-Party Library Usage

If you aren't 100% sure how to use a third-party library, **SEARCH ONLINE** to find the latest documentation and best practices before coding. Prefer primary docs.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, ignoring comments/strings, and can **safely rewrite** code.

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

### Rule of Thumb

- Need correctness or **applying changes** → `ast-grep`
- Need raw speed or **hunting text** → `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

---

## Session Completion

Before ending a work session:

1. Summarize changes clearly
2. Note any remaining risks or follow-ups
3. Provide the exact commands to run for tests/linters (if not run)

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   br sync --flush-only
   git add .beads/
   git commit -m "sync beads"
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

## Issue Tracking with br (beads_rust)

All issue tracking goes through **br** (beads_rust). No other TODO systems.

**Important:** `br` is non-invasive—it NEVER executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

Key invariants:

- `.beads/` is authoritative state and **must always be committed** with code changes.
- Do not edit `.beads/*.jsonl` directly; only via `br`.

### Basics

Check ready work:

```bash
br ready --json
```

Create issues:

```bash
br create "Issue title" -t bug|feature|task -p 0-4 --json
br create "Issue title" -p 1 --deps discovered-from:br-123 --json
```

Update:

```bash
br update br-42 --status in_progress --json
br update br-42 --priority 1 --json
```

Complete:

```bash
br close br-42 --reason "Completed" --json
```

Types:

- `bug`, `feature`, `task`, `epic`, `chore`

Priorities:

- `0` critical (security, data loss, broken builds)
- `1` high
- `2` medium (default)
- `3` low
- `4` backlog

Agent workflow:

1. `br ready` to find unblocked work.
2. Claim: `br update <id> --status in_progress`.
3. Implement + test.
4. If you discover new work, create a new bead with `discovered-from:<parent-id>`.
5. Close when done.
6. Run `br sync --flush-only && git add .beads/` before committing.
7. Commit `.beads/` in the same commit as code changes.

Sync behavior:

- `br sync --flush-only` exports DB to `.beads/issues.jsonl` (NO git operations).
- `br sync --import-only` imports from JSONL when it's newer (e.g. after `git pull`).
- Auto-import runs before read commands; auto-flush runs after write commands.

Never:

- Use markdown TODO lists.
- Use other trackers.
- Duplicate tracking.

---

## Deployment Health Monitoring

Before starting work, check deployment health.

### 1. Check alert files

```bash
ls -la *.alert 2>/dev/null || echo "No alerts - site healthy"
bun scripts/dev/deployment-monitor.ts --skip-browser --verbose
```

Alert patterns (project root):

| Pattern | Severity | Action |
| ---------------------------------------- | -------- | -------------------------- |
| `RED_ALERT_CRITICAL_DEPLOYMENT__*.alert` | P0 | Stop everything; fix now |
| `RED_ALERT_WARNING_DEPLOYMENT__*.alert` | P1 | Investigate before coding |
| `GREEN_RESOLVED__*.alert` | Info | Site recovered; resume dev |

### 2. Responding to critical alerts

If a `RED_ALERT_CRITICAL_DEPLOYMENT__*.alert` exists:

1. Read the file (HTTP status, error details, screenshot paths).
2. Check Vercel logs:

```bash
vercel logs --prod
```

3. List deployments:

```bash
vercel list
```

4. Roll back if needed (no code changes, just routing):

```bash
vercel rollback
```

5. Create a P0 bead:

```bash
br create "P0: Production site down - <description>" -t bug -p 0 --json
br update <id> --status in_progress --json
```

6. After fix, verify:

```bash
bun scripts/dev/deployment-monitor.ts --all-pages --verbose
```

### Automatic monitoring

See `scripts/dev/DEPLOYMENT_MONITOR_SETUP.md`. The monitor runs via cron:

- Every 3 minutes: fast HTTP homepage check.
- Every 15 minutes: multi-page browser check.

Manual checks:

```bash
bun scripts/dev/deployment-monitor.ts --skip-browser
bun scripts/dev/deployment-monitor.ts --all-pages
bun scripts/dev/deployment-monitor.ts --all-pages --verbose
```

Monitor state:

- Requires 2 consecutive failures to alert.
- Requires 3 consecutive successes to mark as recovered.
- State file: `tmp/deployment-monitor-state.json`.

P0 workflow:

1. Detect alert → create P0 bead.
2. Claim bead → `br update <id> --status in_progress`.
3. Fix → deploy to Vercel.
4. Verify → `... --all-pages`.
5. Close bead → `br close <id>`.
6. Sync and commit: `br sync --flush-only && git add .beads/ && git commit`.

Key files:

- `scripts/dev/deployment-monitor.ts`
- `scripts/dev/DEPLOYMENT_MONITOR_SETUP.md`
- `tmp/deployment-monitor-state.json`
- `tmp/deployment-alerts/*.webp`
- `*.alert`

---

### Vercel Deployment Configuration

**Auto-deployments are DISABLED** to reduce Vercel credit consumption. Deployments must be triggered manually.

#### Current Settings (via API)

| Setting | Value | Effect |
|---------|-------|--------|
| `createDeployments` | `disabled` | No auto-deploys on push |
| `enableAffectedProjectsDeployments` | `true` | Smart skipping for monorepos |
| `commandForIgnoringBuildStep` | `bash scripts/vercel-ignore-build.sh` | Custom skip logic |

#### Deploy to Production

```bash
vercel --prod                    # Deploy current branch to production
vercel                           # Preview deploy first
vercel --prod                    # Then promote to production
```

#### Ignore Build Script (`scripts/vercel-ignore-build.sh`)

Only changes to these paths trigger builds:
- `src/`, `public/`, `package.json`, `bun.lock`, `next.config.ts`, `tailwind.config.ts`, `postcss.config.mjs`, `vercel.json`

The script exits 1 (skip) if none of these paths changed, exit 0 (build) otherwise.

#### Re-enable Auto-Deploy (if needed)

```bash
# Via API (uses local Vercel auth token)
curl -X PATCH \
  -H "Authorization: Bearer $(cat ~/.local/share/com.vercel.cli/auth.json | jq -r '.token')" \
  -H "Content-Type: application/json" \
  -d '{"gitProviderOptions":{"createDeployments":"enabled"}}' \
  "https://api.vercel.com/v9/projects/prj_JPFsf1MqNgvHt425JcaPDbjuuNJ9?teamId=team_F5Q3EH8Qxu3nDEOyEZLcQPe6"
```

#### Project IDs

| Key | Value |
|-----|-------|
| Project ID | `prj_JPFsf1MqNgvHt425JcaPDbjuuNJ9` |
| Team ID | `team_F5Q3EH8Qxu3nDEOyEZLcQPe6` |

---

### ast-grep vs ripgrep

**ast-grep**: use when structure matters.

- Good for refactors/codemods:
  - Renaming APIs, changing import forms, rewriting call sites.
- Good for policy checks:
  - Enforcing patterns (`scan` + rules).
- Works with LSP mode; supports `--json` output.

**ripgrep (`rg`)**: use when plain text search is enough.

- Great for:
  - Finding strings, TODOs, log lines, config values.
  - Quick recon and pre-filtering.

Rules of thumb:

- Need correctness and structural awareness or plan to **modify** code → start with `ast-grep`.
- Just hunting a string or regex → start with `rg`.
- Often combine:
  - `rg` to limit files.
  - `ast-grep` for precise matches/rewrites.

Examples:

```bash
ast-grep run -l TypeScript -p 'import $X from "$P"'
ast-grep run -l JavaScript -p 'var $A = $B' -r 'let $A = $B' -U

rg -n 'console\.log\(' -t js
rg -l -t ts 'useQuery\(' | xargs ast-grep run -l TypeScript -p 'useQuery($A)' -r 'useSuspenseQuery($A)' -U
```

Mental model:

- Match unit: `ast-grep` → AST node; `rg` → line.
- False positives: lower with `ast-grep`.
- Rewrites: first-class in `ast-grep`; ad-hoc with `rg` + `sed`/`awk` (riskier).

---

## UBS (Ultimate Bug Scanner)

Run UBS on changed files before every commit.

Install:

```bash
curl -sSL https://raw.githubusercontent.com/Dicklesworthstone/ultimate_bug_scanner/main/install.sh | bash
```

Typical usage:

```bash
ubs file.ts file2.ts # Per-file (fast) – preferred
ubs $(git diff --name-only --cached)
ubs --only=js,ts src/
ubs --ci --fail-on-warning .
ubs sessions --entries 1
ubs . # Whole project (slow; use sparingly)
```

Output:

```
⚠️ Category (N errors)
 file.ts:42:5 – Issue description
 💡 Suggested fix
Exit code: 1
```

Workflow:

1. Read category + suggestion.
2. Jump to `file:line:col`.
3. Confirm real issue (not false positive).
4. Fix root cause.
5. Re-run `ubs <file>` until exit code 0.
6. Then commit.

Focus:

- Always fix critical issues (null/undefined, security, race conditions, resource leaks).
- Treat "important" findings as production bugs, not nits.
- Scope scans to changed files for speed.

---

### Google Cloud CLI (gcloud)

`gcloud` is installed at `./google-cloud-sdk/bin/gcloud`.

Auth:

```bash
./google-cloud-sdk/bin/gcloud auth login
```

Projects:

```bash
./google-cloud-sdk/bin/gcloud projects list
./google-cloud-sdk/bin/gcloud config set project <PROJECT_ID>
```

Billing:

```bash
./google-cloud-sdk/bin/gcloud beta billing accounts list
./google-cloud-sdk/bin/gcloud beta billing projects link <PROJECT_ID> --billing-account=<ACCOUNT_ID>
```

Services:

```bash
./google-cloud-sdk/bin/gcloud services list --enabled
./google-cloud-sdk/bin/gcloud services enable <SERVICE_NAME>
```

Analytics (GA4):

- Script: `scripts/ga4-setup.ts`
- Auth via ADC:

```bash
./google-cloud-sdk/bin/gcloud auth application-default login
export GA4_PROPERTY_ID="<YOUR_NUMERIC_PROPERTY_ID>"
bun scripts/ga4-setup.ts
```

---

### Cloudflare R2 (Vendor Evidence Storage)

We store vendor pricing screenshots in R2:

- Bucket: `midas-edge-bucket`
- Account ID: `abb7d369730a2d0adcb077c8147384e0`
- Endpoint: `https://abb7d369730a2d0adcb077c8147384e0.r2.cloudflarestorage.com`
- Public base URL (if used):

`https://abb7d369730a2d0adcb077c8147384e0.r2.cloudflarestorage.com/midas-edge-bucket`

Env keys (all in Vault path `secret/midas-edge`):

- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ENDPOINT`
- `R2_BUCKET`
- `R2_ACCOUNT_ID`
- `R2_PUBLIC_BASE_URL`

---

### QStash (Job Queue)

We use Upstash QStash for background work (image generation, email, etc.).

- Base URL: `https://qstash.upstash.io`
- Env keys:
  - `QSTASH_URL`
  - `QSTASH_TOKEN`
  - `QSTASH_CURRENT_SIGNING_KEY`
  - `QSTASH_NEXT_SIGNING_KEY`

Use QStash for async jobs; ensure request signing and verification is wired up correctly on webhooks.

---

### Using bv as an AI Sidecar

`bv` is a terminal UI + analysis layer for `.beads/beads.jsonl`. It precomputes graph metrics so you don't have to.

Useful robot commands:

- `bv --robot-help` – overview
- `bv --robot-insights` – graph metrics (PageRank, betweenness, HITS, critical path, cycles)
- `bv --robot-plan` – parallelizable execution plan with unblocks info
- `bv --robot-priority` – priority suggestions with reasoning
- `bv --robot-recipes` – list recipes; apply via `bv --recipe <name>`
- `bv --robot-diff --diff-since <commit|date>` – JSON diff of issue changes

Use `bv` instead of rolling your own dependency graph logic.

---

### Morph Warp Grep — AI-Powered Code Search

Use `mcp__morph-mcp__warp_grep` for "how does X work?" discovery across the codebase.

When to use:

- You don't know where something lives.
- You want data flow across multiple files (API → service → schema → types).
- You want all touchpoints of a cross-cutting concern (e.g., moderation, billing).

Example:

```
mcp__morph-mcp__warp_grep(
  repoPath: "/data/projects/communitai",
  query: "How is the L3 Guardian appeals system implemented?"
)
```

Warp Grep:

- Expands a natural-language query to multiple search patterns.
- Runs targeted greps, reads code, follows imports, then returns concise snippets with line numbers.
- Reduces token usage by returning only relevant slices, not entire files.

When **not** to use Warp Grep:

- You already know the function/identifier name; use `rg`.
- You know the exact file; just open it.
- You only need a yes/no existence check.

Comparison:

| Scenario | Tool |
| ---------------------------------- | ---------- |
| "How is auth session validated?" | warp_grep |
| "Where is `handleSubmit` defined?" | `rg` |
| "Replace `var` with `let`" | `ast-grep` |

---

### cass — Cross-Agent Search

`cass` indexes prior agent conversations (Claude Code, Codex, Cursor, Gemini, ChatGPT, etc.) so we can reuse solved problems.

Rules:

- Never run bare `cass` (TUI). Always use `--robot` or `--json`.

Examples:

```bash
cass health
cass search "authentication error" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
cass expand /path/to/session.jsonl -n 42 -C 3 --json
cass capabilities --json
cass robot-docs guide
```

Tips:

- Use `--fields minimal` for lean output.
- Filter by agent with `--agent`.
- Use `--days N` to limit to recent history.

stdout is data-only, stderr is diagnostics; exit code 0 means success.

Treat cass as a way to avoid re-solving problems other agents already handled.

## Learnings & Troubleshooting (Dec 5, 2025)

### Next.js 16 Middleware Deprecation

**CRITICAL**: Next.js 16 deprecates `middleware.ts` in favor of `proxy.ts`.

- The middleware file is now `src/proxy.ts` (NOT `src/middleware.ts`)
- The exported function is `proxy()` (NOT `middleware()`)
- DO NOT restore or recreate `src/middleware.ts` - it will cause deprecation warnings
- If you see both files, delete `middleware.ts` and keep only `proxy.ts`

- **Tooling Issues**:
  - `mcp-agent-mail` CLI is currently missing from the environment path. Cannot register or check mail.
  - `drizzle-kit generate` may fail with `TypeError: sql2.toQuery is not a function` when `pgPolicy` is used with `sql` template literals in the schema file.
- **Workarounds**:
  - If `drizzle-kit generate` fails on `pgPolicy`, remove the policy definitions from `schema.ts` and implement RLS via raw SQL migrations or manual migration files.
  - Always provide `--name` to `drizzle-kit generate` to avoid interactive prompts.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   br sync --flush-only    # Export beads to JSONL (no git ops)
   git add .beads/         # Stage beads changes
   git add <other files>   # Stage code changes
   git commit -m "..."     # Commit everything
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

**Important:** `br` is non-invasive—it NEVER executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export to JSONL (NO git operations)
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Run `br sync --flush-only` then manually commit

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies
