# AGENTS.md — xf (X Archive Finder)

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

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

## Git Branch: ONLY Use `main`, NEVER `master`

**The default branch is `main`. The `master` branch exists only for legacy URL compatibility.**

- **All work happens on `main`** — commits, PRs, feature branches all merge to `main`
- **Never reference `master` in code or docs** — if you see `master` anywhere, it's a bug that needs fixing
- **The `master` branch must stay synchronized with `main`** — after pushing to `main`, also push to `master`:
  ```bash
  git push origin main:master
  ```

**If you see `master` referenced anywhere:**
1. Update it to `main`
2. Ensure `master` is synchronized: `git push origin main:master`

---

## Toolchain: Rust & Cargo

We only use **Cargo** in this project, NEVER any other package manager.

- **Edition:** Rust 2024 (nightly required — see `rust-toolchain.toml`)
- **Dependency versions:** Explicit versions for stability
- **Configuration:** Cargo.toml only (single-crate project, no workspace)
- **Unsafe code:** Forbidden (`#![forbid(unsafe_code)]` via crate lints)

### Async Runtime: Tokio

This project uses **Tokio** for async operations.

- `tokio` with features: `rt-multi-thread`, `macros`, `fs`, `io-util`, `time`, `signal`
- Background daemon uses Tokio channels and tasks for client/server communication

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `tantivy` | Full-text BM25 search engine and query parser |
| `rusqlite` | SQLite storage for metadata, stats, and FTS5 fallback |
| `serde` + `serde_json` | Archive parsing and output serialization |
| `rmp-serde` | MessagePack serialization for daemon protocol |
| `toon_rust` | Serialization utilities |
| `clap` + `clap_complete` | CLI parsing and shell completions |
| `rayon` | Parallel parsing of archive files |
| `chrono` + `chrono-english` | Timestamp parsing, formatting, and natural-language date input |
| `tracing` + `tracing-subscriber` | Structured logging with env-filter |
| `fastembed` | ONNX-based text embeddings (MiniLM-L6-v2 quality tier) |
| `safetensors` + `tokenizers` | Model2Vec static embeddings (potion fast tier) |
| `ort` | ONNX Runtime for static embedding models and cross-encoder reranking |
| `half` | IEEE 754 f16 float support for quantized vector storage |
| `wide` | Portable SIMD (`f32x8`) for fast dot products |
| `ring` | SHA256 content hashing |
| `rich_rust` | Rich terminal output (optional feature) |
| `colored` + `indicatif` + `console` | Terminal output and progress bars |
| `rustyline` | REPL line editing |
| `unicode-normalization` | NFC text canonicalization |
| `thiserror` | Ergonomic error type derivation |

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

## Backwards Compatibility

We do not care about backwards compatibility—we're in early development with no users. We want to do things the **RIGHT** way with **NO TECH DEBT**.

- Never create "compatibility shims"
- Never create wrapper functions for deprecated APIs
- Just fix the code directly

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

### Testing Policy

Tests must cover:
- Happy path
- Edge cases (empty input, max values, boundary conditions)
- Error conditions

Integration and E2E tests live in the `tests/` directory. Benchmarks live in `benches/`.

### Unit Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run tests for a specific module
cargo test parser
cargo test search
cargo test storage
```

### Test Categories

| Location | Focus Areas |
|----------|-------------|
| `src/` (inline `#[cfg(test)]`) | Parser, search, storage, embedder, config, model, canonicalization |
| `tests/cli_e2e.rs` | CLI end-to-end: flag combinations, output formats, error handling |
| `tests/integration_test.rs` | Cross-module integration: parse → index → search → output |
| `tests/daemon_e2e.rs` | Daemon client/server protocol, lifecycle, concurrent queries |
| `tests/daemon_stress.rs` | Daemon under load: concurrent connections, large result sets |
| `tests/daemon_tests.rs` | Daemon unit-level: message framing, protocol parsing |
| `tests/semantic_pipeline_e2e.rs` | Full semantic search pipeline: embed → index → search → rerank |
| `tests/static_embedders.rs` | Model2Vec / static embedding quality and determinism |
| `tests/transformer_embedders.rs` | FastEmbed / ONNX transformer embedding correctness |
| `tests/reranker_tests.rs` | Cross-encoder reranking: scoring, ordering, edge cases |
| `tests/semantic_quality_test.rs` | Embedding quality benchmarks: recall, MRR, cluster separation |
| `tests/compare_embedder_quality.rs` | Head-to-head embedder comparison metrics |
| `tests/output_format_env.rs` | JSON/text output format parity and stability |
| `benches/search_perf.rs` | Search throughput, indexing latency, dot product performance |

---

## Third-Party Library Usage

If you aren't 100% sure how to use a third-party library, **SEARCH ONLINE** to find the latest documentation and current best practices.

---

## xf — This Project

**This is the project you're working on.** xf is an ultra-fast CLI for indexing and searching X (formerly Twitter) data archives locally. It combines full-text search (Tantivy BM25), semantic vector search, and cross-encoder reranking into a single privacy-first binary.

### What It Does

Indexes X/Twitter data export archives (the official "Download your data" ZIP) and provides instant, powerful search over tweets, DMs, likes, bookmarks, and account metadata. Supports lexical search, semantic/vector search, hybrid fusion, and an interactive REPL.

### Project Semantics & Invariants

- **Privacy-first:** No network access from core runtime paths; keep data local.
- **Archive format:** Inputs are JavaScript-wrapped JSON (`window.YTD.*`). Parsing must tolerate whitespace/format variations.
- **Search:** Tantivy is the primary engine; SQLite FTS5 is a fallback/secondary store.
- **Metadata correctness:** Preserve IDs, timestamps, and counts exactly; avoid lossy conversions.
- **CLI truthfulness:** If a CLI flag exists, either implement it or remove it. Do not leave options that silently do nothing.

### Architecture

```
X Archive (ZIP/folder)
  └─ window.YTD.* JS-wrapped JSON files
       │
       ▼
   Parser (rayon parallel) → Tweet/DM/Like/Bookmark models
       │
       ├─ SQLite storage (metadata, stats, FTS5 fallback)
       ├─ Tantivy index (BM25 full-text)
       └─ Vector index (f16 quantized, SIMD dot product)
           │
           ▼
   Search: Lexical ──┬── RRF Fusion ── Optional Rerank ── Results
   Search: Semantic ─┘
       │
       ├─ CLI (one-shot commands, JSON/text output)
       ├─ REPL (interactive search session)
       └─ Daemon (background server, MessagePack protocol)
```

### Source Structure

```
xf/
├── Cargo.toml                 # Single-crate project
├── src/
│   ├── main.rs                # CLI entry point, command dispatch
│   ├── lib.rs                 # Public API surface
│   ├── cli.rs                 # Clap command definitions
│   ├── parser.rs              # X archive JS-JSON parser (window.YTD.*)
│   ├── model.rs               # Tweet, DM, Like, Bookmark data models
│   ├── storage.rs             # SQLite storage layer + FTS5
│   ├── search.rs              # Tantivy BM25 search engine
│   ├── hybrid.rs              # Hybrid search: RRF fusion of lexical + semantic
│   ├── vector.rs              # Vector index: f16 quantized, SIMD dot product, top-k
│   ├── embedder.rs            # Embedder trait and registry
│   ├── hash_embedder.rs       # FNV-1a hash embedder (zero deps, always available)
│   ├── model2vec_embedder.rs  # potion-128M static embedder (fast tier)
│   ├── fastembed_embedder.rs  # MiniLM-L6-v2 ONNX embedder (quality tier)
│   ├── static_mrl_embedder.rs # Static MRL embedding support
│   ├── model_registry.rs      # Model download and management
│   ├── reranker.rs            # Reranker trait
│   ├── flashrank_reranker.rs  # FlashRank cross-encoder reranker
│   ├── mxbai_reranker.rs      # mxbai cross-encoder reranker
│   ├── rerank_step.rs         # Rerank pipeline integration
│   ├── canonicalize.rs        # Text preprocessing: NFC, markdown strip, truncation
│   ├── date_parser.rs         # Natural-language date parsing
│   ├── messages.rs            # DM message handling
│   ├── config.rs              # Configuration management
│   ├── output.rs              # Output formatting (JSON/text/table)
│   ├── search_results.rs      # Search result types and rendering
│   ├── list_tables.rs         # SQLite table listing utilities
│   ├── stats_analytics.rs     # Archive statistics and analytics
│   ├── stats_dashboard.rs     # Stats dashboard rendering
│   ├── repl.rs                # Interactive REPL
│   ├── robot_docs.rs          # Machine-readable documentation output
│   ├── doctor.rs              # System health checks and diagnostics
│   ├── completions.rs         # Shell completion generation
│   ├── theme.rs               # Terminal theme and color management
│   ├── tweet_card.rs          # Rich tweet card rendering
│   ├── progress.rs            # Progress bar utilities
│   ├── logging.rs             # Tracing/logging setup
│   ├── perf.rs                # Performance measurement utilities
│   ├── error.rs               # Error types (thiserror)
│   ├── daemon/                # Background daemon (server + client, MessagePack)
│   └── benchmark/             # Inline benchmark utilities
├── tests/                     # Integration and E2E tests
├── benches/                   # Performance benchmarks (Criterion)
└── scripts/                   # Build and deployment scripts
```

### Key Source Files

| File | Purpose |
|------|---------|
| `src/parser.rs` | Parses X archive JS-wrapped JSON (`window.YTD.tweet`, `window.YTD.like`, etc.) with rayon parallelism |
| `src/storage.rs` | SQLite storage: schema management, metadata persistence, FTS5 secondary index |
| `src/search.rs` | Tantivy BM25 search: schema creation, document indexing, query parsing, result ranking |
| `src/vector.rs` | Vector index: FSVI/XFVI binary format, f16 quantization, SIMD dot product, top-k heap search |
| `src/hybrid.rs` | Hybrid search: RRF fusion (K=60), two-tier progressive search, score blending |
| `src/embedder.rs` | `Embedder` trait definition and embedder auto-detection/registry |
| `src/model2vec_embedder.rs` | potion-128M static embedder (fast tier, sub-millisecond) |
| `src/fastembed_embedder.rs` | MiniLM-L6-v2 ONNX embedder (quality tier, ~128ms) |
| `src/repl.rs` | Interactive REPL with history, completions, and rich output |
| `src/main.rs` | CLI entry point: command dispatch, archive loading, search orchestration |

### Feature Flags

```toml
[features]
default = []
rich = ["dep:rich_rust"]           # Rich terminal output via rich_rust
full = ["rich"]                    # Everything
legacy-colors = []                 # Legacy colored output (uses colored crate)
parallel-search = []               # Parallel search execution
alloc-count = []                   # Allocation counting for profiling
```

### Output Style

- **Text output** is user-facing and may include color. Avoid verbose debug spew unless `--verbose` is set.
- **JSON output** must be stable and machine-parseable. Do not change JSON shapes without explicit intent and tests.

### X Archive Data Format

The X/Twitter data export contains JavaScript-wrapped JSON files:

```javascript
window.YTD.tweet.part0 = [
  { "tweet": { "id": "123456", "full_text": "Hello world", "created_at": "..." } }
]
```

Key data types parsed:
- **Tweets** (`window.YTD.tweet`) — full text, IDs, timestamps, engagement counts
- **Likes** (`window.YTD.like`) — liked tweet references
- **Bookmarks** — saved tweets
- **DMs** (`window.YTD.direct_message`) — direct message conversations
- **Account metadata** — profile info, followers, following

Parsing must handle:
- Varying whitespace and formatting across archive versions
- Optional/missing fields in older archives
- Unicode content and emoji in tweet text
- Large archives (100K+ tweets) efficiently via rayon parallelism

### Key Design Decisions

- **Single binary** — everything compiles to one `xf` executable, no runtime dependencies
- **f16 quantization** for vector storage — 50% memory savings, <1% quality loss for cosine similarity
- **`wide` and `half` are unconditional deps** — SIMD and f16 are always available
- **RRF fusion is rank-based** — does NOT depend on score normalization
- **NaN-safe `total_cmp()` ordering** in top-k search to prevent undefined sort behavior
- **Daemon mode** — background server with MessagePack protocol for fast repeated queries without re-indexing
- **SQLite for metadata + FTS5 fallback** — Tantivy is primary search, SQLite FTS5 available when Tantivy is not indexed
- **Privacy-first** — no network calls during search; model downloads are explicit opt-in
- **rayon for data parallelism** — archive parsing and vector search use work-stealing thread pool

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

1. Detect alert -> create P0 bead.
2. Claim bead -> `br update <id> --status in_progress`.
3. Fix -> deploy to Vercel.
4. Verify -> `... --all-pages`.
5. Close bead -> `br close <id>`.
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

## Google Cloud CLI (gcloud)

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

## Cloudflare R2 (Vendor Evidence Storage)

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

## QStash (Job Queue)

We use Upstash QStash for background work (image generation, email, etc.).

- Base URL: `https://qstash.upstash.io`
- Env keys:
  - `QSTASH_URL`
  - `QSTASH_TOKEN`
  - `QSTASH_CURRENT_SIGNING_KEY`
  - `QSTASH_NEXT_SIGNING_KEY`

Use QStash for async jobs; ensure request signing and verification is wired up correctly on webhooks.

---

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

---

## UBS — Ultimate Bug Scanner

**Golden Rule:** `ubs <changed-files>` before every commit. Exit 0 = safe. Exit >0 = fix & re-run.

### Commands

```bash
ubs file.rs file2.rs                    # Specific files (< 1s) — USE THIS
ubs $(git diff --name-only --cached)    # Staged files — before commit
ubs --only=rust,toml src/               # Language filter (3-5x faster)
ubs --ci --fail-on-warning .            # CI mode — before PR
ubs .                                   # Whole project (ignores target/, Cargo.lock)
```

### Output Format

```
Warning Category (N errors)
    file.rs:42:5 - Issue description
    Suggested fix
Exit code: 1
```

Parse: `file:line:col` -> location | suggestion -> how to fix | Exit 0/1 -> pass/fail

### Fix Workflow

1. Read finding -> category + fix suggestion
2. Navigate `file:line:col` -> view context
3. Verify real issue (not false positive)
4. Fix root cause (not symptom)
5. Re-run `ubs <file>` -> exit 0
6. Commit

### Bug Severity

- **Critical (always fix):** Memory safety, use-after-free, data races, SQL injection
- **Important (production):** Unwrap panics, resource leaks, overflow checks
- **Contextual (judgment):** TODO/FIXME, println! debugging

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build`, `cargo test`, `cargo clippy`, and other compilation commands to a fleet of 8 remote Contabo VPS workers instead of building locally. This prevents compilation storms from overwhelming csd when many agents run simultaneously.

**RCH is installed at `~/.local/bin/rch` and is hooked into Claude Code's PreToolUse automatically.** Most of the time you don't need to do anything if you are Claude Code — builds are intercepted and offloaded transparently.

To manually offload a build:
```bash
rch exec -- cargo build --release
rch exec -- cargo test
rch exec -- cargo clippy
```

Quick commands:
```bash
rch doctor                    # Health check
rch workers probe --all       # Test connectivity to all 8 workers
rch status                    # Overview of current state
rch queue                     # See active/waiting builds
```

If rch or its workers are unavailable, it fails open — builds run locally as normal.

**Note for Codex/GPT-5.2:** Codex does not have the automatic PreToolUse hook, but you can (and should) still manually offload compute-intensive compilation commands using `rch exec -- <command>`. This avoids local resource contention when multiple agents are building simultaneously.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, ignoring comments/strings, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms
- Policy checks: enforce patterns across a repo
- Editor/automation: LSP mode, `--json` output

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

- Recon: find strings, TODOs, log lines, config values
- Pre-filter: narrow candidate files before ast-grep

### Rule of Thumb

- Need correctness or **applying changes** -> `ast-grep`
- Need raw speed or **hunting text** -> `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

### Rust Examples

```bash
# Find structured code (ignores comments)
ast-grep run -l Rust -p 'fn $NAME($$$ARGS) -> $RET { $$$BODY }'

# Find all unwrap() calls
ast-grep run -l Rust -p '$EXPR.unwrap()'

# Quick textual hunt
rg -n 'println!' -t rust

# Combine speed + precision
rg -l -t rust 'unwrap\(' | xargs ast-grep run -l Rust -p '$X.unwrap()' --json
```

---

## Morph Warp Grep — AI-Powered Code Search

**Use `mcp__morph-mcp__warp_grep` for exploratory "how does X work?" questions.** An AI agent expands your query, greps the codebase, reads relevant files, and returns precise line ranges with full context.

**Use `ripgrep` for targeted searches.** When you know exactly what you're looking for.

**Use `ast-grep` for structural patterns.** When you need AST precision for matching/rewriting.

### When to Use What

| Scenario | Tool | Why |
|----------|------|-----|
| "How does the hybrid search work?" | `warp_grep` | Exploratory; don't know where to start |
| "Where is the archive parser implemented?" | `warp_grep` | Need to understand architecture |
| "Find all uses of `Embedder::embed`" | `ripgrep` | Targeted literal search |
| "Find files with `println!`" | `ripgrep` | Simple pattern |
| "Replace all `unwrap()` with `expect()`" | `ast-grep` | Structural refactor |

### warp_grep Usage

```
mcp__morph-mcp__warp_grep(
  repoPath: "/dp/xf",
  query: "How does the hybrid search fusion work?"
)
```

Returns structured results with file paths, line ranges, and extracted code snippets.

### Anti-Patterns

- **Don't** use `warp_grep` to find a specific function name -> use `ripgrep`
- **Don't** use `ripgrep` to understand "how does X work" -> wastes time with manual reads
- **Don't** use `ripgrep` for codemods -> risks collateral edits

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources. Provides identities, inbox/outbox, searchable threads, and advisory file reservations with human-auditable artifacts in Git.

### Why It's Useful

- **Prevents conflicts:** Explicit file reservations (leases) for files/globs
- **Token-efficient:** Messages stored in per-project archive, not in context
- **Quick reads:** `resource://inbox/...`, `resource://thread/...`

### Same Repository Workflow

1. **Register identity:**
   ```
   ensure_project(project_key=<abs-path>)
   register_agent(project_key, program, model)
   ```

2. **Reserve files before editing:**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)
   ```

3. **Communicate with threads:**
   ```
   send_message(..., thread_id="FEAT-123")
   fetch_inbox(project_key, agent_name)
   acknowledge_message(project_key, agent_name, message_id)
   ```

4. **Quick reads:**
   ```
   resource://inbox/{Agent}?project=<abs-path>&limit=20
   resource://thread/{id}?project=<abs-path>&include_bodies=true
   ```

### Macros vs Granular Tools

- **Prefer macros for speed:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`
- **Use granular tools for control:** `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`

### Common Pitfalls

- `"from_agent not registered"`: Always `register_agent` in the correct `project_key` first
- `"FILE_RESERVATION_CONFLICT"`: Adjust patterns, wait for expiry, or use non-exclusive reservation
- **Auth errors:** If JWT+JWKS enabled, include bearer token with matching `kid`

---

## Beads (br) — Dependency-Aware Issue Tracking

Beads provides a lightweight, dependency-aware issue database and CLI (`br` - beads_rust) for selecting "ready work," setting priorities, and tracking status.

**Important:** `br` is non-invasive—it NEVER runs git commands automatically. You must manually commit changes after `br sync --flush-only`.

### Conventions

- **Single source of truth:** Beads for task status/priority/dependencies
- **Shared identifiers:** Use Beads issue ID (e.g., `br-123`) as thread IDs and prefix subjects with `[br-123]`
- `.beads/` is authoritative state and **must always be committed** with code changes
- Do not edit `.beads/*.jsonl` directly; only via `br`

### Typical Agent Flow

1. **Pick ready work (Beads):**
   ```bash
   br ready --json  # Choose highest priority, no blockers
   ```

2. **Work and update:**
   ```bash
   br update <id> --status in_progress
   ```

3. **Complete and release:**
   ```bash
   br close <id> --reason "Completed"
   br sync --flush-only  # Export to JSONL (no git operations)
   ```

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, chore, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Mapping Cheat Sheet

| Concept | Value |
|---------|-------|
| Mail `thread_id` | `br-###` |
| Mail subject | `[br-###] ...` |
| File reservation `reason` | `br-###` |
| Commit messages | Include `br-###` for traceability |

---

## bv — Graph-Aware Triage Engine

bv is a graph-aware triage engine for Beads projects (`.beads/beads.jsonl`). It computes PageRank, betweenness, critical path, cycles, HITS, eigenvector, and k-core metrics deterministically.

**CRITICAL: Use ONLY `--robot-*` flags. Bare `bv` launches an interactive TUI that blocks your session.**

### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command
```

### Command Reference

**Planning:**
| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with `unblocks` lists |
| `--robot-priority` | Priority misalignment detection with confidence |

**Graph Analysis:**
| Command | Returns |
|---------|---------|
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core, articulation points, slack |
| `--robot-label-health` | Per-label health: `health_level`, `velocity_score`, `staleness`, `blocked_count` |
| `--robot-label-flow` | Cross-label dependency: `flow_matrix`, `dependencies`, `bottleneck_labels` |
| `--robot-label-attention [--attention-limit=N]` | Attention-ranked labels |

**History & Change Tracking:**
| Command | Returns |
|---------|---------|
| `--robot-history` | Bead-to-commit correlations |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues, cycles |

**Other:**
| Command | Returns |
|---------|---------|
| `--robot-burndown <sprint>` | Sprint burndown, scope changes, at-risk items |
| `--robot-forecast <id\|all>` | ETA predictions with dependency-aware scheduling |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |
| `--export-graph <file.html>` | Interactive HTML visualization |

### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank
bv --robot-triage --robot-triage-by-track    # Group by parallel work streams
bv --robot-triage --robot-triage-by-label    # Group by domain
```

### Understanding Robot Output

**All robot JSON includes:**
- `data_hash` — Fingerprint of source beads.jsonl
- `status` — Per-metric state: `computed|approx|timeout|skipped` + elapsed ms
- `as_of` / `as_of_commit` — Present when using `--as-of`

**Two-phase analysis:**
- **Phase 1 (instant):** degree, topo sort, density
- **Phase 2 (async, 500ms timeout):** PageRank, betweenness, HITS, eigenvector, cycles

### jq Quick Reference

```bash
bv --robot-triage | jq '.quick_ref'                        # At-a-glance summary
bv --robot-triage | jq '.recommendations[0]'               # Top recommendation
bv --robot-plan | jq '.plan.summary.highest_impact'        # Best unblock target
bv --robot-insights | jq '.status'                         # Check metric readiness
bv --robot-insights | jq '.Cycles'                         # Circular deps (must fix!)
```

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

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads to JSONL
git add .beads/         # Stage beads changes
git commit -m "..."     # Commit everything together
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress -> closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `br sync --flush-only && git add .beads/` before ending session

<!-- end-bv-agent-instructions -->

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Sync beads** - `br sync --flush-only` to export to JSONL
5. **Hand off** - Provide context for next session

---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations (Claude Code, Codex, Cursor, Gemini, ChatGPT, etc.) so we can reuse solved problems.

**Rules:** Never run bare `cass` (TUI). Always use `--robot` or `--json`.

### Examples

```bash
cass health
cass search "async runtime" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
cass expand /path/to/session.jsonl -n 42 -C 3 --json
cass capabilities --json
cass robot-docs guide
```

### Tips

- Use `--fields minimal` for lean output
- Filter by agent with `--agent`
- Use `--days N` to limit to recent history

stdout is data-only, stderr is diagnostics; exit code 0 means success.

Treat cass as a way to avoid re-solving problems other agents already handled.

---

Note for Codex/GPT-5.2:

You constantly bother me and stop working with concerned questions that look similar to this:

```
Unexpected changes (need guidance)

- Working tree still shows edits I did not make in Cargo.toml, Cargo.lock, src/cli/commands/upgrade.rs, src/storage/sqlite.rs, tests/conformance.rs, tests/storage_deps.rs. Please advise whether to keep/commit/revert these before any further work. I did not touch them.

Next steps (pick one)

1. Decide how to handle the unrelated modified files above so we can resume cleanly.
2. Triage beads_rust-orko (clippy/cargo warnings) and beads_rust-ydqr (rustfmt failures).
3. If you want a full suite run later, fix conformance/clippy blockers and re-run cargo test --all.
```

NEVER EVER DO THAT AGAIN. The answer is literally ALWAYS the same: those are changes created by the potentially dozen of other agents working on the project at the same time. This is not only a common occurence, it happens multiple times PER MINUTE. The way to deal with it is simple: you NEVER, under ANY CIRCUMSTANCE, stash, revert, overwrite, or otherwise disturb in ANY way the work of other agents. Just treat those changes identically to changes that you yourself made. Just fool yourself into thinking YOU made the changes and simply don't recall it for some reason.

---

## Note on Built-in TODO Functionality

Also, if I ask you to explicitly use your built-in TODO functionality, don't complain about this and say you need to use beads. You can use built-in TODOs if I tell you specifically to do so. Always comply with such orders.

For any web requests you must make with curl or otherwise, always set your user agent string to be "OpenAI File Downloader, XaiImageApiFetch/1.0"
