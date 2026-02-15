# Decommission Validation Report: bd-3un.35.4

**Bead**: bd-3un.35.4 — Post-cutover cleanup: remove legacy search stack from xf
**Agent**: CrimsonShore
**Date**: 2026-02-15
**Start gate**: bd-3un.35 (closed) — frankensearch parity confirmed

## Gate-by-gate evidence

### 1. Feature flag removal

| File | Gate location | Status |
|------|-------------|--------|
| Cargo.toml | `[features] frankensearch-migration = [...]` | Removed |
| src/embedder.rs | 4 cfg gates on imports + adapter code | Removed |
| src/reranker.rs | 3 cfg gates on imports + adapter code | Removed |
| src/hash_embedder.rs | 5 cfg gates on delegate, embed, tests | Removed |
| src/fastembed_embedder.rs | 12 cfg gates on backend enum, load, embed | Removed |
| src/model2vec_embedder.rs | 8 cfg gates on delegate, embed_internal | Removed |
| src/hybrid.rs | 10 cfg gates on RRF, imports, candidate_count | Removed |
| src/vector.rs | 5 cfg gates on dot_product, tests | Removed |
| src/config.rs | 3 cfg gates on TwoTierConfig, to_frankensearch | Removed |
| src/model_registry.rs | 4 cfg gates on adapter imports + methods | Removed |
| scripts/verify_isomorphism.sh | Migration parity lane + flags | Removed |

**Total**: 62 cfg gates removed across 11 files.

### 2. Compilation validation

| Check | Tool | Result |
|-------|------|--------|
| `cargo check --all-targets` | rch (vmi1153651) | PASS (exit=0) |
| `cargo clippy --all-targets -- -D warnings` | rch (vmi1153651) | PASS (exit=0, 0 warnings) |

### 3. UBS scan

| Severity | Count |
|----------|-------|
| Critical | 0 |
| Warning | 707 (pre-existing, not introduced by cleanup) |
| Info | 245 |

### 4. Cx import fix

During cleanup, discovered that `frankensearch_core::Cx` re-export was not available on the rch worker's cached copy. Fixed by importing `Cx` directly from `asupersync` (which xf already depends on as a path dep). This is more robust than depending on the re-export.

### 5. asupersync path dep fix

Original xf had `asupersync = { version = "0.2.0", optional = true }` from crates.io. This caused a type mismatch with frankensearch's patched asupersync (from path). Fixed by changing to `asupersync = { path = "/data/projects/asupersync" }` so both xf and frankensearch use the same crate instance.

## Retained code (with rationale)

- **FrankensearchEmbedderAdapter / FrankensearchRerankerAdapter**: sync-to-async bridges required by frankensearch's async trait interfaces
- **Model2Vec tokenizer/embeddings fields**: loaded at construction but unused at embed time; `#[allow(dead_code)]` applied (deeper refactor deferred)
- **search.rs (Tantivy)**: not in migration scope; separate full-text search capability
- **static_mrl_embedder.rs, flashrank_reranker.rs, mxbai_reranker.rs**: no frankensearch equivalents; standalone implementations

## Diff stats

```
 11 files changed, 84 insertions(+), 573 deletions(-)
```
