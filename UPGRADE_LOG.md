# Dependency Upgrade Log

**Date:** 2026-04-27  |  **Project:** xf (X Archive Finder)  |  **Language:** Rust (nightly, edition 2024)

## Baseline
- Toolchain: `rustc 1.97.0-nightly (ca9a134e0 2026-04-26)` / `cargo 1.97.0-nightly`
- Starting version: `xf 0.3.1`
- `cargo check --all-targets`: ✓ clean

## Outdated direct dependencies (pre-update)

| Crate | Current | Latest | Notes |
|-------|---------|--------|-------|
| clap_complete | 4.6.2 | 4.6.3 | patch |
| colored | 2.2.0 | 3.1.1 | major |
| console | 0.15.11 | 0.16.3 | minor |
| criterion (dev) | 0.5.1 | 0.8.2 | major |
| dirs | 5.0.1 | 6.0.0 | major |
| fastembed | 5.13.3 | 5.13.4 | patch |
| indicatif | 0.17.11 | 0.18.4 | minor |
| itertools | 0.13.0 | 0.14.0 | minor |
| nix (dev) | 0.30.1 | 0.31.2 | minor |
| rand | 0.8.6 | 0.10.1 | major (×2) |
| rusqlite | 0.32.1 | 0.39.0 | major |
| rustyline | 12.0.0 | 18.0.0 | many majors |
| safetensors | 0.5.3 | 0.7.0 | minor |
| tantivy | 0.22.1 | 0.26.1 | minor |
| tokenizers | 0.21.4 | 0.23.1 | minor |
| toml | 0.8.23 | 1.1.2 | major |
| tru | 0.2.2 | 0.2.3 | patch |
| which (dev) | 7.0.3 | 8.0.2 | major |
| wide | 0.7.33 | 1.3.0 | major |
| zip | 2.4.2 | 8.6.0 | many majors |

## Updates

### clap_complete: 4.5 → 4.6 (4.6.3)
- **Breaking:** None
- **Tests:** ✓ build clean

### fastembed: 5.13.3 → 5.13.4 (patch via lockfile)
- **Breaking:** None

### tru: 0.2.2 → 0.2.3 (patch via lockfile)
- **Breaking:** None

### tokenizers: 0.21 → 0.23 (0.23.1)
- **Breaking:** None observed in our usage (we don't call deprecated APIs)
- **Side-effect:** Pulled newer indicatif/console transitively

### console: 0.15 → 0.16 (0.16.3)
- **Breaking:** None observed in our usage

### indicatif: 0.17 → 0.18 (0.18.4)
- **Breaking:** None observed in our usage

### itertools: 0.13 → 0.14
- **Breaking:** None observed

### nix (dev): 0.30 → 0.31
- **Breaking:** None observed

### safetensors: 0.5 → 0.7
- **Breaking:** `SafeTensors::names()` now returns `Vec<&str>` directly (no longer wrapping iterator)
- **Migration:** Replaced `.into_iter().cloned().collect()` with `.into_iter().map(String::from).collect()` in `src/model2vec_embedder.rs:134`
- **Tests:** ✓ build clean

### rand: 0.8 → 0.10
- **Breaking:** `Rng::gen_range` removed; `RngExt::random_range` added
- **Migration:** `use rand::{RngExt, ...}`; `gen_range` → `random_range` in `src/benchmark/metrics.rs`
- **Tests:** ✓ build clean

### dirs: 5.0 → 6.0
- **Breaking:** None observed in our usage

### colored: 2.1 → 3.0 (3.1.1)
- **Breaking:** None observed in our usage

### criterion (dev): 0.5 → 0.8
- **Breaking:** `criterion::black_box` deprecated in favor of `std::hint::black_box`
- **Migration:** Switched import in `benches/search_perf.rs`
- **Tests:** ✓ build clean

### wide: 0.7 → 1.3
- **Breaking:** None observed (`f32x8::from(arr)`, `ZERO`, `reduce_add` stable)

### toml: 0.8 → 1.1
- **Breaking:** None observed in our usage (`toml::from_str`, `Table`, `Value::as_*`)

### tantivy: 0.22 → 0.26
- **Breaking:** `TopDocs::with_limit(N)` is no longer a `Collector`. Need `.order_by_score()` for default scoring.
- **Migration:** Three call sites in `src/search.rs` updated.
- **Tests:** ✓ build clean

### rusqlite: 0.32 → 0.39
- **Breaking:** None observed in our usage

### rustyline: 12 → 18
- **Breaking:** `Highlighter::highlight_char` signature gained `CmdKind` parameter
- **Migration:** Added `_kind: rustyline::highlight::CmdKind` arg in `src/repl.rs:731`
- **Tests:** ✓ build clean

### zip: 2.2 → 8.6
- **Breaking:** None observed (we only use `zip::ZipArchive::new`)

### ort: 2.0.0-rc.11 → 2.0.0-rc.12
- **Breaking:** None observed in our usage

### which (dev): 7.0 → 8.0
- **Breaking:** None observed

## Summary

- **Updated:** 21 direct deps + many transitive
- **Skipped:** 0
- **Failed:** 0
- All compile errors resolved; `cargo check --all-targets` clean
