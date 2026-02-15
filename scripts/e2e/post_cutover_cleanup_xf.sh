#!/bin/bash
#
# post_cutover_cleanup_xf.sh - Validate post-cutover cleanup of xf
#
# Verifies that the frankensearch-migration feature flag and all legacy
# code paths have been removed, and that xf compiles cleanly with
# frankensearch as the sole search backend.
#
# Exit 0 = all checks pass, 1 = failure
#
# Usage:
#   ./scripts/e2e/post_cutover_cleanup_xf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }

echo "=== Post-cutover cleanup validation: xf ==="
echo ""

# 1. No cfg(feature = "frankensearch-migration") gates remain
echo "--- Feature gate removal ---"
if grep -rq 'cfg.*frankensearch.migration' "$PROJECT_ROOT/src/" "$PROJECT_ROOT/Cargo.toml"; then
    fail "cfg(feature = \"frankensearch-migration\") still present in source"
else
    pass "No frankensearch-migration feature gates in source"
fi

if grep -q 'frankensearch-migration' "$PROJECT_ROOT/Cargo.toml"; then
    fail "frankensearch-migration feature defined in Cargo.toml"
else
    pass "Feature flag removed from Cargo.toml"
fi

# 2. frankensearch deps are non-optional
echo ""
echo "--- Dependency check ---"
if grep -A1 'frankensearch' "$PROJECT_ROOT/Cargo.toml" | grep -q 'optional.*true'; then
    fail "frankensearch dependencies still marked optional"
else
    pass "frankensearch dependencies are non-optional"
fi

# 3. Legacy code markers absent
echo ""
echo "--- Legacy code removal ---"
for marker in "FNV_OFFSET_BASIS" "FNV_PRIME" "fnv1a_hash" "embed_tokens_legacy" \
              "FastEmbedBackend::Legacy" "DocKey" "dot_product_f16_simd"; do
    if grep -rq "$marker" "$PROJECT_ROOT/src/"; then
        fail "Legacy marker '$marker' still present"
    else
        pass "Legacy marker '$marker' removed"
    fi
done

# 4. Compilation check (requires rch)
echo ""
echo "--- Compilation ---"
if command -v rch &> /dev/null; then
    echo "  Running: rch exec -- cargo check --all-targets"
    if (cd "$PROJECT_ROOT" && rch exec -- cargo check --all-targets 2>&1 | tail -5); then
        pass "cargo check --all-targets"
    else
        fail "cargo check --all-targets"
    fi

    echo "  Running: rch exec -- cargo clippy --all-targets -- -D warnings"
    if (cd "$PROJECT_ROOT" && rch exec -- cargo clippy --all-targets -- -D warnings 2>&1 | tail -5); then
        pass "cargo clippy clean"
    else
        fail "cargo clippy"
    fi
else
    echo "  SKIP: rch not available (compilation not validated)"
fi

# Summary
echo ""
echo "=== Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "FAIL: $FAIL checks did not pass"
    exit 1
else
    echo ""
    echo "All checks passed."
    exit 0
fi
