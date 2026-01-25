#!/bin/bash
# Reranker Implementation E2E Test Script
# Runs comprehensive tests for FlashRank and mxbai rerankers.
#
# Bead: bd-2q67
#
# Usage:
#   ./scripts/benchmark/reranker_e2e.sh
#
# Prerequisites:
#   - Model files must be downloaded and cached in ~/.cache/xf/models/
#   - Run 'xf doctor' to verify model availability

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/tests/e2e/results"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/reranker_$(date +%Y%m%d_%H%M%S).log"
PASS=0
FAIL=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
pass() { ((PASS++)); log "PASS: $*"; }
fail() { ((FAIL++)); log "FAIL: $*"; }

cd "$PROJECT_ROOT"

log "=== Reranker Implementation E2E ==="
log "Log file: $LOG"
log "Project root: $PROJECT_ROOT"
log ""

# Check model availability
log "Checking model availability..."
if [ -d "$HOME/.cache/xf/models/flashrank-nano" ]; then
    log "  FlashRank: found"
else
    log "  FlashRank: NOT FOUND - some tests will be skipped"
fi

if [ -d "$HOME/.cache/xf/models/mxbai-rerank-xsmall" ]; then
    log "  mxbai: found"
else
    log "  mxbai: NOT FOUND - some tests will be skipped"
fi
log ""

# Run unit tests (constants, bad path, etc.)
log "=== Unit Tests (no model required) ==="
if cargo test --test reranker_tests -- --skip ignore 2>&1 | tee -a "$LOG" | tail -20; then
    pass "Unit tests (no model required)"
else
    fail "Unit tests (no model required)"
fi
log ""

# Check if models are available for full tests
MODELS_AVAILABLE=false
if [ -d "$HOME/.cache/xf/models/flashrank-nano" ] || [ -d "$HOME/.cache/xf/models/mxbai-rerank-xsmall" ]; then
    MODELS_AVAILABLE=true
fi

if [ "$MODELS_AVAILABLE" = true ]; then
    log "=== Full Tests (model required) ==="
    log "Running tests that require model files..."

    # Run loading tests
    log "--- Loading Tests ---"
    if cargo test --test reranker_tests loading --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Loading tests"
    else
        fail "Loading tests"
    fi

    # Run relevance tests
    log "--- Relevance Ordering Tests ---"
    if cargo test --test reranker_tests relevance --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Relevance ordering tests"
    else
        fail "Relevance ordering tests"
    fi

    # Run calibration tests
    log "--- Score Calibration Tests ---"
    if cargo test --test reranker_tests calibration --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Score calibration tests"
    else
        fail "Score calibration tests"
    fi

    # Run edge case tests
    log "--- Edge Case Tests ---"
    if cargo test --test reranker_tests edge_cases --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Edge case tests"
    else
        fail "Edge case tests"
    fi

    # Run comparison tests
    log "--- Comparison Tests ---"
    if cargo test --test reranker_tests comparison --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Comparison tests"
    else
        fail "Comparison tests"
    fi

    # Run pipeline integration tests
    log "--- Pipeline Integration Tests ---"
    if cargo test --test reranker_tests pipeline --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Pipeline integration tests"
    else
        fail "Pipeline integration tests"
    fi

    # Run thread safety tests
    log "--- Thread Safety Tests ---"
    if cargo test --test reranker_tests thread_safety --release -- --include-ignored 2>&1 | tee -a "$LOG" | tail -10; then
        pass "Thread safety tests"
    else
        fail "Thread safety tests"
    fi

    # Run performance tests (slower)
    log "--- Performance Tests ---"
    log "Running performance benchmarks (this may take a while)..."
    if cargo test --test reranker_tests performance --release -- --include-ignored --nocapture 2>&1 | tee -a "$LOG" | tail -20; then
        pass "Performance tests"
    else
        fail "Performance tests"
    fi
else
    log "=== Skipping Full Tests (no models available) ==="
    log "Download models using 'xf index --semantic' on a sample archive to populate cache."
fi

log ""
log "========================="
log "Results: $PASS passed, $FAIL failed"
log "Full log: $LOG"

if [ "$FAIL" -gt 0 ]; then
    log "OVERALL: FAILED"
    exit 1
else
    log "OVERALL: PASSED"
    exit 0
fi
