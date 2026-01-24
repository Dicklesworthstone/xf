#!/bin/bash
# Embedding Model Discovery E2E Validation Script
# Bead: bd-e9kz
# Validates the curated embedding candidates list

set -euo pipefail

LOG=/tmp/embedding_discovery_e2e.log
DOCS_DIR="$(dirname "$0")/../../docs"
CURATED_JSON="$DOCS_DIR/embedding_candidates_curated.json"
ANALYSIS_MD="$DOCS_DIR/embedding_candidates_analysis.md"
DISCOVERY_LOG="$DOCS_DIR/embedding_candidates_discovery_log.txt"

echo "========================================" | tee "$LOG"
echo "Embedding Discovery E2E Validation" | tee -a "$LOG"
echo "Date: $(date -Iseconds)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Environment info
echo "" | tee -a "$LOG"
echo "Environment:" | tee -a "$LOG"
echo "  OS: $(uname -s) $(uname -r)" | tee -a "$LOG"
echo "  Host: $(hostname)" | tee -a "$LOG"
echo "  Git SHA: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "  Rust: $(rustc --version 2>/dev/null || echo 'N/A')" | tee -a "$LOG"

# Check required files exist
echo "" | tee -a "$LOG"
echo "Checking required files..." | tee -a "$LOG"

PASS=true

if [[ -f "$CURATED_JSON" ]]; then
    echo "  [PASS] Curated JSON exists: $CURATED_JSON" | tee -a "$LOG"
else
    echo "  [FAIL] Missing: $CURATED_JSON" | tee -a "$LOG"
    PASS=false
fi

if [[ -f "$ANALYSIS_MD" ]]; then
    echo "  [PASS] Analysis MD exists: $ANALYSIS_MD" | tee -a "$LOG"
else
    echo "  [FAIL] Missing: $ANALYSIS_MD" | tee -a "$LOG"
    PASS=false
fi

if [[ -f "$DISCOVERY_LOG" ]]; then
    echo "  [PASS] Discovery log exists: $DISCOVERY_LOG" | tee -a "$LOG"
else
    echo "  [FAIL] Missing: $DISCOVERY_LOG" | tee -a "$LOG"
    PASS=false
fi

# Validate JSON schema
echo "" | tee -a "$LOG"
echo "Validating JSON structure..." | tee -a "$LOG"

if command -v jq &> /dev/null && [[ -f "$CURATED_JSON" ]]; then
    # Check required top-level keys
    REQUIRED_KEYS=("metadata" "summary" "eligible" "baselines" "rejected" "decision_gate")
    for key in "${REQUIRED_KEYS[@]}"; do
        if jq -e ".$key" "$CURATED_JSON" > /dev/null 2>&1; then
            echo "  [PASS] Key '$key' present" | tee -a "$LOG"
        else
            echo "  [FAIL] Key '$key' missing" | tee -a "$LOG"
            PASS=false
        fi
    done

    # Count models
    ELIGIBLE_COUNT=$(jq '.eligible | length' "$CURATED_JSON")
    BASELINE_COUNT=$(jq '.baselines | length' "$CURATED_JSON")
    REJECTED_COUNT=$(jq '.rejected | length' "$CURATED_JSON")

    echo "" | tee -a "$LOG"
    echo "Model counts:" | tee -a "$LOG"
    echo "  Eligible: $ELIGIBLE_COUNT" | tee -a "$LOG"
    echo "  Baselines: $BASELINE_COUNT" | tee -a "$LOG"
    echo "  Rejected: $REJECTED_COUNT" | tee -a "$LOG"

    # Verify cutoff date compliance
    echo "" | tee -a "$LOG"
    echo "Verifying cutoff date compliance..." | tee -a "$LOG"
    CUTOFF="2025-11-01"

    # Check all eligible models are >= cutoff
    ELIGIBLE_DATES=$(jq -r '.eligible[].release_date' "$CURATED_JSON" 2>/dev/null || echo "")
    if [[ -n "$ELIGIBLE_DATES" ]]; then
        while IFS= read -r date; do
            if [[ "$date" > "$CUTOFF" ]] || [[ "$date" == "$CUTOFF"* ]]; then
                echo "  [PASS] Eligible model date $date >= $CUTOFF" | tee -a "$LOG"
            else
                echo "  [FAIL] Eligible model date $date < $CUTOFF" | tee -a "$LOG"
                PASS=false
            fi
        done <<< "$ELIGIBLE_DATES"
    fi

    # Check all baselines are < cutoff
    BASELINE_DATES=$(jq -r '.baselines[].release_date' "$CURATED_JSON" 2>/dev/null || echo "")
    if [[ -n "$BASELINE_DATES" ]]; then
        while IFS= read -r date; do
            if [[ "$date" < "$CUTOFF" ]]; then
                echo "  [PASS] Baseline model date $date < $CUTOFF" | tee -a "$LOG"
            else
                echo "  [WARN] Baseline model date $date >= $CUTOFF (should be eligible?)" | tee -a "$LOG"
            fi
        done <<< "$BASELINE_DATES"
    fi
else
    echo "  [SKIP] jq not available or JSON missing" | tee -a "$LOG"
fi

# Summary
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
if $PASS; then
    echo "RESULT: ALL CHECKS PASSED" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    exit 0
else
    echo "RESULT: SOME CHECKS FAILED" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    exit 1
fi
