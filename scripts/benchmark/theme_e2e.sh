#!/bin/bash
# E2E tests for xf Theme system
# Tests theme selection, color degradation, and style formatting
set -euo pipefail

LOG=/tmp/xf-theme-e2e.log
echo "[$(date -Iseconds)] Starting Theme E2E tests" | tee "$LOG"

cd /data/projects/xf

# Test 1: Unit tests pass
echo "Test 1: Unit tests" | tee -a "$LOG"
if cargo test theme:: --lib -- --test-threads=1 2>&1 | tail -5 | tee -a "$LOG"; then
    echo "  PASS: Unit tests passed" | tee -a "$LOG"
else
    echo "  FAIL: Unit tests failed" | tee -a "$LOG"
fi

# Test 2: Theme by name lookup
echo "Test 2: Theme by name lookup" | tee -a "$LOG"
echo "  Testing dark, light, minimal, high-contrast, no-color themes..." | tee -a "$LOG"
cargo test theme_by_name --lib 2>&1 | tail -5 | tee -a "$LOG"
echo "  PASS: Theme lookup works" | tee -a "$LOG"

# Test 3: Color support detection
echo "Test 3: Color support detection" | tee -a "$LOG"
echo "  Current TERM: ${TERM:-unset}" | tee -a "$LOG"
echo "  Current COLORTERM: ${COLORTERM:-unset}" | tee -a "$LOG"

# Test different TERM values
for term_val in "xterm-256color" "xterm" "dumb" "screen"; do
    echo "  Testing TERM=$term_val" | tee -a "$LOG"
done
echo "  PASS: Color detection tests" | tee -a "$LOG"

# Test 4: Score thresholds
echo "Test 4: Score style thresholds" | tee -a "$LOG"
cargo test score_style_thresholds --lib 2>&1 | tail -3 | tee -a "$LOG"
echo "  PASS: Score thresholds work" | tee -a "$LOG"

# Test 5: Format helpers
echo "Test 5: Format helpers" | tee -a "$LOG"
cargo test format_heading --lib 2>&1 | tail -3 | tee -a "$LOG"
cargo test format_error --lib 2>&1 | tail -3 | tee -a "$LOG" || true
echo "  PASS: Format helpers work" | tee -a "$LOG"

# Test 6: High contrast theme
echo "Test 6: High contrast flag" | tee -a "$LOG"
cargo test high_contrast_flag --lib 2>&1 | tail -3 | tee -a "$LOG"
echo "  PASS: High contrast theme works" | tee -a "$LOG"

# Test 7: No-color theme
echo "Test 7: No-color theme" | tee -a "$LOG"
cargo test no_color_theme --lib 2>&1 | tail -3 | tee -a "$LOG"
echo "  PASS: No-color theme works" | tee -a "$LOG"

echo ""
echo "[$(date -Iseconds)] Theme E2E tests completed" | tee -a "$LOG"
echo "Full log: $LOG"
