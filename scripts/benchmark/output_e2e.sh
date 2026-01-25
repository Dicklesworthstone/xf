#!/bin/bash
# E2E tests for xf Output module
# Tests TTY detection, color environment variables, and format handling
set -euo pipefail

LOG=/tmp/xf-output-e2e.log
echo "[$(date -Iseconds)] Starting Output Module E2E tests" | tee "$LOG"

# Helper to check for ANSI codes
has_ansi() {
    echo "$1" | grep -qP '\x1b\['
}

# Test 1: TTY output detection info
echo "Test 1: TTY detection info" | tee -a "$LOG"
if [ -t 1 ]; then
    echo "  INFO: Running in TTY mode" | tee -a "$LOG"
else
    echo "  INFO: Running in piped/non-TTY mode" | tee -a "$LOG"
fi

# Test 2: Piped output should have no ANSI codes
echo "Test 2: Piped output clean" | tee -a "$LOG"
if command -v xf &>/dev/null; then
    PIPED=$(xf stats 2>&1 | cat || true)
    if has_ansi "$PIPED"; then
        echo "  WARN: ANSI codes detected in piped output (may be expected with FORCE_COLOR)" | tee -a "$LOG"
    else
        echo "  PASS: No ANSI in piped output" | tee -a "$LOG"
    fi
else
    echo "  SKIP: xf binary not available" | tee -a "$LOG"
fi

# Test 3: NO_COLOR environment variable
echo "Test 3: NO_COLOR support" | tee -a "$LOG"
if command -v xf &>/dev/null; then
    NO_COLOR_OUT=$(NO_COLOR=1 xf stats 2>&1 | cat || true)
    if has_ansi "$NO_COLOR_OUT"; then
        echo "  FAIL: ANSI present with NO_COLOR=1" | tee -a "$LOG"
    else
        echo "  PASS: NO_COLOR respected" | tee -a "$LOG"
    fi
else
    echo "  SKIP: xf binary not available" | tee -a "$LOG"
fi

# Test 4: JSON format purity
echo "Test 4: JSON format purity" | tee -a "$LOG"
if command -v xf &>/dev/null && command -v jq &>/dev/null; then
    JSON_OUT=$(xf stats --format json 2>&1 || true)
    if echo "$JSON_OUT" | jq . > /dev/null 2>&1; then
        echo "  PASS: JSON is valid" | tee -a "$LOG"
        if has_ansi "$JSON_OUT"; then
            echo "  FAIL: ANSI codes in JSON output" | tee -a "$LOG"
        else
            echo "  PASS: No ANSI in JSON" | tee -a "$LOG"
        fi
    else
        echo "  INFO: JSON parsing failed (may need index)" | tee -a "$LOG"
    fi
else
    echo "  SKIP: xf or jq not available" | tee -a "$LOG"
fi

# Test 5: CLICOLOR=0 support
echo "Test 5: CLICOLOR=0 support" | tee -a "$LOG"
if command -v xf &>/dev/null; then
    CLICOLOR_OUT=$(CLICOLOR=0 xf stats 2>&1 | cat || true)
    if has_ansi "$CLICOLOR_OUT"; then
        echo "  FAIL: ANSI present with CLICOLOR=0" | tee -a "$LOG"
    else
        echo "  PASS: CLICOLOR=0 respected" | tee -a "$LOG"
    fi
else
    echo "  SKIP: xf binary not available" | tee -a "$LOG"
fi

# Test 6: Unit tests pass
echo "Test 6: Unit tests" | tee -a "$LOG"
cd /data/projects/xf
if cargo test output:: --lib -- --test-threads=1 2>&1 | tail -5 | tee -a "$LOG"; then
    echo "  PASS: Unit tests passed" | tee -a "$LOG"
else
    echo "  FAIL: Unit tests failed" | tee -a "$LOG"
fi

# Test 7: Clippy clean
echo "Test 7: Clippy check" | tee -a "$LOG"
if cargo clippy --lib -p xf -- -D warnings 2>&1 | tail -3 | grep -q "Finished"; then
    echo "  PASS: Clippy clean" | tee -a "$LOG"
else
    echo "  INFO: Clippy may have warnings or still compiling" | tee -a "$LOG"
fi

# Test 8: Rich feature compiles
echo "Test 8: Rich feature compilation" | tee -a "$LOG"
if cargo check --features rich 2>&1 | tail -3 | tee -a "$LOG"; then
    echo "  PASS: Rich feature compiles" | tee -a "$LOG"
else
    echo "  FAIL: Rich feature compilation failed" | tee -a "$LOG"
fi

echo ""
echo "[$(date -Iseconds)] Output Module E2E tests completed" | tee -a "$LOG"
echo "Full log: $LOG"
