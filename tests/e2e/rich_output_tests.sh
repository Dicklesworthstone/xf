#!/usr/bin/env bash
#
# xf Rich Output Integration E2E Test Suite
# Tests all rich_rust integration features across TTY and non-TTY modes
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/test-output"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rich_output_e2e_$(date +%Y%m%d_%H%M%S).log"
DETAIL_LOG="$LOG_FILE"

XF_BIN="${XF_BIN:-$ROOT_DIR/target/debug/xf}"

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
SKIP_COUNT=0

# Logging utilities
log() {
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S.%3N')"
    echo "[$ts] $msg" | tee -a "$LOG_FILE"
}

log_test() {
    local name="$1"
    echo "" | tee -a "$LOG_FILE"
    log "[TEST] $name"
}

log_pass() {
    local msg="$1"
    ((PASS_COUNT++)) || true
    log "  PASS: $msg"
}

log_fail() {
    local msg="$1"
    ((FAIL_COUNT++)) || true
    log "  FAIL: $msg"
}

log_warn() {
    local msg="$1"
    ((WARN_COUNT++)) || true
    log "  WARN: $msg"
}

log_skip() {
    local msg="$1"
    ((SKIP_COUNT++)) || true
    log "  SKIP: $msg"
}

# Check for ANSI escape codes
check_no_ansi() {
    local input="$1"
    local context="$2"
    if echo "$input" | grep -qP '\x1b\[' 2>/dev/null; then
        log_fail "$context: Found ANSI escape codes in output"
        return 1
    fi
    log_pass "$context: No ANSI codes"
    return 0
}

# Check for valid JSON
check_valid_json() {
    local input="$1"
    local context="$2"
    if echo "$input" | jq . > /dev/null 2>&1; then
        log_pass "$context: Valid JSON"
        return 0
    else
        log_fail "$context: Invalid JSON"
        return 1
    fi
}

# Measure execution time in ms
measure_time() {
    local start end duration
    start=$(date +%s%N)
    "$@" > /dev/null 2>&1 || true
    end=$(date +%s%N)
    echo $(( (end - start) / 1000000 ))
}

# Save original env vars for cleanup
save_env() {
    ORIG_NO_COLOR="${NO_COLOR:-}"
    ORIG_FORCE_COLOR="${FORCE_COLOR:-}"
    ORIG_CLICOLOR="${CLICOLOR:-}"
    ORIG_CLICOLOR_FORCE="${CLICOLOR_FORCE:-}"
    ORIG_XF_OUTPUT_FORMAT="${XF_OUTPUT_FORMAT:-}"
}

# Restore original env vars
restore_env() {
    if [ -n "$ORIG_NO_COLOR" ]; then
        export NO_COLOR="$ORIG_NO_COLOR"
    else
        unset NO_COLOR 2>/dev/null || true
    fi
    if [ -n "$ORIG_FORCE_COLOR" ]; then
        export FORCE_COLOR="$ORIG_FORCE_COLOR"
    else
        unset FORCE_COLOR 2>/dev/null || true
    fi
    if [ -n "$ORIG_CLICOLOR" ]; then
        export CLICOLOR="$ORIG_CLICOLOR"
    else
        unset CLICOLOR 2>/dev/null || true
    fi
    if [ -n "$ORIG_CLICOLOR_FORCE" ]; then
        export CLICOLOR_FORCE="$ORIG_CLICOLOR_FORCE"
    else
        unset CLICOLOR_FORCE 2>/dev/null || true
    fi
    if [ -n "$ORIG_XF_OUTPUT_FORMAT" ]; then
        export XF_OUTPUT_FORMAT="$ORIG_XF_OUTPUT_FORMAT"
    else
        unset XF_OUTPUT_FORMAT 2>/dev/null || true
    fi
}

# Clean env vars for isolated testing
clean_env() {
    unset NO_COLOR 2>/dev/null || true
    unset FORCE_COLOR 2>/dev/null || true
    unset CLICOLOR 2>/dev/null || true
    unset CLICOLOR_FORCE 2>/dev/null || true
    unset XF_OUTPUT_FORMAT 2>/dev/null || true
}

# Setup and cleanup
save_env
trap restore_env EXIT

# Build if needed
if [ ! -x "$XF_BIN" ]; then
    log "xf binary not found at $XF_BIN; building"
    (cd "$ROOT_DIR" && cargo build --features rich)
fi

# Check if archive is indexed
HAS_ARCHIVE=false
if "$XF_BIN" stats --format json 2>/dev/null | jq . >/dev/null 2>&1; then
    HAS_ARCHIVE=true
fi

log "=== xf Rich Output E2E Test Suite Starting ==="
log "Log file: $LOG_FILE"
log "XF binary: $XF_BIN"
log "Archive indexed: $HAS_ARCHIVE"

# ============================================================
# Section 1: Output Format Tests (require archive)
# ============================================================
log ""
log "=== Section 1: Output Format Tests ==="

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "1.1 JSON format validity"
    clean_env
    JSON_OUT=$("$XF_BIN" stats --format json 2>&1 || echo "{}")
    check_valid_json "$JSON_OUT" "Stats JSON"

    log_test "1.2 JSON format no ANSI"
    check_no_ansi "$JSON_OUT" "Stats JSON"

    log_test "1.3 CSV format no ANSI"
    clean_env
    CSV_OUT=$("$XF_BIN" stats --format csv 2>&1 || echo "")
    check_no_ansi "$CSV_OUT" "Stats CSV"
else
    log_skip "Section 1: Output format tests require indexed archive"
fi

# ============================================================
# Section 2: Piped Output Tests
# ============================================================
log ""
log "=== Section 2: Piped Output Tests ==="

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "2.1 Piped search output"
    clean_env
    PIPED_SEARCH=$("$XF_BIN" search "test" --limit 5 2>&1 | cat || echo "")
    check_no_ansi "$PIPED_SEARCH" "Piped search"

    log_test "2.2 Piped stats output"
    clean_env
    PIPED_STATS=$("$XF_BIN" stats 2>&1 | cat)
    check_no_ansi "$PIPED_STATS" "Piped stats"
else
    log_skip "2.1-2.2: Piped search/stats tests require indexed archive"
fi

log_test "2.3 Piped doctor output"
clean_env
PIPED_DOCTOR=$("$XF_BIN" doctor 2>&1 | cat || echo "")
check_no_ansi "$PIPED_DOCTOR" "Piped doctor"

log_test "2.4 Piped help output"
clean_env
PIPED_HELP=$("$XF_BIN" --help 2>&1 | cat)
check_no_ansi "$PIPED_HELP" "Piped help"

# ============================================================
# Section 3: Environment Variable Tests
# ============================================================
log ""
log "=== Section 3: Environment Variable Tests ==="

log_test "3.1 NO_COLOR disables styling (help)"
clean_env
export NO_COLOR=1
NO_COLOR_OUT=$("$XF_BIN" --help 2>&1 || echo "")
if check_no_ansi "$NO_COLOR_OUT" "NO_COLOR"; then
    log_pass "NO_COLOR respected"
fi
unset NO_COLOR

log_test "3.2 CLICOLOR=0 disables styling (help)"
clean_env
export CLICOLOR=0
CLICOLOR_OUT=$("$XF_BIN" --help 2>&1 || echo "")
if check_no_ansi "$CLICOLOR_OUT" "CLICOLOR=0"; then
    log_pass "CLICOLOR=0 respected"
fi
unset CLICOLOR

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "3.3 XF_OUTPUT_FORMAT=json works"
    clean_env
    export XF_OUTPUT_FORMAT=json
    ENV_JSON=$("$XF_BIN" stats 2>&1 || echo "{}")
    check_valid_json "$ENV_JSON" "XF_OUTPUT_FORMAT=json"
    unset XF_OUTPUT_FORMAT
else
    log_skip "3.3: XF_OUTPUT_FORMAT test requires indexed archive"
fi

# ============================================================
# Section 4: Verbosity Tests
# ============================================================
log ""
log "=== Section 4: Verbosity Tests ==="

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "4.1 Quiet mode"
    clean_env
    QUIET_OUT=$("$XF_BIN" stats -q 2>&1 || echo "")
    NORMAL_OUT=$("$XF_BIN" stats 2>&1 || echo "")
    QUIET_LINES=$(echo "$QUIET_OUT" | wc -l)
    NORMAL_LINES=$(echo "$NORMAL_OUT" | wc -l)
    if [ "$QUIET_LINES" -le "$NORMAL_LINES" ]; then
        log_pass "Quiet mode reduces output ($QUIET_LINES <= $NORMAL_LINES)"
    else
        log_warn "Quiet mode unclear"
    fi

    log_test "4.2 Verbose mode"
    clean_env
    VERBOSE_OUT=$("$XF_BIN" stats -v 2>&1 || echo "")
    VERBOSE_LINES=$(echo "$VERBOSE_OUT" | wc -l)
    if [ "$VERBOSE_LINES" -ge "$NORMAL_LINES" ]; then
        log_pass "Verbose mode increases output ($VERBOSE_LINES >= $NORMAL_LINES)"
    else
        log_warn "Verbose mode unclear"
    fi
else
    log_skip "Section 4: Verbosity tests require indexed archive"
fi

# ============================================================
# Section 5: Command-Specific Tests
# ============================================================
log ""
log "=== Section 5: Command-Specific Tests ==="

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "5.1 Search command output"
    clean_env
    SEARCH_OUT=$("$XF_BIN" search "test" --limit 3 2>&1 || echo "")
    if [ -n "$SEARCH_OUT" ]; then
        log_pass "Search command works"
    else
        log_warn "Search returned empty (may be no matches)"
    fi

    log_test "5.2 Stats command output"
    clean_env
    STATS_OUT=$("$XF_BIN" stats 2>&1 || echo "")
    if echo "$STATS_OUT" | grep -qiE 'tweet|document|total|count|indexed'; then
        log_pass "Stats shows expected fields"
    else
        log_warn "Stats format unclear"
    fi
else
    log_skip "5.1-5.2: Search/stats tests require indexed archive"
fi

log_test "5.3 Doctor command output"
clean_env
DOCTOR_OUT=$("$XF_BIN" doctor 2>&1 || true)
if echo "$DOCTOR_OUT" | grep -qiE 'OK|PASS|FAIL|WARN|healthy|warning|error|check|archive'; then
    log_pass "Doctor shows status indicators"
else
    log_warn "Doctor format unclear"
fi

log_test "5.4 Help command output"
clean_env
HELP_OUT=$("$XF_BIN" --help 2>&1)
if echo "$HELP_OUT" | grep -qiE 'usage|command|options'; then
    log_pass "Help shows expected sections"
else
    log_fail "Help missing expected content"
fi

log_test "5.5 Robot-docs command output"
clean_env
ROBOT_OUT=$("$XF_BIN" robot-docs 2>&1 || echo "{}")
if check_valid_json "$ROBOT_OUT" "Robot-docs JSON"; then
    log_pass "Robot-docs produces valid JSON"
fi

# ============================================================
# Section 6: Performance Tests
# ============================================================
log ""
log "=== Section 6: Performance Tests ==="

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "6.1 Stats performance"
    clean_env
    STATS_TIME=$(measure_time "$XF_BIN" stats)
    log "  Stats: ${STATS_TIME}ms"
    if [ "$STATS_TIME" -lt 5000 ]; then
        log_pass "Stats < 5000ms (${STATS_TIME}ms)"
    else
        log_warn "Stats slow: ${STATS_TIME}ms"
    fi
else
    log_skip "6.1: Stats performance test requires indexed archive"
fi

log_test "6.2 Doctor performance"
clean_env
DOCTOR_TIME=$(measure_time "$XF_BIN" doctor)
log "  Doctor: ${DOCTOR_TIME}ms"
if [ "$DOCTOR_TIME" -lt 5000 ]; then
    log_pass "Doctor < 5000ms (${DOCTOR_TIME}ms)"
else
    log_warn "Doctor slow: ${DOCTOR_TIME}ms"
fi

log_test "6.3 Help performance"
clean_env
HELP_TIME=$(measure_time "$XF_BIN" --help)
log "  Help: ${HELP_TIME}ms"
if [ "$HELP_TIME" -lt 1000 ]; then
    log_pass "Help < 1000ms (${HELP_TIME}ms)"
else
    log_warn "Help slow: ${HELP_TIME}ms"
fi

# ============================================================
# Section 7: Format Consistency Tests
# ============================================================
log ""
log "=== Section 7: Format Consistency Tests ==="

if [ "$HAS_ARCHIVE" = "true" ]; then
    log_test "7.1 Multiple runs produce consistent format"
    clean_env
    RUN1=$("$XF_BIN" stats --format json 2>&1 || echo "{}")
    RUN2=$("$XF_BIN" stats --format json 2>&1 || echo "{}")
    RUN1_KEYS=$(echo "$RUN1" | jq -r 'keys | join(",")' 2>/dev/null || echo "")
    RUN2_KEYS=$(echo "$RUN2" | jq -r 'keys | join(",")' 2>/dev/null || echo "")
    if [ "$RUN1_KEYS" = "$RUN2_KEYS" ]; then
        log_pass "JSON structure is consistent"
    else
        log_warn "JSON structure varies between runs"
    fi

    log_test "7.2 JSON and text have same data"
    clean_env
    JSON_STATS=$("$XF_BIN" stats --format json 2>&1 || echo "{}")
    TEXT_STATS=$("$XF_BIN" stats 2>&1 || echo "")
    # Extract a count from JSON and verify it appears in text
    JSON_TOTAL=$(echo "$JSON_STATS" | jq -r '.total_documents // .tweet_count // "0"' 2>/dev/null || echo "0")
    if [ "$JSON_TOTAL" != "0" ] && echo "$TEXT_STATS" | grep -q "$JSON_TOTAL"; then
        log_pass "JSON data matches text output"
    else
        log_warn "Could not verify JSON/text consistency"
    fi
else
    log_skip "Section 7: Format consistency tests require indexed archive"
fi

# ============================================================
# Summary
# ============================================================
log ""
log "========================================"
log "TEST SUMMARY"
log "========================================"
log "Passed:   $PASS_COUNT"
log "Failed:   $FAIL_COUNT"
log "Warnings: $WARN_COUNT"
log "Skipped:  $SKIP_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    log "RESULT: FAILED"
    exit 1
else
    log "RESULT: PASSED"
    exit 0
fi
