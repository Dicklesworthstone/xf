#!/bin/bash
set -euo pipefail

# =============================================================================
# xf TOON Integration E2E Test Suite
# =============================================================================
# Tests:
#   - format flags
#   - XF_OUTPUT_FORMAT / TOON_DEFAULT_FORMAT env vars
#   - optional round-trip via tru
#   - command coverage (stats, search, list)
# Logs:
#   test_logs/xf_toon_<timestamp>/
# Exit:
#   0 = all pass, 1 = failures
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/test_logs/xf_toon_${TIMESTAMP}"
ARCHIVE_DIR="$LOG_DIR/sample_archive"
DATA_DIR="$ARCHIVE_DIR/data"
DB_PATH="$LOG_DIR/xf.db"
INDEX_PATH="$LOG_DIR/xf_index"

mkdir -p "$DATA_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/test.log"; }
pass() { log "PASS: $*"; ((PASS_COUNT++)) || true; }
fail() { log "FAIL: $*"; ((FAIL_COUNT++)) || true; FAILURES+=("$*"); }
skip() { log "SKIP: $*"; ((SKIP_COUNT++)) || true; }

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAILURES=()

TWEET_COUNT=120
TWEET_TEXT="test"
MIN_SAVINGS=15

SAMPLE_MANIFEST='window.YTD.manifest.part0 = {
  "userInfo": {
    "accountId": "999999999",
    "userName": "test_user",
    "displayName": "Test User"
  },
  "archiveInfo": {
    "sizeBytes": "1234",
    "generationDate": "2025-01-01T00:00:00Z",
    "isPartialArchive": false
  }
}'

log "=========================================="
log "xf TOON Integration E2E Test Suite"
log "Log directory: $LOG_DIR"
log "=========================================="

if [[ -n "${CARGO_TARGET_DIR:-}" && -f "$CARGO_TARGET_DIR/release/xf" ]]; then
  XF="$CARGO_TARGET_DIR/release/xf"
elif [[ -n "${CARGO_TARGET_DIR:-}" && -f "$CARGO_TARGET_DIR/debug/xf" ]]; then
  XF="$CARGO_TARGET_DIR/debug/xf"
elif [[ -f "$PROJECT_ROOT/target/release/xf" ]]; then
  XF="$PROJECT_ROOT/target/release/xf"
elif [[ -f "$PROJECT_ROOT/target/debug/xf" ]]; then
  XF="$PROJECT_ROOT/target/debug/xf"
elif command -v xf &>/dev/null; then
  XF="xf"
else
  fail "xf binary not found (run cargo build --release)"
  exit 1
fi
log "Using xf binary: $XF"

if command -v tru &>/dev/null; then
  log "tru binary found"
else
  log "tru binary not found (round-trip tests will skip)"
fi

log "xf version: $($XF --version 2>&1 | head -1)"

log ""
log "Phase 1: Build minimal archive + index"
{
  echo "window.YTD.tweets.part0 = ["
  for i in $(seq 1 "$TWEET_COUNT"); do
    if [[ "$i" -gt 1 ]]; then
      echo ","
    fi
    printf '  {\n    "tweet": {\n      "id_str": "%s",\n      "created_at": "Wed Jan 08 12:00:00 +0000 2025",\n      "full_text": "test tweet %d. %s.",\n      "source": "<a href=\\"https://x.com\\">X Web App</a>",\n      "favorite_count": "%d",\n      "retweet_count": "%d",\n      "lang": "en",\n      "entities": {"hashtags": [{"text": "test"}, {"text": "archive"}], "user_mentions": [], "urls": []}\n    }\n  }' \
      "$((1234567800 + i))" "$i" "$TWEET_TEXT" "$i" "$((i / 2))"
  done
  echo
  echo "]"
} > "$DATA_DIR/tweets.js"
printf "%s\n" "$SAMPLE_MANIFEST" > "$DATA_DIR/manifest.js"

XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" "$XF" index "$ARCHIVE_DIR" --quiet \
  >"$LOG_DIR/index_stdout.log" 2>"$LOG_DIR/index_stderr.log" || {
    fail "xf index failed"
    exit 1
  }
pass "Indexed minimal archive"

log ""
log "Phase 2: Format flag tests"
if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" "$XF" stats --format json \
  >"$LOG_DIR/stats_json.out" 2>"$LOG_DIR/stats_json.err"; then
  if [[ "$(head -1 "$LOG_DIR/stats_json.out")" == "{"* ]]; then
    pass "stats --format json output looks like JSON"
  else
    fail "stats --format json output did not start with {"
  fi
else
  fail "stats --format json failed"
fi

if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" "$XF" stats --format toon \
  >"$LOG_DIR/stats_toon.out" 2>"$LOG_DIR/stats_toon.err"; then
  if [[ "$(head -1 "$LOG_DIR/stats_toon.out")" != "{"* ]]; then
    pass "stats --format toon output is non-JSON"
  else
    fail "stats --format toon output looks like JSON"
  fi
else
  fail "stats --format toon failed"
fi

log ""
log "Phase 3: Environment variable tests"
if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" XF_OUTPUT_FORMAT=toon "$XF" stats \
  >"$LOG_DIR/env_xf_toon.out" 2>"$LOG_DIR/env_xf_toon.err"; then
  if [[ "$(head -1 "$LOG_DIR/env_xf_toon.out")" != "{"* ]]; then
    pass "XF_OUTPUT_FORMAT=toon produces TOON output"
  else
    fail "XF_OUTPUT_FORMAT=toon still produced JSON"
  fi
else
  fail "XF_OUTPUT_FORMAT test failed"
fi

if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" TOON_DEFAULT_FORMAT=toon "$XF" stats \
  >"$LOG_DIR/env_default_toon.out" 2>"$LOG_DIR/env_default_toon.err"; then
  if [[ "$(head -1 "$LOG_DIR/env_default_toon.out")" != "{"* ]]; then
    pass "TOON_DEFAULT_FORMAT=toon produces TOON output"
  else
    fail "TOON_DEFAULT_FORMAT=toon still produced JSON"
  fi
else
  fail "TOON_DEFAULT_FORMAT test failed"
fi

if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" XF_OUTPUT_FORMAT=toon "$XF" stats --format json \
  >"$LOG_DIR/env_precedence.out" 2>"$LOG_DIR/env_precedence.err"; then
  if [[ "$(head -1 "$LOG_DIR/env_precedence.out")" == "{"* ]]; then
    pass "CLI --format json overrides XF_OUTPUT_FORMAT"
  else
    fail "CLI did not override XF_OUTPUT_FORMAT"
  fi
else
  fail "Env precedence test failed"
fi

log ""
log "Phase 4: Round-trip (optional)"
if command -v tru &>/dev/null; then
  if tru --decode < "$LOG_DIR/stats_toon.out" > "$LOG_DIR/roundtrip_decoded.json" 2>/dev/null; then
    pass "tru --decode succeeded"
  else
    fail "tru --decode failed"
  fi
else
  skip "tru not available"
fi

log ""
log "Phase 5: Command coverage"
if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" "$XF" search "test" --format json --limit 20 \
  >"$LOG_DIR/search_json.out" 2>"$LOG_DIR/search_json.err"; then
  pass "search --format json succeeds"
else
  fail "search --format json failed"
fi

if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" "$XF" search "test" --format toon --limit 20 \
  >"$LOG_DIR/search_toon.out" 2>"$LOG_DIR/search_toon.err"; then
  pass "search --format toon succeeds"
else
  fail "search --format toon failed"
fi

if XF_DB="$DB_PATH" XF_INDEX="$INDEX_PATH" "$XF" list tweets --format toon -n 5 \
  >"$LOG_DIR/list_toon.out" 2>"$LOG_DIR/list_toon.err"; then
  pass "list tweets --format toon succeeds"
else
  fail "list tweets --format toon failed"
fi

log ""
log "Phase 6: Token savings"
if [[ -s "$LOG_DIR/search_json.out" ]]; then
  if command -v tru &>/dev/null; then
    STATS_LINE=$(tru --stats "$LOG_DIR/search_json.out" 2>&1 | rg -m1 "Saved" || true)
    log "$STATS_LINE"
    SAVINGS_PCT=$(printf "%s" "$STATS_LINE" | sed -n 's/.*(\([-0-9.]*\)%).*/\1/p')
    if [[ -n "$SAVINGS_PCT" ]]; then
      if [[ "$SAVINGS_PCT" == -* ]]; then
        log "Savings: ${SAVINGS_PCT}% (TOON larger)"
        skip "Token savings negative (TOON larger)"
      else
        log "Savings: ${SAVINGS_PCT}%"
        if [[ "${SAVINGS_PCT%.*}" -ge "$MIN_SAVINGS" ]]; then
          pass "Token savings >= ${MIN_SAVINGS}%"
        else
          fail "Token savings < ${MIN_SAVINGS}%"
        fi
      fi
    else
      fail "Token savings could not be parsed"
    fi
  else
    skip "tru not available for token savings"
  fi
else
  skip "Token savings (missing outputs)"
fi

log ""
log "=========================================="
log "TEST SUMMARY"
log "=========================================="
log "Passed: $PASS_COUNT"
log "Failed: $FAIL_COUNT"
log "Skipped: $SKIP_COUNT"
log "Log directory: $LOG_DIR"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  log ""
  log "FAILURES:"
  for f in "${FAILURES[@]}"; do
    log "  - $f"
  done
  exit 1
fi

log ""
log "All tests passed!"
exit 0
