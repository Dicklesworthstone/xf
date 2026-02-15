#!/bin/bash
#
# verify_isomorphism.sh - Verify xf outputs match golden files
#
# This script runs xf commands and compares outputs against expected golden files.
# Exit code 0 = all outputs match, 1 = mismatch found
#
# Usage:
#   ./scripts/verify_isomorphism.sh [--update]
#
# Options:
#   --update    Update golden files with current outputs (use after intentional changes)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CORPUS_DIR="$PROJECT_ROOT/tests/fixtures/perf_corpus"
GOLDEN_DIR="$PROJECT_ROOT/tests/fixtures/golden_outputs"
TEMP_DIR=$(mktemp -d)

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

UPDATE_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update)
            UPDATE_MODE=true
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./scripts/verify_isomorphism.sh [--update]"
            exit 1
            ;;
    esac
    shift
done

if [[ "$UPDATE_MODE" == "true" ]]; then
    echo -e "${YELLOW}Running in UPDATE mode - golden files will be updated${NC}"
fi

# Check if xf binary exists.
if ! command -v xf &> /dev/null; then
    # Try to find it in target directory
    if [[ -f "$PROJECT_ROOT/target/release/xf" ]]; then
        XF="$PROJECT_ROOT/target/release/xf"
    elif [[ -f "$PROJECT_ROOT/target/debug/xf" ]]; then
        XF="$PROJECT_ROOT/target/debug/xf"
    else
        echo -e "${RED}Error: xf binary not found. Run 'cargo build --release' first.${NC}"
        exit 1
    fi
else
    XF="xf"
fi

echo "Using xf binary: $XF"

# Create temp database and index for testing
TEST_DB="$TEMP_DIR/test.db"
TEST_INDEX="$TEMP_DIR/test_index"

# Index the corpus
echo "Indexing test corpus..."
XF_DB="$TEST_DB" XF_INDEX="$TEST_INDEX" $XF index "$CORPUS_DIR" --quiet 2>/dev/null || {
    echo -e "${RED}Failed to index corpus${NC}"
    exit 1
}
echo "Index complete."

# Track results
PASSED=0
FAILED=0
UPDATED=0

# Compare or update golden file
check_golden() {
    local name="$1"
    local actual_file="$2"
    local golden_file="$GOLDEN_DIR/$name"

    if [[ "$UPDATE_MODE" == "true" ]]; then
        cp "$actual_file" "$golden_file"
        echo -e "${YELLOW}Updated: $name${NC}"
        ((UPDATED++))
        return 0
    fi

    if [[ ! -f "$golden_file" ]]; then
        echo -e "${RED}MISSING: $name (golden file not found)${NC}"
        ((FAILED++))
        return 1
    fi

    # For JSON files, use jq to normalize before comparing
    if [[ "$name" == *.json ]]; then
        if jq -S '.' "$actual_file" > "$TEMP_DIR/actual_normalized.json" 2>/dev/null && \
           jq -S '.' "$golden_file" > "$TEMP_DIR/golden_normalized.json" 2>/dev/null; then
            if diff -q "$TEMP_DIR/actual_normalized.json" "$TEMP_DIR/golden_normalized.json" > /dev/null 2>&1; then
                echo -e "${GREEN}PASS: $name${NC}"
                ((PASSED++))
                return 0
            else
                echo -e "${RED}FAIL: $name${NC}"
                echo "Differences:"
                diff "$TEMP_DIR/golden_normalized.json" "$TEMP_DIR/actual_normalized.json" | head -20
                ((FAILED++))
                return 1
            fi
        fi
    fi

    # For text files, direct comparison
    if diff -q "$actual_file" "$golden_file" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS: $name${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAIL: $name${NC}"
        echo "Differences:"
        diff "$golden_file" "$actual_file" | head -20
        ((FAILED++))
        return 1
    fi
}

# Run xf command and save output
run_and_check() {
    local name="$1"
    shift
    local output_file="$TEMP_DIR/$name"
    local error_file="$TEMP_DIR/$name.stderr"

    echo -n "Running: xf $*... "
    XF_DB="$TEST_DB" XF_INDEX="$TEST_INDEX" NO_COLOR=1 $XF "$@" > "$output_file" 2> "$error_file" || true

    check_golden "$name" "$output_file"
}

echo ""
echo "=== Verifying search outputs ==="

# Search tests - lexical mode
run_and_check "search_lexical_machine.json" search "machine" --mode lexical --limit 20 --format json

# Search tests - hybrid mode (default)
run_and_check "search_hybrid_rust.json" search "rust" --limit 20 --format json

# Stats tests
run_and_check "stats_basic.txt" stats
run_and_check "stats_detailed.json" stats --detailed --format json

echo ""
echo "=== Summary ==="
if [[ "$UPDATE_MODE" == "true" ]]; then
    echo -e "Updated: $UPDATED golden files"
else
    echo -e "Passed: $PASSED"
    echo -e "Failed: $FAILED"

    if [[ $FAILED -gt 0 ]]; then
        echo -e "\n${RED}Some outputs differ from golden files.${NC}"
        echo "Run with --update to regenerate golden files if changes are intentional."
        exit 1
    else
        echo -e "\n${GREEN}All outputs match golden files.${NC}"
        exit 0
    fi
fi
