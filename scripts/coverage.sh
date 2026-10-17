#!/bin/bash
# Test coverage analysis script for xf
#
# Prerequisites:
#   - cargo install cargo-llvm-cov
#
# Usage:
#   ./scripts/coverage.sh          # Generate HTML report
#   ./scripts/coverage.sh summary  # Show summary only
#   ./scripts/coverage.sh lcov     # Generate LCOV report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== xf Test Coverage Analysis ===${NC}"
echo ""

# Check if cargo-llvm-cov is installed
if ! command -v cargo-llvm-cov &> /dev/null; then
    echo -e "${RED}Error: cargo-llvm-cov not found${NC}"
    echo "Install with: cargo install cargo-llvm-cov"
    exit 1
fi

MODE="${1:-html}"

case "$MODE" in
    summary)
        echo -e "${YELLOW}Generating coverage summary...${NC}"
        cargo llvm-cov --features full,parallel-search --summary-only
        ;;

    lcov)
        echo -e "${YELLOW}Generating LCOV report...${NC}"
        cargo llvm-cov --features full,parallel-search --lcov --output-path lcov.info
        echo -e "${GREEN}LCOV report saved to: lcov.info${NC}"
        ;;

    html)
        echo -e "${YELLOW}Generating HTML coverage report...${NC}"
        cargo llvm-cov --features full,parallel-search --html --output-dir coverage
        echo -e "${GREEN}HTML report saved to: coverage/html/index.html${NC}"
        echo ""
        echo "Open with: open coverage/html/index.html"
        ;;

    json)
        echo -e "${YELLOW}Generating JSON coverage report...${NC}"
        cargo llvm-cov --features full,parallel-search --json --output-path coverage.json
        echo -e "${GREEN}JSON report saved to: coverage.json${NC}"
        ;;

    *)
        echo "Usage: $0 [summary|lcov|html|json]"
        echo ""
        echo "Options:"
        echo "  summary  - Show coverage summary in terminal"
        echo "  lcov     - Generate LCOV format for CI/Codecov"
        echo "  html     - Generate HTML report (default)"
        echo "  json     - Generate JSON report"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}=== Coverage Analysis Complete ===${NC}"
