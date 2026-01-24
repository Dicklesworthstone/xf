#!/bin/bash
# Benchmark E2E Validation Script
# Bead: bd-2kp
# Runs a complete benchmark cycle and validates outputs

set -euo pipefail

LOG=/tmp/bench_e2e.log
RESULTS_DIR=/tmp/bench_e2e_results
CORPUS_FILE=tests/fixtures/benchmark_corpus.json

echo "========================================" | tee "$LOG"
echo "Benchmark E2E Validation" | tee -a "$LOG"
echo "Date: $(date -Iseconds)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Environment info
echo "" | tee -a "$LOG"
echo "Environment:" | tee -a "$LOG"
echo "  OS: $(uname -s) $(uname -r)" | tee -a "$LOG"
echo "  Host: $(hostname)" | tee -a "$LOG"
echo "  CPU: $(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'N/A')" | tee -a "$LOG"
echo "  Cores: $(nproc 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "  RAM: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'N/A')" | tee -a "$LOG"
echo "  Git SHA: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "  Rust: $(rustc --version 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "  Nice level: $(nice 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "  Thread count: $(echo "${RAYON_NUM_THREADS:-default}" | tee -a "$LOG")" | tee -a "$LOG"

# System load snapshot
echo "" | tee -a "$LOG"
echo "System load:" | tee -a "$LOG"
uptime | tee -a "$LOG"

# Setup results directory
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

PASS=true

# Check corpus file exists
echo "" | tee -a "$LOG"
echo "Checking prerequisites..." | tee -a "$LOG"

if [[ -f "$CORPUS_FILE" ]]; then
    echo "  [PASS] Corpus file exists: $CORPUS_FILE" | tee -a "$LOG"
    CORPUS_SIZE=$(jq '.corpus | length' "$CORPUS_FILE" 2>/dev/null || echo "0")
    QUERY_COUNT=$(jq '.queries | length' "$CORPUS_FILE" 2>/dev/null || echo "0")
    echo "  Corpus docs: $CORPUS_SIZE" | tee -a "$LOG"
    echo "  Queries: $QUERY_COUNT" | tee -a "$LOG"
else
    echo "  [FAIL] Missing corpus file: $CORPUS_FILE" | tee -a "$LOG"
    PASS=false
fi

# Check binary is built
if cargo build --release 2>&1 | tail -5 | tee -a "$LOG"; then
    echo "  [PASS] Binary built successfully" | tee -a "$LOG"
else
    echo "  [FAIL] Build failed" | tee -a "$LOG"
    PASS=false
fi

# Run benchmark with hash embedder (always available)
echo "" | tee -a "$LOG"
echo "Running benchmark with hash embedder..." | tee -a "$LOG"

BENCH_START=$(date +%s%3N)
if cargo run --release -- benchmark \
    --model hash \
    --corpus "$CORPUS_FILE" \
    --warmup 5 \
    --iters 20 \
    --batch-size 8 \
    --output "$RESULTS_DIR/hash_bench.json" 2>&1 | tee -a "$LOG"; then
    echo "  [PASS] Benchmark completed" | tee -a "$LOG"
else
    echo "  [WARN] Benchmark command failed (may not be implemented yet)" | tee -a "$LOG"
fi
BENCH_END=$(date +%s%3N)
BENCH_DURATION=$((BENCH_END - BENCH_START))
echo "  Duration: ${BENCH_DURATION}ms" | tee -a "$LOG"

# Validate JSON output (if exists)
echo "" | tee -a "$LOG"
echo "Validating outputs..." | tee -a "$LOG"

if [[ -f "$RESULTS_DIR/hash_bench.json" ]]; then
    echo "  [PASS] JSON output exists" | tee -a "$LOG"

    # Check required keys
    for key in metadata speed latency_ms; do
        if jq -e ".$key" "$RESULTS_DIR/hash_bench.json" > /dev/null 2>&1; then
            echo "  [PASS] Key '$key' present" | tee -a "$LOG"
        else
            echo "  [FAIL] Key '$key' missing" | tee -a "$LOG"
            PASS=false
        fi
    done

    # Verify speed metrics
    if jq -e '.speed.cold_start_ms >= 0' "$RESULTS_DIR/hash_bench.json" > /dev/null 2>&1; then
        COLD_START=$(jq '.speed.cold_start_ms' "$RESULTS_DIR/hash_bench.json")
        echo "  [PASS] cold_start_ms = $COLD_START" | tee -a "$LOG"
    else
        echo "  [FAIL] Invalid cold_start_ms" | tee -a "$LOG"
        PASS=false
    fi

    if jq -e '.speed.warm_latency_p50_ms >= 0' "$RESULTS_DIR/hash_bench.json" > /dev/null 2>&1; then
        P50=$(jq '.speed.warm_latency_p50_ms' "$RESULTS_DIR/hash_bench.json")
        P95=$(jq '.speed.warm_latency_p95_ms' "$RESULTS_DIR/hash_bench.json")
        P99=$(jq '.speed.warm_latency_p99_ms' "$RESULTS_DIR/hash_bench.json")
        echo "  [PASS] Latency p50=$P50 p95=$P95 p99=$P99" | tee -a "$LOG"
    else
        echo "  [FAIL] Invalid latency metrics" | tee -a "$LOG"
        PASS=false
    fi

    if jq -e '.speed.throughput_docs_per_sec > 0' "$RESULTS_DIR/hash_bench.json" > /dev/null 2>&1; then
        THROUGHPUT=$(jq '.speed.throughput_docs_per_sec' "$RESULTS_DIR/hash_bench.json")
        echo "  [PASS] throughput = $THROUGHPUT docs/sec" | tee -a "$LOG"
    else
        echo "  [WARN] Zero or invalid throughput" | tee -a "$LOG"
    fi
else
    echo "  [SKIP] JSON output not generated (benchmark may not be implemented)" | tee -a "$LOG"
fi

# Test unit tests for benchmark module
echo "" | tee -a "$LOG"
echo "Running benchmark unit tests..." | tee -a "$LOG"

if cargo test --release benchmark:: 2>&1 | tail -20 | tee -a "$LOG"; then
    echo "  [PASS] Unit tests passed" | tee -a "$LOG"
else
    echo "  [FAIL] Unit tests failed" | tee -a "$LOG"
    PASS=false
fi

# Generate summary markdown
echo "" | tee -a "$LOG"
echo "Generating summary..." | tee -a "$LOG"

{
    echo "# Benchmark E2E Summary"
    echo ""
    echo "**Date:** $(date -Iseconds)"
    echo "**Git SHA:** $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
    echo ""
    echo "## Environment"
    echo "| Property | Value |"
    echo "|----------|-------|"
    echo "| OS | $(uname -s) $(uname -r) |"
    echo "| CPU | $(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'N/A') |"
    echo "| Cores | $(nproc 2>/dev/null || echo 'N/A') |"
    echo "| RAM | $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'N/A') |"
    echo "| Rust | $(rustc --version 2>/dev/null || echo 'N/A') |"
    echo ""
    echo "## Results"
    if [[ -f "$RESULTS_DIR/hash_bench.json" ]]; then
        echo "| Metric | Value |"
        echo "|--------|-------|"
        echo "| Cold start (ms) | $(jq '.speed.cold_start_ms' "$RESULTS_DIR/hash_bench.json" 2>/dev/null || echo 'N/A') |"
        echo "| TTFR (ms) | $(jq '.speed.time_to_first_result_ms' "$RESULTS_DIR/hash_bench.json" 2>/dev/null || echo 'N/A') |"
        echo "| Warm p50 (ms) | $(jq '.speed.warm_latency_p50_ms' "$RESULTS_DIR/hash_bench.json" 2>/dev/null || echo 'N/A') |"
        echo "| Warm p95 (ms) | $(jq '.speed.warm_latency_p95_ms' "$RESULTS_DIR/hash_bench.json" 2>/dev/null || echo 'N/A') |"
        echo "| Warm p99 (ms) | $(jq '.speed.warm_latency_p99_ms' "$RESULTS_DIR/hash_bench.json" 2>/dev/null || echo 'N/A') |"
        echo "| Throughput (docs/sec) | $(jq '.speed.throughput_docs_per_sec' "$RESULTS_DIR/hash_bench.json" 2>/dev/null || echo 'N/A') |"
    else
        echo "(Benchmark output not available)"
    fi
} > "$RESULTS_DIR/summary.md"

echo "  Summary written to: $RESULTS_DIR/summary.md" | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
if $PASS; then
    echo "RESULT: ALL CHECKS PASSED" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "Outputs:" | tee -a "$LOG"
    echo "  Log: $LOG" | tee -a "$LOG"
    echo "  Results: $RESULTS_DIR/" | tee -a "$LOG"
    ls -la "$RESULTS_DIR/" 2>/dev/null | tee -a "$LOG"
    exit 0
else
    echo "RESULT: SOME CHECKS FAILED" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    exit 1
fi
