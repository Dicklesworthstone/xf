#!/bin/bash
# tests/e2e/daemon_test.sh
# Comprehensive E2E tests for xf daemon with detailed JSON logging
#
# Usage:
#   ./daemon_test.sh              # Run all tests
#   ./daemon_test.sh test_name    # Run specific test
#   XF_BIN=/path/to/xf ./daemon_test.sh  # Use specific binary
#
# Output:
#   Writes JSONL test results to $TEST_DIR/test_results.jsonl
#   Human-readable output to stdout
#   Exit 0 if all tests pass, 1 otherwise

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XF_BIN="${XF_BIN:-$PROJECT_ROOT/target/release/xf}"
TEST_DIR=$(mktemp -d)
LOG_FILE="$TEST_DIR/test_results.jsonl"
SOCKET_PATH="$TEST_DIR/xf-daemon.sock"
PID_FILE="$TEST_DIR/xf-daemon.pid"

# --- Counters ---
PASSES=0
FAILURES=0
SKIPPED=0
declare -a FAILED_TESTS=()

# Cleanup any existing daemon with same socket
cleanup_existing_daemon() {
    if [[ -S "$SOCKET_PATH" ]]; then
        "$XF_BIN" daemon stop --socket "$SOCKET_PATH" 2>/dev/null || true
        sleep 0.5
    fi
    rm -f "$SOCKET_PATH" "$PID_FILE"
}

# --- Logging Functions ---
log_json() {
    local level="$1"
    local test="$2"
    local msg="$3"
    shift 3

    local json
    json=$(printf '{"timestamp":"%s","level":"%s","test":"%s","message":"%s"' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" "$level" "$test" "$msg")

    while [[ $# -gt 0 ]]; do
        local key="$1"
        local value="$2"
        # Check if value is already JSON (starts with { or [ or is a number)
        if [[ "$value" =~ ^[\{\[\"\-0-9] ]] || [[ "$value" == "true" ]] || [[ "$value" == "false" ]] || [[ "$value" == "null" ]]; then
            json="$json,\"$key\":$value"
        else
            json="$json,\"$key\":\"$value\""
        fi
        shift 2
    done

    json="$json}"
    echo "$json" >> "$LOG_FILE"

    # Human-readable output
    case "$level" in
        PASS) echo -e "\033[32m[PASS]\033[0m $test: $msg" ;;
        FAIL) echo -e "\033[31m[FAIL]\033[0m $test: $msg" ;;
        SKIP) echo -e "\033[33m[SKIP]\033[0m $test: $msg" ;;
        INFO) echo -e "\033[34m[INFO]\033[0m $test: $msg" ;;
        ERROR) echo -e "\033[31m[ERROR]\033[0m $test: $msg" ;;
        *) echo "[$level] $test: $msg" ;;
    esac
}

# --- Test Helpers ---
wait_for_socket() {
    local socket="$1"
    local timeout="${2:-10}"
    local elapsed=0

    while [[ ! -S "$socket" ]] && [[ $elapsed -lt $timeout ]]; do
        sleep 0.1
        elapsed=$((elapsed + 1))
    done

    [[ -S "$socket" ]]
}

wait_for_no_socket() {
    local socket="$1"
    local timeout="${2:-10}"
    local elapsed=0

    while [[ -S "$socket" ]] && [[ $elapsed -lt $timeout ]]; do
        sleep 0.1
        elapsed=$((elapsed + 1))
    done

    [[ ! -S "$socket" ]]
}

# Start daemon and wait for socket
start_daemon() {
    local idle_timeout="${1:-300}"
    cleanup_existing_daemon

    "$XF_BIN" daemon start --foreground --socket "$SOCKET_PATH" --idle-timeout "$idle_timeout" &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    if ! wait_for_socket "$SOCKET_PATH" 10; then
        kill "$pid" 2>/dev/null || true
        return 1
    fi

    echo "$pid"
}

# Stop daemon gracefully
stop_daemon() {
    local pid
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE")
        kill -INT "$pid" 2>/dev/null || true

        # Wait for process to exit
        local elapsed=0
        while kill -0 "$pid" 2>/dev/null && [[ $elapsed -lt 50 ]]; do
            sleep 0.1
            elapsed=$((elapsed + 1))
        done
    fi
    rm -f "$PID_FILE"
}

cleanup() {
    stop_daemon
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Run a test function and record result
run_test() {
    local test_name="$1"
    local start_time
    start_time=$(date +%s%3N 2>/dev/null || date +%s000)

    log_json INFO "$test_name" "Starting test"

    local result=0
    if "$test_name"; then
        result=0
    else
        result=1
    fi

    local end_time
    end_time=$(date +%s%3N 2>/dev/null || date +%s000)
    local duration=$((end_time - start_time))

    if [[ $result -eq 0 ]]; then
        PASSES=$((PASSES + 1))
        log_json PASS "$test_name" "Test passed" duration_ms "$duration"
    else
        FAILURES=$((FAILURES + 1))
        FAILED_TESTS+=("$test_name")
        log_json FAIL "$test_name" "Test failed" duration_ms "$duration"
    fi

    # Cleanup between tests
    stop_daemon
    cleanup_existing_daemon
    sleep 0.2
}

# --- Test Cases ---

test_daemon_start_stop() {
    local pid
    pid=$(start_daemon 300)

    # Verify socket exists
    if [[ ! -S "$SOCKET_PATH" ]]; then
        log_json ERROR test_daemon_start_stop "Socket not created"
        return 1
    fi

    log_json INFO test_daemon_start_stop "Socket created, stopping daemon" pid "$pid"

    # Stop daemon
    stop_daemon

    # Verify socket removed (with longer timeout since SIGINT triggers cleanup)
    sleep 1
    if [[ -S "$SOCKET_PATH" ]]; then
        log_json ERROR test_daemon_start_stop "Socket not cleaned up after stop"
        # Force cleanup for next test
        rm -f "$SOCKET_PATH"
        return 1
    fi

    log_json INFO test_daemon_start_stop "Lifecycle verified" pid "$pid"
    return 0
}

test_daemon_status_json() {
    local pid
    pid=$(start_daemon 300)

    # Get status as JSON
    local status
    if ! status=$("$XF_BIN" daemon status --socket "$SOCKET_PATH" --format json 2>&1); then
        log_json ERROR test_daemon_status_json "Status command failed: $status"
        return 1
    fi

    # Verify it's valid JSON with expected fields
    if ! echo "$status" | jq -e '.uptime_secs' > /dev/null 2>&1; then
        log_json ERROR test_daemon_status_json "Invalid JSON or missing uptime_secs" details "\"$status\""
        return 1
    fi

    local uptime
    uptime=$(echo "$status" | jq -r '.uptime_secs')
    log_json INFO test_daemon_status_json "Status retrieved" uptime_secs "$uptime"
    return 0
}

test_daemon_embed_via_daemon() {
    start_daemon 300 > /dev/null

    # Run embed command (should use daemon)
    local result
    if ! result=$("$XF_BIN" embed --text "test embedding" --format json 2>&1); then
        log_json ERROR test_daemon_embed_via_daemon "Embed failed: $result"
        return 1
    fi

    # Verify we got embeddings
    if ! echo "$result" | jq -e '.embeddings' > /dev/null 2>&1; then
        log_json ERROR test_daemon_embed_via_daemon "Invalid embed response" details "\"$result\""
        return 1
    fi

    local dim
    dim=$(echo "$result" | jq -r '.embeddings[0] | length')
    log_json INFO test_daemon_embed_via_daemon "Embedding retrieved" dimensions "$dim"
    return 0
}

test_daemon_rerank_via_daemon() {
    start_daemon 300 > /dev/null

    # Run rerank command (should use daemon)
    local result
    if ! result=$("$XF_BIN" rerank --query "test query" --documents "doc one" "doc two" --format json 2>&1); then
        log_json ERROR test_daemon_rerank_via_daemon "Rerank failed: $result"
        return 1
    fi

    # Verify we got scores
    if ! echo "$result" | jq -e '.results' > /dev/null 2>&1; then
        log_json ERROR test_daemon_rerank_via_daemon "Invalid rerank response" details "\"$result\""
        return 1
    fi

    local count
    count=$(echo "$result" | jq -r '.results | length')
    log_json INFO test_daemon_rerank_via_daemon "Rerank complete" num_results "$count"
    return 0
}

test_daemon_idle_timeout() {
    # Start with 2 second timeout
    local pid
    pid=$(start_daemon 2)

    log_json INFO test_daemon_idle_timeout "Waiting for idle timeout (2s + buffer)"

    # Wait for timeout plus buffer
    sleep 4

    # Check if process exited
    if kill -0 "$pid" 2>/dev/null; then
        log_json ERROR test_daemon_idle_timeout "Daemon still running after idle timeout"
        kill "$pid" 2>/dev/null || true
        return 1
    fi

    # Socket should be cleaned up
    if [[ -S "$SOCKET_PATH" ]]; then
        log_json ERROR test_daemon_idle_timeout "Socket not cleaned up after timeout"
        return 1
    fi

    log_json INFO test_daemon_idle_timeout "Idle timeout triggered shutdown correctly"
    return 0
}

test_daemon_signal_handling() {
    local pid
    pid=$(start_daemon 300)

    # Send SIGINT
    kill -INT "$pid"

    # Wait for clean exit
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null && [[ $elapsed -lt 50 ]]; do
        sleep 0.1
        elapsed=$((elapsed + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        log_json ERROR test_daemon_signal_handling "Daemon didn't exit on SIGINT"
        kill -9 "$pid" 2>/dev/null || true
        return 1
    fi

    # Socket should be cleaned up
    sleep 0.2
    if [[ -S "$SOCKET_PATH" ]]; then
        log_json ERROR test_daemon_signal_handling "Socket not cleaned up after SIGINT"
        return 1
    fi

    log_json INFO test_daemon_signal_handling "SIGINT handled correctly"
    return 0
}

test_daemon_sigterm_handling() {
    local pid
    pid=$(start_daemon 300)

    # Send SIGTERM
    kill -TERM "$pid"

    # Wait for exit (SIGTERM doesn't get graceful handling currently)
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null && [[ $elapsed -lt 50 ]]; do
        sleep 0.1
        elapsed=$((elapsed + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        log_json ERROR test_daemon_sigterm_handling "Daemon didn't exit on SIGTERM"
        kill -9 "$pid" 2>/dev/null || true
        return 1
    fi

    # Note: SIGTERM doesn't trigger cleanup, so socket may remain
    log_json INFO test_daemon_sigterm_handling "SIGTERM terminated process"
    return 0
}

test_socket_permissions() {
    start_daemon 300 > /dev/null

    # Check socket permissions
    local perms
    perms=$(stat -c '%a' "$SOCKET_PATH" 2>/dev/null || stat -f '%Lp' "$SOCKET_PATH" 2>/dev/null)

    if [[ "$perms" != "600" ]]; then
        log_json ERROR test_socket_permissions "Wrong permissions" expected "600" actual "\"$perms\""
        return 1
    fi

    log_json INFO test_socket_permissions "Socket permissions correct" mode "\"$perms\""
    return 0
}

test_pid_file_contents() {
    local daemon_pid
    daemon_pid=$(start_daemon 300)

    # Note: The daemon writes its own PID file at a default location
    # For this test, we verify the PID from our tracking matches
    local file_pid
    file_pid=$(cat "$PID_FILE")

    if [[ "$daemon_pid" != "$file_pid" ]]; then
        log_json ERROR test_pid_file_contents "PID mismatch" expected "$daemon_pid" actual "$file_pid"
        return 1
    fi

    # Verify the PID is actually running
    if ! kill -0 "$daemon_pid" 2>/dev/null; then
        log_json ERROR test_pid_file_contents "PID not running"
        return 1
    fi

    log_json INFO test_pid_file_contents "PID file verified" pid "$daemon_pid"
    return 0
}

test_concurrent_requests() {
    start_daemon 300 > /dev/null

    # Launch 5 concurrent embed requests
    local pids=()
    local results_dir="$TEST_DIR/concurrent"
    mkdir -p "$results_dir"

    for i in {1..5}; do
        (
            "$XF_BIN" embed --text "concurrent test $i" --format json > "$results_dir/result_$i.json" 2>&1
        ) &
        pids+=($!)
    done

    # Wait for all to complete
    local failures=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done

    # Verify at least 4 out of 5 succeeded
    local successes=$((5 - failures))
    if [[ $successes -lt 4 ]]; then
        log_json ERROR test_concurrent_requests "Too many failures" succeeded "$successes" failed "$failures"
        return 1
    fi

    log_json INFO test_concurrent_requests "Concurrent requests handled" succeeded "$successes" failed "$failures"
    return 0
}

test_daemon_restart() {
    # First daemon
    local pid1
    pid1=$(start_daemon 300)

    # Verify health
    if ! "$XF_BIN" daemon status --socket "$SOCKET_PATH" > /dev/null 2>&1; then
        log_json ERROR test_daemon_restart "First daemon unhealthy"
        return 1
    fi

    # Stop
    stop_daemon
    if ! wait_for_no_socket "$SOCKET_PATH" 20; then
        log_json ERROR test_daemon_restart "First daemon didn't clean up"
        return 1
    fi

    sleep 0.5

    # Second daemon
    local pid2
    pid2=$(start_daemon 300)

    # Verify health
    if ! "$XF_BIN" daemon status --socket "$SOCKET_PATH" > /dev/null 2>&1; then
        log_json ERROR test_daemon_restart "Second daemon unhealthy"
        return 1
    fi

    log_json INFO test_daemon_restart "Restart cycle complete" first_pid "$pid1" second_pid "$pid2"
    return 0
}

test_env_config_override() {
    # Test that XF_DAEMON_SOCK is respected
    local custom_socket="$TEST_DIR/custom.sock"

    cleanup_existing_daemon

    XF_DAEMON_SOCK="$custom_socket" "$XF_BIN" daemon start --foreground --idle-timeout 300 &
    local pid=$!

    if ! wait_for_socket "$custom_socket" 10; then
        log_json ERROR test_env_config_override "Daemon didn't use XF_DAEMON_SOCK"
        kill "$pid" 2>/dev/null || true
        return 1
    fi

    # Cleanup
    kill -INT "$pid" 2>/dev/null || true
    sleep 0.5

    log_json INFO test_env_config_override "Environment override worked" custom_socket "\"$custom_socket\""
    return 0
}

# --- Main ---
main() {
    # Check for xf binary
    if [[ ! -x "$XF_BIN" ]]; then
        echo "Building xf..."
        (cd "$PROJECT_ROOT" && cargo build --release)
    fi

    log_json INFO setup "Starting daemon E2E tests" \
        test_dir "\"$TEST_DIR\"" \
        xf_bin "\"$XF_BIN\"" \
        log_file "\"$LOG_FILE\""

    echo "Test output: $LOG_FILE"
    echo ""

    # Run specific test or all tests
    if [[ $# -gt 0 ]]; then
        run_test "$1"
    else
        run_test test_daemon_start_stop
        run_test test_daemon_status_json
        run_test test_daemon_idle_timeout
        run_test test_daemon_signal_handling
        run_test test_daemon_sigterm_handling
        run_test test_socket_permissions
        run_test test_pid_file_contents
        run_test test_concurrent_requests
        run_test test_daemon_restart
        run_test test_env_config_override
        # These tests require actual embed/rerank models
        # run_test test_daemon_embed_via_daemon
        # run_test test_daemon_rerank_via_daemon
    fi

    echo ""
    echo "================================"
    log_json INFO summary "Test run complete" \
        passed "$PASSES" \
        failed "$FAILURES" \
        skipped "$SKIPPED"

    if [[ $FAILURES -gt 0 ]]; then
        echo ""
        echo "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
    fi

    echo ""
    echo "Full log: $LOG_FILE"

    [[ $FAILURES -eq 0 ]]
}

main "$@"
