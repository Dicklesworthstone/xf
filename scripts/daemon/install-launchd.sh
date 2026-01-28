#!/bin/bash
# Install xf-daemon as a launchd user agent (macOS)
set -euo pipefail

PLIST_NAME="com.dicklesworthstone.xf-daemon"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs"

echo "Installing xf-daemon launchd agent..."

# Check if xf is installed
if ! command -v xf &>/dev/null; then
    echo "Error: xf binary not found in PATH"
    echo "Please install xf first: cargo install --path ."
    exit 1
fi

# Create directories
mkdir -p "$PLIST_DIR" "$LOG_DIR"

# Copy plist file
cp "$SCRIPT_DIR/$PLIST_NAME.plist" "$PLIST_DIR/"

# Update log paths to use home directory
sed -i "" "s|/tmp/xf-daemon.log|$LOG_DIR/xf-daemon.log|g" "$PLIST_DIR/$PLIST_NAME.plist"
sed -i "" "s|/tmp/xf-daemon.err|$LOG_DIR/xf-daemon.err|g" "$PLIST_DIR/$PLIST_NAME.plist"

# Unload if already loaded (ignore errors)
launchctl bootout "gui/$(id -u)/$PLIST_NAME" 2>/dev/null || true

# Load the agent
launchctl bootstrap "gui/$(id -u)" "$PLIST_DIR/$PLIST_NAME.plist"

echo ""
echo "Service installed and started!"
echo ""
echo "Usage:"
echo "  launchctl kickstart -k gui/$(id -u)/$PLIST_NAME   # Force restart"
echo "  launchctl print gui/$(id -u)/$PLIST_NAME          # Check status"
echo "  tail -f $LOG_DIR/xf-daemon.log                    # Follow logs"
echo "  xf daemon status                                   # Health check"
echo ""
echo "To uninstall:"
echo "  launchctl bootout gui/$(id -u)/$PLIST_NAME"
echo "  # Then manually remove: $PLIST_DIR/$PLIST_NAME.plist"
