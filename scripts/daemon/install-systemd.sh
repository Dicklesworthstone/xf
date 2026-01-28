#!/bin/bash
# Install xf-daemon as a systemd user service
set -euo pipefail

SERVICE_NAME="xf-daemon"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

echo "Installing xf-daemon systemd user service..."

# Check if xf is installed
if ! command -v xf &>/dev/null; then
    echo "Error: xf binary not found in PATH"
    echo "Please install xf first: cargo install --path ."
    exit 1
fi

# Create user systemd directory
mkdir -p "$USER_UNIT_DIR"

# Copy service file
cp "$SCRIPT_DIR/xf-daemon.service" "$USER_UNIT_DIR/"

# Substitute home directory placeholder
sed -i "s|%h|$HOME|g" "$USER_UNIT_DIR/xf-daemon.service"

# Reload systemd
systemctl --user daemon-reload

echo ""
echo "Service installed successfully!"
echo ""
echo "Usage:"
echo "  systemctl --user enable xf-daemon   # Start on login"
echo "  systemctl --user start xf-daemon    # Start now"
echo "  systemctl --user status xf-daemon   # Check status"
echo "  systemctl --user stop xf-daemon     # Stop daemon"
echo "  journalctl --user -u xf-daemon -f   # Follow logs"
echo ""
echo "To uninstall:"
echo "  systemctl --user disable xf-daemon"
echo "  systemctl --user stop xf-daemon"
echo "  # Then manually remove: $USER_UNIT_DIR/xf-daemon.service"
