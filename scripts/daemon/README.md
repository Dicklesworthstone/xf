# xf Daemon Service Files

This directory contains service definitions for running the xf model daemon as a system service.

## Overview

The xf daemon keeps embedding and reranker models warm in memory for fast inference. Running it as a service ensures it starts automatically and stays running.

## Linux (systemd)

### Install as User Service

```bash
./install-systemd.sh
```

This installs the daemon as a user service (runs when you log in, not at boot).

### Commands

```bash
# Enable to start on login
systemctl --user enable xf-daemon

# Start now
systemctl --user start xf-daemon

# Check status
systemctl --user status xf-daemon

# Stop
systemctl --user stop xf-daemon

# View logs
journalctl --user -u xf-daemon -f
```

### Resource Limits

The systemd service is configured with:
- Nice level: 10 (lower priority)
- IO scheduling: idle class
- Memory limit: 2GB max, 1.5GB high
- CPU quota: 200% (2 cores max)

## macOS (launchd)

### Install as User Agent

```bash
./install-launchd.sh
```

This installs and starts the daemon as a user agent.

### Commands

```bash
# Force restart
launchctl kickstart -k gui/$(id -u)/com.dicklesworthstone.xf-daemon

# Check status
launchctl print gui/$(id -u)/com.dicklesworthstone.xf-daemon

# View logs
tail -f ~/Library/Logs/xf-daemon.log

# Health check
xf daemon status
```

### Uninstall

```bash
launchctl bootout gui/$(id -u)/com.dicklesworthstone.xf-daemon
# Remove plist manually if desired
```

## Manual Control

You can always control the daemon manually without service files:

```bash
# Start in background
xf daemon start

# Start in foreground (for debugging)
xf daemon start --foreground

# Check status
xf daemon status

# Stop
xf daemon stop
```

## Troubleshooting

### Daemon not starting

1. Check if xf is in PATH: `which xf`
2. Try running manually: `xf daemon start --foreground`
3. Check logs (see commands above)

### Socket permission errors

The daemon creates a socket at `/tmp/xf-daemon-$USER.sock`. Ensure:
- /tmp is writable
- No stale socket file exists

### Memory issues

If the daemon is using too much memory:
1. Edit the service file to lower `MemoryMax`
2. Reduce max loaded models in xf config
