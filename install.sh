#!/usr/bin/env bash
#
# xf installer
#
# One-liner install (with cache buster):
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash
#
# Or without cache buster:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --from-source      Build from source instead of downloading binary
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#   --insecure-skip-checksum   Skip SHA256 verification (NOT recommended)
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

VERSION="${VERSION:-}"
OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-xf}"
DEST_DEFAULT="$HOME/.local/bin"
DEST="${DEST:-$DEST_DEFAULT}"
EASY=0
QUIET=0
VERIFY=0
FROM_SOURCE=0
CHECKSUM="${CHECKSUM:-}"
CHECKSUM_URL="${CHECKSUM_URL:-}"
SKIP_CHECKSUM="${SKIP_CHECKSUM:-0}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
LOCK_FILE="/tmp/xf-install.lock"
SYSTEM=0
NO_GUM=0

# Detect gum for fancy output (https://github.com/charmbracelet/gum)
HAS_GUM=0
if command -v gum &> /dev/null && [ -t 1 ]; then
  HAS_GUM=1
fi

# Logging functions with optional gum formatting
log() { [ "$QUIET" -eq 1 ] && return 0; echo -e "$@"; }

info() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 39 "→ $*"
  else
    echo -e "\033[0;34m→\033[0m $*"
  fi
}

ok() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 42 "✓ $*"
  else
    echo -e "\033[0;32m✓\033[0m $*"
  fi
}

warn() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 214 "⚠ $*"
  else
    echo -e "\033[1;33m⚠\033[0m $*"
  fi
}

err() {
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 196 "✗ $*"
  else
    echo -e "\033[0;31m✗\033[0m $*"
  fi
}

# Spinner wrapper for long operations
run_with_spinner() {
  local title="$1"
  shift
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ] && [ "$QUIET" -eq 0 ]; then
    gum spin --spinner dot --title "$title" -- "$@"
  else
    info "$title"
    "$@"
  fi
}

resolve_version() {
  if [ -n "$VERSION" ]; then return 0; fi

  info "Resolving latest version..."
  local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
  local tag
  if ! tag=$(curl -fsSL -H "Accept: application/vnd.github.v3+json" "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); then
    tag=""
  fi

  if [ -n "$tag" ]; then
    VERSION="$tag"
    info "Resolved latest version: $VERSION"
  else
    # Try redirect-based resolution as fallback
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if tag=$(curl -fsSL -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||'); then
      # Validate: tag must be non-empty, start with 'v' + digit, and not contain URL chars
      if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *"/"* ]]; then
        VERSION="$tag"
        info "Resolved latest version via redirect: $VERSION"
        return 0
      fi
    fi
    err "Could not resolve the latest release version (GitHub API and redirect lookup both failed)."
    err "This is usually a transient network issue or GitHub API rate limiting."
    err "Retry later, or pin a version explicitly, e.g.: install.sh --version v0.4.0"
    err "Releases: https://github.com/${OWNER}/${REPO}/releases"
    exit 1
  fi
}

maybe_add_path() {
  case ":$PATH:" in
    *:"$DEST":*) return 0;;
    *)
      if [ "$EASY" -eq 1 ]; then
        UPDATED=0
        for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
          if [ -e "$rc" ] && [ -w "$rc" ]; then
            if ! grep -F "$DEST" "$rc" >/dev/null 2>&1; then
              echo "export PATH=\"$DEST:\$PATH\"" >> "$rc"
            fi
            UPDATED=1
          fi
        done
        if [ "$UPDATED" -eq 1 ]; then
          warn "PATH updated in ~/.zshrc/.bashrc; restart shell to use xf"
        else
          warn "Add $DEST to PATH to use xf"
        fi
      else
        warn "Add $DEST to PATH to use xf"
      fi
    ;;
  esac
}

ensure_rust() {
  if [ "${RUSTUP_INIT_SKIP:-0}" != "0" ]; then
    info "Skipping rustup install (RUSTUP_INIT_SKIP set)"
    return 0
  fi
  if command -v cargo >/dev/null 2>&1 && rustc --version 2>/dev/null | grep -q nightly; then return 0; fi
  if [ "$EASY" -ne 1 ]; then
    if [ -t 0 ]; then
      echo -n "Install Rust nightly via rustup? (y/N): "
      read -r ans
      case "$ans" in y|Y) :;; *) warn "Skipping rustup install"; return 0;; esac
    fi
  fi
  info "Installing rustup (nightly)"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  rustup component add rustfmt clippy || true
}

usage() {
  cat <<EOFU
Usage: install.sh [--version vX.Y.Z] [--dest DIR] [--system] [--easy-mode] [--verify] \\
                  [--artifact-url URL] [--checksum HEX] [--checksum-url URL] \\
                  [--insecure-skip-checksum] [--quiet] [--no-gum]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --from-source      Build from source instead of downloading binary
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
  --insecure-skip-checksum   Skip SHA256 verification (NOT recommended)
EOFU
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --system) SYSTEM=1; DEST="/usr/local/bin"; shift;;
    --easy-mode) EASY=1; shift;;
    --verify) VERIFY=1; shift;;
    --artifact-url) ARTIFACT_URL="$2"; shift 2;;
    --checksum) CHECKSUM="$2"; shift 2;;
    --checksum-url) CHECKSUM_URL="$2"; shift 2;;
    --from-source) FROM_SOURCE=1; shift;;
    --insecure-skip-checksum) SKIP_CHECKSUM=1; shift;;
    --quiet|-q) QUIET=1; shift;;
    --no-gum) NO_GUM=1; shift;;
    -h|--help) usage; exit 0;;
    *) shift;;
  esac
done

# Show fancy header
if [ "$QUIET" -eq 0 ]; then
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style \
      --border normal \
      --border-foreground 39 \
      --padding "0 1" \
      --margin "1 0" \
      "$(gum style --foreground 42 --bold 'xf installer')" \
      "$(gum style --foreground 245 'Ultra-fast X data archive search')"
  else
    echo ""
    echo -e "\033[1;32mxf installer\033[0m"
    echo -e "\033[0;90mUltra-fast Twitter/X archive search\033[0m"
    echo ""
  fi
fi

resolve_version

mkdir -p "$DEST"
OS=$(uname -s | tr 'A-Z' 'a-z')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *) warn "Unknown arch $ARCH, using as-is" ;;
esac

TARGET=""
case "${OS}-${ARCH}" in
  linux-x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
  linux-aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
  darwin-x86_64) TARGET="x86_64-apple-darwin" ;;
  darwin-aarch64) TARGET="aarch64-apple-darwin" ;;
  *) :;;
esac

# Prefer prebuilt artifact when we know the target or the caller supplied a direct URL.
TAR=""
URL=""
if [ "$FROM_SOURCE" -eq 0 ]; then
  if [ -n "$ARTIFACT_URL" ]; then
    TAR=$(basename "$ARTIFACT_URL")
    URL="$ARTIFACT_URL"
  elif [ -n "$TARGET" ]; then
    TAR="xf-${TARGET}.tar.gz"
    URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${TAR}"
  else
    warn "No prebuilt artifact for ${OS}/${ARCH}; falling back to build-from-source"
    FROM_SOURCE=1
  fi
fi

# Cross-platform locking using mkdir (atomic on all POSIX systems including macOS)
LOCK_DIR="${LOCK_FILE}.d"
LOCKED=0
if mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCKED=1
  echo $$ > "$LOCK_DIR/pid"
else
  # Check if existing lock is stale (process no longer running)
  if [ -f "$LOCK_DIR/pid" ]; then
    OLD_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ! kill -0 "$OLD_PID" 2>/dev/null; then
      rm -rf "$LOCK_DIR"
      if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCKED=1
        echo $$ > "$LOCK_DIR/pid"
      fi
    fi
  fi
  if [ "$LOCKED" -eq 0 ]; then
    err "Another installer is running (lock $LOCK_DIR)"
    exit 1
  fi
fi

cleanup() {
  rm -rf "$TMP"
  if [ "$LOCKED" -eq 1 ]; then rm -rf "$LOCK_DIR"; fi
}

TMP=$(mktemp -d)
trap cleanup EXIT

if [ "$FROM_SOURCE" -eq 0 ]; then
  info "Downloading $URL"
  if ! curl -fsSL "$URL" -o "$TMP/$TAR"; then
    warn "Artifact download failed; falling back to build-from-source"
    FROM_SOURCE=1
  fi
fi

if [ "$FROM_SOURCE" -eq 1 ]; then
  info "Building from source (requires git, rust nightly)"
  ensure_rust
  git clone --depth 1 "https://github.com/${OWNER}/${REPO}.git" "$TMP/src"
  (cd "$TMP/src" && cargo build --release)
  BIN="$TMP/src/target/release/xf"
  [ -x "$BIN" ] || { err "Build failed"; exit 1; }
  install -m 0755 "$BIN" "$DEST/xf"
  ok "Installed to $DEST/xf (source build)"
  maybe_add_path
  if [ "$VERIFY" -eq 1 ]; then
    "$DEST/xf" --version || true
    ok "Self-test complete"
  fi
  ok "Done. Binary at: $DEST/xf"
  exit 0
fi

if [ "$SKIP_CHECKSUM" -eq 1 ]; then
  warn "SHA256 verification DISABLED (--insecure-skip-checksum); installing unverified artifact"
else
  if [ -z "$CHECKSUM" ]; then
    # Use SHA256SUMS file from release (contains checksums for all artifacts)
    if [ -z "$CHECKSUM_URL" ]; then
      CHECKSUM_URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/SHA256SUMS"
    fi
    info "Fetching checksum from ${CHECKSUM_URL}"
    CHECKSUM_FILE="$TMP/SHA256SUMS"
    if ! curl -fsSL "$CHECKSUM_URL" -o "$CHECKSUM_FILE"; then
      err "Could not fetch checksums from ${CHECKSUM_URL}"
      err "Refusing to install an unverified binary."
      err "Retry later, supply one via --checksum HEX / --checksum-url URL,"
      err "or (NOT recommended) bypass verification with --insecure-skip-checksum."
      exit 1
    fi
    # Extract checksum for our specific file from SHA256SUMS.
    # `|| true` guards set -e -o pipefail: a missing entry must reach the
    # explicit error below, not kill the script mid-pipeline with no message.
    CHECKSUM=$(grep "  ${TAR}$" "$CHECKSUM_FILE" 2>/dev/null | awk '{print $1}' || true)
    if [ -z "$CHECKSUM" ]; then
      # Try alternate format (single space)
      CHECKSUM=$(grep " ${TAR}$" "$CHECKSUM_FILE" 2>/dev/null | awk '{print $1}' || true)
    fi
    if [ -z "$CHECKSUM" ]; then
      err "No checksum entry for ${TAR} in ${CHECKSUM_URL}"
      err "Refusing to install an unverified binary."
      err "Supply one via --checksum HEX / --checksum-url URL,"
      err "or (NOT recommended) bypass verification with --insecure-skip-checksum."
      exit 1
    fi
  fi
  echo "$CHECKSUM  $TMP/$TAR" | sha256sum -c - || { err "Checksum mismatch"; exit 1; }
  ok "Checksum verified"
fi

info "Extracting"
tar -xf "$TMP/$TAR" -C "$TMP"
BIN="$TMP/xf"
if [ ! -x "$BIN" ] && [ -n "$TARGET" ]; then
  BIN="$TMP/xf-${TARGET}/xf"
fi
if [ ! -x "$BIN" ]; then
  BIN=$(find "$TMP" -maxdepth 3 -type f -name "xf" -perm -111 | head -n 1)
fi

[ -x "$BIN" ] || { err "Binary not found in tar"; exit 1; }
install -m 0755 "$BIN" "$DEST/xf"
ok "Installed to $DEST/xf"
maybe_add_path

if [ "$VERIFY" -eq 1 ]; then
  "$DEST/xf" --version || true
  ok "Self-test complete"
fi

ok "Done. Binary at: $DEST/xf"
echo ""
info "Quick start:"
echo "  1. Download your data from x.com/settings/download_your_data"
echo "  2. Run: xf index /path/to/your-archive"
echo "  3. Search: xf search \"your query\""
echo ""
info "Tip: For automatic updates, install via Homebrew: brew install dicklesworthstone/tap/xf"
