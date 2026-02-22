# xf

<div align="center">
  <img src="xf_illustration.webp" alt="xf - Ultra-fast CLI for searching your X data archive">
</div>

<div align="center">

[![CI](https://github.com/Dicklesworthstone/xf/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicklesworthstone/xf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT%2BOpenAI%2FAnthropic%20Rider-blue.svg)](./LICENSE)

</div>

Ultra-fast CLI for searching and querying your X data archive with sub-millisecond latency.

<div align="center">
<h3>Quick Install</h3>

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash
```

**Or via package managers:**

```bash
# macOS/Linux (Homebrew)
brew install dicklesworthstone/tap/xf

# Windows (Scoop)
scoop bucket add dicklesworthstone https://github.com/Dicklesworthstone/scoop-bucket
scoop install dicklesworthstone/xf
```

<p><em>Works on Linux, macOS, and Windows. Auto-detects your platform and downloads the right binary.</em></p>
</div>

---

## 🤖 Agent Quickstart (JSON)

**Use `--format json` in agent contexts.** stdout = data, stderr = diagnostics, exit 0 = success.

```bash
# 1) Index once (required before search)
xf index ~/x-archive

# 2) Search (machine-readable)
xf search "machine learning" --format json --limit 20

# 3) Archive stats (machine-readable)
xf stats --format json
```

## TL;DR

**The Problem**: X lets you download all your data, but actually *finding* anything in that archive is painful. The built-in HTML viewer is slow and clunky, there's no real search, and your data is scattered across separate files.

**The Solution**: `xf` indexes your X (formerly Twitter) data export and provides blazingly fast full-text search across tweets, likes, DMs, and Grok conversations—all from the command line.

### Why Use xf?

| Feature | What It Does |
|---------|--------------|
| **Sub-Millisecond Search** | Tantivy-powered full-text search with BM25 ranking |
| **Vector Similarity (default: hash)** | Finds content with overlapping vocabulary; best when queries share words with target content |
| **True Semantic Search (optional ML)** | Uses MiniLM embeddings when indexed with `--semantic` for synonym-level matching |
| **Hybrid Search** | Combines keyword + vector similarity (hash by default, ML when indexed with `--semantic`) |
| **Search Everything** | Tweets, likes, DMs, and Grok conversations in one place |
| **Rich Query Syntax** | Phrases, wildcards, boolean operators (`AND`, `OR`, `NOT`) |
| **DM Context** | View full conversation threads with search matches highlighted |
| **Multiple Formats** | JSON, CSV, compact, or colorized terminal output |
| **Privacy-First** | All data stays local on your machine—nothing sent anywhere |
| **Fast Indexing** | ~10,000 documents/second with parallel parsing |

Note: Semantic mode uses hash-based vocabulary similarity **by default**. Run `xf index --semantic` to build true semantic embeddings (MiniLM). If you switch modes, re-index so the vector index matches the embedder.

### Quick Example

```bash
# Index your archive (default: hash-based embeddings)
$ xf index ~/x-archive

# Optional: true semantic embeddings (downloads ~80MB on first use)
$ xf index ~/x-archive --semantic

# Search across everything (hybrid mode by default)
$ xf search "machine learning"

# Semantic search (vector similarity; true semantic if indexed with --semantic)
$ xf search "feeling overwhelmed at work" --mode semantic

# Keyword-only search (classic BM25)
$ xf search "rust async" --mode lexical

# Search only your DMs with full conversation context
$ xf search "meeting tomorrow" --types dm --context

# Export results as JSON
$ xf search "rust async" --format json --limit 50
```
---
## Prepared Blurb for AGENTS.md Files:
```
## xf — X Archive Search

Ultra-fast local search for X (Twitter) data archives. Parses `window.YTD.*` JavaScript format from X data exports. Hybrid search combining keyword (BM25) + vector similarity (hash by default; ML when indexed with `--semantic`) via RRF fusion.

### Core Workflow

```bash
# 1. Index archive (one-time, ~5-30 seconds)
xf index ~/x-archive
xf index ~/x-archive --force          # Rebuild from scratch
xf index ~/x-archive --only tweet,dm  # Index specific types
xf index ~/x-archive --skip grok      # Skip specific types
xf index ~/x-archive --semantic       # True semantic embeddings (MiniLM; slower)

# 2. Search
xf search "machine learning"          # Hybrid search (default)
xf search "feeling stressed" --mode semantic  # Vector similarity (hash default, ML if indexed with --semantic)
xf search "rust async" --mode lexical # Keyword-only (BM25)
xf search "meeting" --types dm        # DMs only
xf search "article" --types like      # Liked tweets only

Search Modes

--mode hybrid   # Default: BM25 + vector similarity (hash default, ML with --semantic index)
--mode lexical  # Keyword-only (BM25), best for exact terms
--mode semantic # Vector similarity (hash default, ML with --semantic index)

Search Syntax (lexical mode)

xf search "exact phrase"              # Phrase match (quotes matter)
xf search "rust AND async"            # Boolean AND
xf search "python OR javascript"      # Boolean OR
xf search "python NOT snake"          # Exclusion
xf search "rust*"                     # Wildcard prefix

Key Flags

--format json                         # Machine-readable output (use this!)
--format csv                          # Spreadsheet export
--limit 50                            # Results count (default: 20)
--offset 20                           # Pagination
--context                             # Full DM conversation thread (--types dm only)
--since "2024-01-01"                  # Date filter (supports natural language)
--until "last week"                   # Date filter
--sort date|date_desc|relevance|engagement

Other Commands

xf stats                              # Archive overview (counts, date range)
xf stats --detailed                   # Full analytics (temporal, engagement, content)
xf stats --format json                # Machine-readable stats
xf tweet <id>                         # Show specific tweet by ID
xf tweet <id> --engagement            # Include engagement metrics
xf list tweets --limit 20             # Browse indexed tweets
xf list dms                           # Browse DM conversations
xf doctor                             # Health checks (archive, DB, index)
xf shell                              # Interactive REPL

Data Types

tweet (your posts), like (liked tweets), dm (direct messages), grok (AI chats), follower, following, block, mute

Storage

- Database: ~/.local/share/xf/xf.db (override: XF_DB env)
- Index: ~/.local/share/xf/xf_index/ (override: XF_INDEX env)
- Archive format: Expects data/ directory with tweets.js, like.js, direct-messages.js, etc.

Notes

- First search after restart may be slower (index loading). Subsequent searches <10ms.
- Semantic mode uses hash-based similarity by default. Run `xf index --semantic` for true semantic embeddings.
- --context only works with --types dm — shows full conversation around matches.
- All data stays local. No network access during search; optional model download only when you enable `--semantic`.
```

---

## Design Philosophy

`xf` is built around several core principles that inform every design decision:

### Local-First, Privacy-Always

Your social media history is deeply personal. `xf` processes everything locally:

- **No network calls during search**: Zero telemetry, no analytics, no "phone home" (optional model download only if you enable `--semantic`)
- **No cloud dependencies**: Works completely offline after installation
- **No API keys**: Unlike tools that query X's API, `xf` works entirely from your downloaded archive
- **Your data stays yours**: The SQLite database and search index live on your machine

### Zero-Configuration Similarity

Getting started should take seconds, not hours:

- **Sensible defaults**: Hybrid search, 20 results, colorized output—just works
- **Auto-detection**: Finds archive structure automatically, handles format variations
- **No model downloads by default**: The hash embedder means no waiting for ML model files (unless you opt into `--semantic`)
- **Platform detection**: Install script handles OS/architecture differences

### Composition Over Complexity

`xf` is designed to play well with Unix philosophy:

```bash
# Pipe to jq for custom JSON processing
xf search "machine learning" --format json | jq '.[] | .text'

# Count tweets by year
xf search "coffee" --format json --limit 1000 | jq -r '.[].created_at[:4]' | sort | uniq -c

# Export to clipboard (macOS)
xf tweet 1234567890 --format json | pbcopy

# Feed into other tools
xf search "interesting" --types like --format json | ./my-analysis-script.py
```

### Speed as a Feature

Performance isn't an afterthought—it's a core feature:

- **Sub-millisecond lexical search**: Faster than you can blink
- **Memory-mapped indices**: OS-level caching, minimal RAM overhead
- **Parallel everything**: Parsing, indexing, embedding generation
- **Lazy initialization**: Pay only for what you use

## How xf Compares

| Feature | xf | X's HTML Viewer | grep/ripgrep | Elasticsearch |
|---------|-----|-----------------|--------------|---------------|
| Full-text search | ✅ BM25 + vector similarity (hash default; ML optional) | ❌ None | ⚠️ Basic regex | ✅ Full |
| Similarity search | ✅ Hash embedder | ❌ | ❌ | ⚠️ With plugins |
| Search speed | ✅ <10ms | ❌ Manual scrolling | ⚠️ Depends on size | ✅ Fast |
| Setup time | ✅ ~10 seconds | ✅ Just open HTML | ✅ None | ❌ Hours |
| Dependencies | ✅ Single binary | ✅ Browser | ✅ None | ❌ JVM, config |
| Offline use | ✅ Fully offline | ✅ | ✅ | ⚠️ Usually |
| Privacy | ✅ 100% local | ✅ | ✅ | ⚠️ Depends |
| DM search | ✅ With context | ❌ | ⚠️ Raw files | ✅ If indexed |
| Date filtering | ✅ Natural language | ❌ | ❌ | ✅ |
| Export formats | ✅ JSON/CSV/text | ❌ | ⚠️ Raw text | ✅ |

**When to use xf:**
- You want fast, comprehensive search across your entire archive
- You value privacy and want everything local
- You want similarity search without cloud APIs
- You prefer CLI tools that compose with Unix pipelines

**When xf might not be ideal:**
- You only need to find one specific tweet (just Ctrl+F in the HTML viewer)
- You need real-time access to X (use the app/website)
- You want collaborative features (xf is single-user by design)

## Origins & Authors

This project was created by Jeffrey Emanuel after realizing that X's data export, while comprehensive, lacks any useful search functionality.

- **[Jeffrey Emanuel](https://github.com/Dicklesworthstone)** - Creator and maintainer

## Getting Your X Data Archive

Before using `xf`, you need to download your data from X. Here's the complete process:

### Step 1: Request Your Archive

1. **Log into X** at [x.com](https://x.com) or [twitter.com](https://twitter.com)
2. **Navigate to Settings**:
   - Click "More" (...) in the left sidebar
   - Select "Settings and Support" -> "Settings and privacy"
   - Or go directly to: [x.com/settings/download_your_data](https://x.com/settings/download_your_data)
3. **Request your archive**:
   - Under "Download an archive of your data", click "Request archive"
   - You may need to verify your identity (password, 2FA)
   - Select what data you want (recommend "All data" for complete archive)

### Step 2: Wait for Processing

X needs time to compile your archive:
- **Typical wait time**: 24-48 hours (can be longer for large accounts)
- You'll receive an **email notification** when it's ready
- You can also check the same settings page for status updates
- The link expires after a few days, so download promptly!

### Step 3: Download and Extract

1. **Download**: Click the link in your email or on the settings page
   - File will be named something like `twitter-2026-01-09-abc123.zip`
   - Size varies: typically 50MB to several GB depending on your activity and media
2. **Extract**: Unzip the archive to a folder
   ```bash
   unzip twitter-2026-01-09-abc123.zip -d ~/x-archive
   ```

### What's Inside the Archive

Your extracted archive contains:

```
x-archive/
├── Your archive.html      # Browser viewer (open this to explore manually)
├── data/
│   ├── tweets.js          # All your tweets
│   ├── like.js            # Tweets you've liked
│   ├── direct-messages.js # DM conversations
│   ├── follower.js        # Your followers
│   ├── following.js       # Accounts you follow
│   ├── grok-chat-item.js   # Grok AI chats (if any)
│   ├── account.js         # Account info
│   ├── profile.js         # Profile data
│   └── ...                # Many other data files
└── assets/
    └── images/            # Media files (can be large!)
```

The data files use a JavaScript format like:
```javascript
window.YTD.tweets.part0 = [
  { "tweet": { "id": "123...", "full_text": "Hello world!", ... } },
  ...
]
```

`xf` knows how to parse this format and extract all your content.

### ⚠️ Important: What's NOT in Your Archive

Your X data archive only contains **your own data**—content you created or directly interacted with. This is a limitation of X's export, not xf.

**What IS included:**
| Data Type | Description |
|-----------|-------------|
| Your tweets | Everything you posted (including replies you made to others) |
| Your likes | Tweets you liked (with full text if available) |
| Your DMs | Direct message conversations you participated in |
| Your Grok chats | Conversations with Grok AI |
| Followers/Following | Lists of accounts (usernames only, not their tweets) |

**What is NOT included:**
| Data Type | Why It's Missing |
|-----------|------------------|
| **Replies to your tweets** | Other people's replies are *their* data, not yours |
| **Quote tweets of you** | Same reason—belongs to whoever quoted you |
| **Mentions of you** | Tweets mentioning @you are owned by others |
| **Others' tweets** | You only get tweets you liked, not random tweets you viewed |
| **Analytics/impressions** | Detailed view counts aren't in the standard export |

**Why this matters:** If you're hoping to find "what did people say in response to my tweet about X?"—that data isn't in your archive. You'd need to use the X API or third-party tools to fetch replies in real-time.

**What you CAN do:**
- Search your own replies to others: `xf search "query" --replies-only`
- Find conversations in your DMs: `xf search "topic" --types dm --context`
- See tweets you engaged with via likes: `xf search "topic" --types like`

## Installation

### Quick Install (Recommended)

**Recommended: Homebrew (macOS/Linux)**
```bash
brew install dicklesworthstone/tap/xf
```

**Windows: Scoop**
```powershell
scoop bucket add dicklesworthstone https://github.com/Dicklesworthstone/scoop-bucket
scoop install dicklesworthstone/xf
```

**Alternative: Install Script**
The easiest way to install without a package manager is using the install script, which downloads a prebuilt binary for your platform:

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash
```

**With options:**

Easy mode (auto-update PATH in shell rc files):

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash -s -- --easy-mode
```

Install specific version:

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash -s -- --version v0.1.0
```

Install to /usr/local/bin (system-wide, requires sudo):

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | sudo bash -s -- --system
```

Build from source instead of downloading binary:

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash -s -- --from-source
```

> **Note:** If you have [gum](https://github.com/charmbracelet/gum) installed, the installer will use it for fancy terminal formatting.

The install script:
- Automatically detects your OS and architecture
- Downloads the appropriate prebuilt binary
- Verifies SHA256 checksums for security
- Falls back to building from source if no prebuilt is available
- Offers to update your PATH

### From Source (requires Rust nightly)

This project uses Rust Edition 2024 features and requires the nightly toolchain. The repository includes a `rust-toolchain.toml` that automatically selects the correct toolchain.

```bash
# Install Rust nightly if you don't have it
rustup install nightly

# Install directly from GitHub
cargo +nightly install --git https://github.com/Dicklesworthstone/xf.git
```

### Manual Build

```bash
git clone https://github.com/Dicklesworthstone/xf.git
cd xf
# rust-toolchain.toml automatically selects nightly
cargo build --release
cp target/release/xf ~/.local/bin/
```

### Prebuilt Binaries

Prebuilt binaries are available for:
- Linux x86_64 (`x86_64-unknown-linux-gnu`)
- Linux ARM64 (`aarch64-unknown-linux-gnu`)
- macOS Intel (`x86_64-apple-darwin`)
- macOS Apple Silicon (`aarch64-apple-darwin`)

Download from [GitHub Releases](https://github.com/Dicklesworthstone/xf/releases) and verify the SHA256 checksum.

## Quick Start

### 1. Index your archive

```bash
xf index ~/x-archive
```

This parses all your data and builds a searchable index. On a typical archive, this takes 10-30 seconds.

### 2. Search!

```bash
# Basic search
xf search "machine learning"

# Search only tweets
xf search "python" --types tweet

# Search DMs
xf search "meeting" --types dm

# Search likes
xf search "interesting article" --types like

# JSON output
xf search "rust" --format json

# Limit results
xf search "AI" --limit 5
```

## Commands

### `xf index <archive_path>`

Index an X data archive.

```bash
xf index ~/Downloads/x-archive

# Force re-index (clear existing data)
xf index ~/Downloads/x-archive --force

# Build true semantic embeddings (MiniLM; downloads ~80MB on first use)
xf index ~/Downloads/x-archive --semantic

# Index only specific data types
xf index ~/Downloads/x-archive --only tweet,like

# Skip certain data types
xf index ~/Downloads/x-archive --skip dm,grok
```

### `xf search <query>`

Search the indexed archive.

```bash
# Basic search (hybrid mode by default)
xf search "your query"

# Search modes
xf search "query" --mode hybrid    # Default: combines keyword + vector similarity (hash default; ML if indexed with --semantic)
xf search "query" --mode lexical   # Keyword-only (BM25)
xf search "query" --mode semantic  # Vector similarity (hash default; ML if indexed with --semantic)

# Filter by type
xf search "query" --types tweet,dm

# Pagination
xf search "query" --limit 20 --offset 40

# Output formats
xf search "query" --format json
xf search "query" --format csv
xf search "query" --format compact

# DM context: show full conversation with matches highlighted
xf search "meeting" --types dm --context
xf search "meeting" --types dm --context --format json
```

**Search Modes:**

| Mode | Best For | How It Works |
|------|----------|--------------|
| `hybrid` | General use (default) | Combines keyword + vector similarity (hash default; ML with `--semantic`) |
| `lexical` | Exact terms, boolean queries | Classic BM25 keyword matching |
| `semantic` | Similar wording | Vector similarity (hash default; ML with `--semantic`) |

**Query syntax:**
- Simple terms: `machine learning`
- Phrases: `"exact phrase"`
- Boolean: `rust AND async`
- Exclusion: `python NOT snake`

### `xf stats`

Show archive statistics.

```bash
xf stats

# JSON output
xf stats --format json

# Detailed breakdown
xf stats --detailed
```

### `xf tweet <id>`

Show details for a specific tweet.

```bash
xf tweet 1234567890

# Show engagement metrics
xf tweet 1234567890 --engagement
```

### `xf config`

Manage configuration.

```bash
# Show current config
xf config --show
```

### `xf update`

Check for updates.

```bash
xf update
```

### `xf completions <shell>`

Generate shell completions.

```bash
# Bash
xf completions bash > ~/.local/share/bash-completion/completions/xf

# Zsh
xf completions zsh > ~/.zfunc/_xf

# Fish
xf completions fish > ~/.config/fish/completions/xf.fish
```

## Output Formats

| Format | Description |
|--------|-------------|
| `text` | Human-readable with colors (default) |
| `json` | Compact JSON |
| `json-pretty` | Pretty-printed JSON |
| `csv` | Comma-separated values |
| `compact` | One result per line |

## Data Types

| Type | Description |
|------|-------------|
| `tweet` | Your tweets |
| `like` | Tweets you've liked |
| `dm` | Direct messages |
| `grok` | Grok AI conversations |
| `follower` | Your followers |
| `following` | Accounts you follow |
| `block` | Blocked accounts |
| `mute` | Muted accounts |

## Storage Locations

By default, `xf` stores data in:

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/xf/` |
| Linux | `~/.local/share/xf/` |
| Windows | `%LOCALAPPDATA%\xf\` |

Override with environment variables:
- `XF_DB`: Path to SQLite database
- `XF_INDEX`: Path to search index directory

## Data Model

### What Gets Indexed

Each document type has specific fields indexed for search:

#### Tweets

| Field | Indexed | Stored | Notes |
|-------|---------|--------|-------|
| `id` | ✅ Term | ✅ | Tweet ID for lookup |
| `full_text` | ✅ Full-text | ✅ | Main search content |
| `created_at` | ✅ Date | ✅ | For date filtering |
| `favorite_count` | ❌ | ✅ | Likes received |
| `retweet_count` | ❌ | ✅ | Retweets received |
| `in_reply_to_status_id` | ✅ Term | ✅ | For thread detection |
| `hashtags` | ❌ | ✅ | Extracted from text |
| `mentions` | ❌ | ✅ | @usernames mentioned |
| `urls` | ❌ | ✅ | Expanded URLs |
| `media` | ❌ | ✅ | Media attachments |

#### Likes

| Field | Indexed | Stored | Notes |
|-------|---------|--------|-------|
| `tweet_id` | ✅ Term | ✅ | Liked tweet's ID |
| `full_text` | ✅ Full-text | ✅ | If available in export |
| `expanded_url` | ❌ | ✅ | Link to original |

#### Direct Messages

| Field | Indexed | Stored | Notes |
|-------|---------|--------|-------|
| `id` | ✅ Term | ✅ | Message ID |
| `conversation_id` | ✅ Term | ✅ | For grouping context |
| `text` | ✅ Full-text | ✅ | Message content |
| `sender_id` | ✅ Term | ✅ | Who sent it |
| `recipient_id` | ❌ | ✅ | Who received it |
| `created_at` | ✅ Date | ✅ | Timestamp |

#### Grok Conversations

| Field | Indexed | Stored | Notes |
|-------|---------|--------|-------|
| `chat_id` | ✅ Term | ✅ | Conversation ID |
| `message` | ✅ Full-text | ✅ | Message content |
| `sender` | ✅ Term | ✅ | "user" or "grok" |
| `created_at` | ✅ Date | ✅ | Timestamp |

### Embedding Strategy

All content is stored and indexed in full—nothing is truncated. For vector embeddings, text is canonicalized (Unicode normalization, markdown stripped, whitespace collapsed) before embedding (hash or ML).

| Type | Text Source | Notes |
|------|-------------|-------|
| Tweet | `full_text` | Full content including long-form tweets |
| Like | `full_text` | If available from archive |
| DM | `text` | Full message text |
| Grok | `message` | Full response text |

Empty or trivial messages (e.g., "OK", "Thanks") are filtered from embeddings but still searchable via keyword search.

## Security & Privacy

### Your Data Never Leaves Your Machine

`xf` is designed with privacy as a non-negotiable requirement:

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR MACHINE                            │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  X Archive  │───▶│  xf binary  │───▶│  Local DB   │      │
│  │  (input)    │    │  (process)  │    │  (output)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                             │
│  ❌ No network calls                                        │
│  ❌ No telemetry                                            │
│  ❌ No cloud sync                                           │
│  ❌ No API keys required                                    │
└─────────────────────────────────────────────────────────────┘
```

### What's Stored Where

| Location | Contents | Sensitive? |
|----------|----------|------------|
| `~/.local/share/xf/xf.db` | Full tweet text, DMs, metadata | ⚠️ **Yes** |
| `~/.local/share/xf/xf_index/` | Tokenized search index | ⚠️ Yes (reversible) |
| Embeddings (in DB) | Numerical vectors | Low (hard to reverse) |

**Recommendations:**

1. **Encrypt your disk**: Use full-disk encryption (FileVault, LUKS, BitLocker)
2. **Secure permissions**: The database is created with user-only permissions (0600)
3. **Backup carefully**: When backing up, treat xf's data directory as sensitive
4. **Delete when done**: `rm -rf ~/.local/share/xf/` removes all indexed data

### No Network Access

xf makes zero network calls during normal search operations:

- **No update checks**: Use `xf update` explicitly when you want to update
- **No telemetry**: No usage stats, no error reporting, no analytics
- **No model downloads by default**: The hash embedder is pure Rust (unless you opt into `xf index --semantic`)
- **No API calls**: Works entirely from your local archive export

The only network access is during:
1. **Installation**: Downloading the binary from GitHub Releases
2. **`xf update`**: Checking for and downloading updates (user-initiated)
3. **Optional semantic indexing**: Downloading the MiniLM model when you run `xf index --semantic`

### Secure Deletion

To completely remove all xf data:

```bash
# Remove database and index
rm -rf ~/.local/share/xf/

# Or on macOS
rm -rf ~/Library/Application\ Support/xf/

# Remove the binary
rm ~/.local/bin/xf
# or
rm /usr/local/bin/xf
```

This permanently deletes all indexed content. The original archive is unaffected.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       X Data Archive                            │
│   (tweets.js, like.js, direct-messages.js, etc.)                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Parser (parser.rs)                           │
│   Handles window.YTD.* JavaScript format with rayon parallelism │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ SQLite           │ │ Tantivy          │ │ Vector Index     │
│ (storage.rs)     │ │ (search.rs)      │ │ (vector.rs)      │
│ - Metadata       │ │ - Full-text      │ │ - Embeddings     │
│ - Statistics     │ │ - BM25 ranking   │ │ - SIMD search    │
│ - FTS5 fallback  │ │ - Phrase queries │ │ - F16 storage    │
│ - Tweet lookup   │ │ - Boolean ops    │ │ - Cosine sim     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                   │                   │
        │                   ▼                   │
        │         ┌──────────────────┐          │
        │         │ Hybrid Fusion    │◀─────────┘
        │         │ (hybrid.rs)      │
        │         │ - RRF algorithm  │
        │         │ - Score fusion   │
        │         └────────┬─────────┘
        │                  │
        └────────┬─────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLI (cli.rs)                               │
│   clap-based command parsing with rich colored output           │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

**Stage 1: Archive Parsing**
- Reads JavaScript files from the archive's `data/` directory
- Strips `window.YTD.<type>.part0 = ` prefix to extract JSON
- Uses `rayon` for parallel parsing of large files

**Stage 2: Storage**
- Normalizes data into structured models (`Tweet`, `Like`, `DirectMessage`, etc.)
- Stores in SQLite with FTS5 virtual tables for fallback search
- Maintains statistics and metadata

**Stage 3: Keyword Indexing**
- Feeds content to Tantivy search engine
- Creates inverted index with BM25 scoring
- Supports prefix queries via edge n-grams

**Stage 4: Embedding Generation**
- Canonicalizes text (strips markdown, normalizes whitespace, filters noise)
- Generates 384-dimensional embeddings via:
  - **Default**: FNV-1a hash embedder (fast, zero external dependencies)
  - **Optional**: MiniLM via FastEmbed when indexed with `--semantic` (true semantic, slower)
- Stores embeddings with F16 quantization (50% size reduction)
- Content hashing (SHA256) enables incremental re-indexing

**Stage 5: Search**
- **Lexical mode**: Tantivy BM25 keyword matching
- **Semantic mode**: Vector similarity via SIMD dot product (hash or ML embeddings)
- **Hybrid mode**: RRF fusion of both result sets for optimal relevance
- Joins with SQLite for full metadata retrieval

### Search Algorithms

`xf` implements three distinct search strategies, each optimized for different use cases:

#### Lexical Search (BM25)

The classic information retrieval approach, powered by [Tantivy](https://github.com/quickwit-oss/tantivy):

- **Algorithm**: BM25 (Best Match 25) with saturation term frequency
- **Strengths**: Exact keyword matching, phrase queries, boolean operators
- **Use case**: When you know the exact words you're looking for

```bash
xf search "async await" --mode lexical
```

#### Semantic Search (Hash or ML Embeddings)

`xf` supports two semantic embedding modes that share the same vector index format:

**A) Default: Hash-Based Vector Similarity**  
Finds content with overlapping vocabulary rather than exact keyword matches:

- **Embedder**: FNV-1a hash-based embeddings (zero external dependencies)
- **Dimensions**: 384-dimensional vectors
- **Similarity**: Cosine similarity via SIMD-accelerated dot product
- **Storage**: F16 quantization reduces memory by 50%

```bash
# Hash-based similarity (default index)
xf search "feeling overwhelmed at work" --mode semantic
```

**How the Hash Embedder Works:**

Unlike neural network embedders (Word2Vec, BERT), xf uses a deterministic hash-based approach:

1. **Tokenize**: Split text on word boundaries
2. **Hash**: FNV-1a 64-bit hash for each token
3. **Project**: Hash determines vector index (`hash % 384`) and sign (MSB)
4. **Normalize**: L2 normalization for cosine similarity

This approach is:
- **Fast**: ~0ms per embedding (no GPU needed)
- **Deterministic**: Same input always produces same output
- **Zero dependencies**: No model files to download

**B) Optional: True Semantic (MiniLM via FastEmbed)**  
When you index with `--semantic`, xf builds MiniLM embeddings for synonym-level matching:

```bash
# Build ML embeddings (downloads ~80MB on first use)
xf index ~/x-archive --semantic

# True semantic similarity
xf search "feeling overwhelmed at work" --mode semantic
```

This mode is:
- **Semantic**: "happy" and "joyful" can match
- **Slower to index**: ~100 items/sec on CPU
- **Larger downloads**: ~80MB model weights on first use

#### Hybrid Search (RRF Fusion)

Combines the best of both approaches using Reciprocal Rank Fusion:

```
                     User Query
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    ┌──────────────┐           ┌──────────────┐
    │   Tantivy    │           │   Vector     │
    │   (BM25)     │           │  (Cosine)    │
    └──────┬───────┘           └──────┬───────┘
           │ Rank 0,1,2...            │ Rank 0,1,2...
           │                          │
           └────────────┬─────────────┘
                        ▼
                ┌───────────────┐
                │  RRF Fusion   │
                │  K=60         │
                └───────┬───────┘
                        ▼
                  Final Results
```

**RRF Algorithm:**

```
Score(doc) = Σ 1/(K + rank + 1)
```

Where:
- **K = 60**: Empirically optimal constant that balances score distribution
- **rank**: 0-indexed position in each result list
- Documents appearing in both lists get scores from both, naturally boosting multi-signal matches

**Why RRF?**

1. **Score normalization**: BM25 scores (0-20+) and cosine similarity (0-1) are incompatible. RRF uses ranks, not scores.
2. **Robust fusion**: Outperforms simple score averaging or max-pooling
3. **No tuning needed**: K=60 works well across diverse datasets
4. **Deterministic**: Tie-breaking by doc ID ensures consistent ordering

```bash
# Default mode—best of both worlds
xf search "productivity tips"
```

### Text Canonicalization

Before embedding, text passes through a normalization pipeline:

1. **Unicode NFC**: Normalize composed characters
2. **Strip Markdown**: Remove `**bold**`, `*italic*`, `[links](url)`, headers
3. **Collapse Code Blocks**: Keep first 20 + last 10 lines of code
4. **Normalize Whitespace**: Collapse runs of spaces/newlines
5. **Filter Low-Signal**: Skip trivial content ("OK", "Thanks", "Done")
6. **Truncate**: Cap at 2000 characters for consistent embedding dimensions

This ensures semantically equivalent text produces identical embeddings.

## Real-World Recipes

Here are practical examples for common tasks:

### Finding That Tweet You Vaguely Remember

```bash
# You remember talking about "that one coffee shop in Brooklyn"
xf search "coffee brooklyn" --mode hybrid

# You remember the vibe but not the words
xf search "cozy morning routine" --mode semantic

# Combine with date if you remember roughly when
xf search "vacation" --since "2023-06" --until "2023-09"
```

### Analyzing Your Posting Patterns

```bash
# Most engaged tweets (by likes + retweets)
xf search "" --types tweet --sort engagement --limit 20

# Your tweets from a specific era
xf search "" --since "2020-03" --until "2020-06" --types tweet

# Detailed stats about your archive
xf stats --detailed
```

### Exporting Data for Analysis

```bash
# Export all tweets as JSON for external processing
xf search "" --types tweet --limit 100000 --format json > all_tweets.json

# Export to CSV for spreadsheets
xf search "project" --format csv > project_tweets.csv

# Get tweets as JSONL (one per line) for streaming processing
xf search "" --types tweet --format json | jq -c '.[]' > tweets.jsonl
```

### Searching DM Conversations

```bash
# Find DMs about a topic with full conversation context
xf search "dinner plans" --types dm --context

# Export a specific conversation thread
xf search "project update" --types dm --context --format json > project_thread.json
```

### Scripting and Automation

```bash
# Count tweets containing "rust" by year
xf search "rust" --format json --limit 10000 | \
  jq -r '.[].created_at[:4]' | sort | uniq -c

# Find all unique hashtags you've used
xf search "" --types tweet --format json --limit 100000 | \
  jq -r '.[].text' | grep -oE '#\w+' | sort | uniq -c | sort -rn | head -20

# Daily tweet count (requires jq)
xf search "" --types tweet --format json --limit 100000 | \
  jq -r '.[].created_at[:10]' | sort | uniq -c

# Backup your indexed data
tar -czvf xf-backup.tar.gz ~/.local/share/xf/
```

### Shell Integration

```bash
# Add to your shell aliases (~/.bashrc or ~/.zshrc)
alias xs='xf search'
alias xst='xf search --types tweet'
alias xsd='xf search --types dm --context'
alias xsl='xf search --types like'

# Function to search and copy first result
xfirst() {
  xf search "$@" --limit 1 --format json | jq -r '.[0].text'
}

# Quick stats check
alias xinfo='xf stats --format json | jq'
```

## Technical Deep Dives

### Why BM25 Over TF-IDF?

Traditional TF-IDF (Term Frequency–Inverse Document Frequency) has a flaw: term frequency grows linearly forever. A document mentioning "rust" 100 times scores 10x higher than one mentioning it 10 times—but is it really 10x more relevant?

BM25 adds **saturation**: after a point, additional occurrences contribute diminishing returns.

```
BM25 score = IDF × (tf × (k₁ + 1)) / (tf + k₁ × (1 - b + b × (docLen/avgDocLen)))
```

Where:
- **k₁ = 1.2**: Controls term frequency saturation
- **b = 0.75**: Controls document length normalization

This means:
- Short tweets aren't penalized for being short
- Repetitive content doesn't dominate results
- Relevance better matches human intuition

### Why FNV-1a for Hashing?

The embedder uses FNV-1a (Fowler–Noll–Vo) rather than cryptographic hashes:

| Property | FNV-1a | SHA256 | MurmurHash3 |
|----------|--------|--------|-------------|
| Speed | ⚡ Fastest | 🐢 Slow | ⚡ Fast |
| Distribution | Good | Excellent | Excellent |
| Deterministic | ✅ Yes | ✅ Yes | ⚠️ Seed-dependent |
| Simplicity | ✅ ~10 lines | ❌ Complex | ⚠️ Medium |

FNV-1a's key advantage: **simplicity with good distribution**. For embedding purposes, we need consistent hashing that spreads tokens across dimensions—not cryptographic security.

```rust
// FNV-1a in ~5 lines
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}
```

### Why 384 Dimensions?

The embedding dimension (384) is chosen to match common ML embedders:

- **MiniLM-L6**: 384 dimensions
- **all-MiniLM-L6-v2**: 384 dimensions
- **paraphrase-MiniLM-L6-v2**: 384 dimensions

This means if you later want to swap in a neural embedder, the vector index structure remains compatible. It's also a sweet spot:
- **Large enough**: Good representation capacity
- **Small enough**: Fast dot products, reasonable storage
- **Power of 2 adjacent**: 384 = 256 + 128, good for SIMD alignment

### F16 Quantization Trade-offs

Embeddings are stored as 16-bit floats (F16) rather than 32-bit (F32):

| Format | Size per Vector | Precision | Speed Impact |
|--------|-----------------|-----------|--------------|
| F32 | 1,536 bytes | Full | Baseline |
| F16 | 768 bytes | ~3 decimal places | ~Same |
| INT8 | 384 bytes | ~2 decimal places | Faster |

Why F16?
- **50% storage reduction**: 768 bytes vs 1,536 bytes per embedding
- **Negligible precision loss**: Cosine similarity differences < 0.001
- **Fast conversion**: Hardware F16↔F32 conversion on modern CPUs
- **Good enough**: Personal archives don't need INT8's extra compression

### SIMD Dot Product Optimization

Vector similarity uses SIMD (Single Instruction, Multiple Data) for parallel computation:

```rust
use wide::f32x8;

pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let va = f32x8::from(&a[i*8..][..8]);
        let vb = f32x8::from(&b[i*8..][..8]);
        sum += va * vb;
    }

    // Horizontal sum + handle remainder
    sum.reduce_add() + a[chunks*8..].iter()
        .zip(&b[chunks*8..])
        .map(|(x, y)| x * y)
        .sum::<f32>()
}
```

This processes 8 floats per instruction, achieving:
- **~8x throughput** on supported CPUs
- **Portable**: Uses `wide` crate for cross-platform SIMD
- **Fallback**: Scalar loop for non-aligned remainders

### SQLite Performance Tuning

The database uses aggressive performance settings:

```sql
PRAGMA journal_mode = WAL;      -- Write-Ahead Logging: concurrent reads
PRAGMA synchronous = NORMAL;    -- Balanced durability vs speed
PRAGMA foreign_keys = ON;       -- Referential integrity
PRAGMA cache_size = -64000;     -- 64MB page cache
PRAGMA temp_store = MEMORY;     -- Temp tables in RAM
```

**Why WAL mode?**
- Readers don't block writers
- Writers don't block readers
- Better performance for read-heavy workloads (search is read-heavy)

**Why -64000 cache?**
- Negative values = KB (so -64000 = 64MB)
- Keeps hot pages in memory
- Reduces disk I/O for repeated queries

## Performance

`xf` is designed for speed:

- **Indexing (hash)**: ~10,000 documents/second
- **Indexing (semantic ML)**: ~100 documents/second (CPU, model-dependent)
- **Search**: Sub-millisecond for most lexical queries; semantic adds embedding cost
- **Memory**: Efficient memory-mapped index files
- **Parallelism**: Multi-threaded parsing via rayon

### Benchmarks

On a typical archive (12,000 tweets, 40,000 likes):

| Operation | Time |
|-----------|------|
| Index + embed (hash) | ~8 seconds |
| Index + embed (semantic ML) | ~100 items/sec (CPU, model-dependent) |
| Lexical search | <1ms |
| Semantic search (hash) | <5ms |
| Semantic search (ML) | higher latency (embedding cost; model-dependent) |
| Hybrid search | <10ms (hash), higher with ML |

| Storage | Size |
|---------|------|
| SQLite database | ~10MB |
| Tantivy index | ~15MB |
| Embeddings (F16) | ~3MB |

### Performance Optimizations

**1. Lazy Static Initialization**
- Regex patterns and search readers are compiled once on first use
- Subsequent operations reuse compiled resources

**2. Parallel Parsing**
- Uses `rayon` to parse archive files in parallel
- Takes full advantage of multi-core CPUs
- Automatically scales to available cores

**3. Memory-Mapped Index**
- Tantivy uses memory-mapped files for the search index
- OS manages caching automatically
- Subsequent searches benefit from warm cache

**4. SIMD Vector Operations**
- Dot products use `wide` crate for 8-float SIMD operations
- 8x theoretical throughput improvement
- Portable across x86_64 and ARM64

**5. F16 Quantization**
- Embeddings stored as 16-bit floats
- 50% memory reduction with negligible precision loss
- Fast hardware conversion on modern CPUs

**6. Content Hashing for Dedup**
- SHA256 hash of canonicalized text
- Skip re-embedding unchanged content on re-index
- Incremental updates are fast

**7. Release Profile**

```toml
[profile.release]
opt-level = "z"     # Optimize for size (lean binary)
lto = true          # Link-time optimization across crates
codegen-units = 1   # Single codegen unit for better optimization
panic = "abort"     # Smaller binary, no unwinding overhead
strip = true        # Remove debug symbols
```

### Scaling Characteristics

| Archive Size | Index Time | Search Time | Memory (Runtime) |
|--------------|------------|-------------|------------------|
| 1K docs | ~1s | <1ms | ~10MB |
| 10K docs | ~3s | <1ms | ~20MB |
| 50K docs | ~10s | <5ms | ~50MB |
| 100K docs | ~20s | <10ms | ~100MB |

*Tested on M2 MacBook Pro. Times vary by CPU and disk speed.*

## Building from Source

Requirements:
- Rust nightly (automatically selected via `rust-toolchain.toml`)
- Git

```bash
git clone https://github.com/Dicklesworthstone/xf.git
cd xf
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

### Performance Corpus & Golden Outputs

`xf` includes a deterministic performance corpus under `tests/fixtures/perf_corpus/`.
To regenerate it locally:

```bash
python3 scripts/generate_perf_corpus.py --seed 42 --output-dir tests/fixtures/perf_corpus
```

Golden outputs for isomorphism checks live in `tests/fixtures/golden_outputs/` and can
be refreshed with:

```bash
./scripts/verify_isomorphism.sh --update
```

## Troubleshooting

### "No archive indexed yet"

You need to run `xf index` before searching:

```bash
xf index ~/path/to/your/x-archive
```

The archive should contain a `data/` directory with files like `tweets.js`.

### "Search index missing"

The Tantivy index got corrupted or deleted. Rebuild it:

```bash
xf index ~/path/to/your/x-archive --force
```

### Slow first search after restart

This is normal. The first search loads the index into memory (~100-500ms). Subsequent searches are <10ms. The OS caches the memory-mapped files.

### No results for a query I know should match

Try different search modes:

```bash
# If lexical finds nothing, try semantic
xf search "that thing about coffee" --mode semantic

# Check if the content type is indexed
xf stats  # Shows counts by type

# Try broader terms
xf search "coffee" --mode lexical
```

### "Failed to parse archive"

The archive might be incomplete or from an unexpected format. Check:

```bash
# Verify the archive structure
ls ~/x-archive/data/

# Should see: tweets.js, like.js, direct-messages.js, etc.

# Try the doctor command
xf doctor --archive ~/x-archive
```

### High memory usage

For very large archives (100K+ documents), memory usage during indexing can spike. After indexing completes, runtime memory is minimal since indices are memory-mapped.

If indexing runs out of memory:
1. Close other applications
2. Consider indexing specific types: `xf index ~/archive --only tweet,like`
3. The embedding generation is the most memory-intensive phase

### Embeddings missing (semantic search returns nothing)

Re-index to generate embeddings:

```bash
xf index ~/x-archive --force
```

Check embedding count:
```bash
xf stats --format json | jq '.embeddings'
```

## Limitations

### What xf Doesn't Do

- **Real-time sync**: xf works on static archive exports, not live data
- **Multi-archive**: Only one archive at a time (re-index to switch)
- **Media search**: Can't search image/video content (only text metadata)
- **True synonyms (hash mode)**: Hash embedder finds related words, not true synonyms ("car" won't find "automobile" unless they co-occur in your tweets). Use `xf index --semantic` to enable ML embeddings.
- **Incremental updates**: Re-indexing processes the entire archive (fast enough that it rarely matters)

### Known Limitations of the Hash Embedder

The hash-based embedder is fast and dependency-free, but has limitations compared to neural embedders (MiniLM is available via `xf index --semantic`):

| Capability | Hash Embedder | Neural (BERT/MiniLM) |
|------------|---------------|----------------------|
| Word co-occurrence | ✅ Yes | ✅ Yes |
| Synonyms | ❌ No | ✅ Yes |
| Typo tolerance | ❌ No | ⚠️ Sometimes |
| Context understanding | ❌ No | ✅ Yes |
| Sentence meaning | ⚠️ Bag-of-words | ✅ Full context |
| Speed | ✅ ~0ms | 🐢 ~10-100ms |
| Dependencies | ✅ None | ❌ Model files |

**When this matters**: If you search "automobile" hoping to find tweets about "cars", the hash embedder won't help. Use lexical search with explicit synonyms: `xf search "car OR automobile OR vehicle"`.

**When it doesn't matter**: For personal archives, you typically remember some of the words you used. Hash-based similarity helps when your query shares vocabulary with the target text (e.g., "stressed deadlines" matches "deadline stress").

### Archive Format Dependencies

xf expects the standard X data export format:
- `data/` directory structure
- `window.YTD.*` JavaScript prefix
- JSON arrays of tweet/DM/like objects

If X changes their export format significantly, xf may need updates to parse it correctly.

## FAQ

### Why "xf"?

`xf` stands for "x_find" - a fast way to find things in your X (formerly Twitter) data.

### Is my data safe?

Yes! All data stays on your local machine. `xf` never sends data anywhere. The search index and database are stored locally.

### Can I search old tweets?

Yes, if they're in your archive. X includes all your tweets in the data export.

### What about deleted tweets?

X includes recently deleted tweets (within the last 30 days) in a separate file. `xf` can index these too.

### How do I update?

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh?$(date +%s)" | bash
```

Or use the built-in command:

```bash
xf update
```

### The search is slow. What's wrong?

First search after restart may be slower as the index loads. Subsequent searches should be sub-millisecond. If consistently slow, try rebuilding the index with `xf index --force`.

### Can I search multiple archives?

Currently, `xf` supports one archive at a time. To switch archives, re-run `xf index` with the new path (use `--force` to clear the old data).

### What query syntax is supported?

Tantivy's query parser supports:
- Terms: `word`
- Phrases: `"multiple words"`
- Boolean: `term1 AND term2`, `term1 OR term2`
- Exclusion: `term1 NOT term2`
- Wildcards: `rust*`
- Field-specific: `type:tweet text:rust`

### When should I use semantic vs lexical search?

**Use lexical (`--mode lexical`) when:**
- You know the exact words or phrases
- You need boolean operators (`AND`, `OR`, `NOT`)
- You're searching for specific names, hashtags, or technical terms

**Use semantic (`--mode semantic`) when:**
- You want vector similarity instead of exact keyword matching
- **Default (hash)**: broader recall based on word overlap
- **With `xf index --semantic`**: synonym-level matching (true semantic)

**Use hybrid (default) when:**
- You're not sure which approach is best
- You want the most comprehensive results
- Hybrid combines both and uses RRF to rank results optimally

### How does semantic search work?

`xf` supports two embedding modes:

**Default (hash-based):** no model downloads. Each word is hashed (FNV-1a) to deterministically select which dimensions to activate in a 384-dimensional vector. This approach:

- Requires **no model download** (zero bytes of ML weights)
- Runs in **~0ms** (no GPU needed)
- Produces **deterministic** results (same input = same output)
- Works well for **word overlap** and **topic similarity**

Tradeoff: it won't understand pure synonyms (e.g., "car" vs "automobile").

**Optional (ML-based):** run `xf index --semantic` to build MiniLM embeddings. This enables true semantic matching but is slower to index and requires a one-time model download (~80MB).

### Why is hybrid search the default?

Hybrid search gives you the best of both worlds:

1. **Lexical catches exact matches** — important for names, hashtags, URLs
2. **Semantic catches related content** — via vector similarity (hash by default, ML when indexed with `--semantic`)
3. **RRF fusion prioritizes documents that score well in both** — naturally surfacing the most relevant results

If a document ranks #1 in both lexical and semantic results, it's almost certainly what you're looking for.

### Does semantic search require re-indexing?

Yes. Embeddings are generated automatically during `xf index`, but the **embedder choice is fixed at index time**:

- Default: hash embeddings
- Optional: ML embeddings via `xf index --semantic`

If you switch between hash and ML, re-run indexing so the vector index matches the embedder. Use:

```bash
xf index ~/x-archive --force            # rebuild with hash embeddings
xf index ~/x-archive --semantic --force # rebuild with ML embeddings
```

## Contributing

*About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

MIT License (with OpenAI/Anthropic Rider) — see [LICENSE](LICENSE) for details.

---

Built with Rust, Tantivy, and SQLite. Features hybrid search combining keyword matching with semantic similarity via RRF fusion. Inspired by the need to actually search through years of tweets.
