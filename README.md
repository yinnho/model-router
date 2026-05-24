<div align="center">

# Model Router

### All-in-One Manager for Claude Code, Codex, Gemini CLI & More

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/yinnho/model-router)
[![Built with Tauri](https://img.shields.io/badge/built%20with-Tauri%202-orange.svg)](https://tauri.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

English | [中文](README_ZH.md)

</div>

## What is Model Router?

Model Router is a desktop app that lets you manage API providers for AI coding tools — Claude Code, Codex, Gemini CLI, OpenCode, OpenClaw, and Hermes — from a single interface.

Instead of manually editing JSON, TOML, or `.env` files every time you switch providers, Model Router gives you one-click switching, 50+ built-in provider presets, and a built-in local proxy that supports protocol translation between different APIs.

## Key Features

- **One app for all CLI tools** — Manage providers for Claude Code, Codex, Gemini CLI, OpenCode, OpenClaw, and Hermes
- **One-click switching** — Switch providers instantly from the UI or system tray
- **50+ provider presets** — AWS Bedrock, DeepSeek, Qwen, community relays, and more
- **Local proxy with protocol translation** — Transparently convert between OpenAI Responses API, Chat Completions, and Anthropic Messages formats
- **Auto failover & circuit breaker** — Automatically switch to a backup provider when the primary fails
- **Unified MCP & Skills management** — One panel for all apps, with bidirectional sync
- **Usage tracking** — Monitor spending, requests, and token usage across providers
- **Cloud sync** — Sync via Dropbox, OneDrive, iCloud, or WebDAV
- **Cross-platform** — Windows, macOS, and Linux, built with Tauri 2

## Protocol Translation

Model Router includes a built-in HTTP proxy (port 15721) that handles API format conversion:

| Upstream API | Translation | Use Case |
|---|---|---|
| OpenAI Responses → Chat Completions | Codex CLI → DeepSeek | DeepSeek only supports `/v1/chat/completions` |
| OpenAI Responses → Anthropic Messages | Codex CLI → Qwen | Qwen uses Anthropic-compatible endpoint |
| OpenAI Responses (passthrough) | Codex CLI → OpenAI | Direct proxy, no conversion |
| Anthropic Messages (passthrough) | Claude → Anthropic | Direct proxy with hot-switch |

This means you can use **Codex CLI with non-OpenAI providers** (like DeepSeek or Qwen) without any external bridge process.

## Quick Start

### Installation

Download the latest release from [GitHub Releases](https://github.com/yinnho/model-router/releases):

- **macOS**: `.dmg` installer
- **Windows**: `.msi` installer or portable `.zip`
- **Linux**: `.deb`, `.rpm`, or `.AppImage`

### Build from Source

```bash
# Prerequisites: Node.js 18+, pnpm 8+, Rust 1.85+

git clone https://github.com/yinnho/model-router.git
cd model-router
pnpm install
pnpm tauri build
```

### Basic Usage

1. **Add a provider** — Click "Add Provider" and choose a preset or enter custom config
2. **Switch providers** — Select a provider and click "Enable", or use the system tray
3. **For Codex with DeepSeek/Qwen** — Enable the local proxy, then point Codex at `http://127.0.0.1:15721`
4. **Restart your terminal** — Most CLI tools need a restart to pick up config changes (Claude Code supports hot-switch)

## Data Storage

- **Database**: `~/.model-router/model-router.db` (SQLite)
- **Settings**: `~/.model-router/settings.json`
- **Backups**: `~/.model-router/backups/`
- **Skills**: `~/.model-router/skills/`

If you're upgrading from cc-switch, Model Router will automatically migrate data from `~/.cc-switch/` on first launch.

## Architecture

```
┌──────────────────────────────────────────┐
│           Frontend (React + TS)           │
│   Components · Hooks · TanStack Query    │
└──────────────────┬───────────────────────┘
                   │ Tauri IPC
┌──────────────────▼───────────────────────┐
│           Backend (Tauri + Rust)          │
│  Commands → Services → DAO → Database    │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │        Proxy Server (Axum)          │ │
│  │  Responses↔ChatCompletions↔Anthropic │ │
│  │  Circuit Breaker · Failover         │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

## Development

```bash
pnpm dev          # Dev mode with hot reload
pnpm typecheck    # Type check frontend
pnpm test:unit    # Run frontend tests
cd src-tauri
cargo test        # Run backend tests
cargo clippy      # Lint Rust code
```

## License

MIT — Forked from [cc-switch](https://github.com/farion1231/cc-switch) (MIT, © farion1231)
