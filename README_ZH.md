<div align="center">

# Model Router

### Claude Code、Codex、Gemini CLI 等AI编程工具的统一管理器

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/yinnho/model-router)
[![Built with Tauri](https://img.shields.io/badge/built%20with-Tauri%202-orange.svg)](https://tauri.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md) | 中文

</div>

## Model Router 是什么？

Model Router 是一个桌面应用，让你在一个界面里管理所有 AI 编程工具的 API 供应商——Claude Code、Codex、Gemini CLI、OpenCode、OpenClaw 和 Hermes。

不用每次切换供应商时手动编辑 JSON、TOML 或 `.env` 文件，Model Router 提供一键切换、50+ 内置供应商预设，以及支持不同 API 协议自动翻译的本地代理。

## 核心功能

- **一个应用管理所有工具** — Claude Code、Codex、Gemini CLI、OpenCode、OpenClaw、Hermes
- **一键切换** — 从界面或系统托盘即时切换供应商
- **50+ 供应商预设** — AWS Bedrock、DeepSeek、Qwen、社区中转等
- **本地代理 + 协议翻译** — 自动在 OpenAI Responses API、Chat Completions 和 Anthropic Messages 格式之间转换
- **自动故障转移 & 熔断器** — 主供应商失败时自动切换到备用供应商
- **统一 MCP & Skills 管理** — 一个面板管理所有应用，双向同步
- **用量追踪** — 跨供应商监控支出、请求数和 Token 用量
- **云同步** — 通过 Dropbox、OneDrive、iCloud 或 WebDAV 同步
- **跨平台** — Windows、macOS、Linux，基于 Tauri 2

## 协议翻译

Model Router 内置 HTTP 代理服务器（端口 15721），自动处理 API 格式转换：

| 上游 API | 翻译方式 | 使用场景 |
|---|---|---|
| OpenAI Responses → Chat Completions | Codex CLI → DeepSeek | DeepSeek 只支持 `/v1/chat/completions` |
| OpenAI Responses → Anthropic Messages | Codex CLI → Qwen | Qwen 使用 Anthropic 兼容端点 |
| OpenAI Responses（透传） | Codex CLI → OpenAI | 直接代理，无需转换 |
| Anthropic Messages（透传） | Claude → Anthropic | 直接代理，支持热切换 |

这意味着你可以**在 Codex CLI 中使用 DeepSeek 或 Qwen 等非 OpenAI 供应商**，无需任何外部桥接进程。

## 快速开始

### 安装

从 [GitHub Releases](https://github.com/yinnho/model-router/releases) 下载最新版本：

- **macOS**：`.dmg` 安装包
- **Windows**：`.msi` 安装包或便携版 `.zip`
- **Linux**：`.deb`、`.rpm` 或 `.AppImage`

### 从源码构建

```bash
# 前置条件：Node.js 18+、pnpm 8+、Rust 1.85+

git clone https://github.com/yinnho/model-router.git
cd model-router
pnpm install
pnpm tauri build
```

### 基本使用

1. **添加供应商** — 点击"添加供应商"，选择预设或输入自定义配置
2. **切换供应商** — 选中供应商点击"启用"，或使用系统托盘
3. **Codex 使用 DeepSeek/Qwen** — 启用本地代理，将 Codex 指向 `http://127.0.0.1:15721`
4. **重启终端** — 大多数 CLI 工具需要重启才能生效（Claude Code 支持热切换）

## 数据存储

- **数据库**：`~/.model-router/model-router.db`（SQLite）
- **设置**：`~/.model-router/settings.json`
- **备份**：`~/.model-router/backups/`
- **Skills**：`~/.model-router/skills/`

如果从 cc-switch 升级，Model Router 会在首次启动时自动从 `~/.cc-switch/` 迁移数据。

## 架构

```
┌──────────────────────────────────────────┐
│           前端 (React + TypeScript)        │
│     Components · Hooks · TanStack Query  │
└──────────────────┬───────────────────────┘
                   │ Tauri IPC
┌──────────────────▼───────────────────────┐
│           后端 (Tauri + Rust)             │
│  Commands → Services → DAO → Database   │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │        代理服务器 (Axum)             │ │
│  │  Responses↔ChatCompletions↔Anthropic │ │
│  │  熔断器 · 故障转移                   │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

## 开发

```bash
pnpm dev          # 开发模式（热重载）
pnpm typecheck    # 前端类型检查
pnpm test:unit    # 运行前端测试
cd src-tauri
cargo test        # 运行后端测试
cargo clippy      # Rust 代码检查
```

## 开源协议

MIT — 衍生自 [cc-switch](https://github.com/farion1231/cc-switch)（MIT，© farion1231）
