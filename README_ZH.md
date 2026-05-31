# Model Router

> AI 模型代理 + 协议转换 — 在 Anthropic Messages、OpenAI Chat Completions 和 OpenAI Responses API 之间无缝路由

Model Router 是一个轻量级桌面应用，位于你的 AI 客户端（如 Claude Code）和各种 AI 模型提供商之间，透明地进行协议转换和智能路由。

## 核心功能

- **协议转换** — 在 Anthropic Messages / OpenAI Chat / OpenAI Responses API 之间自动转换
- **智能路由** — 通过可配置的标签（opus/sonnet/haiku/auto）路由请求，支持从模型名自动解析
- **Claude Code 接管** — 一键接管 Claude Code，所有流量自动经过 Model Router
- **桌面应用** — 原生 macOS 应用（Tauri v2），系统托盘常驻，关闭到托盘
- **流式支持** — 全 SSE 流式协议转换
- **模型保护** — 自动用原始请求模型名替换 provider 模型名，防止客户端的反馈循环
- **Web 管理界面** — 内置管理界面 `http://127.0.0.1:8082`

## 工作原理

```
Claude Code / App
      │
      ▼  Anthropic Messages (SSE)
Model Router
      │
      ├─── OpenAI Chat Completions (如 DeepSeek, SiliconFlow)
      ├─── OpenAI Responses API (如 DashScope Qwen)
      ├─── Anthropic Messages 透传 (如 Baidu, Zhipu)
      └─── ...
```

请求流程：
1. 客户端发送带模型名的请求（如 `opus`、`sonnet`、`haiku`、`auto`）
2. Model Router 解析模型名为标签，找到匹配的路由
3. 请求转换为 provider 原生格式并转发
4. 响应转换回客户端期望的格式
5. 模型名字段被标准化，防止客户端反馈循环

## 快速开始

```bash
# 编译运行（开发模式）
cd src-tauri && cargo run

# 或者通过 Tauri
npm --prefix web run tauri dev
```

打开 `http://127.0.0.1:8082` 访问管理界面。

## 配置

编辑 `~/.model-router/config.yaml`:

```yaml
port: 8082
current_tag: auto

tags:
  - name: opus
    color: "#A855F7"
  - name: sonnet
    color: "#3B82F6"
  - name: haiku
    color: "#22C55E"
  - name: auto
    color: "#F59E0B"
    is_auto: true

providers:
  baidu:
    name: Baidu
    base_url: "https://qianfan.baidubce.com/anthropic/coding"
    api_key: "your-key"
    auth_type: bearer

routes:
  - endpoint: /v1/messages
    model: qianfan-code-latest
    provider: baidu
    tags: [opus]
    format: anthropic
```

## 架构

- **Rust 后端**: axum HTTP 服务器、Tauri v2 桌面壳、流式 SSE 转换
- **React 前端**: 暗色主题管理界面，实时日志
- **系统托盘**: 常驻后台运行，关闭到托盘

## License

MIT
