# Model Router

> AI model proxy with protocol translation — seamlessly route between Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses API.

Model Router is a lightweight desktop application that sits between your AI client (e.g. Claude Code) and various AI model providers. It transparently translates protocols and routes requests based on configurable tags.

## Key Features

- **Protocol Translation** — Transparently convert between Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses API
- **Smart Routing** — Route requests by configurable tags (opus/sonnet/haiku/auto), or auto-resolve from model name
- **Claude Code Takeover** — One-click takeover of Claude Code's settings to route all traffic through Model Router
- **Desktop App** — Native macOS app (Tauri v2) with system tray and close-to-tray behavior
- **Streaming Support** — Full SSE streaming conversion between all protocols
- **Model Preservation** — Provider model names are replaced with the original request model to prevent client feedback loops
- **Web UI** — Built-in management interface at `http://127.0.0.1:8082`

## How It Works

```
Claude Code / App
      │
      ▼  Anthropic Messages (SSE)
Model Router
      │
      ├─── OpenAI Chat Completions (e.g. DeepSeek, SiliconFlow)
      ├─── OpenAI Responses API (e.g. DashScope Qwen)
      ├─── Anthropic Messages passthrough (e.g. Baidu, Zhipu)
      └─── ...
```

Request flow:
1. Client sends a request with a model name (e.g. `opus`, `sonnet`, `haiku`, `auto`)
2. Model Router resolves the model to a tag, then finds the matching route
3. Request is converted to the provider's native format and forwarded
4. Response is converted back to the client's expected format
5. Model field is normalized to prevent client feedback loops

## Quick Start

```bash
# Build and run (development)
cd src-tauri && cargo run

# Or via Tauri
npm --prefix web run tauri dev
```

Open `http://127.0.0.1:8082` in your browser to access the management UI.

## Configuration

Edit `~/.model-router/config.yaml`:

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

## Architecture

- **Rust backend**: axum HTTP server, Tauri v2 desktop shell, streaming SSE conversion
- **React frontend**: Dark-themed management UI with real-time logs
- **System tray**: Always-on background operation, close-to-tray behavior

## License

MIT
