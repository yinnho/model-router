# Model Router：让 Codex CLI 用上 DeepSeek 和 Qwen

## 起因

Codex CLI 是 OpenAI 推出的 AI 编程工具，设计上只支持 OpenAI 的 Responses API。问题在于——不是所有模型提供商都支持这个协议。

DeepSeek 只认 `/v1/chat/completions`，Qwen 走的是 Anthropic Messages 格式。如果你想用 Codex CLI 接入这些更便宜或更适合自己的模型，官方没有给出方案。

## 之前的笨办法

社区里最早的解法是跑一个 Node.js 桥接服务（bridge），它充当中间人：

```
Codex CLI → Node.js Bridge → DeepSeek API
```

这个 bridge 做两件事：把 Codex 发出的 Responses API 请求翻译成 Chat Completions 格式，再把上游的流式响应翻译回 Responses API 的 SSE 格式。

同时还需要 cc-switch——一个 Tauri 桌面应用，用来管理不同 provider 的切换，修改配置文件。

于是你的电脑上同时跑着两套系统：cc-switch 负责切换，Node.js bridge 负责翻译。切一次 provider，bridge 可能没跟上；bridge 挂了，cc-switch 不知道。两套系统各管各的，配置不同步是家常便饭。

## 还有一个坑：tool_calls 排序

DeepSeek 对消息格式要求极其严格——assistant 消息里的 tool_calls 之后，必须紧跟对应的 tool 结果消息。但 Codex 生成的对话历史经常不守这个规矩：tool_calls 后面可能跟着用户消息，或者 tool 结果和 tool_calls 对不上号。

不加修复直接发给 DeepSeek，直接报错。

## Model Router 做了什么

Model Router 把 bridge 的协议翻译能力直接内置到了 cc-switch 的代理服务器里。现在只需要一个应用：

```
Codex CLI → Model Router (内置代理) → DeepSeek / Qwen / OpenAI
```

核心是三个翻译层：

### Responses → Chat Completions（DeepSeek 路径）

- 把 Codex 的 Responses API 请求体转为 Chat Completions 格式
- 流式响应从 Chat Completions SSE 逐块转回 Responses API SSE
- 自动修复 tool_calls 消息排序：补缺失的 tool 结果、删孤立的 tool 消息、确保不以 assistant 结尾
- 处理 `reasoning_content`：DeepSeek 要求带 tool_calls 的 assistant 消息必须包含 reasoning 字段
- 映射 `developer` 角色为 `user`（DeepSeek 不认 developer）

### Responses → Anthropic Messages（Qwen 路径）

- 把 Responses 请求转为 Anthropic Messages 格式
- `instructions` → `system`，`function_call` → `tool_use`，`function_call_output` → `tool_result`
- Auth header 从 `Authorization: Bearer` 转为 `x-api-key`
- 流式响应从 Anthropic SSE 转回 Responses API SSE

### Responses 透传（OpenAI 路径）

- 直接代理，支持热切换和故障转移

加上熔断器、自动故障转移、模型发现端点（`/v1/models`），一个应用覆盖了之前两套系统的全部能力。

## 其他功能

Model Router 保留了 cc-switch 原有的全部功能：

- **Provider 管理** — 50+ 预设，一键切换，系统托盘快速访问
- **统一 MCP & Skills 管理** — 一个面板管理所有 CLI 工具的 MCP 服务器和技能
- **用量追踪** — 跨供应商监控支出和 Token 用量
- **云同步** — WebDAV / Dropbox / iCloud 跨设备同步
- **跨平台** — Windows、macOS、Linux

## 安全

从 cc-switch fork 后做了一次完整的安全审计，修复了以下问题：

- SSRF 防护：代理目标地址校验，禁止指向内网 IP
- SSE 缓冲区限制：50MB 上限防止内存耗尽
- 输出项和文本长度限制：防止单个流式响应无限膨胀
- 上游错误体截断：日志中只保留前 4KB，防止敏感信息泄露
- API key 校验：无效字符直接报错而非静默丢弃

## 用法

1. 启动 Model Router，添加 DeepSeek 或 Qwen provider
2. 启用本地代理（默认端口 15721）
3. Codex 配置指向 `http://127.0.0.1:15721`
4. 正常使用 Codex，协议翻译完全透明

不需要额外的 Node.js 进程，不需要手动改配置文件，不需要担心两套系统不同步。

## 开源

MIT 协议，代码在 [github.com/yinnho/model-router](https://github.com/yinnho/model-router)。
