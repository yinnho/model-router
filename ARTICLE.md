# Claude Code 新版不能用国产模型了？这个开源工具完美解决

## 背景：更新即「阵痛」

Claude Code 最近更新到了 2.1.x 版本（当前 2.1.158），带来了 Opus 4.8 等重磅更新。但很多使用国内模型（DeepSeek、智谱 GLM、通义千问、Kimi 等）的用户发现——**更新后没法用了**。

社区里常见的「解决方案」只有一个：**降级到旧版本**。

但降级不是长久之计。新版本的 Opus 4.8、动态工作流（Dynamic Workflows）、Fast Mode 等特性都用不了。

为什么会这样？问题的根源在哪里？

## Opus 4.8 带来的三个「坑」

![Model Router 日志界面 — 实时请求日志，彩色 Tag 标签](./web/src/assets/screenshots/logs.png)

### 坑一：Thinking Blocks 协议变更

2.1.154 引入 Opus 4.8，随之而来的是 **thinking blocks（思考块）**的大量使用。这是 Anthropic Messages 协议的新特性，在 API 响应中新增了 `type: "thinking"` 的内容块。

2.1.156 的 changelog 明确写道：

> *"Fixed an issue when using Opus 4.8 where thinking blocks were modified, leading to API errors."*

问题在于：**国内支持 Anthropic 格式的 provider（如百度文心、智谱 GLM），不一定支持 thinking blocks**。当 Claude Code 发送包含思考参数的请求，或者收到包含 thinking 块的响应时，解析就会出错。

### 坑二：模型反馈循环

这是另一个隐蔽但破坏力巨大的问题。

Claude Code 2.1.x 引入了 `ANTHROPIC_DEFAULT_OPUS_MODEL` / `ANTHROPIC_DEFAULT_SONNET_MODEL` / `ANTHROPIC_DEFAULT_HAIKU_MODEL` 三个环境变量，允许用户为 opus/sonnet/haiku 别名指定不同的模型。

工作流程是这样的：

1. 你设 `ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.1`
2. Claude Code 发请求，model 字段是 `glm-5.1`
3. 智谱返回 `model: "glm-5.1"`
4. **Claude Code 记住了这个模型名**，下次直接发 `model: "glm-5.1"`

问题来了：你的路由配置只认 `opus` / `sonnet` / `haiku` / `auto` 这些标签，`glm-5.1` 这个值该走哪个路由？

这就导致**路由逻辑彻底失效**，每次请求都可能走错 provider，甚至报错。

### 坑三：协议格式不兼容

Claude Code 使用的是 **Anthropic Messages API 格式**。但国内模型提供商（DeepSeek、Moonshot、SiliconFlow 等）大多只支持 **OpenAI Chat Completions 格式**，有些支持 OpenAI Responses 格式。

```
Claude Code 发送：
POST /v1/messages
{
  "model": "haiku",
  "messages": [{"role": "user", "content": "Hi"}],
  "max_tokens": 4096
}

DeepSeek 需要：
POST /v1/chat/completions
{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "Hi"}]
}
```

没有协议转换层，Claude Code 根本无法和国内模型通信。

## Model Router：一劳永逸的解决方案

**Model Router** 是一个开源桌面应用（Tauri v2 + Rust），位于 Claude Code 和模型 provider 之间，同时解决以上三个问题。

![Model Router 界面总览 — 页头 + Takeover 开关](./web/src/assets/screenshots/takeover.png)

### 解决问题一：Thinking Blocks 兼容

Model Router 在流式响应中自动处理 thinking 块：

- 如果 provider 返回了 thinking 块，正确透传，保证 Claude Code 不报错
- 流式转换中对 thinking 块的 `block_index` 做严格的状态机管理，防止索引错乱
- 支持 `reasoning_content`（DeepSeek 等 OpenAI 格式的思考内容）→ `thinking` 块的自动转换

再也不会出现 "Content block not found" 之类的 API 错误。

### 解决问题二：模型名保护（切断反馈循环）

对于所有非 streaming 和 streaming 响应，Model Router **在返回给 Claude Code 之前，将 provider 返回的 model 字段替换为原始请求的模型名**：

```
provider 返回: model: "glm-5.1"    ← 原始值
Model Router 替换后: model: "sonnet"  ← 原始请求别名
```

Claude Code 永远只看到 `opus` / `sonnet` / `haiku` / `auto` 这些它认识的模型别名，**永远不会学习到 provider 的模型名**。反馈循环被彻底切断。

### 解决问题三：三向协议转换

![Provider 配置界面 — 管理多个模型提供商](./web/src/assets/screenshots/providers.png)

Model Router 支持三种主流 API 格式的任意互转：

| 客户端格式 | Provider 格式 | 典型场景 |
|-----------|-------------|---------|
| Anthropic Messages | OpenAI Chat | DeepSeek、Moonshot、SiliconFlow |
| Anthropic Messages | OpenAI Responses | 通义千问 DashScope |
| OpenAI Chat | OpenAI Responses | 跨格式兼容 |
| Anthropic Messages | Anthropic（透传） | 百度文心、智谱 GLM |

**流式 SSE 转换同样完整支持**，thinking 块、text 块、tool_use 块全部正确处理。

### 四、标签路由系统

![路由配置界面 — 每个路由绑定 Tag、Provider、模型名](./web/src/assets/screenshots/routes.png)

支持自定义路由规则：

```
opus → 百度文心 (qianfan-code-latest)
sonnet → 智谱 GLM (glm-5.1)
haiku → DeepSeek (deepseek-v4-pro)
auto → Moonshot Kimi (K2.6)
```

**任何未识别的模型名（如 `glm-5.1`、`deepseek-v4-pro`）自动走 `auto` 路由**，不会再丢失任何请求。

### 五、一键接管 / 恢复

![Tags 配置界面 — 自定义标签和颜色](./web/src/assets/screenshots/tags.png)

不需要手动配置环境变量。Model Router 管理界面点击 "Takeover"，自动写入 Claude Code 配置：

```json
{
  "env": {
    "ANTHROPIC_MODEL": "auto",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082/anthropic",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "opus",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku",
    "ANTHROPIC_API_KEY": "sk-ant-placeholder-model-router"
  }
}
```

点击 "Restore" 一键恢复原始配置，零风险。

## 架构设计

Model Router 是 **Tauri v2 桌面应用**，内嵌 axum HTTP 服务器：

```
┌──────────┐     HTTP      ┌─────────────────────────────────────┐
│ Claude   │◄────────────►│  Model Router (Tauri + axum)       │
│ Code     │                │                                     │
└──────────┘                │  ┌────────────┐  ┌──────────────┐  │
                            │  │ Protocol   │  │ Tag Router   │  │
                            │  │ Converter  │  │ Engine       │  │
                            │  └────────────┘  └──────────────┘  │
                            │         │               │          │
                            └─────────┼───────────────┼──────────┘
                                      │               │
                              ┌───────▼───┐   ┌───────▼──────┐
                              │ DeepSeek  │   │ Zhipu GLM   │ ...
                              │ (OpenAI)  │   │ (Anthropic) │
                              └───────────┘   └─────────────┘
```

- **系统托盘常驻**：关闭窗口不退出，后台持续运行
- **管理界面**：浏览器打开 `http://127.0.0.1:8082`，实时日志、配置管理
- **零依赖部署**：构建为 macOS .app，双击即用

## 和其他方案对比

| 方案 | Thinking Blocks | 模型反馈 | 协议转换 | 一键接管 | 开源 |
|-----|----------------|---------|---------|---------|------|
| **降级旧版本** | ❌ 无新功能 | ❌ | ❌ | ❌ | - |
| **手动配环境变量** | ❌ 会报错 | ❌ 反馈循环 | ❌ | ❌ | - |
| **普通 HTTP 代理** | ⚠️ 透传 | ❌ 反馈循环 | ❌ 必须同协议 | ❌ | - |
| **Model Router** | ✅ 自动处理 | ✅ 模型名替换 | ✅ 三向互转 | ✅ 一键切换 | ✅ MIT |

## 快速上手

```bash
# 克隆
git clone https://github.com/yinnho/model-router

# 开发模式运行
npm --prefix web run tauri dev

# 或构建 .app
npm --prefix web run tauri build
```

编辑 `~/.model-router/config.yaml`，配置你的 provider 和路由规则，然后在管理界面点 "Takeover" 接管 Claude Code——完成。

## 写在最后

Model Router 解决的是一个具体的痛点：**Claude Code 很强，但它只原生支持 Anthropic 生态。通过 Model Router，你可以把任何兼容的 API 接到 Claude Code 上，同时享受新版本的所有特性。**

不需要降级。不需要折腾环境变量。不需要担心协议不兼容。

---

项目地址：[https://github.com/yinnho/model-router](https://github.com/yinnho/model-router)

开源免费，欢迎 Star、Issue、PR。
