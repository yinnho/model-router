# Model Router: 从切换器到路由器

## 1. 现状

当前 Model Router 本质是一个 **切换器**：

- 用户在 UI 上选择一个 provider → 改写 CLI 配置文件 → CLI 请求发到固定 provider
- Proxy 层只做协议转换和故障转移，不参与路由决策
- `ProviderRouter.select_providers()` 固定返回当前 provider 或 failover 队列

## 2. 目标

将 proxy 改造为 **路由器**：

- CLI 请求进来 → proxy 根据当前路由模式决定目标 provider → 转发
- 用户不再手动切换 provider，而是选择 **路由模式**
- 三种模式：**auto** / **opus 4.7** / **gpt 5.5**

## 3. 路由模式

| 模式 | 行为 | 请求路径 |
|------|------|----------|
| **opus 4.7** | 所有请求强制走 opus 4.7 | 请求 → proxy → opus 4.7 provider |
| **gpt 5.5** | 所有请求强制走 gpt 5.5 | 请求 → proxy → gpt 5.5 provider |
| **auto** | 轻量 LLM 判断复杂度后路由 | 请求 → proxy → haiku 判断 → 路由到对应模型 |

### auto 模式路由逻辑

```
请求进来
  ↓
提取 prompt 内容（取最后一条 user message）
  ↓
调用轻量 LLM（haiku）快速分类：
  - light  → 路由到便宜/快速模型（如 deepseek-v3、qwen-turbo）
  - medium → 路由到平衡模型（如 sonnet、deepseek-r1）
  - heavy  → 路由到最强模型（如 opus 4.7、o3）
  ↓
转发到目标 provider
```

判断 prompt 本身也是一个 LLM 调用，需要：
- 选择一个轻量、便宜、快速的模型做判断
- prompt 固定，输出结构化（只返回 light/medium/heavy）
- 延迟控制在 1-2 秒内
- 判断请求本身也记录 usage

## 4. 架构变更

### 4.1 新增概念

```
RouterMode:
  - auto      # LLM 自动判断
  - fixed     # 固定路由到指定模型

RouterTarget:
  - id: string           # 如 "opus-4.7", "gpt-5.5"
  - display_name: string # 显示名
  - provider_id: string  # 对应的 provider
  - app_type: AppType    # 对应的 app

RouterConfig:
  - mode: RouterMode
  - current_target: Option<RouterTarget>  # fixed 模式下的目标
  - auto_classifier: ClassifierConfig      # auto 模式的分类器配置

ClassifierConfig:
  - classifier_provider_id: string  # 做判断用的 provider
  - classifier_model: string        # 做判断用的模型
  - categories: Vec<ClassifierCategory>

ClassifierCategory:
  - name: string          # light / medium / heavy
  - target_provider_id: string
  - target_model: string
```

### 4.2 Proxy 层改动

**现有流程：**
```
RequestContext::new()
  → ProviderRouter::select_providers()  // 返回固定 provider 链
  → RequestForwarder::forward_with_retry()
```

**改造后：**
```
RequestContext::new()
  → Router::resolve_target(mode, request)  // 新增路由决策层
    → if fixed mode: 返回指定 provider
    → if auto mode: 调用分类器 LLM → 返回对应 provider
  → ProviderRouter::select_providers()     // 仍负责 failover 链
  → RequestForwarder::forward_with_retry()
```

关键改动点：

1. **proxy/router.rs**（新增）— 路由决策核心
   - `resolve_target()` 根据模式返回目标 provider
   - `classify_request()` auto 模式下调用分类 LLM
   - 缓存分类结果（同一 session 内相同复杂度不重复判断）

2. **proxy/handler_context.rs** — RequestContext 增加路由步骤
   - 在 `select_providers()` 之前插入 `router.resolve_target()`
   - 路由决策结果影响 provider 选择

3. **proxy/providers/adapter.rs** — 无需改动
   - 路由层决定用哪个 provider，adapter 层只管协议转换

### 4.3 数据库改动

新增表 `router_config`：

```sql
CREATE TABLE router_config (
  app_type TEXT PRIMARY KEY,
  mode TEXT NOT NULL DEFAULT 'fixed',           -- auto / fixed
  current_target_id TEXT,                        -- fixed 模式的目标
  classifier_provider_id TEXT,                   -- 分类器 provider
  classifier_model TEXT DEFAULT 'claude-haiku-4-5-20251001',
  categories TEXT NOT NULL DEFAULT '{}',         -- JSON: { light, medium, heavy }
  created_at INTEGER,
  updated_at INTEGER
);
```

### 4.4 Tauri Commands 新增

```rust
// 路由模式管理
get_router_config(app_type) -> RouterConfig
set_router_mode(app_type, mode) -> ()
set_router_target(app_type, target_id) -> ()       // fixed 模式选目标
update_classifier_config(app_type, config) -> ()   // auto 模式配置分类器

// 路由目标管理
get_router_targets(app_type) -> Vec<RouterTarget>
add_router_target(target) -> ()
remove_router_target(id) -> ()
update_router_target(target) -> ()
```

### 4.5 前端改动

**UI 从"切换 provider"变成"选路由模式"：**

1. **Header 区域** — 当前是 ProxyToggle + FailoverToggle
   - 改为路由模式选择器：三个按钮 `[Auto] [Opus 4.7] [GPT 5.5]`
   - 当前选中模式高亮，一键切换

2. **路由设置面板**（新页面或 settings 子 tab）
   - 路由目标配置：添加/编辑/删除目标（关联 provider + 模型）
   - Auto 模式配置：
     - 分类器选择（用哪个 provider + 模型做判断）
     - 分类规则编辑（light → 目标, medium → 目标, heavy → 目标）

3. **Provider 列表** — 保留但角色变化
   - 不再是"切换"入口，而是"管理可用 provider"
   - Provider 的 meta 里标注它可以作为路由目标

## 5. 分类器 Prompt 设计

```markdown
You are a request complexity classifier. Given a user message, classify its complexity.

Rules:
- light: Simple questions, formatting, small edits, translations, quick lookups
- medium: Normal coding tasks, debugging, feature implementation, code review
- heavy: Architecture design, complex reasoning, large refactors, multi-step planning

Respond with ONLY one word: light, medium, or heavy.
```

这个 prompt 非常短，haiku 处理起来很快（< 1s），成本极低。

## 6. 兼容性考虑

- **现有 failover 机制保留**：路由决定目标 provider 后，failover 链仍然生效
- **Takeover 模式不变**：proxy 接管配置文件的方式不变
- **非路由模式**：如果用户不启用路由，行为和现在完全一致（向后兼容）
- **协议转换不受影响**：路由层在协议转换之前，只决定"发给谁"

## 7. 实施优先级

1. **P0**：路由模式 + fixed 模式（opus 4.7 / gpt 5.5）— 这部分改动小，主要是 UI + proxy 层加路由决策
2. **P1**：auto 模式（LLM 分类器）— 需要新增分类器调用逻辑
3. **P2**：分类结果缓存、session 级路由记忆 — 优化体验
