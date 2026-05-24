//! Codex Responses → Anthropic Messages conversion.
//!
//! Used when the Codex client sends Responses API requests but the upstream
//! provider only exposes an Anthropic-compatible Messages endpoint
//! (e.g., Qwen token-plan, DeepSeek Anthropic gateway).

use crate::proxy::error::ProxyError;
use serde_json::{json, Value};

/// Convert an OpenAI Responses request into an Anthropic Messages request.
pub fn responses_to_anthropic_messages(body: Value) -> Result<Value, ProxyError> {
    let mut result = json!({});

    // Model
    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }

    // Instructions → system
    if let Some(instructions) = body.get("instructions") {
        let text = instruction_text(instructions);
        if !text.is_empty() {
            result["system"] = json!(text);
        }
    }

    // Input → messages
    let mut messages = Vec::new();
    if let Some(input) = body.get("input") {
        append_responses_input_as_anthropic_messages(input, &mut messages)?;
    }

    // Fix message ordering for Anthropic strict validation
    fix_anthropic_message_ordering(&mut messages);

    result["messages"] = json!(messages);

    // Max tokens
    let max_tokens = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(16384);
    result["max_tokens"] = json!(max_tokens);

    // Stream
    if let Some(stream) = body.get("stream") {
        result["stream"] = stream.clone();
    } else {
        result["stream"] = json!(true);
    }

    // Thinking config
    let thinking_enabled = body
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|v| v.as_str())
        != Some("disabled");
    if thinking_enabled {
        result["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": 10000
        });
        // Anthropic requires max_tokens >= budget_tokens when thinking is enabled
        if max_tokens < 16000 {
            result["max_tokens"] = json!(16000);
        }
    }

    // Tools
    if let Some(tools) = body.get("tools") {
        let anthropic_tools = responses_tools_to_anthropic_tools(tools);
        if !anthropic_tools.is_empty() {
            result["tools"] = json!(anthropic_tools);
        }
    }

    Ok(result)
}

/// Convert an Anthropic Messages response into an OpenAI Responses response (non-streaming).
pub fn anthropic_messages_to_response(body: Value) -> Result<Value, ProxyError> {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
    let response_id = format!("resp_{}", chrono_like_id());

    let mut output_items = Vec::new();
    let mut output_text = String::new();

    if let Some(content) = body.get("content").and_then(|v| v.as_array()) {
        for block in content {
            let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match block_type {
                "thinking" => {
                    let thinking = block.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                    let item_id = format!("rs_{}", chrono_like_id());
                    output_items.push(json!({
                        "id": item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": thinking}]
                    }));
                }
                "text" => {
                    let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    output_text.push_str(text);
                    let item_id = format!("msg_{}", chrono_like_id());
                    output_items.push(json!({
                        "id": item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]
                    }));
                }
                "tool_use" => {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    let args = serde_json::to_string(&input).unwrap_or_default();
                    output_items.push(json!({
                        "id": id,
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": args,
                        "status": "completed"
                    }));
                }
                _ => {}
            }
        }
    }

    let usage = body.get("usage").map(|u| {
        json!({
            "input_tokens": u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "output_tokens": u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "total_tokens": u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
                + u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
        })
    });

    let stop_reason = body.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("end_turn");
    let status = if stop_reason == "end_turn" || stop_reason == "stop" { "completed" } else { "completed" };

    Ok(json!({
        "id": response_id,
        "object": "response",
        "created_at": chrono_like_id(),
        "status": status,
        "model": model,
        "output": output_items,
        "output_text": output_text,
        "usage": usage
    }))
}

fn instruction_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| part.as_str())
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        other => other.as_str().unwrap_or_default().to_string(),
    }
}

fn append_responses_input_as_anthropic_messages(
    input: &Value,
    messages: &mut Vec<Value>,
) -> Result<(), ProxyError> {
    let mut pending_tool_calls: Vec<Value> = Vec::new();
    let mut pending_reasoning = String::new();

    match input {
        Value::String(text) => {
            messages.push(json!({"role": "user", "content": text}));
        }
        Value::Array(items) => {
            for item in items {
                let item_type = item.get("type").and_then(|v| v.as_str());
                match item_type {
                    Some("reasoning") => {
                        let text = extract_reasoning_text(item);
                        if !text.is_empty() {
                            pending_reasoning.push_str(&text);
                        }
                    }
                    Some("function_call") => {
                        let call_id = item.get("call_id").or_else(|| item.get("id"))
                            .and_then(|v| v.as_str()).unwrap_or("");
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let args = item.get("arguments").cloned().unwrap_or(json!({}));
                        let input_val = match args {
                            Value::String(s) => serde_json::from_str(&s).unwrap_or(json!({})),
                            v => v,
                        };
                        pending_tool_calls.push(json!({
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": input_val
                        }));
                    }
                    Some("function_call_output") => {
                        flush_pending_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
                        let call_id = item.get("call_id").or_else(|| item.get("tool_call_id"))
                            .and_then(|v| v.as_str()).unwrap_or("");
                        let output = match item.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) => serde_json::to_string(v).unwrap_or_default(),
                            None => item.get("content").map(|v| match v {
                                Value::String(s) => s.clone(),
                                other => serde_json::to_string(other).unwrap_or_default(),
                            }).unwrap_or_default(),
                        };
                        // Anthropic: tool_result goes inside a user message
                        messages.push(json!({
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}]
                        }));
                    }
                    Some("message") | None => {
                        flush_pending_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
                        let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                        let text = content_to_text(item.get("content"));

                        match role {
                            "developer" => {
                                // Anthropic has no developer role, map to user
                                if !text.is_empty() {
                                    messages.push(json!({"role": "user", "content": text}));
                                }
                            }
                            "user" => {
                                pending_reasoning.clear();
                                if !text.is_empty() {
                                    messages.push(json!({"role": "user", "content": text}));
                                }
                            }
                            "assistant" => {
                                let mut content: Vec<Value> = Vec::new();
                                if !pending_reasoning.is_empty() {
                                    content.push(json!({"type": "thinking", "thinking": std::mem::take(&mut pending_reasoning)}));
                                }
                                if !text.is_empty() {
                                    content.push(json!({"type": "text", "text": text}));
                                }
                                if !content.is_empty() {
                                    messages.push(json!({"role": "assistant", "content": content}));
                                } else if !pending_reasoning.is_empty() {
                                    // Already consumed above
                                }
                            }
                            _ => {
                                if !text.is_empty() {
                                    messages.push(json!({"role": role, "content": text}));
                                }
                            }
                        }
                    }
                    _ => {
                        flush_pending_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
                    }
                }
            }
        }
        _ => {}
    }

    flush_pending_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
    Ok(())
}

fn flush_pending_anthropic_tool_calls(
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
    pending_reasoning: &mut String,
) {
    if pending_tool_calls.is_empty() {
        return;
    }
    // Anthropic: tool_use blocks go inside assistant content
    let mut content: Vec<Value> = Vec::new();
    if !pending_reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": std::mem::take(pending_reasoning)}));
    }
    content.extend(std::mem::take(pending_tool_calls));
    messages.push(json!({"role": "assistant", "content": content}));
}

fn responses_tools_to_anthropic_tools(tools: &Value) -> Vec<Value> {
    let Some(arr) = tools.as_array() else { return Vec::new() };
    arr.iter()
        .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"))
        .filter(|t| t.get("name").is_some())
        .map(|t| {
            let mut schema = json!({"type": "object", "properties": {}});
            if let Some(params) = t.get("parameters") {
                if let Some(obj) = params.as_object() {
                    let mut clean = serde_json::Map::new();
                    for (k, v) in obj {
                        if k != "additionalProperties" && k != "strict" {
                            clean.insert(k.clone(), v.clone());
                        }
                    }
                    schema = Value::Object(clean);
                }
            }
            json!({
                "name": t["name"],
                "description": t.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                "input_schema": schema
            })
        })
        .collect()
}

fn extract_reasoning_text(item: &Value) -> String {
    let from_content = item.get("content")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter()
            .filter(|c| c.get("type").and_then(|v| v.as_str()) == Some("reasoning_text")
                || c.get("type").and_then(|v| v.as_str()) == Some("reasoning"))
            .filter_map(|c| c.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(""))
        .unwrap_or_default();

    let from_summary = item.get("summary")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter()
            .filter(|c| c.get("type").and_then(|v| v.as_str()) == Some("summary_text"))
            .filter_map(|c| c.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(""))
        .unwrap_or_default();

    if !from_content.is_empty() { from_content } else { from_summary }
}

fn content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts.iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()).or_else(|| p.as_str()))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Fix Anthropic message ordering constraints:
/// - assistant with tool_use must be followed by user with tool_result
/// - messages must not end with assistant
/// - consecutive user messages should be merged
fn fix_anthropic_message_ordering(messages: &mut Vec<Value>) {
    let mut fixed: Vec<Value> = Vec::with_capacity(messages.len());

    for msg in messages.drain(..) {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");

        // If previous assistant has tool_use, current must be user with tool_result
        if let Some(prev) = fixed.last() {
            if prev.get("role").and_then(|v| v.as_str()) == Some("assistant") {
                let has_tool_use = prev.get("content")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().any(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use")))
                    .unwrap_or(false);
                if has_tool_use && role != "user" {
                    // Insert empty tool_results for all tool_use blocks
                    let tool_use_ids: Vec<String> = prev.get("content")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter()
                            .filter(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
                            .filter_map(|b| b.get("id").and_then(|v| v.as_str()).map(String::from))
                            .collect())
                        .unwrap_or_default();
                    if !tool_use_ids.is_empty() {
                        let results: Vec<Value> = tool_use_ids.iter()
                            .map(|id| json!({"type": "tool_result", "tool_use_id": id, "content": ""}))
                            .collect();
                        fixed.push(json!({"role": "user", "content": results}));
                    }
                }
            }
        }

        // Merge consecutive user messages
        if role == "user" {
            if let Some(prev) = fixed.last_mut() {
                if prev.get("role").and_then(|v| v.as_str()) == Some("user") {
                    // Merge content
                    let prev_content = prev.get_mut("content");
                    match prev_content {
                        Some(Value::Array(arr)) => {
                            if let Some(Value::Array(new_arr)) = msg.get("content") {
                                arr.extend(new_arr.clone());
                            } else if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                                arr.push(json!({"type": "text", "text": text}));
                            }
                            continue;
                        }
                        _ => {}
                    }
                }
            }
        }

        fixed.push(msg);
    }

    // Ensure not ending with assistant
    while fixed.last().map_or(false, |m| m.get("role").and_then(|v| v.as_str()) == Some("assistant")) {
        let last = fixed.last().unwrap();
        let has_tool_use = last.get("content")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().any(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use")))
            .unwrap_or(false);
        if has_tool_use {
            let tool_use_ids: Vec<String> = last.get("content")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                    .filter(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
                    .filter_map(|b| b.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect())
                .unwrap_or_default();
            let results: Vec<Value> = tool_use_ids.iter()
                .map(|id| json!({"type": "tool_result", "tool_use_id": id, "content": ""}))
                .collect();
            fixed.push(json!({"role": "user", "content": results}));
        } else {
            fixed.pop();
        }
    }

    // Ensure at least one user message
    if !fixed.iter().any(|m| m.get("role").and_then(|v| v.as_str()) == Some("user")) {
        fixed.push(json!({"role": "user", "content": "Continue."}));
    }

    *messages = fixed;
}

fn chrono_like_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn responses_to_anthropic_maps_messages_and_tools() {
        let body = json!({
            "model": "qwen3.7-max",
            "instructions": "You are helpful.",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi there!"},
                {"type": "message", "role": "user", "content": "What is 2+2?"}
            ],
            "tools": [{
                "type": "function",
                "name": "calculator",
                "description": "Calculate",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "strict": true}
            }],
            "max_output_tokens": 4096
        });

        let result = responses_to_anthropic_messages(body).unwrap();
        assert_eq!(result["model"], "qwen3.7-max");
        assert_eq!(result["system"], "You are helpful.");
        assert!(result["stream"] == true);

        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");

        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "calculator");
        // strict should be filtered out
        assert!(tools[0]["input_schema"].as_object().unwrap().get("strict").is_none());
    }

    #[test]
    fn responses_to_anthropic_handles_tool_calls() {
        let body = json!({
            "model": "qwen3.7-max",
            "input": [
                {"type": "message", "role": "user", "content": "Read file"},
                {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{\"path\":\"/tmp/x\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "file contents"}
            ]
        });

        let result = responses_to_anthropic_messages(body).unwrap();
        let messages = result["messages"].as_array().unwrap();
        // Should have: user, assistant(tool_use), user(tool_result)
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
    }

    #[test]
    fn responses_to_anthropic_handles_reasoning() {
        let body = json!({
            "model": "qwen3.7-max",
            "input": [
                {"type": "message", "role": "user", "content": "Think"},
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Let me think..."}]},
                {"type": "message", "role": "assistant", "content": "The answer is 42"}
            ]
        });

        let result = responses_to_anthropic_messages(body).unwrap();
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages[1]["role"], "assistant");
        let content = messages[1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[1]["type"], "text");
    }
}
