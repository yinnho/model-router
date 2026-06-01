use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Anthropic → OpenAI: Request conversion
// ---------------------------------------------------------------------------

/// Convert an Anthropic Messages API request body to OpenAI Chat Completions format.
pub fn anthropic_to_openai_request(body: &Value, target_model: &str) -> Value {
    // Make a defensive normalized copy of the incoming Anthropic body so the
    // converter can assume messages have valid roles and non-null content.
    let mut norm = body.clone();
    if let Some(msgs) = norm.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for msg in msgs.iter_mut() {
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                if role == "tool" || role == "system" {
                    msg["role"] = Value::String("assistant".to_string());
                }
            }

            match msg.get("content") {
                Some(c) if c.is_null() => {
                    msg["content"] = Value::String(String::new());
                }
                Some(c) if !(c.is_string() || c.is_array()) => {
                    let s = if c.is_object() || c.is_number() || c.is_boolean() {
                        serde_json::to_string(c).unwrap_or_default()
                    } else {
                        String::new()
                    };
                    msg["content"] = Value::Array(vec![json!({"type":"text","text":s})]);
                }
                None => {
                    msg["content"] = Value::String(String::new());
                }
                _ => {}
            }
        }
    }

    let mut out = serde_json::Map::new();
    out.insert("model".into(), Value::String(target_model.to_string()));

    // system → prepend as system message
    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        let system_text = extract_text_from_content(system);
        if !system_text.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_text
            }));
        }
    }

    // Convert messages
    if let Some(msgs) = norm.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match role {
                "assistant" => {
                    let mut openai_msg = json!({"role": "assistant"});
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();
                    let mut reasoning_parts = Vec::new();

                    if let Some(c) = content {
                        if let Some(arr) = c.as_array() {
                            for block in arr {
                                match block.get("type").and_then(|t| t.as_str()) {
                                    Some("text") => {
                                        if let Some(t) = block.get("text").and_then(|t| t.as_str())
                                        {
                                            text_parts.push(t.to_string());
                                        }
                                    }
                                    Some("thinking") => {
                                        if let Some(t) = block.get("thinking").and_then(|t| t.as_str()) {
                                            reasoning_parts.push(t.to_string());
                                        }
                                    }
                                    Some("tool_use") => {
                                        let id = block
                                            .get("id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let name = block
                                            .get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let input = block.get("input").cloned().unwrap_or(json!({}));
                                        tool_calls.push(json!({
                                            "id": id,
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": serde_json::to_string(&input).unwrap_or_default()
                                            }
                                        }));
                                    }
                                    _ => {}
                                }
                            }
                        } else if let Some(s) = c.as_str() {
                            text_parts.push(s.to_string());
                        }
                    }

                    let text = text_parts.join("");
                    // Always send a string for content (empty string if no text)
                    openai_msg["content"] = Value::String(text);

                    // Convert thinking blocks → reasoning_content
                    if !reasoning_parts.is_empty() {
                        openai_msg["reasoning_content"] = Value::String(reasoning_parts.join(""));
                    }

                    if !tool_calls.is_empty() {
                        openai_msg["tool_calls"] = Value::Array(tool_calls);
                    }
                    messages.push(openai_msg);
                }
                "user" => {
                    // Check for tool_result blocks
                    if let Some(c) = content {
                        if let Some(arr) = c.as_array() {
                            let mut has_tool_result = false;
                            let mut text_parts = Vec::new();
                            let mut tool_results = Vec::new();

                            for block in arr {
                                match block.get("type").and_then(|t| t.as_str()) {
                                    Some("tool_result") => {
                                        has_tool_result = true;
                                        let tool_use_id = block
                                            .get("tool_use_id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let result_content =
                                            extract_text_from_content(block.get("content").unwrap_or(&Value::Null));
                                        tool_results.push(json!({
                                            "role": "tool",
                                            "tool_call_id": tool_use_id,
                                            "content": result_content
                                        }));
                                    }
                                    Some("text") => {
                                        if let Some(t) = block.get("text").and_then(|t| t.as_str())
                                        {
                                            text_parts.push(t.to_string());
                                        }
                                    }
                                    _ => {
                                        // Pass through other content blocks as text
                                        text_parts.push(block.to_string());
                                    }
                                }
                            }

                            // If there are tool results, split into separate messages
                            if has_tool_result {
                                // Tool results MUST come first (right after assistant's tool_calls)
                                for tr in tool_results {
                                    messages.push(tr);
                                }
                                // Any remaining text goes after tool results
                                let text = text_parts.join("");
                                if !text.is_empty() {
                                    messages.push(json!({
                                        "role": "user",
                                        "content": text
                                    }));
                                }
                            } else {
                                let text = text_parts.join("");
                                messages.push(json!({
                                    "role": "user",
                                    "content": text
                                }));
                            }
                        } else {
                            // Simple string content
                            messages.push(json!({
                                "role": "user",
                                "content": c.clone()
                            }));
                        }
                    } else {
                        messages.push(json!({"role": "user", "content": ""}));
                    }
                }
                _ => {
                    // Other roles → user
                    if let Some(c) = content {
                        messages.push(json!({"role": "user", "content": c.clone()}));
                    }
                }
            }
        }
    }

    out.insert("messages".into(), Value::Array(messages));

    // max_tokens
    if let Some(mt) = body.get("max_tokens") {
        out.insert("max_tokens".into(), mt.clone());
    }

    // temperature
    if let Some(t) = body.get("temperature") {
        out.insert("temperature".into(), t.clone());
    }

    // stream
    if let Some(s) = body.get("stream") {
        out.insert("stream".into(), s.clone());
    }

    // tools: Anthropic → OpenAI
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                if tool.get("type").and_then(|t| t.as_str()) == Some("function") {
                    // Already OpenAI format
                    tool.clone()
                } else {
                    // Anthropic format: {name, description, input_schema}
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                            "description": tool.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                            "parameters": tool.get("input_schema").cloned().unwrap_or(json!({"type": "object", "properties": {}}))
                        }
                    })
                }
            })
            .collect();
        out.insert("tools".into(), Value::Array(openai_tools));
    }

    // tool_choice: Anthropic → OpenAI
    if let Some(tc) = body.get("tool_choice") {
        let openai_tc = match tc {
            Value::String(s) => match s.as_str() {
                "auto" => json!("auto"),
                "any" => json!("required"),
                "none" => json!("none"),
                _ => json!("auto"),
            },
            Value::Object(obj) => {
                if obj.get("type").and_then(|t| t.as_str()) == Some("tool") {
                    json!({
                        "type": "function",
                        "function": {"name": obj.get("name").and_then(|n| n.as_str()).unwrap_or("")}
                    })
                } else {
                    json!("auto")
                }
            }
            _ => json!("auto"),
        };
        out.insert("tool_choice".into(), openai_tc);
    }

    Value::Object(out)
}

// ---------------------------------------------------------------------------
// OpenAI Chat → Anthropic Messages: Request conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Chat Completions request to Anthropic Messages format.
pub fn openai_to_anthropic_request(body: &Value, target_model: &str) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("model".into(), Value::String(target_model.to_string()));

    // Extract system message from messages list (Anthropic uses top-level "system")
    let mut system_parts: Vec<String> = Vec::new();
    let mut anthropic_messages: Vec<Value> = Vec::new();

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        system_parts.push(content.to_string());
                    }
                    // Don't add system messages to the messages array
                }
                "user" => {
                    // Convert user content from string to content array if tool_result-like
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": content
                    }));
                }
                "assistant" => {
                    let mut blocks: Vec<Value> = Vec::new();
                    // Text content
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            blocks.push(serde_json::json!({"type": "text", "text": content}));
                        }
                    }
                    // Tool calls
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tool_calls {
                            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let func = tc.get("function").unwrap_or(&Value::Null);
                            let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let args_str = func.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
                            let input: Value = serde_json::from_str(args_str).unwrap_or(Value::Object(serde_json::Map::new()));
                            blocks.push(serde_json::json!({
                                "type": "tool_use",
                                "id": tc_id,
                                "name": name,
                                "input": input
                            }));
                        }
                    }
                    if !blocks.is_empty() {
                        anthropic_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": blocks
                        }));
                    } else {
                        // Plain text assistant message
                        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        anthropic_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": content
                        }));
                    }
                }
                "tool" => {
                    let tool_call_id = msg.get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content}]
                    }));
                }
                _ => {
                    // Unknown role → user
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": content
                    }));
                }
            }
        }
    }

    // Set system field if we collected system messages
    if !system_parts.is_empty() {
        out.insert("system".into(), Value::String(system_parts.join("\n")));
    }

    out.insert("messages".into(), Value::Array(anthropic_messages));

    // Convert tools format
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let mut anthropic_tools: Vec<Value> = Vec::new();
        for tool in tools {
            let func = tool.get("function").unwrap_or(tool);
            let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let desc = func.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let params = func.get("parameters").or_else(|| func.get("input_schema"));
            let mut at = serde_json::Map::new();
            at.insert("name".into(), Value::String(name.to_string()));
            at.insert("description".into(), Value::String(desc.to_string()));
            at.insert("input_schema".into(), params.cloned().unwrap_or(Value::Object(serde_json::Map::new())));
            anthropic_tools.push(Value::Object(at));
        }
        out.insert("tools".into(), Value::Array(anthropic_tools));
    }

    // Copy common fields with OpenAI→Anthropic name mapping
    for (key, dest) in &[
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop_sequences"),
        ("stream", "stream"),
        ("metadata", "metadata"),
    ] {
        if let Some(val) = body.get(*key) {
            out.insert(dest.to_string(), val.clone());
        }
    }

    Value::Object(out)
}

// ---------------------------------------------------------------------------
// OpenAI → Anthropic: Non-streaming response conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Chat Completions response to Anthropic Messages response format.
pub fn openai_to_anthropic_response(body: &Value, request_model: &str) -> Value {
    let mut content: Vec<Value> = Vec::new();

    if let Some(choices) = body.get("choices").and_then(|c| c.as_array()) {
        if let Some(choice) = choices.first() {
            let message = choice.get("message");

            // Reasoning content → thinking block (must come first in content)
            if let Some(reasoning) = message.and_then(|m| m.get("reasoning_content")).and_then(|r| r.as_str()) {
                if !reasoning.is_empty() {
                    content.push(json!({
                        "type": "thinking",
                        "thinking": reasoning
                    }));
                }
            }

            // Text content
            if let Some(text) = message.and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    content.push(json!({
                        "type": "text",
                        "text": text
                    }));
                }
            }

            // Tool calls → tool_use blocks
            if let Some(tool_calls) = message.and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()) {
                for (idx, tc) in tool_calls.iter().enumerate() {
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = tc.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    let args_str = tc.get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}");
                    let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

                    content.push(json!({
                        "type": "tool_use",
                        "id": if id.is_empty() { format!("toolu_{}", idx) } else { id.to_string() },
                        "name": name,
                        "input": input
                    }));
                }
            }
        }
    }

    // If no content was extracted, add an empty text block
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    // Map finish_reason
    let stop_reason = body
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .map(|r| match r {
            "stop" => "end_turn",
            "tool_calls" => "tool_use",
            "length" => "max_tokens",
            _ => "end_turn",
        })
        .unwrap_or("end_turn");

    // Map usage
    let usage = body.get("usage").map(|u| {
        json!({
            "input_tokens": u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "output_tokens": u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        })
    }).unwrap_or(json!({
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0
    }));

    json!({
        "id": body.get("id").and_then(|v| v.as_str()).unwrap_or("msg_unknown"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": request_model,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": usage
    })
}

// ---------------------------------------------------------------------------
// Anthropic Messages → OpenAI Chat: Non-streaming response conversion
// ---------------------------------------------------------------------------

/// Convert an Anthropic Messages response to OpenAI Chat Completions response format.
pub fn anthropic_to_openai_response(body: &Value, request_model: &str) -> Value {
    let mut content = String::new();
    let mut reasoning: Option<String> = None;
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(blocks) = body.get("content").and_then(|c| c.as_array()) {
        for block in blocks {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("thinking") => {
                    if let Some(text) = block.get("thinking").and_then(|t| t.as_str()) {
                        reasoning = Some(reasoning.unwrap_or_default() + text);
                    }
                }
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        content.push_str(text);
                    }
                }
                Some("tool_use") => {
                    let tc_id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = block.get("input").unwrap_or(&Value::Null);
                    let args_str = if input.is_object() {
                        serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string())
                    } else if let Some(s) = input.as_str() {
                        s.to_string()
                    } else {
                        "{}".to_string()
                    };
                    tool_calls.push(json!({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_str}
                    }));
                }
                _ => {}
            }
        }
    }

    // Build message
    let mut message = serde_json::Map::new();
    message.insert("role".into(), Value::String("assistant".into()));
    message.insert("content".into(), Value::String(content));
    if let Some(r) = reasoning {
        message.insert("reasoning_content".into(), Value::String(r));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }

    // Map stop_reason
    let finish_reason = match body.get("stop_reason").and_then(|v| v.as_str()) {
        Some("end_turn") | Some("stop_sequence") => "stop",
        Some("tool_use") => "tool_calls",
        Some("max_tokens") => "length",
        _ => "stop",
    };

    let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("msg_unknown");

    // Usage mapping
    let usage = body.get("usage").map(|u| {
        let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        json!({
            "prompt_tokens": input,
            "completion_tokens": output,
            "total_tokens": input + output
        })
    }).unwrap_or_else(|| json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}));

    json!({
        "id": id,
        "object": "chat.completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u64,
        "model": request_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": usage
    })
}

// ---------------------------------------------------------------------------
// OpenAI SSE → Anthropic SSE: Streaming conversion
// ---------------------------------------------------------------------------

/// State for the streaming SSE converter state machine.
struct StreamState {
    started: bool,
    finished: bool,
    block_index: u32,
    thinking_block_open: bool,
    text_block_open: bool,
    tool_blocks_open: Vec<bool>,
    msg_id: String,
    model: String,
    usage_input_tokens: u64,
    usage_output_tokens: u64,
}

/// Convert an OpenAI SSE byte stream into Anthropic SSE events.
pub fn convert_openai_stream_to_anthropic(
    upstream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: String,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let state = Arc::new(Mutex::new(StreamState {
        started: false,
        finished: false,
        block_index: 0,
        thinking_block_open: false,
        text_block_open: false,
        tool_blocks_open: Vec::new(),
        msg_id: format!("msg_{:x}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()),
        model: request_model,
        usage_input_tokens: 0,
        usage_output_tokens: 0,
    }));

    let buffer = Arc::new(Mutex::new(String::new()));

    let stream = upstream.flat_map(move |chunk_result| {
        let state = state.clone();
        let buffer = buffer.clone();

        async_stream::stream! {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let text = String::from_utf8_lossy(&chunk);
            let mut local_buf;
            {
                let mut guard = buffer.lock().await;
                guard.push_str(&text);
                local_buf = std::mem::take(&mut *guard);
            }

            // Process complete lines
            while let Some(newline_pos) = local_buf.find('\n') {
                let line = local_buf[..newline_pos].trim().to_string();
                local_buf = local_buf[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if line == "data: [DONE]" {
                    let mut st = state.lock().await;
                    if st.finished {
                        continue;
                    }
                    // Close any open blocks and finish
                    log::info!("[Stream] [DONE] received, closing blocks (thinking={}, text={}, blk_idx={})",
                        st.thinking_block_open, st.text_block_open, st.block_index);
                    if st.thinking_block_open {
                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                        st.thinking_block_open = false;
                        st.block_index += 1;
                    }
                    if st.text_block_open {
                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                        st.text_block_open = false;
                        st.block_index += 1;
                    }
                    let tool_count = st.tool_blocks_open.len();
                    for i in 0..tool_count {
                        if st.tool_blocks_open[i] {
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.block_index += 1;
                        }
                    }

                    // message_delta + message_stop
                    st.finished = true;
                    let msg_delta = json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": Value::Null},
                        "usage": {"output_tokens": st.usage_output_tokens}
                    });
                    yield Ok(Bytes::from(format!("event: message_delta\ndata: {}\n\n", msg_delta)));
                    yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string()));
                    continue;
                }

                let data = if let Some(stripped) = line.strip_prefix("data: ") {
                    stripped.to_string()
                } else if let Some(stripped) = line.strip_prefix("data:") {
                    stripped.trim().to_string()
                } else {
                    continue;
                };

                let parsed: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let mut st = state.lock().await;

                // Emit message_start on first chunk
                if !st.started {
                    st.started = true;
                    log::info!("[Stream] first chunk received, emitting message_start");
                    // Try to extract usage from first chunk
                    if let Some(u) = parsed.get("usage") {
                        st.usage_input_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                        if let Some(ct) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                            st.usage_output_tokens = ct;
                        }
                    }
                    let msg_start = json!({
                        "type": "message_start",
                        "message": {
                            "id": st.msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": st.model,
                            "stop_reason": Value::Null,
                            "stop_sequence": Value::Null,
                            "usage": {
                                "input_tokens": st.usage_input_tokens,
                                "output_tokens": 1,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0
                            }
                        }
                    });
                    yield Ok(Bytes::from(format!("event: message_start\ndata: {}\n\n", msg_start)));
                }

                // Extract delta from choices
                let choices = parsed.get("choices").and_then(|c| c.as_array());
                if let Some(choices) = choices {
                    if let Some(choice) = choices.first() {
                        let delta = choice.get("delta");
                        let finish_reason = choice.get("finish_reason").and_then(|f| f.as_str());

                        // Debug: log every delta for troubleshooting
                        let has_reasoning = delta.and_then(|d| d.get("reasoning_content")).is_some();
                        let has_content = delta.and_then(|d| d.get("content")).map(|c| !c.is_null()).unwrap_or(false);
                        let has_tool_calls = delta.and_then(|d| d.get("tool_calls")).is_some();
                        log::info!("[Stream] delta: reasoning={} content={} tools={} finish={:?} blk_idx={} thinking_open={} text_open={}",
                                has_reasoning, has_content, has_tool_calls, finish_reason,
                                st.block_index, st.thinking_block_open, st.text_block_open);

                        // Handle reasoning_content → thinking block
                        if let Some(reasoning) = delta.and_then(|d| d.get("reasoning_content")).and_then(|r| r.as_str()) {
                            if !reasoning.is_empty() {
                                if !st.thinking_block_open {
                                    let start = json!({
                                        "type": "content_block_start",
                                        "index": st.block_index,
                                        "content_block": {"type": "thinking", "thinking": ""}
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", start)));
                                    st.thinking_block_open = true;
                                }

                                let delta_event = json!({
                                    "type": "content_block_delta",
                                    "index": st.block_index,
                                    "delta": {"type": "thinking_delta", "thinking": reasoning}
                                });
                                yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", delta_event)));
                            }
                        } else if st.thinking_block_open {
                            // reasoning_content went away → close thinking block
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.thinking_block_open = false;
                            st.block_index += 1;
                        }

                        // Handle text content
                        if let Some(content) = delta.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                            if !content.is_empty() {
                                if !st.text_block_open {
                                    // Close any open tool blocks before opening text block
                                    for i in 0..st.tool_blocks_open.len() {
                                        if st.tool_blocks_open[i] {
                                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                            st.tool_blocks_open[i] = false;
                                            st.block_index += 1;
                                        }
                                    }
                                    // Open text block
                                    let start = json!({
                                        "type": "content_block_start",
                                        "index": st.block_index,
                                        "content_block": {"type": "text", "text": ""}
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", start)));
                                    st.text_block_open = true;
                                }

                                let delta_event = json!({
                                    "type": "content_block_delta",
                                    "index": st.block_index,
                                    "delta": {"type": "text_delta", "text": content}
                                });
                                yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", delta_event)));
                            }
                        }

                        // Handle tool calls
                        if let Some(tool_calls) = delta.and_then(|d| d.get("tool_calls")).and_then(|t| t.as_array()) {
                            for tc in tool_calls {
                                let tc_index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

                                // Ensure we have enough slots
                                while st.tool_blocks_open.len() <= tc_index {
                                    // Close thinking block if open
                                    if st.thinking_block_open {
                                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                        st.thinking_block_open = false;
                                        st.block_index += 1;
                                    }
                                    // Close text block if open
                                    if st.text_block_open {
                                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                        st.text_block_open = false;
                                        st.block_index += 1;
                                    }
                                    st.tool_blocks_open.push(false);
                                }

                                // Open tool block if not yet open
                                if !st.tool_blocks_open[tc_index] {
                                    // Close thinking block first if open
                                    if st.thinking_block_open {
                                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                        st.thinking_block_open = false;
                                        st.block_index += 1;
                                    }
                                    // Close text block first if open
                                    if st.text_block_open {
                                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                        st.text_block_open = false;
                                        st.block_index += 1;
                                    }
                                    let tool_name = tc.get("function")
                                        .and_then(|f| f.get("name"))
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("");
                                    let tool_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                                    let tool_start = json!({
                                        "type": "content_block_start",
                                        "index": st.block_index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tool_id,
                                            "name": tool_name,
                                            "input": {}
                                        }
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", tool_start)));
                                    st.tool_blocks_open[tc_index] = true;
                                }

                                // Tool arguments delta
                                if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()) {
                                    if !args.is_empty() {
                                        let args_delta = json!({
                                            "type": "content_block_delta",
                                            "index": st.block_index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": args
                                            }
                                        });
                                        yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", args_delta)));
                                    }
                                }
                            }
                        }

                        // Handle finish_reason
                        if let Some(reason) = finish_reason {
                            if !reason.is_empty() && reason != "null" && !st.finished {
                                log::info!("[Stream] finish_reason={}, closing blocks (thinking={}, text={}, blk_idx={})",
                                    reason, st.thinking_block_open, st.text_block_open, st.block_index);
                                let stop_reason = match reason {
                                    "stop" => "end_turn",
                                    "tool_calls" => "tool_use",
                                    "length" => "max_tokens",
                                    _ => "end_turn",
                                };

                                // Close any open content blocks
                                if st.thinking_block_open {
                                    yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                    st.thinking_block_open = false;
                                    st.block_index += 1;
                                }
                                if st.text_block_open {
                                    yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                    st.text_block_open = false;
                                    st.block_index += 1;
                                }
                                let tool_count = st.tool_blocks_open.len();
                                let mut has_open_tool = false;
                                for i in 0..tool_count {
                                    if st.tool_blocks_open[i] {
                                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                        st.tool_blocks_open[i] = false;
                                        has_open_tool = true;
                                        st.block_index += 1;
                                    }
                                }

                                // If finish_reason is "tool_calls" but no tool blocks were actually
                                // created, fall back to "end_turn" to avoid confusing the client
                                let stop_reason = if stop_reason == "tool_use" && !has_open_tool {
                                    log::info!("[Stream] finish_reason was tool_calls but no tool blocks found, using end_turn");
                                    "end_turn"
                                } else {
                                    stop_reason
                                };

                                // Extract usage from final chunk if available
                                if let Some(u) = parsed.get("usage") {
                                    if let Some(ct) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                                        st.usage_output_tokens = ct;
                                    }
                                }

                                let msg_delta = json!({
                                    "type": "message_delta",
                                    "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                                    "usage": {"output_tokens": st.usage_output_tokens}
                                });
                                yield Ok(Bytes::from(format!("event: message_delta\ndata: {}\n\n", msg_delta)));
                                yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string()));
                                st.finished = true;
                            }
                        }
                    }
                }
            }

            // Save remaining buffer
            *buffer.lock().await = local_buf;
        }
    });

    Box::pin(stream)
}

// ---------------------------------------------------------------------------
// Anthropic SSE → OpenAI Chat SSE: Streaming conversion
// ---------------------------------------------------------------------------

/// Convert an Anthropic Messages SSE stream into an OpenAI Chat Completions SSE stream.
pub fn convert_anthropic_stream_to_openai(
    upstream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: String,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    Box::pin(async_stream::stream! {
        let mut buffer = String::new();
        let mut event_type = String::new();
        let mut response_id = String::new();
        let mut model = request_model.clone();
        let mut chunk_id = 0usize;
        let mut open_blocks: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        let mut block_index_to_type: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        // Track tool calls per tool block
        let mut tool_blocks: std::collections::HashMap<u32, serde_json::Map<String, Value>> = std::collections::HashMap::new();

        tokio::pin!(upstream);

        while let Some(chunk) = upstream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let text = String::from_utf8_lossy(&bytes);
            buffer.push_str(&text);

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer = buffer[newline_pos + 1..].to_string();
                let trimmed = line.trim().to_string();

                if trimmed.is_empty() { continue; }

                // Track event: type from line
                if let Some(ev) = trimmed.strip_prefix("event: ") {
                    event_type = ev.trim().to_string();
                    continue;
                }

                let data_str = if let Some(s) = trimmed.strip_prefix("data: ") {
                    s.trim().to_string()
                } else if let Some(s) = trimmed.strip_prefix("data:") {
                    s.trim().to_string()
                } else {
                    continue;
                };

                if data_str == "[DONE]" { continue; }

                let data: Value = match serde_json::from_str(&data_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Determine event type from event: line or data["type"]
                let ev = if !event_type.is_empty() {
                    event_type.clone()
                } else {
                    data.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string()
                };
                event_type.clear();

                match ev.as_str() {
                    "message_start" => {
                        if let Some(msg) = data.get("message") {
                            response_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("msg_unknown").to_string();
                            model = request_model.clone();
                        }
                    }
                    "content_block_start" => {
                        let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let block = data.get("content_block").unwrap_or(&Value::Null);
                        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        block_index_to_type.insert(idx, block_type.clone());

                        match block_type.as_str() {
                            "text" => {
                                let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                open_blocks.insert(idx, text.to_string());
                            }
                            "thinking" => {
                                let text = block.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                open_blocks.insert(idx, text.to_string());
                            }
                            "tool_use" => {
                                let tc_id = block.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let mut tc = serde_json::Map::new();
                                tc.insert("id".to_string(), Value::String(tc_id));
                                tc.insert("type".to_string(), Value::String("function".to_string()));
                                let mut func = serde_json::Map::new();
                                func.insert("name".to_string(), Value::String(name));
                                func.insert("arguments".to_string(), Value::String(String::new()));
                                tc.insert("function".to_string(), Value::Object(func));
                                tool_blocks.insert(idx, tc);
                                open_blocks.insert(idx, String::new());
                            }
                            _ => {}
                        }
                    }
                    "content_block_delta" => {
                        let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let delta = data.get("delta").unwrap_or(&Value::Null);
                        let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");

                        match delta_type {
                            "text_delta" => {
                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                    let chat_chunk = chat_delta_chunk(&response_id, &model, 0, chunk_id, false, text, None);
                                    chunk_id += 1;
                                    yield Ok(chat_chunk);
                                }
                            }
                            "thinking_delta" => {
                                if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                                    if !text.is_empty() {
                                        let chat_chunk = chat_delta_chunk(&response_id, &model, 0, chunk_id, true, text, None);
                                        chunk_id += 1;
                                        yield Ok(chat_chunk);
                                    }
                                }
                            }
                            "input_json_delta" => {
                                if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                    if let Some(tc) = tool_blocks.get_mut(&idx) {
                                        if let Some(func) = tc.get_mut("function").and_then(|f| f.as_object_mut()) {
                                            let args = func.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                                            let new_args = args.to_string() + partial;
                                            func.insert("arguments".to_string(), Value::String(new_args));
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let block_type = block_index_to_type.get(&idx).map(|s| s.as_str()).unwrap_or("");

                        if block_type == "tool_use" {
                            // Emit the tool call as a delta chunk
                            if let Some(tc_map) = tool_blocks.get(&idx) {
                                let tc_val = Value::Object(tc_map.clone());
                                // Construct tool call delta
                                let tc_delta = json!([{
                                    "index": 0,
                                    "id": tc_val.get("id").unwrap_or(&Value::Null),
                                    "type": tc_val.get("type").unwrap_or(&Value::Null),
                                    "function": tc_val.get("function").unwrap_or(&Value::Null)
                                }]);
                                let chat_chunk = chat_delta_chunk(&response_id, &model, 0, chunk_id, false, "", Some(tc_delta));
                                chunk_id += 1;
                                yield Ok(chat_chunk);
                            }
                        }

                        open_blocks.remove(&idx);
                        block_index_to_type.remove(&idx);
                        tool_blocks.remove(&idx);
                    }
                    "message_delta" => {
                        // Capture stop_reason for final chunk
                        // We'll use it when message_stop arrives
                    }
                    "message_stop" => {
                        // Emit final chunk with finish_reason
                        // We don't have access to the stop_reason directly here
                        // since it was in message_delta which we may have skipped tracking
                        let final_chunk = json!({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": 0, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        });
                        yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&final_chunk).unwrap_or_default())));
                        yield Ok(Bytes::from("data: [DONE]\n\n"));
                    }
                    _ => {}
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract plain text from Anthropic content (string or array of content blocks).
fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut texts = Vec::new();
            for block in blocks {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    texts.push(t.to_string());
                } else if let Some(s) = block.as_str() {
                    texts.push(s.to_string());
                }
            }
            texts.join("")
        }
        _ => String::new(),
    }
}

// ===========================================================================
// OpenAI Responses API conversions
// ===========================================================================
//
// The Responses API (used by Codex CLI) differs from Chat Completions:
//   - Input  uses `input` (array of mixed items), not `messages`
//   - Output uses `output` (array of items), not `choices`
//   - Tool calls use `call_id`, not `id`
//   - System prompt goes in `instructions`, not a system message
//   - Streaming events: response.output_text.delta, etc.
//   - `max_output_tokens` instead of `max_tokens`
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Anthropic → OpenAI Responses: Request conversion
// ---------------------------------------------------------------------------

/// Convert an Anthropic Messages API request body to OpenAI Responses API format.
pub fn anthropic_to_responses_request(body: &Value, target_model: &str) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("model".into(), Value::String(target_model.to_string()));

    // system → instructions
    if let Some(system) = body.get("system") {
        let system_text = extract_text_from_content(system);
        if !system_text.is_empty() {
            out.insert("instructions".into(), Value::String(system_text));
        }
    }

    // Convert messages → input
    let mut input: Vec<Value> = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match role {
                "assistant" => {
                    // Collect text parts and tool_use blocks separately
                    let mut text_parts: Vec<Value> = Vec::new();
                    let mut tool_uses: Vec<Value> = Vec::new();

                    if let Some(c) = content {
                        if let Some(arr) = c.as_array() {
                            for block in arr {
                                match block.get("type").and_then(|t| t.as_str()) {
                                    Some("text") => {
                                        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                                            text_parts.push(json!({"type": "output_text", "text": t}));
                                        }
                                    }
                                    Some("tool_use") => {
                                        let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                        let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                        let input_val = block.get("input").cloned().unwrap_or(json!({}));
                                        let args_str = serde_json::to_string(&input_val).unwrap_or_default();
                                        tool_uses.push(json!({
                                            "type": "function_call",
                                            "call_id": id,
                                            "name": name,
                                            "arguments": args_str,
                                        }));
                                    }
                                    Some("thinking") => {
                                        // Skip thinking blocks
                                    }
                                    _ => {
                                        // Pass through as text
                                        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                                            text_parts.push(json!({"type": "output_text", "text": t}));
                                        }
                                    }
                                }
                            }
                        } else if let Some(s) = c.as_str() {
                            if !s.is_empty() {
                                text_parts.push(json!({"type": "output_text", "text": s}));
                            }
                        }
                    }

                    // Emit assistant message (only if there's text content)
                    if !text_parts.is_empty() {
                        input.push(json!({
                            "role": "assistant",
                            "content": text_parts
                        }));
                    }

                    // Emit function_call items
                    for tu in tool_uses {
                        input.push(tu);
                    }
                }
                "user" => {
                    // Check for tool_result blocks
                    if let Some(c) = content {
                        if let Some(arr) = c.as_array() {
                            let mut has_tool_result = false;
                            let mut text_parts: Vec<Value> = Vec::new();
                            let mut tool_results: Vec<Value> = Vec::new();

                            for block in arr {
                                match block.get("type").and_then(|t| t.as_str()) {
                                    Some("tool_result") => {
                                        has_tool_result = true;
                                        let call_id = block
                                            .get("tool_use_id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let result_text =
                                            extract_text_from_content(block.get("content").unwrap_or(&Value::Null));
                                        tool_results.push(json!({
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": result_text
                                        }));
                                    }
                                    Some("text") => {
                                        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                                            text_parts.push(Value::String(t.to_string()));
                                        }
                                    }
                                    _ => {
                                        text_parts.push(Value::String(block.to_string()));
                                    }
                                }
                            }

                            if has_tool_result {
                                // function_call_output items first (must follow function_call)
                                for tr in tool_results {
                                    input.push(tr);
                                }
                                // Any remaining text goes after
                                let text = text_parts.iter().filter_map(|t| t.as_str()).collect::<Vec<_>>().join("");
                                if !text.is_empty() {
                                    input.push(json!({
                                        "role": "user",
                                        "content": text
                                    }));
                                }
                            } else {
                                // Regular user message
                                let text = text_parts.iter().filter_map(|t| t.as_str()).collect::<Vec<_>>().join("");
                                if !text.is_empty() {
                                    input.push(json!({"role": "user", "content": text}));
                                }
                            }
                        } else {
                            input.push(json!({"role": "user", "content": c.clone()}));
                        }
                    } else {
                        input.push(json!({"role": "user", "content": ""}));
                    }
                }
                _ => {
                    // Other roles → user
                    if let Some(c) = content {
                        input.push(json!({"role": "user", "content": c.clone()}));
                    }
                }
            }
        }
    }

    out.insert("input".into(), Value::Array(input));

    // max_tokens → max_output_tokens
    if let Some(mt) = body.get("max_tokens") {
        out.insert("max_output_tokens".into(), mt.clone());
    }

    // temperature
    if let Some(t) = body.get("temperature") {
        out.insert("temperature".into(), t.clone());
    }

    // stream
    if let Some(s) = body.get("stream") {
        out.insert("stream".into(), s.clone());
    }

    // tools: Anthropic → Responses format
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let resp_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                if tool.get("type").and_then(|t| t.as_str()) == Some("function") {
                    // Already function format
                    tool.clone()
                } else {
                    // Anthropic format: {name, description, input_schema}
                    json!({
                        "type": "function",
                        "name": tool.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "description": tool.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(json!({"type": "object", "properties": {}}))
                    })
                }
            })
            .collect();
        out.insert("tools".into(), Value::Array(resp_tools));
    }

    // tool_choice
    if let Some(tc) = body.get("tool_choice") {
        let resp_tc = match tc {
            Value::String(s) => match s.as_str() {
                "auto" => json!("auto"),
                "any" => json!("required"),
                "none" => json!("none"),
                _ => json!("auto"),
            },
            Value::Object(obj) => {
                if obj.get("type").and_then(|t| t.as_str()) == Some("tool") {
                    json!({
                        "type": "function",
                        "name": obj.get("name").and_then(|n| n.as_str()).unwrap_or("")
                    })
                } else {
                    json!("auto")
                }
            }
            _ => json!("auto"),
        };
        out.insert("tool_choice".into(), resp_tc);
    }

    Value::Object(out)
}

// ---------------------------------------------------------------------------
// OpenAI Responses → Anthropic: Non-streaming response conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Responses API response to Anthropic Messages response format.
pub fn responses_to_anthropic_response(body: &Value, request_model: &str) -> Value {
    let mut content: Vec<Value> = Vec::new();

    if let Some(output) = body.get("output").and_then(|o| o.as_array()) {
        for item in output {
            match item.get("type").and_then(|t| t.as_str()) {
                Some("message") => {
                    // Extract text content from message items
                    if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                        for part in parts {
                            match part.get("type").and_then(|t| t.as_str()) {
                                Some("output_text") => {
                                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                        if !text.is_empty() {
                                            content.push(json!({
                                                "type": "text",
                                                "text": text
                                            }));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args_str = item.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                    let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

                    content.push(json!({
                        "type": "tool_use",
                        "id": if call_id.is_empty() { "call_0".to_string() } else { call_id.to_string() },
                        "name": name,
                        "input": input
                    }));
                }
                _ => {}
            }
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    // Map status → stop_reason
    let stop_reason = body
        .get("status")
        .and_then(|s| s.as_str())
        .map(|s| match s {
            "completed" => "end_turn",
            "incomplete" => "max_tokens",
            "failed" => "end_turn",
            _ => "end_turn",
        })
        .unwrap_or("end_turn");

    // Check if there are function_call outputs → tool_use stop reason
    let has_tool_calls = body
        .get("output")
        .and_then(|o| o.as_array())
        .map(|arr| arr.iter().any(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call")))
        .unwrap_or(false);
    let stop_reason = if has_tool_calls { "tool_use" } else { stop_reason };

    // Map usage
    let usage = body.get("usage").map(|u| {
        json!({
            "input_tokens": u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "output_tokens": u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        })
    }).unwrap_or(json!({
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0
    }));

    json!({
        "id": body.get("id").and_then(|v| v.as_str()).unwrap_or("msg_unknown"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": request_model,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": usage
    })
}

// ---------------------------------------------------------------------------
// OpenAI Responses SSE → Anthropic SSE: Streaming conversion
// ---------------------------------------------------------------------------

/// State for the Responses SSE → Anthropic SSE converter.
struct ResponsesStreamState {
    started: bool,
    finished: bool,
    block_index: u32,
    thinking_block_open: bool,
    text_block_open: bool,
    tool_block_open: bool,  // one tool at a time in Responses
    msg_id: String,
    model: String,
    usage_input_tokens: u64,
    usage_output_tokens: u64,
}

/// Convert an OpenAI Responses SSE byte stream into Anthropic SSE events.
pub fn convert_responses_stream_to_anthropic(
    upstream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: String,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let state = Arc::new(Mutex::new(ResponsesStreamState {
        started: false,
        finished: false,
        block_index: 0,
        thinking_block_open: false,
        text_block_open: false,
        tool_block_open: false,
        msg_id: format!("msg_{:x}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()),
        model: request_model,
        usage_input_tokens: 0,
        usage_output_tokens: 0,
    }));

    let buffer = Arc::new(Mutex::new(String::new()));

    let stream = upstream.flat_map(move |chunk_result| {
        let state = state.clone();
        let buffer = buffer.clone();

        async_stream::stream! {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let text = String::from_utf8_lossy(&chunk);
            let mut local_buf;
            {
                let mut guard = buffer.lock().await;
                guard.push_str(&text);
                local_buf = std::mem::take(&mut *guard);
            }

            while let Some(newline_pos) = local_buf.find('\n') {
                let line = local_buf[..newline_pos].trim().to_string();
                local_buf = local_buf[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                // Responses API uses SSE events like:
                //   event: response.output_text.delta
                //   data: {"type":"response.output_text.delta", ...}
                // We only parse the data lines; event lines are informational.
                let data = if let Some(stripped) = line.strip_prefix("data: ") {
                    stripped.to_string()
                } else if let Some(stripped) = line.strip_prefix("data:") {
                    stripped.trim().to_string()
                } else {
                    continue; // skip event: lines and others
                };

                if data == "[DONE]" {
                    let mut st = state.lock().await;
                    if st.finished {
                        continue;
                    }
                    // Close any open blocks
                    if st.thinking_block_open {
                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                        st.thinking_block_open = false;
                        st.block_index += 1;
                    }
                    if st.text_block_open {
                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                        st.text_block_open = false;
                        st.block_index += 1;
                    }
                    if st.tool_block_open {
                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                        st.tool_block_open = false;
                        st.block_index += 1;
                    }
                    st.finished = true;
                    let msg_delta = json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": Value::Null},
                        "usage": {"output_tokens": st.usage_output_tokens}
                    });
                    yield Ok(Bytes::from(format!("event: message_delta\ndata: {}\n\n", msg_delta)));
                    yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string()));
                    continue;
                }

                let parsed: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let mut st = state.lock().await;

                match event_type {
                    "response.created" | "response.in_progress" => {
                        if !st.started {
                            st.started = true;
                            // Extract response id and usage if available
                            if let Some(resp) = parsed.get("response") {
                                if let Some(id) = resp.get("id").and_then(|v| v.as_str()) {
                                    st.msg_id = id.to_string();
                                }
                                if let Some(u) = resp.get("usage") {
                                    st.usage_input_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                }
                            }
                            let msg_start = json!({
                                "type": "message_start",
                                "message": {
                                    "id": st.msg_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [],
                                    "model": st.model,
                                    "stop_reason": Value::Null,
                                    "stop_sequence": Value::Null,
                                    "usage": {
                                        "input_tokens": st.usage_input_tokens,
                                        "output_tokens": 1,
                                        "cache_creation_input_tokens": 0,
                                        "cache_read_input_tokens": 0
                                    }
                                }
                            });
                            yield Ok(Bytes::from(format!("event: message_start\ndata: {}\n\n", msg_start)));
                        }
                    }
                    "response.output_text.delta" => {
                        if let Some(delta_text) = parsed.get("delta").and_then(|d| d.as_str()) {
                            if !delta_text.is_empty() {
                                if !st.text_block_open {
                                    // Close thinking block first if open
                                    if st.thinking_block_open {
                                        yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                        st.thinking_block_open = false;
                                        st.block_index += 1;
                                    }
                                    let start = json!({
                                        "type": "content_block_start",
                                        "index": st.block_index,
                                        "content_block": {"type": "text", "text": ""}
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", start)));
                                    st.text_block_open = true;
                                }

                                let delta_event = json!({
                                    "type": "content_block_delta",
                                    "index": st.block_index,
                                    "delta": {"type": "text_delta", "text": delta_text}
                                });
                                yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", delta_event)));
                            }
                        }
                    }
                    "response.output_text.done" => {
                        if st.text_block_open {
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.text_block_open = false;
                            st.block_index += 1;
                        }
                    }
                    "response.output_item.added" => {
                        let item_type = parsed.get("item")
                            .and_then(|i| i.get("type"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");

                        if item_type == "function_call" {
                            // Close thinking block if open
                            if st.thinking_block_open {
                                yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                st.thinking_block_open = false;
                                st.block_index += 1;
                            }
                            // Close text block if open
                            if st.text_block_open {
                                yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                st.text_block_open = false;
                                st.block_index += 1;
                            }

                            let call_id = parsed.get("item")
                                .and_then(|i| i.get("call_id"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("");
                            let name = parsed.get("item")
                                .and_then(|i| i.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("");

                            let tool_start = json!({
                                "type": "content_block_start",
                                "index": st.block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call_id,
                                    "name": name,
                                    "input": {}
                                }
                            });
                            yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", tool_start)));
                            st.tool_block_open = true;
                        }
                        // For "message" type items, we don't need to do anything special
                        // text deltas will handle the content
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(args_delta) = parsed.get("delta").and_then(|d| d.as_str()) {
                            if !args_delta.is_empty() {
                                let args_event = json!({
                                    "type": "content_block_delta",
                                    "index": st.block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args_delta
                                    }
                                });
                                yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", args_event)));
                            }
                        }
                    }
                    "response.function_call_arguments.done" | "response.output_item.done" => {
                        // Close tool block if this is a function_call item done
                        if st.tool_block_open {
                            // Only close on output_item.done for function_call
                            if event_type == "response.output_item.done" {
                                let item_type = parsed.get("item")
                                    .and_then(|i| i.get("type"))
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");
                                if item_type == "function_call" {
                                    yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                                    st.tool_block_open = false;
                                    st.block_index += 1;
                                }
                            }
                        }
                    }
                    "response.completed" => {
                        if st.finished {
                            continue;
                        }
                        // Extract usage if available
                        if let Some(u) = parsed.get("response").and_then(|r| r.get("usage")) {
                            st.usage_output_tokens = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(st.usage_output_tokens);
                        }

                        // Close any remaining open blocks
                        if st.thinking_block_open {
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.thinking_block_open = false;
                            st.block_index += 1;
                        }
                        if st.text_block_open {
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.text_block_open = false;
                            st.block_index += 1;
                        }
                        if st.tool_block_open {
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.tool_block_open = false;
                            st.block_index += 1;
                        }

                        // Determine stop_reason from the response
                        let stop_reason = parsed.get("response")
                            .and_then(|r| r.get("status"))
                            .and_then(|s| s.as_str())
                            .map(|s| match s {
                                "completed" => "end_turn",
                                "incomplete" => "max_tokens",
                                _ => "end_turn",
                            })
                            .unwrap_or("end_turn");

                        // Check for function_calls in output
                        let has_tool_calls = parsed.get("response")
                            .and_then(|r| r.get("output"))
                            .and_then(|o| o.as_array())
                            .map(|arr| arr.iter().any(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call")))
                            .unwrap_or(false);
                        let stop_reason = if has_tool_calls { "tool_use" } else { stop_reason };

                        st.finished = true;
                        let msg_delta = json!({
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                            "usage": {"output_tokens": st.usage_output_tokens}
                        });
                        yield Ok(Bytes::from(format!("event: message_delta\ndata: {}\n\n", msg_delta)));
                        yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string()));
                    }
                    "response.reasoning_summary_text.delta" => {
                        if let Some(delta_text) = parsed.get("delta").and_then(|d| d.as_str()) {
                            if !delta_text.is_empty() {
                                if !st.thinking_block_open {
                                    let start = json!({
                                        "type": "content_block_start",
                                        "index": st.block_index,
                                        "content_block": {"type": "thinking", "thinking": ""}
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", start)));
                                    st.thinking_block_open = true;
                                }
                                let delta_event = json!({
                                    "type": "content_block_delta",
                                    "index": st.block_index,
                                    "delta": {"type": "thinking_delta", "thinking": delta_text}
                                });
                                yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", delta_event)));
                            }
                        }
                    }
                    "response.reasoning_summary_text.done" => {
                        if st.thinking_block_open {
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n", st.block_index)));
                            st.thinking_block_open = false;
                            st.block_index += 1;
                        }
                    }
                    _ => {
                        // Ignore other event types (response.content_part.added, etc.)
                    }
                }
            }

            *buffer.lock().await = local_buf;
        }
    });

    Box::pin(stream)
}

// ---------------------------------------------------------------------------
// OpenAI Chat → OpenAI Responses: Request conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Chat Completions request body to OpenAI Responses API format.
pub fn openai_to_responses_request(body: &Value, target_model: &str) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("model".into(), Value::String(target_model.to_string()));

    let mut input: Vec<Value> = Vec::new();
    let mut instructions: Option<String> = None;

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match role {
                "system" => {
                    // First system message → instructions
                    if instructions.is_none() {
                        let text = match content {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => String::new(),
                        };
                        if !text.is_empty() {
                            instructions = Some(text);
                        }
                    } else {
                        // Additional system messages → user
                        let text = match content {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => String::new(),
                        };
                        input.push(json!({"role": "user", "content": text}));
                    }
                }
                "assistant" => {
                    let mut text_parts: Vec<Value> = Vec::new();
                    let mut tool_calls: Vec<Value> = Vec::new();

                    if let Some(c) = content {
                        if !c.is_null() {
                            if let Some(s) = c.as_str() {
                                if !s.is_empty() {
                                    text_parts.push(json!({"type": "output_text", "text": s}));
                                }
                            }
                        }
                    }

                    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = tc.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("");
                            let args = tc.get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                .unwrap_or("{}");
                            tool_calls.push(json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": args
                            }));
                        }
                    }

                    if !text_parts.is_empty() {
                        input.push(json!({
                            "role": "assistant",
                            "content": text_parts
                        }));
                    }
                    for tc in tool_calls {
                        input.push(tc);
                    }
                }
                "tool" => {
                    // tool message → function_call_output
                    let call_id = msg.get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let output_text = match content {
                        Some(Value::String(s)) => s.clone(),
                        Some(other) => serde_json::to_string(other).unwrap_or_default(),
                        None => String::new(),
                    };
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text
                    }));
                }
                _ => {
                    // user or other → user message
                    input.push(json!({"role": role, "content": content.cloned().unwrap_or(Value::String(String::new()))}));
                }
            }
        }
    }

    if let Some(instr) = instructions {
        out.insert("instructions".into(), Value::String(instr));
    }
    out.insert("input".into(), Value::Array(input));

    // max_tokens → max_output_tokens
    if let Some(mt) = body.get("max_tokens") {
        out.insert("max_output_tokens".into(), mt.clone());
    }
    if let Some(mt) = body.get("max_completion_tokens") {
        out.insert("max_output_tokens".into(), mt.clone());
    }

    if let Some(t) = body.get("temperature") {
        out.insert("temperature".into(), t.clone());
    }
    if let Some(s) = body.get("stream") {
        out.insert("stream".into(), s.clone());
    }

    // tools passthrough (already in OpenAI function format)
    if let Some(tools) = body.get("tools") {
        out.insert("tools".into(), tools.clone());
    }
    if let Some(tc) = body.get("tool_choice") {
        out.insert("tool_choice".into(), tc.clone());
    }

    Value::Object(out)
}

// ---------------------------------------------------------------------------
// OpenAI Responses → OpenAI Chat: Response conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Responses API response to OpenAI Chat Completions response format.
pub fn responses_to_openai_response(body: &Value, request_model: &str) -> Value {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(output) = body.get("output").and_then(|o| o.as_array()) {
        for item in output {
            match item.get("type").and_then(|t| t.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                        for part in parts {
                            if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    text_parts.push(text.to_string());
                                }
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = item.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");

                    tool_calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let content_text = text_parts.join("");
    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        body.get("status").and_then(|s| s.as_str()).map(|s| match s {
            "completed" => "stop",
            "incomplete" => "length",
            _ => "stop",
        }).unwrap_or("stop")
    };

    let mut message = json!({
        "role": "assistant",
        "content": if content_text.is_empty() { Value::Null } else { Value::String(content_text) }
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    json!({
        "id": body.get("id").and_then(|v| v.as_str()).unwrap_or("chatcmpl_unknown"),
        "object": "chat.completion",
        "model": request_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": body.get("usage").and_then(|u| u.get("input_tokens")).and_then(|v| v.as_u64()).unwrap_or(0),
            "completion_tokens": body.get("usage").and_then(|u| u.get("output_tokens")).and_then(|v| v.as_u64()).unwrap_or(0),
            "total_tokens": body.get("usage").and_then(|u| u.get("total_tokens")).and_then(|v| v.as_u64()).unwrap_or(0)
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Codex: Responses API → Chat Completions request conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Responses API request body to OpenAI Chat Completions format.
/// Used when the Codex client sends Responses requests but the upstream provider
/// only supports Chat Completions (e.g., DeepSeek, Moonshot).
pub fn responses_to_chat_request(body: &Value, target_model: &str) -> Value {
    let mut result = serde_json::Map::new();
    result.insert("model".into(), Value::String(target_model.to_string()));

    let mut messages = Vec::new();

    // instructions → system message
    if let Some(instructions) = body.get("instructions") {
        let text = codex_instruction_text(instructions);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    // input → messages array
    if let Some(input) = body.get("input") {
        codex_append_responses_input_as_chat_messages(input, &mut messages);
    }

    // Fix ordering for strict providers (DeepSeek)
    codex_fix_chat_message_ordering(&mut messages);
    result.insert("messages".into(), json!(messages));

    // Token limits
    if let Some(max_tokens) = body.get("max_output_tokens") {
        result.insert("max_tokens".into(), max_tokens.clone());
    }
    if let Some(max_tokens) = body.get("max_tokens") {
        result.insert("max_tokens".into(), max_tokens.clone());
    }
    if let Some(max_tokens) = body.get("max_completion_tokens") {
        result.insert("max_completion_tokens".into(), max_tokens.clone());
    }

    // Passthrough fields
    for key in &["temperature", "top_p", "stream"] {
        if let Some(value) = body.get(*key) {
            result.insert((*key).to_string(), value.clone());
        }
    }

    // Tools conversion: Responses {type:"function", name, parameters} → Chat {type:"function", function:{name, parameters}}
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let chat_tools: Vec<Value> = tools.iter().filter_map(codex_responses_tool_to_chat_tool).collect();
        if !chat_tools.is_empty() {
            result.insert("tools".into(), json!(chat_tools));
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        result.insert("tool_choice".into(), tool_choice.clone());
    }

    // Extra passthrough fields
    for key in &[
        "frequency_penalty", "logit_bias", "logprobs", "metadata", "n",
        "parallel_tool_calls", "presence_penalty", "response_format", "seed",
        "service_tier", "stop", "stream_options", "top_logprobs", "user",
    ] {
        if let Some(value) = body.get(*key) {
            result.insert((*key).to_string(), value.clone());
        }
    }

    Value::Object(result)
}

fn codex_instruction_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()).or_else(|| p.as_str()))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        other => other.as_str().unwrap_or_default().to_string(),
    }
}

/// State for tracking pending tool_calls and reasoning during conversion.
struct CodexChatConversionState {
    pending_reasoning: String,
    last_reasoning: String,
    pending_tool_calls: Vec<Value>,
    last_flushed_tool_call_ids: Vec<String>,
}

impl CodexChatConversionState {
    fn new() -> Self {
        Self {
            pending_reasoning: String::new(),
            last_reasoning: String::new(),
            pending_tool_calls: Vec::new(),
            last_flushed_tool_call_ids: Vec::new(),
        }
    }

    fn consume_reasoning(&mut self) -> String {
        if !self.pending_reasoning.is_empty() {
            self.last_reasoning = self.pending_reasoning.clone();
            return std::mem::take(&mut self.pending_reasoning);
        }
        self.last_reasoning.clone()
    }

    fn flush_tool_calls(&mut self, messages: &mut Vec<Value>) {
        if self.pending_tool_calls.is_empty() {
            return;
        }
        let reasoning = self.consume_reasoning();
        let mut msg = json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": std::mem::take(&mut self.pending_tool_calls)
        });
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        self.last_flushed_tool_call_ids = msg["tool_calls"]
            .as_array()
            .map(|tcs| {
                tcs.iter()
                    .filter_map(|tc| tc.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        messages.push(msg);
    }

    fn fill_missing_tool_results(&mut self, messages: &mut Vec<Value>) {
        if self.last_flushed_tool_call_ids.is_empty() {
            return;
        }
        let answered: std::collections::HashSet<String> = messages
            .iter()
            .rev()
            .take_while(|m| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
            .filter_map(|m| m.get("tool_call_id").and_then(|v| v.as_str()).map(String::from))
            .collect();
        let missing: Vec<String> = self.last_flushed_tool_call_ids.iter()
            .filter(|id| !answered.contains(id.as_str()))
            .cloned()
            .collect();
        for id in missing {
            messages.push(json!({"role": "tool", "tool_call_id": id, "content": ""}));
        }
        self.last_flushed_tool_call_ids.clear();
    }
}

fn codex_append_responses_input_as_chat_messages(input: &Value, messages: &mut Vec<Value>) {
    let mut state = CodexChatConversionState::new();

    match input {
        Value::String(text) => {
            messages.push(json!({"role": "user", "content": text}));
        }
        Value::Array(items) => {
            for item in items {
                codex_append_responses_item_as_chat_message(item, messages, &mut state);
            }
        }
        Value::Object(_) => {
            codex_append_responses_item_as_chat_message(input, messages, &mut state);
        }
        _ => {}
    }

    state.flush_tool_calls(messages);
}

fn codex_extract_reasoning_text(item: &Value) -> String {
    let from_content = item.get("content")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|c| {
                    c.get("type").and_then(|v| v.as_str()) == Some("reasoning_text")
                        || c.get("type").and_then(|v| v.as_str()) == Some("reasoning")
                })
                .filter_map(|c| c.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let from_summary = item.get("summary")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|c| c.get("type").and_then(|v| v.as_str()) == Some("summary_text"))
                .filter_map(|c| c.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    if !from_content.is_empty() { from_content } else { from_summary }
}

fn codex_append_responses_item_as_chat_message(
    item: &Value,
    messages: &mut Vec<Value>,
    state: &mut CodexChatConversionState,
) {
    let item_type = item.get("type").and_then(|v| v.as_str());

    match item_type {
        Some("reasoning") => {
            let text = codex_extract_reasoning_text(item);
            if !text.is_empty() {
                state.pending_reasoning.push_str(&text);
            }
        }
        Some("function_call") => {
            state.pending_tool_calls.push(codex_responses_function_call_to_chat_tool_call(item));
        }
        Some("function_call_output") => {
            state.flush_tool_calls(messages);
            let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
            let output = match item.get("output") {
                Some(Value::String(s)) => s.clone(),
                Some(v) => serde_json::to_string(v).unwrap_or_default(),
                None => String::new(),
            };
            messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
        }
        Some("message") | None => {
            state.flush_tool_calls(messages);
            state.fill_missing_tool_results(messages);

            if item.get("role").is_some() || item.get("content").is_some() {
                let raw_role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let role = match raw_role {
                    "developer" | "system" => "user",
                    other => other,
                };

                if role == "user" {
                    state.pending_reasoning.clear();
                    state.last_reasoning.clear();
                }

                let content = item.get("content")
                    .map(|v| codex_responses_content_to_chat_content(role, v))
                    .unwrap_or(Value::Null);

                let mut msg = json!({"role": role, "content": content});

                if role == "assistant" {
                    let reasoning = state.consume_reasoning();
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = json!(reasoning);
                    }
                }

                // Skip empty assistant messages
                if role == "assistant"
                    && (content.is_null() || content.as_str().map_or(false, |s| s.is_empty()))
                    && !msg.get("reasoning_content").is_some()
                {
                    return;
                }

                messages.push(msg);
            }
        }
        _ => {
            state.flush_tool_calls(messages);
            state.fill_missing_tool_results(messages);
        }
    }
}

fn codex_responses_content_to_chat_content(_role: &str, content: &Value) -> Value {
    if content.is_null() || content.is_string() {
        return content.clone();
    }
    let Some(parts) = content.as_array() else {
        return content.clone();
    };

    let mut chat_parts: Vec<Value> = Vec::new();
    let mut has_non_text = false;

    for part in parts {
        let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match part_type {
            "input_text" | "output_text" | "text" => {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        chat_parts.push(json!({"type": "text", "text": text}));
                    }
                }
            }
            "refusal" => {
                if let Some(text) = part.get("refusal").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        chat_parts.push(json!({"type": "text", "text": text}));
                    }
                }
            }
            "input_image" => {
                if let Some(image_url) = part.get("image_url") {
                    let image_url = if image_url.is_object() {
                        image_url.clone()
                    } else {
                        json!({"url": image_url.as_str().unwrap_or_default()})
                    };
                    chat_parts.push(json!({"type": "image_url", "image_url": image_url}));
                    has_non_text = true;
                }
            }
            _ => {}
        }
    }

    if !has_non_text {
        Value::String(
            chat_parts.iter()
                .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        Value::Array(chat_parts)
    }
}

fn codex_responses_function_call_to_chat_tool_call(item: &Value) -> Value {
    let call_id = item.get("call_id").or_else(|| item.get("id"))
        .and_then(|v| v.as_str()).unwrap_or("");
    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = match item.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => serde_json::to_string(v).unwrap_or_default(),
        None => "{}".to_string(),
    };
    json!({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    })
}

fn codex_responses_tool_to_chat_tool(tool: &Value) -> Option<Value> {
    if tool.get("type").and_then(|v| v.as_str()) != Some("function") {
        return None;
    }
    // If already has nested "function" key, use as-is
    if tool.get("function").is_some() {
        return Some(tool.clone());
    }
    // Otherwise wrap name/description/parameters into function object
    let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let description = tool.get("description").cloned().unwrap_or(Value::Null);
    let parameters = tool.get("parameters").cloned().unwrap_or_else(|| json!({}));
    let mut function = json!({"name": name, "description": description, "parameters": parameters});
    if let Some(strict) = tool.get("strict") {
        function["strict"] = strict.clone();
    }
    Some(json!({"type": "function", "function": function}))
}

/// Fix message ordering for strict providers (e.g., DeepSeek).
fn codex_fix_chat_message_ordering(messages: &mut Vec<Value>) {
    let mut fixed: Vec<Value> = Vec::new();

    for msg in messages.iter() {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");

        // If previous assistant has tool_calls and current is NOT tool, fill missing results
        if let Some(prev_msg) = fixed.last() {
            let prev_role = prev_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if prev_role == "assistant" {
                if let Some(tcs) = prev_msg.get("tool_calls").and_then(|v| v.as_array()) {
                    if !tcs.is_empty() && role != "tool" {
                        let answered: std::collections::HashSet<String> = fixed.iter()
                            .rev()
                            .take_while(|m| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
                            .filter_map(|m| m.get("tool_call_id").and_then(|v| v.as_str()).map(String::from))
                            .collect();
                        let missing: Vec<String> = tcs.iter()
                            .filter_map(|tc| tc.get("id").and_then(|v| v.as_str()).map(String::from))
                            .filter(|id| !answered.contains(id.as_str()))
                            .collect();
                        for id in &missing {
                            fixed.push(json!({"role": "tool", "tool_call_id": id, "content": ""}));
                        }
                    }
                }
            }
        }

        // Skip orphan tool messages
        if role == "tool" {
            if let Some(call_id) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
                let mut found = false;
                for m in fixed.iter().rev() {
                    let m_role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    if m_role == "assistant" {
                        if m.get("tool_calls").and_then(|v| v.as_array())
                            .map(|tcs| tcs.iter().any(|tc| tc.get("id").and_then(|v| v.as_str()) == Some(call_id)))
                            .unwrap_or(false)
                        {
                            found = true;
                            break;
                        }
                    }
                    if m_role == "user" || m_role == "system" {
                        break;
                    }
                }
                if !found {
                    continue;
                }
            }
        }

        // Skip consecutive empty assistant messages
        if role == "assistant" {
            if let Some(prev_msg) = fixed.last() {
                let prev_role = prev_msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
                if prev_role == "assistant"
                    && msg.get("content").map_or(true, |c| c.is_null() || c.as_str().map_or(true, |s| s.is_empty()))
                    && msg.get("tool_calls").is_none()
                    && msg.get("reasoning_content").is_none()
                {
                    continue;
                }
            }
        }

        fixed.push(msg.clone());
    }

    // Remove trailing assistant messages without tool_calls
    while fixed.last().map_or(false, |m| {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        role == "assistant" && m.get("tool_calls").and_then(|v| v.as_array()).map_or(true, |tcs| tcs.is_empty())
    }) {
        fixed.pop();
    }

    // If trailing assistant has tool_calls, add empty tool results
    if fixed.last().map_or(false, |m| {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        role == "assistant" && m.get("tool_calls").and_then(|v| v.as_array()).map_or(false, |tcs| !tcs.is_empty())
    }) {
        let tool_call_ids: Vec<String> = fixed.last()
            .and_then(|m| m.get("tool_calls").and_then(|v| v.as_array()))
            .map(|tcs| tcs.iter().filter_map(|tc| tc.get("id").and_then(|v| v.as_str()).map(String::from)).collect())
            .unwrap_or_default();
        for id in tool_call_ids {
            fixed.push(json!({"role": "tool", "tool_call_id": id, "content": ""}));
        }
    }

    // Ensure at least one user or tool message
    if !fixed.iter().any(|m| {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        role == "user" || role == "tool"
    }) {
        fixed.insert(0, json!({"role": "user", "content": "Continue."}));
    }

    *messages = fixed;
}

// ---------------------------------------------------------------------------
// Codex: Responses API → Anthropic Messages request conversion
// ---------------------------------------------------------------------------

/// Convert an OpenAI Responses API request body to Anthropic Messages format.
/// Used when the Codex client sends Responses requests but the upstream provider
/// only supports Anthropic Messages (e.g., Qwen, Zhipu GLM).
pub fn responses_to_anthropic_request(body: &Value, target_model: &str) -> Value {
    let mut result = serde_json::Map::new();
    result.insert("model".into(), Value::String(target_model.to_string()));

    // instructions → system
    if let Some(instructions) = body.get("instructions") {
        let text = codex_instruction_text(instructions);
        if !text.is_empty() {
            result.insert("system".into(), json!(text));
        }
    }

    // input → messages
    let mut messages = Vec::new();
    if let Some(input) = body.get("input") {
        codex_append_responses_input_as_anthropic_messages(input, &mut messages);
    }
    codex_fix_anthropic_message_ordering(&mut messages);
    result.insert("messages".into(), json!(messages));

    // Max tokens
    let max_tokens = body.get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(16384);
    result.insert("max_tokens".into(), json!(max_tokens));

    // Stream
    if let Some(stream) = body.get("stream") {
        result.insert("stream".into(), stream.clone());
    }

    // Thinking config
    let thinking_enabled = body.get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|v| v.as_str())
        != Some("disabled");
    if thinking_enabled {
        let budget_tokens = body.get("thinking")
            .and_then(|t| t.get("budget_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(10000);
        result.insert("thinking".into(), json!({"type": "enabled", "budget_tokens": budget_tokens}));
        let min_max_tokens = (budget_tokens + 6000) as u64;
        if max_tokens < min_max_tokens {
            result.insert("max_tokens".into(), json!(min_max_tokens));
        }
    }

    // Tools: Responses {type:"function", name, parameters} → Anthropic {name, description, input_schema}
    if let Some(tools) = body.get("tools") {
        let anthropic_tools = codex_responses_tools_to_anthropic_tools(tools);
        if !anthropic_tools.is_empty() {
            result.insert("tools".into(), json!(anthropic_tools));
        }
    }

    Value::Object(result)
}

fn codex_append_responses_input_as_anthropic_messages(input: &Value, messages: &mut Vec<Value>) {
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
                        let text = codex_extract_reasoning_text(item);
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
                            "type": "tool_use", "id": call_id, "name": name, "input": input_val
                        }));
                    }
                    Some("function_call_output") => {
                        codex_flush_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
                        let call_id = item.get("call_id").or_else(|| item.get("tool_call_id"))
                            .and_then(|v| v.as_str()).unwrap_or("");
                        let output = match item.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) => serde_json::to_string(v).unwrap_or_default(),
                            None => String::new(),
                        };
                        messages.push(json!({
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}]
                        }));
                    }
                    Some("message") | None => {
                        codex_flush_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
                        let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                        let text = codex_anthropic_content_to_text(item.get("content"));

                        match role {
                            "developer" => {
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
                                }
                            }
                            _ => {
                                if !text.is_empty() {
                                    messages.push(json!({"role": "user", "content": text}));
                                }
                            }
                        }
                    }
                    _ => {
                        codex_flush_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
                    }
                }
            }
        }
        _ => {}
    }

    codex_flush_anthropic_tool_calls(messages, &mut pending_tool_calls, &mut pending_reasoning);
}

fn codex_flush_anthropic_tool_calls(
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
    pending_reasoning: &mut String,
) {
    if pending_tool_calls.is_empty() {
        return;
    }
    let mut content: Vec<Value> = Vec::new();
    if !pending_reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": std::mem::take(pending_reasoning)}));
    }
    content.extend(std::mem::take(pending_tool_calls));
    messages.push(json!({"role": "assistant", "content": content}));
}

fn codex_anthropic_content_to_text(content: Option<&Value>) -> String {
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

fn codex_responses_tools_to_anthropic_tools(tools: &Value) -> Vec<Value> {
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

fn codex_fix_anthropic_message_ordering(messages: &mut Vec<Value>) {
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
                    if let Some(Value::Array(arr)) = prev.get_mut("content") {
                        if let Some(Value::Array(new_arr)) = msg.get("content") {
                            arr.extend(new_arr.clone());
                            continue;
                        } else if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                            arr.push(json!({"type": "text", "text": text}));
                            continue;
                        }
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
            fixed.push(json!({"role": "user", "content": " "}));
        }
    }

    // Ensure at least one user message
    if !fixed.iter().any(|m| m.get("role").and_then(|v| v.as_str()) == Some("user")) {
        fixed.push(json!({"role": "user", "content": "Continue."}));
    }

    *messages = fixed;
}

// ---------------------------------------------------------------------------
// Codex: Chat Completions → Responses API response conversion (non-streaming)
// ---------------------------------------------------------------------------

/// Convert a non-streaming Chat Completions response to Responses API format.
pub fn chat_to_responses_response(body: &Value, request_model: &str) -> Value {
    let choices = body.get("choices").and_then(|v| v.as_array());
    let choice = choices.and_then(|c| c.first());
    let message = choice.and_then(|c| c.get("message"));

    let response_id = codex_response_id_from_chat_id(body.get("id").and_then(|v| v.as_str()));
    // Always return the request model so Codex never sees the upstream provider's model name.
    let model = request_model;
    let created_at = body.get("created").and_then(|v| v.as_u64()).unwrap_or(0);
    let finish_reason = choice.and_then(|c| c.get("finish_reason").and_then(|v| v.as_str()));

    let mut output = Vec::new();

    if let Some(msg) = message {
        // Reasoning item
        if let Some(reasoning) = codex_chat_reasoning_text(msg) {
            if !reasoning.is_empty() {
                output.push(json!({
                    "id": format!("rs_{}", response_id),
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning}]
                }));
            }
        }

        // Message item
        if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
            let text = codex_split_leading_think_block(text)
                .map(|(_, answer)| answer)
                .unwrap_or_else(|| text.to_string());
            if !text.is_empty() {
                output.push(json!({
                    "id": format!("{}_msg", response_id),
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                }));
            }
        }

        // Function call items
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            for (index, tc) in tool_calls.iter().enumerate() {
                let call_id = tc.get("id").and_then(|v| v.as_str()).filter(|v| !v.is_empty())
                    .unwrap_or_else(|| "");
                let function = tc.get("function").unwrap_or(&Value::Null);
                let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = match function.get("arguments") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => serde_json::to_string(v).unwrap_or_default(),
                    None => "{}".to_string(),
                };
                let id = if call_id.is_empty() { format!("call_{index}") } else { call_id.to_string() };
                output.push(json!({
                    "id": format!("fc_{}", id),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments
                }));
            }
        }
    }

    let status = match finish_reason {
        Some("length") => "incomplete",
        _ => "completed",
    };

    let usage = codex_chat_usage_to_responses_usage(body.get("usage"));

    let mut response = json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage
    });

    if finish_reason == Some("length") {
        response["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }

    response
}

fn codex_chat_reasoning_text(message: &Value) -> Option<String> {
    for key in &["reasoning_content", "reasoning"] {
        if let Some(text) = message.get(*key).and_then(|v| v.as_str()) {
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    // Check for inline <think> blocks
    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
        if let Some((reasoning, _)) = codex_split_leading_think_block(content) {
            if !reasoning.is_empty() {
                return Some(reasoning);
            }
        }
    }
    None
}

fn codex_split_leading_think_block(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("轶事") {
        return None;
    }
    // Look for <think> or <![CDATA[<think...>]]>
    let open_idx = text.find("轶事")?;
    let after_open = &text[open_idx + 6..]; // skip "轶事"
    let close_idx = after_open.find("轶事")?;
    let reasoning = after_open[..close_idx].trim().to_string();
    let answer = after_open[close_idx + 7..].trim_start_matches(['\r', '\n', '\t', ' ']).to_string();
    Some((reasoning, answer))
}

fn codex_response_id_from_chat_id(id: Option<&str>) -> String {
    let id = id.unwrap_or("modelrouter");
    if id.starts_with("resp_") { id.to_string() } else { format!("resp_{id}") }
}

fn codex_chat_usage_to_responses_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage.filter(|v| v.is_object() && !v.is_null()) else {
        return json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0});
    };
    let input = usage.get("prompt_tokens").or_else(|| usage.get("input_tokens"))
        .and_then(|v| v.as_u64()).unwrap_or(0);
    let output = usage.get("completion_tokens").or_else(|| usage.get("output_tokens"))
        .and_then(|v| v.as_u64()).unwrap_or(0);
    let total = usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(input + output);
    json!({"input_tokens": input, "output_tokens": output, "total_tokens": total})
}

// ---------------------------------------------------------------------------
// Codex: Anthropic Messages → Responses API response conversion (non-streaming)
// ---------------------------------------------------------------------------

/// Convert a non-streaming Anthropic Messages response to Responses API format.
pub fn anthropic_to_responses_response(body: &Value, request_model: &str) -> Value {
    // Always return the request model so Codex never sees the upstream provider's model name.
    let model = request_model;
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
                        "id": item_id, "type": "reasoning", "status": "completed",
                        "summary": [], "content": [{"type": "reasoning_text", "text": thinking}]
                    }));
                }
                "text" => {
                    let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    output_text.push_str(text);
                    let item_id = format!("msg_{}", chrono_like_id());
                    output_items.push(json!({
                        "id": item_id, "type": "message", "status": "completed",
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
                        "id": id, "type": "function_call", "call_id": id,
                        "name": name, "arguments": args, "status": "completed"
                    }));
                }
                _ => {}
            }
        }
    }

    let usage = body.get("usage").map(|u| {
        let input = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        json!({"input_tokens": input, "output_tokens": output, "total_tokens": input + output})
    }).unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}));

    json!({
        "id": response_id,
        "object": "response",
        "created_at": chrono_like_id(),
        "status": "completed",
        "model": model,
        "output": output_items,
        "output_text": output_text,
        "usage": usage
    })
}

fn chrono_like_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Codex: Chat Completions SSE → Responses API SSE (streaming)
// ---------------------------------------------------------------------------

/// Convert a Chat Completions SSE stream into a Responses API SSE stream.
pub fn convert_chat_stream_to_responses(
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: &str,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let request_model = request_model.to_string();
    let stream = async_stream::stream! {
        let mut buffer = String::new();
        let mut state = CodexChatToResponsesState::default();
        // Always report the request model to the client, never the upstream model
        state.model = request_model.clone();
        let mut stream_failed = false;
        let mut event_count: u64 = 0;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        let data_line = if let Some(stripped) = line.strip_prefix("data: ") {
                            stripped.trim().to_string()
                        } else if let Some(stripped) = line.strip_prefix("data:") {
                            stripped.trim().to_string()
                        } else {
                            continue;
                        };
                        if data_line == "[DONE]" {
                            for event in state.finalize() {
                                event_count += 1;
                                let event_str = String::from_utf8_lossy(&event);
                                log::info!("[Chat→Responses] SSE event #{} (DONE finalize): {}", event_count, event_str.lines().next().unwrap_or(""));
                                yield Ok(event);
                            }
                            continue;
                        }
                        let chunk: Value = match serde_json::from_str(&data_line) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if chunk.get("error").is_some() {
                            log::warn!("[Chat→Responses] upstream error in chunk");
                            yield Ok(state.failed_event("Upstream error".to_string()));
                            stream_failed = true;
                            break;
                        }
                        for event in state.handle_chat_chunk(&chunk) {
                            event_count += 1;
                            let event_str = String::from_utf8_lossy(&event);
                            log::info!("[Chat→Responses] SSE event #{}: {}", event_count, event_str.lines().next().unwrap_or(""));
                            yield Ok(event);
                        }
                    }

                    if stream_failed { break; }
                }
                Err(e) => {
                    log::warn!("[Chat→Responses] stream error: {e}");
                    yield Ok(state.failed_event(format!("Stream error: {e}")));
                    stream_failed = true;
                    break;
                }
            }
        }

        if !stream_failed {
            for event in state.finalize() {
                event_count += 1;
                let event_str = String::from_utf8_lossy(&event);
                log::info!("[Chat→Responses] SSE event #{} (finalize): {}", event_count, event_str.lines().next().unwrap_or(""));
                yield Ok(event);
            }
        }
    };

    Box::pin(stream)
}

// ---------------------------------------------------------------------------
// Responses API SSE → OpenAI Chat Completions SSE (streaming)
// ---------------------------------------------------------------------------

/// Convert an OpenAI Responses API SSE stream into an OpenAI Chat Completions SSE stream.
pub fn convert_responses_stream_to_chat(
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: &str,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let request_model = request_model.to_string();
    Box::pin(async_stream::stream! {
        let mut buffer = String::new();
        let mut response_id = String::new();
        let mut created = 0u64;
        let mut model = request_model.clone();
        let mut _first_chunk_sent = false;
        let mut chunk_id = 0usize;
        let mut event_type = String::new();

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let text = String::from_utf8_lossy(&bytes);
            buffer.push_str(&text);

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer = buffer[newline_pos + 1..].to_string();
                let trimmed = line.trim().to_string();

                if trimmed.is_empty() {
                    continue;
                }

                // Track event type from event: line
                if let Some(ev) = trimmed.strip_prefix("event: ") {
                    event_type = ev.trim().to_string();
                    continue;
                }

                // Extract data: content
                let data_str = if let Some(s) = trimmed.strip_prefix("data: ") {
                    s.trim().to_string()
                } else if let Some(s) = trimmed.strip_prefix("data:") {
                    s.trim().to_string()
                } else {
                    continue;
                };

                if data_str == "[DONE]" {
                    continue;
                }

                let data: Value = match serde_json::from_str(&data_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Determine the event name: either from event: line or from data["type"]
                let ev = if !event_type.is_empty() {
                    event_type.clone()
                } else {
                    data.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string()
                };
                event_type.clear();

                match ev.as_str() {
                    "response.created" | "response.in_progress" => {
                        if let Some(resp) = data.get("response") {
                            if response_id.is_empty() {
                                response_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            }
                            if created == 0 {
                                created = resp.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
                            }
                            model = resp.get("model").and_then(|v| v.as_str()).unwrap_or(&request_model).to_string();
                        }
                    }
                    "response.reasoning_summary_text.delta" => {
                        // Map summary text delta → reasoning_content in Chat
                        if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                            if !delta.is_empty() {
                                let chat_chunk = chat_delta_chunk(&response_id, &model, created, chunk_id, true, delta, None);
                                chunk_id += 1;
                                yield Ok(chat_chunk);
                            }
                        }
                    }
                    "response.output_text.delta" => {
                        if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                            if !delta.is_empty() {
                                let chat_chunk = chat_delta_chunk(&response_id, &model, created, chunk_id, false, delta, None);
                                chunk_id += 1;
                                yield Ok(chat_chunk);
                            }
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                            if !delta.is_empty() {
                                // Tool call index comes from output_index
                                let tool_idx = data.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0);
                                let tc = json!([{
                                    "index": tool_idx,
                                    "function": {"arguments": delta, "name": null}
                                }]);
                                let chat_chunk = chat_delta_chunk(&response_id, &model, created, chunk_id, false, "", Some(tc));
                                chunk_id += 1;
                                yield Ok(chat_chunk);
                            }
                        }
                    }
                    "response.output_item.added" => {
                        let item_type = data.get("item").and_then(|v| v.get("type")).and_then(|v| v.as_str()).unwrap_or("");
                        if item_type == "function_call" {
                            let idx = data.get("output_index").and_then(|v| v.as_u64()).unwrap_or(0);
                            let name = data.get("item").and_then(|v| v.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                            let tc = json!([{
                                "index": idx,
                                "id": data.get("item").and_then(|v| v.get("id")).and_then(|v| v.as_str()).unwrap_or(""),
                                "type": "function",
                                "function": {"name": name, "arguments": ""}
                            }]);
                            let chat_chunk = chat_delta_chunk(&response_id, &model, created, chunk_id, false, "", Some(tc));
                            chunk_id += 1;
                            yield Ok(chat_chunk);
                        }
                    }
                    "response.completed" => {
                        // Determine finish_reason from status
                        let status = data.get("response").and_then(|v| v.get("status")).and_then(|v| v.as_str()).unwrap_or("completed");
                        let finish_reason = match status {
                            "incomplete" => Some("length"),
                            "failed" => Some("stop"),
                            _ => {
                                // Check for function_call items → tool_calls
                                let has_tools = data.get("response")
                                    .and_then(|v| v.get("output"))
                                    .and_then(|v| v.as_array())
                                    .map(|arr| arr.iter().any(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call")))
                                    .unwrap_or(false);
                                if has_tools { Some("tool_calls") } else { Some("stop") }
                            }
                        };
                        // Final chunk with empty delta and finish_reason
                        let final_chunk = json!({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                        });
                        yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&final_chunk).unwrap_or_default())));
                        yield Ok(Bytes::from("data: [DONE]\n\n"));
                    }
                    "response.failed" => {
                        let err_msg = data.get("response")
                            .and_then(|v| v.get("error"))
                            .and_then(|v| v.get("message"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("stream failed");
                        // Emit error as a Chat completion chunk with finish_reason
                        let err_chunk = json!({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        });
                        yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&err_chunk).unwrap_or_default())));
                        yield Ok(Bytes::from("data: [DONE]\n\n"));
                        log::warn!("[Responses→Chat] stream failed: {}", err_msg);
                    }
                    _ => {}
                }
            }
        }
    })
}

/// Build a Chat Completions SSE delta chunk.
fn chat_delta_chunk(id: &str, model: &str, created: u64, chunk_id: usize, is_reasoning: bool, content: &str, tool_calls: Option<Value>) -> Bytes {
    let mut delta = serde_json::Map::new();
    if chunk_id == 0 || content.is_empty() {
        // First chunk includes role
        delta.insert("role".to_string(), Value::String("assistant".to_string()));
    }
    if is_reasoning {
        delta.insert("reasoning_content".to_string(), Value::String(content.to_string()));
    } else if !content.is_empty() {
        delta.insert("content".to_string(), Value::String(content.to_string()));
    }
    if let Some(tc) = tool_calls {
        delta.insert("tool_calls".to_string(), tc);
    }

    let chunk = json!({
        "id": id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]
    });
    Bytes::from(format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap_or_default()))
}

/// State machine for Chat SSE → Responses SSE conversion.
#[derive(Default)]
struct CodexChatToResponsesState {
    response_started: bool,
    completed: bool,
    response_id: String,
    model: String,
    created_at: u64,
    next_output_index: u32,
    sequence_number: u64,
    text_started: bool,
    text_output_index: u32,
    text_item_id: String,
    text_content: String,
    reasoning_started: bool,
    reasoning_output_index: u32,
    reasoning_item_id: String,
    reasoning_content: String,
    tools: std::collections::BTreeMap<usize, CodexToolCallState>,
    output_items: Vec<Value>,
    latest_usage: Option<Value>,
    finish_reason: Option<String>,
}

#[derive(Default)]
struct CodexToolCallState {
    output_index: Option<u32>,
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    added: bool,
}

impl CodexChatToResponsesState {
    fn next_output_index(&mut self) -> u32 {
        let idx = self.next_output_index;
        self.next_output_index += 1;
        idx
    }

    fn sse_event(&mut self, event: &str, data: Value) -> Bytes {
        self.sequence_number += 1;
        // Responses API SSE events require "type" and "sequence_number".
        // For response lifecycle events, wrap the response object in "response" field.
        let payload = if matches!(event, "response.created" | "response.in_progress" | "response.completed" | "response.failed") {
            serde_json::json!({
                "type": event,
                "sequence_number": self.sequence_number,
                "response": data
            })
        } else {
            match data {
                Value::Object(mut map) => {
                    map.insert("type".to_string(), Value::String(event.to_string()));
                    map.insert("sequence_number".to_string(), Value::Number(self.sequence_number.into()));
                    Value::Object(map)
                }
                other => {
                    serde_json::json!({
                        "type": event,
                        "sequence_number": self.sequence_number,
                        "data": other
                    })
                }
            }
        };
        Bytes::from(format!("event: {event}\ndata: {}\n\n", serde_json::to_string(&payload).unwrap_or_default()))
    }

    fn ensure_response_started(&mut self) -> Vec<Bytes> {
        if self.response_started { return Vec::new(); }
        self.response_started = true;
        let response = json!({
            "id": self.response_id, "object": "response", "created_at": self.created_at,
            "status": "in_progress", "model": self.model, "output": []
        });
        vec![
            self.sse_event("response.created", response.clone()),
            self.sse_event("response.in_progress", response),
        ]
    }

    fn handle_chat_chunk(&mut self, chunk: &Value) -> Vec<Bytes> {
        let mut events = Vec::new();

        if let Some(id) = chunk.get("id").and_then(|v| v.as_str()) {
            self.response_id = codex_response_id_from_chat_id(Some(id));
        }
        // Do NOT override model — keep the request model so Codex sees
        // the model name it sent, not the upstream provider's model name.
        if let Some(created) = chunk.get("created").and_then(|v| v.as_u64()) {
            self.created_at = created;
        }

        events.extend(self.ensure_response_started());

        if let Some(usage) = chunk.get("usage").filter(|v| !v.is_null()) {
            self.latest_usage = Some(codex_chat_usage_to_responses_usage(Some(usage)));
        }

        let Some(choice) = chunk.get("choices").and_then(|v| v.as_array()).and_then(|c| c.first()) else {
            return events;
        };

        if let Some(delta) = choice.get("delta") {
            // Reasoning content
            for key in &["reasoning_content", "reasoning"] {
                if let Some(text) = delta.get(*key).and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        events.extend(self.push_reasoning_delta(text));
                    }
                }
            }

            // Text content
            if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                if !content.is_empty() {
                    events.extend(self.push_text_delta(content));
                }
            }

            // Tool calls
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                events.extend(self.finalize_reasoning());
                for tc in tool_calls {
                    events.extend(self.push_tool_call_delta(tc));
                }
            }
        }

        if let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            self.finish_reason = Some(finish_reason.to_string());
        }

        events
    }

    fn push_reasoning_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let mut events = Vec::new();
        if !self.reasoning_started {
            let idx = self.next_output_index();
            let item_id = format!("rs_{}", self.response_id);
            self.reasoning_started = true;
            self.reasoning_output_index = idx;
            self.reasoning_item_id = item_id.clone();
            events.push(self.sse_event("response.output_item.added", json!({
                "output_index": idx, "item": {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []}
            })));
            events.push(self.sse_event("response.reasoning_summary_part.added", json!({
                "item_id": self.reasoning_item_id, "output_index": idx, "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}
            })));
        }
        self.reasoning_content.push_str(delta);
        events.push(self.sse_event("response.reasoning_summary_text.delta", json!({
            "item_id": self.reasoning_item_id, "output_index": self.reasoning_output_index,
            "summary_index": 0, "delta": delta
        })));
        events
    }

    fn finalize_reasoning(&mut self) -> Vec<Bytes> {
        if !self.reasoning_started { return Vec::new(); }
        let idx = self.reasoning_output_index;
        let text = std::mem::take(&mut self.reasoning_content);
        let item = json!({"id": self.reasoning_item_id, "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}]});
        self.output_items.push(item.clone());
        self.reasoning_started = false;
        vec![
            self.sse_event("response.reasoning_summary_text.done", json!({
                "item_id": self.reasoning_item_id, "output_index": idx, "summary_index": 0, "text": text
            })),
            self.sse_event("response.reasoning_summary_part.done", json!({
                "item_id": self.reasoning_item_id, "output_index": idx, "summary_index": 0,
                "part": {"type": "summary_text", "text": text}
            })),
            self.sse_event("response.output_item.done", json!({"output_index": idx, "item": item})),
        ]
    }

    fn push_text_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let mut events = Vec::new();
        if !self.text_started {
            events.extend(self.finalize_reasoning());
            let idx = self.next_output_index();
            let item_id = format!("{}_msg", self.response_id);
            self.text_started = true;
            self.text_output_index = idx;
            self.text_item_id = item_id.clone();
            events.push(self.sse_event("response.output_item.added", json!({
                "output_index": idx, "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
            })));
            events.push(self.sse_event("response.content_part.added", json!({
                "item_id": self.text_item_id, "output_index": idx, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}
            })));
        }
        self.text_content.push_str(delta);
        events.push(self.sse_event("response.output_text.delta", json!({
            "item_id": self.text_item_id, "output_index": self.text_output_index,
            "content_index": 0, "delta": delta
        })));
        events
    }

    fn finalize_text(&mut self) -> Vec<Bytes> {
        if !self.text_started { return Vec::new(); }
        let idx = self.text_output_index;
        let text = std::mem::take(&mut self.text_content);
        let part = json!({"type": "output_text", "text": text, "annotations": []});
        let item = json!({"id": self.text_item_id, "type": "message", "status": "completed", "role": "assistant", "content": [part]});
        self.output_items.push(item.clone());
        self.text_started = false;
        vec![
            self.sse_event("response.output_text.done", json!({
                "item_id": self.text_item_id, "output_index": idx, "content_index": 0, "text": text
            })),
            self.sse_event("response.content_part.done", json!({
                "item_id": self.text_item_id, "output_index": idx, "content_index": 0, "part": part
            })),
            self.sse_event("response.output_item.done", json!({"output_index": idx, "item": item})),
        ]
    }

    fn push_tool_call_delta(&mut self, tool_call: &Value) -> Vec<Bytes> {
        let chat_index = tool_call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let id_delta = tool_call.get("id").and_then(|v| v.as_str()).map(str::to_string);
        let function = tool_call.get("function").unwrap_or(&Value::Null);
        let name_delta = function.get("name").and_then(|v| v.as_str()).map(str::to_string);
        let args_delta = function.get("arguments").and_then(|v| v.as_str()).unwrap_or("").to_string();

        // Update state first, then extract data for events
        {
            let state = self.tools.entry(chat_index).or_default();
            if let Some(id) = id_delta { state.call_id = id; }
            if let Some(name) = name_delta { state.name = name; }
            if !args_delta.is_empty() { state.arguments.push_str(&args_delta); }
        }

        let mut events = Vec::new();
        let state = self.tools.get(&chat_index).unwrap();

        if !state.added && (!state.call_id.is_empty() || !state.name.is_empty()) {
            // Need to add this tool — extract data then mutate
            let mut call_id = state.call_id.clone();
            let mut name = state.name.clone();
            let arguments = state.arguments.clone();
            let _ = state;

            let assigned = self.next_output_index();
            if call_id.is_empty() { call_id = format!("call_{chat_index}"); }
            if name.is_empty() { name = "unknown_tool".to_string(); }
            let item_id = format!("fc_{}", call_id);

            let state = self.tools.get_mut(&chat_index).unwrap();
            state.added = true;
            state.call_id = call_id.clone();
            state.name = name.clone();
            state.output_index = Some(assigned);
            state.item_id = item_id.clone();

            events.push(self.sse_event("response.output_item.added", json!({
                "output_index": assigned, "item": {
                    "id": item_id, "type": "function_call", "status": "in_progress",
                    "call_id": call_id, "name": name, "arguments": ""
                }
            })));

            if !arguments.is_empty() {
                events.push(self.sse_event("response.function_call_arguments.delta", json!({
                    "item_id": item_id, "output_index": assigned, "delta": arguments
                })));
            }
        } else {
            let added = state.added;
            let item_id = state.item_id.clone();
            let output_index = state.output_index;
            let _ = state;

            if added && !args_delta.is_empty() {
                if let Some(output_index) = output_index {
                    events.push(self.sse_event("response.function_call_arguments.delta", json!({
                        "item_id": item_id, "output_index": output_index, "delta": args_delta
                    })));
                }
            }
        }

        events
    }

    fn finalize_tools(&mut self) -> Vec<Bytes> {
        let mut events = Vec::new();
        let keys: Vec<usize> = self.tools.keys().copied().collect();
        for key in keys {
            // Clone data first to avoid borrow conflicts
            let tc = self.tools.get(&key).unwrap();
            let tc_call_id = tc.call_id.clone();
            let tc_name = tc.name.clone();
            let tc_arguments = tc.arguments.clone();
            let tc_output_index = tc.output_index;
            let tc_item_id = tc.item_id.clone();
            let _ = tc;

            let item = json!({
                "id": tc_item_id, "type": "function_call", "status": "completed",
                "call_id": tc_call_id, "name": tc_name, "arguments": tc_arguments
            });
            self.output_items.push(item.clone());
            events.push(self.sse_event("response.function_call_arguments.done", json!({
                "item_id": tc_item_id, "output_index": tc_output_index, "arguments": tc_arguments
            })));
            events.push(self.sse_event("response.output_item.done", json!({"output_index": tc_output_index, "item": item})));
        }
        events
    }

    fn finalize(&mut self) -> Vec<Bytes> {
        if self.completed { return Vec::new(); }
        let mut events = self.ensure_response_started();
        events.extend(self.finalize_reasoning());
        events.extend(self.finalize_text());
        events.extend(self.finalize_tools());

        let status = match self.finish_reason.as_deref() {
            Some("length") => "incomplete",
            _ => "completed",
        };
        let usage = self.latest_usage.clone().unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}));
        let mut response = json!({
            "id": self.response_id, "object": "response", "created_at": self.created_at,
            "status": status, "model": self.model, "output": self.output_items, "usage": usage
        });
        if status == "incomplete" {
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        events.push(self.sse_event("response.completed", response));
        self.completed = true;
        events
    }

    fn failed_event(&mut self, message: String) -> Bytes {
        self.completed = true;
        self.sse_event("response.failed", json!({
            "id": self.response_id, "object": "response", "status": "failed",
            "error": {"message": message}
        }))
    }
}

// ---------------------------------------------------------------------------
// Codex: Anthropic Messages SSE → Responses API SSE (streaming)
// ---------------------------------------------------------------------------

/// Convert an Anthropic Messages SSE stream into a Responses API SSE stream.
pub fn convert_anthropic_stream_to_responses(
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: &str,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let request_model = request_model.to_string();
    let stream = async_stream::stream! {
        let mut buffer = String::new();
        let mut state = CodexAnthropicToResponsesState::default();
        // Always report the request model to the client, never the upstream model
        state.model = request_model.clone();
        let mut stream_failed = false;
        let mut event_count: u64 = 0;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        let data_line = if let Some(stripped) = line.strip_prefix("data: ") {
                            stripped.trim().to_string()
                        } else if let Some(stripped) = line.strip_prefix("data:") {
                            stripped.trim().to_string()
                        } else {
                            continue;
                        };
                        if data_line == "[DONE]" {
                            for event in state.finalize() {
                                event_count += 1;
                                let event_str = String::from_utf8_lossy(&event);
                                log::info!("[Anthropic→Responses] SSE event #{}: {}", event_count, event_str.lines().next().unwrap_or(""));
                                yield Ok(event);
                            }
                            continue;
                        }
                        let chunk: Value = match serde_json::from_str(&data_line) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if chunk.get("error").is_some() {
                            log::warn!("[Anthropic→Responses] upstream error in chunk");
                            yield Ok(state.failed_event("Upstream error".to_string()));
                            stream_failed = true;
                            break;
                        }
                        // Detect event type from the data's "type" field
                        let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        for event in state.handle_anthropic_event(event_type, &chunk) {
                            event_count += 1;
                            let event_str = String::from_utf8_lossy(&event);
                            log::info!("[Anthropic→Responses] SSE event #{}: {}", event_count, event_str.lines().next().unwrap_or(""));
                            yield Ok(event);
                        }
                    }

                    if stream_failed { break; }
                }
                Err(e) => {
                    log::warn!("[Anthropic→Responses] stream error: {e}");
                    yield Ok(state.failed_event(format!("Stream error: {e}")));
                    stream_failed = true;
                    break;
                }
            }
        }

        if !stream_failed {
            for event in state.finalize() {
                event_count += 1;
                let event_str = String::from_utf8_lossy(&event);
                log::info!("[Anthropic→Responses] SSE event #{} (finalize): {}", event_count, event_str.lines().next().unwrap_or(""));
                yield Ok(event);
            }
        }
    };

    Box::pin(stream)
}

/// State machine for Anthropic SSE → Responses SSE conversion.
#[derive(Default)]
struct CodexAnthropicToResponsesState {
    response_id: String,
    model: String,
    created_at: u64,
    output_index: u32,
    output_items: Vec<Value>,
    input_tokens: u64,
    output_tokens: u64,
    response_started: bool,
    stream_terminated: bool,
    sequence_number: u64,
    stop_reason: Option<String>,
    phase: AnthropicPhase,
    current_reasoning: Option<AnthropicItemState>,
    current_message: Option<AnthropicItemState>,
    current_tool: Option<AnthropicToolCallState>,
}

#[derive(Default, PartialEq)]
enum AnthropicPhase {
    #[default]
    Idle,
    Reasoning,
    Message,
    ToolCall,
}

struct AnthropicItemState {
    id: String,
    output_index: u32,
    text: String,
}

struct AnthropicToolCallState {
    id: String,
    name: String,
    args: String,
    output_index: u32,
}

impl CodexAnthropicToResponsesState {
    fn next_output_index(&mut self) -> u32 {
        let idx = self.output_index;
        self.output_index += 1;
        idx
    }

    fn sse_event(&mut self, event: &str, data: Value) -> Bytes {
        self.sequence_number += 1;
        // Responses API SSE events require "type" and "sequence_number".
        // For response lifecycle events (created/in_progress/completed/failed),
        // the response object must be wrapped in a "response" field.
        // For other events (delta/done/added), the fields are at the top level.
        let payload = if matches!(event, "response.created" | "response.in_progress" | "response.completed" | "response.failed") {
            serde_json::json!({
                "type": event,
                "sequence_number": self.sequence_number,
                "response": data
            })
        } else {
            match data {
                Value::Object(mut map) => {
                    map.insert("type".to_string(), Value::String(event.to_string()));
                    map.insert("sequence_number".to_string(), Value::Number(self.sequence_number.into()));
                    Value::Object(map)
                }
                other => {
                    serde_json::json!({
                        "type": event,
                        "sequence_number": self.sequence_number,
                        "data": other
                    })
                }
            }
        };
        Bytes::from(format!("event: {event}\ndata: {}\n\n", serde_json::to_string(&payload).unwrap_or_default()))
    }

    fn ensure_response_started(&mut self) -> Vec<Bytes> {
        if self.response_started { return Vec::new(); }
        self.response_started = true;
        if self.response_id.is_empty() {
            self.response_id = format!("resp_{}", chrono_like_id());
        }
        if self.created_at == 0 {
            self.created_at = chrono_like_id() / 1000;
        }
        let response = json!({
            "id": self.response_id, "object": "response", "created_at": self.created_at,
            "status": "in_progress", "model": self.model, "output": []
        });
        vec![
            self.sse_event("response.created", response.clone()),
            self.sse_event("response.in_progress", response),
        ]
    }

    fn start_reasoning_item(&mut self) -> Vec<Bytes> {
        let idx = self.next_output_index();
        let item_id = format!("rs_{}_{}", chrono_like_id(), idx);
        self.current_reasoning = Some(AnthropicItemState { id: item_id.clone(), output_index: idx, text: String::new() });
        self.phase = AnthropicPhase::Reasoning;
        vec![
            self.sse_event("response.output_item.added", json!({
                "output_index": idx, "item": {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []}
            })),
            self.sse_event("response.reasoning_summary_part.added", json!({
                "item_id": item_id, "output_index": idx, "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}
            })),
        ]
    }

    fn append_reasoning_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let (output_index, item_id) = if let Some(ref item) = self.current_reasoning {
            (item.output_index, item.id.clone())
        } else {
            return Vec::new();
        };
        if let Some(ref mut item) = self.current_reasoning {
            item.text.push_str(delta);
        }
        vec![self.sse_event("response.reasoning_summary_text.delta", json!({
            "item_id": item_id, "output_index": output_index, "summary_index": 0, "delta": delta
        }))]
    }

    fn close_reasoning_item(&mut self) -> Vec<Bytes> {
        if let Some(item) = self.current_reasoning.take() {
            let completed = json!({
                "id": item.id, "type": "reasoning", "status": "completed",
                "summary": [{"type": "summary_text", "text": item.text}]
            });
            self.output_items.push(completed.clone());
            vec![
                self.sse_event("response.reasoning_summary_text.done", json!({
                    "item_id": item.id, "output_index": item.output_index, "summary_index": 0, "text": item.text
                })),
                self.sse_event("response.reasoning_summary_part.done", json!({
                    "item_id": item.id, "output_index": item.output_index, "summary_index": 0,
                    "part": {"type": "summary_text", "text": item.text}
                })),
                self.sse_event("response.output_item.done", json!({"output_index": item.output_index, "item": completed})),
            ]
        } else { Vec::new() }
    }

    fn start_message_item(&mut self) -> Vec<Bytes> {
        let idx = self.next_output_index();
        let item_id = format!("msg_{}_{}", chrono_like_id(), idx);
        self.current_message = Some(AnthropicItemState { id: item_id.clone(), output_index: idx, text: String::new() });
        self.phase = AnthropicPhase::Message;
        vec![
            self.sse_event("response.output_item.added", json!({
                "output_index": idx, "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
            })),
            self.sse_event("response.content_part.added", json!({
                "output_index": idx, "item_id": item_id, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}
            })),
        ]
    }

    fn append_message_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let (output_index, item_id) = if let Some(ref item) = self.current_message {
            (item.output_index, item.id.clone())
        } else {
            return Vec::new();
        };
        if let Some(ref mut item) = self.current_message {
            item.text.push_str(delta);
        }
        vec![self.sse_event("response.output_text.delta", json!({
            "output_index": output_index, "item_id": item_id, "content_index": 0, "delta": delta
        }))]
    }

    fn close_message_item(&mut self) -> Vec<Bytes> {
        if let Some(item) = self.current_message.take() {
            let part = json!({"type": "output_text", "text": item.text, "annotations": []});
            let completed = json!({"id": item.id, "type": "message", "status": "completed", "role": "assistant", "content": [part]});
            self.output_items.push(completed.clone());
            vec![
                self.sse_event("response.output_text.done", json!({
                    "output_index": item.output_index, "item_id": item.id, "content_index": 0, "text": item.text
                })),
                self.sse_event("response.content_part.done", json!({
                    "output_index": item.output_index, "item_id": item.id, "content_index": 0, "part": part
                })),
                self.sse_event("response.output_item.done", json!({"output_index": item.output_index, "item": completed})),
            ]
        } else { Vec::new() }
    }

    fn open_tool_call_item(&mut self, tool_id: &str, name: &str) -> Vec<Bytes> {
        let idx = self.next_output_index();
        self.current_tool = Some(AnthropicToolCallState {
            id: tool_id.to_string(), name: name.to_string(), args: String::new(), output_index: idx,
        });
        self.phase = AnthropicPhase::ToolCall;
        vec![self.sse_event("response.output_item.added", json!({
            "output_index": idx, "item": {"id": tool_id, "type": "function_call", "call_id": tool_id, "name": name, "arguments": "", "status": "in_progress"}
        }))]
    }

    fn append_tool_call_args(&mut self, delta: &str) -> Vec<Bytes> {
        let (output_index, item_id) = if let Some(ref tc) = self.current_tool {
            (tc.output_index, tc.id.clone())
        } else {
            return Vec::new();
        };
        if let Some(ref mut tc) = self.current_tool {
            tc.args.push_str(delta);
        }
        vec![self.sse_event("response.function_call_arguments.delta", json!({
            "output_index": output_index, "item_id": item_id, "delta": delta
        }))]
    }

    fn close_tool_call_item(&mut self) -> Vec<Bytes> {
        if let Some(tc) = self.current_tool.take() {
            let completed = json!({
                "id": tc.id, "type": "function_call", "call_id": tc.id,
                "name": tc.name, "arguments": tc.args, "status": "completed"
            });
            self.output_items.push(completed.clone());
            vec![
                self.sse_event("response.function_call_arguments.done", json!({
                    "output_index": tc.output_index, "item_id": tc.id, "arguments": tc.args
                })),
                self.sse_event("response.output_item.done", json!({"output_index": tc.output_index, "item": completed})),
            ]
        } else { Vec::new() }
    }

    fn close_current(&mut self) -> Vec<Bytes> {
        match self.phase {
            AnthropicPhase::Reasoning => self.close_reasoning_item(),
            AnthropicPhase::Message => self.close_message_item(),
            _ => Vec::new(),
        }
    }

    fn handle_anthropic_event(&mut self, event_type: &str, data: &Value) -> Vec<Bytes> {
        let mut events = self.ensure_response_started();

        match event_type {
            "message_start" => {
                if let Some(msg) = data.get("message") {
                    // Extract upstream message ID for response_id
                    if let Some(msg_id) = msg.get("id").and_then(|v| v.as_str()) {
                        if !msg_id.is_empty() {
                            self.response_id = msg_id.to_string();
                        }
                    }
                    // Do NOT override model — keep the request model so Codex sees
                    // the model name it sent, not the upstream provider's model name.
                    if let Some(usage) = msg.get("usage") {
                        self.input_tokens = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                        self.output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                    }
                }
            }
            "content_block_start" => {
                let block = data.get("content_block").unwrap_or(&Value::Null);
                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match block_type {
                    "thinking" => {
                        events.extend(self.close_current());
                        events.extend(self.start_reasoning_item());
                    }
                    "text" => {
                        events.extend(self.close_current());
                        events.extend(self.close_tool_call_item());
                        events.extend(self.start_message_item());
                    }
                    "tool_use" => {
                        events.extend(self.close_current());
                        let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        events.extend(self.open_tool_call_item(id, name));
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let delta = data.get("delta").unwrap_or(&Value::Null);
                let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match delta_type {
                    "thinking_delta" => {
                        if self.phase != AnthropicPhase::Reasoning {
                            events.extend(self.close_current());
                            events.extend(self.start_reasoning_item());
                        }
                        if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                            events.extend(self.append_reasoning_delta(text));
                        }
                    }
                    "text_delta" => {
                        if self.phase != AnthropicPhase::Message {
                            events.extend(self.close_current());
                            events.extend(self.close_tool_call_item());
                            events.extend(self.start_message_item());
                        }
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            events.extend(self.append_message_delta(text));
                        }
                    }
                    "input_json_delta" => {
                        if self.phase != AnthropicPhase::ToolCall {
                            events.extend(self.close_current());
                        }
                        if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            events.extend(self.append_tool_call_args(partial));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                events.extend(self.close_current());
                events.extend(self.close_tool_call_item());
                self.phase = AnthropicPhase::Idle;
            }
            "message_delta" => {
                if let Some(delta) = data.get("delta") {
                    if let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                        self.stop_reason = Some(reason.to_string());
                    }
                }
                if let Some(usage) = data.get("usage") {
                    self.output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(self.output_tokens);
                }
            }
            "message_stop" => {
                events.extend(self.close_current());
                events.extend(self.close_tool_call_item());
            }
            _ => {}
        }

        events
    }

    fn finalize(&mut self) -> Vec<Bytes> {
        if self.stream_terminated { return Vec::new(); }
        self.stream_terminated = true;
        let mut events = self.close_current();
        events.extend(self.close_tool_call_item());

        let usage = json!({"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "total_tokens": self.input_tokens + self.output_tokens});
        let msg_outputs: Vec<&Value> = self.output_items.iter()
            .filter(|it| it.get("type").and_then(|v| v.as_str()) == Some("message"))
            .collect();
        let empty_content = Vec::new();
        let output_text: String = msg_outputs.iter()
            .flat_map(|it| it.get("content").and_then(|v| v.as_array()).unwrap_or(&empty_content).iter())
            .filter_map(|c| if c.get("type").and_then(|v| v.as_str()) == Some("output_text") { c.get("text").and_then(|v| v.as_str()) } else { None })
            .collect();

        let status = match self.stop_reason.as_deref() {
            Some("max_tokens") => "incomplete",
            _ => "completed",
        };
        let mut response = json!({
            "id": self.response_id, "object": "response", "created_at": self.created_at,
            "status": status, "model": self.model, "output": self.output_items,
            "output_text": output_text, "usage": usage
        });
        if status == "incomplete" {
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        events.push(self.sse_event("response.completed", response));
        events
    }

    fn failed_event(&mut self, message: String) -> Bytes {
        self.stream_terminated = true;
        self.sse_event("response.failed", json!({
            "id": self.response_id, "object": "response", "status": "failed",
            "error": {"message": message}
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_to_openai_simple_messages() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "max_tokens": 1024
        });
        let result = anthropic_to_openai_request(&body, "deepseek-v4-pro");

        assert_eq!(result["model"], "deepseek-v4-pro");
        assert_eq!(result["max_tokens"], 1024);

        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Hi there");
    }

    #[test]
    fn test_anthropic_to_openai_with_system() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });
        let result = anthropic_to_openai_request(&body, "deepseek-v4-pro");

        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
        assert!(result.get("system").is_none());
    }

    #[test]
    fn test_anthropic_to_openai_with_tool_use() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "List files"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I'll list files."},
                    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1.txt\nfile2.txt"}
                ]}
            ],
            "tools": [
                {"name": "bash", "description": "Run bash", "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
            ]
        });
        let result = anthropic_to_openai_request(&body, "deepseek-v4-pro");

        let messages = result["messages"].as_array().unwrap();
        // user, assistant (text + tool_calls), tool result
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "I'll list files.");
        assert_eq!(messages[1]["tool_calls"][0]["type"], "function");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "bash");
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "toolu_1");
        assert_eq!(messages[2]["content"], "file1.txt\nfile2.txt");

        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "bash");
    }

    #[test]
    fn test_openai_to_anthropic_simple_response() {
        let body = json!({
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let result = openai_to_anthropic_response(&body, "claude-sonnet-4-6");

        assert_eq!(result["type"], "message");
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["model"], "claude-sonnet-4-6");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_openai_to_anthropic_tool_calls_response() {
        let body = json!({
            "id": "chatcmpl-456",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}
        });
        let result = openai_to_anthropic_response(&body, "claude-sonnet-4-6");

        assert_eq!(result["stop_reason"], "tool_use");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "call_1");
        assert_eq!(content[0]["name"], "bash");
        assert_eq!(content[0]["input"]["cmd"], "ls");
    }

    #[test]
    fn test_anthropic_to_openai_strips_thinking_fields() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 1024,
            "anthropic_version": "2023-06-01",
            "stream": true
        });
        let result = anthropic_to_openai_request(&body, "deepseek-v4-pro");

        assert!(result.get("anthropic_version").is_none());
        assert_eq!(result["stream"], true);
    }

    // =======================================================================
    // OpenAI Responses conversion tests
    // =======================================================================

    #[test]
    fn test_anthropic_to_responses_simple() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "max_tokens": 1024
        });
        let result = anthropic_to_responses_request(&body, "gpt-4o");

        assert_eq!(result["model"], "gpt-4o");
        assert_eq!(result["instructions"], "You are helpful");
        assert_eq!(result["max_output_tokens"], 1024);

        let input = result["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "Hello");
        assert_eq!(input[1]["role"], "assistant");
    }

    #[test]
    fn test_anthropic_to_responses_with_tool_use() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "List files"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I'll list files."},
                    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1.txt\nfile2.txt"}
                ]}
            ],
            "tools": [
                {"name": "bash", "description": "Run bash", "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
            ]
        });
        let result = anthropic_to_responses_request(&body, "gpt-4o");

        let input = result["input"].as_array().unwrap();
        // user, assistant message, function_call, function_call_output
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "toolu_1");
        assert_eq!(input[2]["name"], "bash");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "toolu_1");

        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "bash");
    }

    #[test]
    fn test_responses_to_anthropic_text_response() {
        let body = json!({
            "id": "resp_123",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello!"}
                    ]
                }
            ],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let result = responses_to_anthropic_response(&body, "claude-sonnet-4-6");

        assert_eq!(result["type"], "message");
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_responses_to_anthropic_function_call_response() {
        let body = json!({
            "id": "resp_456",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Let me check."}
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "bash",
                    "arguments": "{\"cmd\":\"ls\"}"
                }
            ],
            "status": "completed",
            "usage": {"input_tokens": 20, "output_tokens": 10}
        });
        let result = responses_to_anthropic_response(&body, "claude-sonnet-4-6");

        assert_eq!(result["stop_reason"], "tool_use");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me check.");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["id"], "call_abc");
        assert_eq!(content[1]["name"], "bash");
        assert_eq!(content[1]["input"]["cmd"], "ls");
    }

    #[test]
    fn test_openai_to_responses_simple() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "max_tokens": 1024
        });
        let result = openai_to_responses_request(&body, "gpt-4o");

        assert_eq!(result["model"], "gpt-4o");
        assert_eq!(result["instructions"], "You are helpful");
        assert_eq!(result["max_output_tokens"], 1024);

        let input = result["input"].as_array().unwrap();
        assert_eq!(input.len(), 2); // system extracted as instructions
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "assistant");
    }

    #[test]
    fn test_openai_to_responses_with_tool_calls() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "List files"},
                {"role": "assistant", "content": "I'll check.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"}
            ]
        });
        let result = openai_to_responses_request(&body, "gpt-4o");

        let input = result["input"].as_array().unwrap();
        // user, assistant, function_call, function_call_output
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
    }

    #[test]
    fn test_responses_to_openai_text_response() {
        let body = json!({
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello!"}
                    ]
                }
            ],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let result = responses_to_openai_response(&body, "gpt-4o");

        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["model"], "gpt-4o");
        assert_eq!(result["choices"][0]["message"]["role"], "assistant");
        assert_eq!(result["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
        assert_eq!(result["usage"]["prompt_tokens"], 10);
        assert_eq!(result["usage"]["completion_tokens"], 5);
    }

    #[test]
    fn test_responses_to_openai_function_call_response() {
        let body = json!({
            "id": "resp_456",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "bash",
                    "arguments": "{\"cmd\":\"ls\"}"
                }
            ],
            "status": "completed",
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        });
        let result = responses_to_openai_response(&body, "gpt-4o");

        assert_eq!(result["choices"][0]["finish_reason"], "tool_calls");
        let tool_calls = result["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls[0]["id"], "call_abc");
        assert_eq!(tool_calls[0]["function"]["name"], "bash");
        assert_eq!(tool_calls[0]["function"]["arguments"], "{\"cmd\":\"ls\"}");
    }
}
