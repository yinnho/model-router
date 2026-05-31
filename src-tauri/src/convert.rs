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
