//! Anthropic Messages SSE → Codex Responses API SSE conversion.
//!
//! Converts an upstream Anthropic Messages SSE stream into a Codex Responses API
//! SSE stream, used when the Codex client talks to CC Switch via the Responses API
//! but the upstream provider only exposes an Anthropic Messages endpoint.

use futures::stream::{Stream, StreamExt};
use bytes::Bytes;
use serde_json::{json, Value};

const MAX_OUTPUT_ITEMS: usize = 500;
const MAX_TEXT_LEN: usize = 1_000_000; // 1MB per text field

/// State machine for converting Anthropic SSE → Responses SSE.
#[derive(Default)]
struct AnthropicToResponsesState {
    response_id: String,
    model: String,
    created_at: u64,
    output_index: u32,
    seq: u64,
    output_items: Vec<Value>,
    input_tokens: u64,
    output_tokens: u64,
    reasoning_chars: usize,
    text_chars: usize,
    response_started: bool,
    stream_terminated: bool,

    // Current item tracking
    phase: Phase,
    current_reasoning: Option<ItemState>,
    current_message: Option<ItemState>,
    current_tool: Option<ToolCallState>,
}

#[derive(Default, PartialEq)]
enum Phase {
    #[default]
    Idle,
    Reasoning,
    Message,
    ToolCall,
}

struct ItemState {
    id: String,
    output_index: u32,
    text: String,
}

struct ToolCallState {
    id: String,
    name: String,
    args: String,
    output_index: u32,
    #[allow(dead_code)]
    added: bool,
}

impl AnthropicToResponsesState {
    fn next_output_index(&mut self) -> u32 {
        let idx = self.output_index;
        self.output_index += 1;
        idx
    }

    fn emit(&mut self, event_type: &str, data: Value) -> Option<Bytes> {
        let mut event = data;
        event["sequence_number"] = json!(self.seq);
        self.seq += 1;
        Some(Bytes::from(format!(
            "event: {event_type}\ndata: {}\n\n",
            event
        )))
    }

    fn ensure_response_started(&mut self) -> Vec<Bytes> {
        if self.response_started {
            return Vec::new();
        }
        self.response_started = true;
        let response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": []
        });
        let mut events = Vec::new();
        events.extend(self.emit("response.created", response.clone()));
        events.extend(self.emit("response.in_progress", response));
        events
    }

    fn start_reasoning_item(&mut self) -> Vec<Bytes> {
        let idx = self.next_output_index();
        let item_id = format!("rs_{}_{}", now_ms(), idx);
        self.current_reasoning = Some(ItemState {
            id: item_id.clone(),
            output_index: idx,
            text: String::new(),
        });
        self.phase = Phase::Reasoning;
        let mut events = Vec::new();
        events.extend(self.emit("response.output_item.added", json!({
            "output_index": idx,
            "item": {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": [], "content": []}
        })));
        events.extend(self.emit("response.content_part.added", json!({
            "output_index": idx, "item_id": item_id, "content_index": 0,
            "part": {"type": "reasoning_text", "text": ""}
        })));
        events
    }

    fn append_reasoning_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let (output_index, item_id) = if let Some(ref item) = self.current_reasoning {
            (item.output_index, item.id.clone())
        } else {
            return Vec::new();
        };
        if let Some(ref mut item) = self.current_reasoning {
            if item.text.len() + delta.len() <= MAX_TEXT_LEN {
                item.text.push_str(delta);
            }
        }
        self.reasoning_chars += delta.len();
        vec![self.emit("response.reasoning_text.delta", json!({
            "output_index": output_index, "item_id": item_id, "content_index": 0,
            "delta": delta
        })).unwrap()]
    }

    fn close_reasoning_item(&mut self) -> Vec<Bytes> {
        if let Some(item) = self.current_reasoning.take() {
            let mut events = Vec::new();
            events.extend(self.emit("response.reasoning_text.done", json!({
                "output_index": item.output_index, "item_id": item.id, "content_index": 0, "text": item.text
            })));
            events.extend(self.emit("response.content_part.done", json!({
                "output_index": item.output_index, "item_id": item.id, "content_index": 0,
                "part": {"type": "reasoning_text", "text": item.text}
            })));
            let completed = json!({
                "id": item.id, "type": "reasoning", "status": "completed",
                "summary": [], "content": [{"type": "reasoning_text", "text": item.text}]
            });
            events.extend(self.emit("response.output_item.done", json!({
                "output_index": item.output_index, "item": completed
            })));
            if self.output_items.len() < MAX_OUTPUT_ITEMS {
                self.output_items.push(completed);
            } else {
                log::warn!("output_items exceeded {} limit, dropping item", MAX_OUTPUT_ITEMS);
            }
            return events;
        }
        Vec::new()
    }

    fn start_message_item(&mut self) -> Vec<Bytes> {
        let idx = self.next_output_index();
        let item_id = format!("msg_{}_{}", now_ms(), idx);
        self.current_message = Some(ItemState {
            id: item_id.clone(),
            output_index: idx,
            text: String::new(),
        });
        self.phase = Phase::Message;
        let mut events = Vec::new();
        events.extend(self.emit("response.output_item.added", json!({
            "output_index": idx,
            "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
        })));
        events.extend(self.emit("response.content_part.added", json!({
            "output_index": idx, "item_id": item_id, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        })));
        events
    }

    fn append_message_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let (output_index, item_id) = if let Some(ref item) = self.current_message {
            (item.output_index, item.id.clone())
        } else {
            return Vec::new();
        };
        if let Some(ref mut item) = self.current_message {
            if item.text.len() + delta.len() <= MAX_TEXT_LEN {
                item.text.push_str(delta);
            }
        }
        self.text_chars += delta.len();
        vec![self.emit("response.output_text.delta", json!({
            "output_index": output_index, "item_id": item_id, "content_index": 0,
            "delta": delta
        })).unwrap()]
    }

    fn close_message_item(&mut self) -> Vec<Bytes> {
        if let Some(item) = self.current_message.take() {
            let mut events = Vec::new();
            events.extend(self.emit("response.output_text.done", json!({
                "output_index": item.output_index, "item_id": item.id, "content_index": 0, "text": item.text
            })));
            let part = json!({"type": "output_text", "text": item.text, "annotations": []});
            events.extend(self.emit("response.content_part.done", json!({
                "output_index": item.output_index, "item_id": item.id, "content_index": 0, "part": part
            })));
            let completed = json!({
                "id": item.id, "type": "message", "status": "completed",
                "role": "assistant", "content": [part]
            });
            events.extend(self.emit("response.output_item.done", json!({
                "output_index": item.output_index, "item": completed
            })));
            if self.output_items.len() < MAX_OUTPUT_ITEMS {
                self.output_items.push(completed);
            } else {
                log::warn!("output_items exceeded {} limit, dropping item", MAX_OUTPUT_ITEMS);
            }
            return events;
        }
        Vec::new()
    }

    fn open_tool_call_item(&mut self, tool_id: &str, name: &str) -> Vec<Bytes> {
        let idx = self.next_output_index();
        self.current_tool = Some(ToolCallState {
            id: tool_id.to_string(),
            name: name.to_string(),
            args: String::new(),
            output_index: idx,
            added: true,
        });
        self.phase = Phase::ToolCall;
        vec![self.emit("response.output_item.added", json!({
            "output_index": idx,
            "item": {"id": tool_id, "type": "function_call", "call_id": tool_id, "name": name, "arguments": "", "status": "in_progress"}
        })).unwrap()]
    }

    fn append_tool_call_args(&mut self, delta: &str) -> Vec<Bytes> {
        let (output_index, item_id) = if let Some(ref tc) = self.current_tool {
            (tc.output_index, tc.id.clone())
        } else {
            return Vec::new();
        };
        if let Some(ref mut tc) = self.current_tool {
            if tc.args.len() + delta.len() <= MAX_TEXT_LEN {
                tc.args.push_str(delta);
            }
        }
        vec![self.emit("response.function_call_arguments.delta", json!({
            "output_index": output_index, "item_id": item_id, "delta": delta
        })).unwrap()]
    }

    fn close_tool_call_item(&mut self) -> Vec<Bytes> {
        if let Some(tc) = self.current_tool.take() {
            let mut events = Vec::new();
            events.extend(self.emit("response.function_call_arguments.done", json!({
                "output_index": tc.output_index, "item_id": tc.id, "arguments": tc.args
            })));
            let completed = json!({
                "id": tc.id, "type": "function_call", "call_id": tc.id,
                "name": tc.name, "arguments": tc.args, "status": "completed"
            });
            events.extend(self.emit("response.output_item.done", json!({
                "output_index": tc.output_index, "item": completed
            })));
            if self.output_items.len() < MAX_OUTPUT_ITEMS {
                self.output_items.push(completed);
            } else {
                log::warn!("output_items exceeded {} limit, dropping item", MAX_OUTPUT_ITEMS);
            }
            return events;
        }
        Vec::new()
    }

    fn close_current(&mut self) -> Vec<Bytes> {
        let mut events = Vec::new();
        match self.phase {
            Phase::Reasoning => events.extend(self.close_reasoning_item()),
            Phase::Message => events.extend(self.close_message_item()),
            _ => {}
        }
        self.phase = Phase::Idle;
        events
    }

    fn finalize(&mut self) -> Vec<Bytes> {
        if self.stream_terminated {
            return Vec::new();
        }
        self.stream_terminated = true;
        let mut events = self.close_current();
        events.extend(self.close_tool_call_item());

        let usage = json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens
        });
        let msg_outputs: Vec<&Value> = self.output_items.iter()
            .filter(|it| it.get("type").and_then(|v| v.as_str()) == Some("message"))
            .collect();
        let empty_content = Vec::new();
        let output_text: String = msg_outputs.iter()
            .flat_map(|it| it.get("content").and_then(|v| v.as_array()).unwrap_or(&empty_content).iter())
            .filter_map(|c| if c.get("type").and_then(|v| v.as_str()) == Some("output_text") { c.get("text").and_then(|v| v.as_str()) } else { None })
            .collect();

        let response = json!({
            "id": self.response_id, "object": "response", "created_at": self.created_at,
            "status": "completed", "model": self.model, "output": self.output_items,
            "output_text": output_text, "usage": usage
        });
        events.extend(self.emit("response.completed", response));
        events
    }

    fn failed_event(&mut self, message: String, error_type: Option<String>) -> Bytes {
        self.stream_terminated = true;
        Bytes::from(format!(
            "event: response.failed\ndata: {}\n\n",
            json!({
                "response": {"id": self.response_id, "object": "response", "status": "failed",
                    "error": {"code": error_type.unwrap_or_else(|| "unknown".to_string()), "message": message}}
            })
        ))
    }

    /// Handle an Anthropic SSE event and return Responses SSE events.
    fn handle_anthropic_event(&mut self, event_type: &str, data: &Value) -> Vec<Bytes> {
        let mut events = self.ensure_response_started();

        match event_type {
            "message_start" => {
                if let Some(msg) = data.get("message") {
                    if let Some(model) = msg.get("model").and_then(|v| v.as_str()) {
                        self.model = model.to_string();
                    }
                    if let Some(usage) = msg.get("usage") {
                        self.input_tokens = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                        self.output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                    }
                }
            }
            "content_block_start" => {
                let empty_obj = json!({});
                let block = data.get("content_block").unwrap_or(&empty_obj);
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
                let empty_obj = json!({});
                let delta = data.get("delta").unwrap_or(&empty_obj);
                let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match delta_type {
                    "thinking_delta" => {
                        if self.phase != Phase::Reasoning {
                            events.extend(self.close_current());
                            events.extend(self.start_reasoning_item());
                        }
                        if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                            events.extend(self.append_reasoning_delta(text));
                        }
                    }
                    "text_delta" => {
                        if self.phase != Phase::Message {
                            events.extend(self.close_current());
                            events.extend(self.close_tool_call_item());
                            events.extend(self.start_message_item());
                        }
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            events.extend(self.append_message_delta(text));
                        }
                    }
                    "input_json_delta" => {
                        if self.phase != Phase::ToolCall {
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
                self.phase = Phase::Idle;
            }
            "message_delta" => {
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
}

/// Create a Responses API SSE stream from an Anthropic Messages SSE stream.
pub fn create_responses_sse_stream_from_anthropic<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();
        let mut state = AnthropicToResponsesState {
            response_id: format!("resp_{}", now_ms()),
            created_at: now_ms() / 1000,
            ..Default::default()
        };
        let mut stream_failed = false;
        const MAX_BUFFER_SIZE: usize = 50 * 1024 * 1024; // 50 MB

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    crate::proxy::sse::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    if buffer.len() > MAX_BUFFER_SIZE {
                        log::error!("[Anthropic→Responses] SSE buffer exceeded {} bytes, aborting", MAX_BUFFER_SIZE);
                        yield Ok(state.failed_event("SSE buffer overflow".to_string(), Some("buffer_overflow".to_string())));
                        stream_failed = true;
                        break;
                    }

                    while let Some(block) = crate::proxy::sse::take_sse_block(&mut buffer) {
                        if block.trim().is_empty() {
                            continue;
                        }

                        let mut event_name: Option<String> = None;
                        let mut data_parts: Vec<String> = Vec::new();
                        for line in block.lines() {
                            if let Some(event) = crate::proxy::sse::strip_sse_field(line, "event") {
                                event_name = Some(event.trim().to_string());
                            }
                            if let Some(data) = crate::proxy::sse::strip_sse_field(line, "data") {
                                data_parts.push(data.to_string());
                            }
                        }

                        if data_parts.is_empty() {
                            continue;
                        }

                        let data = data_parts.join("\n");
                        if data.trim() == "[DONE]" {
                            for event in state.finalize() {
                                yield Ok(event);
                            }
                            continue;
                        }

                        let chunk: Value = match serde_json::from_str(&data) {
                            Ok(value) => value,
                            Err(_) => continue,
                        };

                        if event_name.as_deref() == Some("error") || chunk.get("error").is_some() {
                            let message = chunk.get("error")
                                .and_then(|e| e.get("message").or_else(|| e.get("detail")))
                                .and_then(|v| v.as_str())
                                .unwrap_or("Unknown error");
                            yield Ok(state.failed_event(message.to_string(), None));
                            stream_failed = true;
                            break;
                        }

                        let evt_type = event_name.as_deref().unwrap_or("");
                        for event in state.handle_anthropic_event(evt_type, &chunk) {
                            yield Ok(event);
                        }
                    }

                    if stream_failed {
                        break;
                    }
                }
                Err(e) => {
                    yield Ok(state.failed_event(
                        format!("Stream error: {e}"),
                        Some("stream_error".to_string()),
                    ));
                    stream_failed = true;
                    break;
                }
            }
        }

        if !stream_failed {
            for event in state.finalize() {
                yield Ok(event);
            }
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
