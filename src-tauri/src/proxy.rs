use crate::config::{AppState, ProviderFormat, RequestLog};
use crate::convert;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use http_body_util::BodyExt;
use serde::Serialize;
use serde_json::{json, Value};

/// Extract tag from Claude model name by keyword matching.
/// e.g. "claude-opus-4-8" → "opus", "claude-sonnet-4-6" → "sonnet", "claude-haiku-4-5" → "haiku"
/// Also handles plain values like "opus", "sonnet", "haiku" (set via ANTHROPIC_DEFAULT_*_MODEL).

/// Forward client headers to the upstream request, filtering out
/// hop-by-hop headers, auth headers we already set, and
/// anthropic-beta thinking flags that most providers don't support.
fn forward_client_headers(
    headers: &HeaderMap,
    req_builder: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    let mut builder = req_builder;
    for (name, value) in headers.iter() {
        let lower = name.as_str().to_lowercase();
        if matches!(
            lower.as_str(),
            "host" | "content-length" | "transfer-encoding" | "connection"
                | "accept-encoding" | "authorization" | "x-api-key" | "x-goog-api-key"
        ) {
            continue;
        }
        if lower == "anthropic-beta" {
            if let Ok(v) = value.to_str() {
                let filtered: Vec<&str> = v
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.contains("thinking"))
                    .collect();
                if !filtered.is_empty() {
                    builder = builder.header("anthropic-beta", filtered.join(", "));
                }
                continue;
            }
        }
        if let Ok(v) = value.to_str() {
            builder = builder.header(name.as_str(), v);
        }
    }
    builder
}

fn resolve_tag_from_model(model: &str) -> Option<String> {
    let lower = model.to_lowercase();
    // Split by common separators for component-level matching,
    // avoiding false positives like "octopus" matching "opus".
    let parts: Vec<&str> = lower.split(&['-', '_', '.', ' ', '/'][..]).collect();

    // Direct tag matches (opus, sonnet, haiku)
    if parts.contains(&"opus") {
        return Some("opus".to_string());
    }
    if parts.contains(&"sonnet") {
        return Some("sonnet".to_string());
    }
    if parts.contains(&"haiku") {
        return Some("haiku".to_string());
    }

    // Provider-specific heuristics — only applied when no explicit tag.
    // These are opinionated quality-to-tag mappings and can be overridden
    // by setting ANTHROPIC_DEFAULT_*_MODEL to an explicit tag name.
    if parts.contains(&"glm") {
        Some("sonnet".to_string())
    } else if parts.contains(&"deepseek") {
        Some("haiku".to_string())
    } else if parts.iter().any(|p| *p == "k2" || p.starts_with("kimi")) {
        Some("sonnet".to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Anthropic client handlers
// ---------------------------------------------------------------------------

pub async fn handle_anthropic_messages(
    state: State<AppState>,
    request: Request,
) -> Result<Response, ProxyError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let body = parse_body(body).await?;
    handle_proxy("anthropic", state, headers, axum::Json(body)).await
}

pub async fn handle_anthropic_count_tokens(
    state: State<AppState>,
    request: Request,
) -> Result<Response, ProxyError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let body = parse_body(body).await?;
    handle_count_tokens("anthropic", state, headers, axum::Json(body)).await
}

// ---------------------------------------------------------------------------
// OpenAI client handlers
// ---------------------------------------------------------------------------

pub async fn handle_openai_chat(
    state: State<AppState>,
    request: Request,
) -> Result<Response, ProxyError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let body = parse_body(body).await?;
    handle_proxy("openai", state, headers, axum::Json(body)).await
}

pub async fn handle_openai_responses(
    state: State<AppState>,
    request: Request,
) -> Result<Response, ProxyError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let body = parse_body(body).await?;
    handle_proxy("openai_responses", state, headers, axum::Json(body)).await
}

/// Parse request body as JSON, accepting any content-type.
/// More forgiving than `axum::Json` which requires `Content-Type: application/json`.
async fn parse_body(body: Body) -> Result<Value, ProxyError> {
    let bytes = body
        .collect()
        .await
        .map_err(|e| ProxyError::Upstream(format!("failed to read request body: {}", e)))?
        .to_bytes();
    if bytes.is_empty() {
        return Ok(Value::Object(serde_json::Map::new()));
    }
    serde_json::from_slice(&bytes)
        .map_err(|e| ProxyError::Upstream(format!("invalid JSON body: {}", e)))
}

// ---------------------------------------------------------------------------
// Core proxy logic (protocol-aware)
// ---------------------------------------------------------------------------

async fn handle_proxy(
    client_protocol: &str,
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::Json<Value>,
) -> Result<Response, ProxyError> {
    let body = body.0;
    let request_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    let msg_count = body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let is_streaming = body
        .as_object()
        .and_then(|o| o.get("stream"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    log::info!(
        "[Proxy] incoming request: protocol={}, model={}, {} messages, stream={}",
        client_protocol,
        request_model,
        msg_count,
        is_streaming
    );

    let config = state.config.read().await;

    // 1. Resolve tag from model
    // Only opus/sonnet/haiku/auto are recognized tags
    // Any other model name (e.g. "glm-5.1" from upstream feedback) falls back to auto
    let tag = resolve_tag_from_model(&request_model)
        .unwrap_or_else(|| config.current_tag.clone());

    // 2. Find matching route (with fallback to auto then first route)
    let route = config
        .routes
        .iter()
        .find(|r| r.tags.contains(&tag))
        .or_else(|| {
            log::warn!("[Proxy] no route for tag '{}', falling back to auto/default", tag);
            config.routes.iter().find(|r| r.tags.iter().any(|t| t == "auto"))
        })
        .or_else(|| config.routes.first())
        .ok_or_else(|| ProxyError::NoRoute(tag.clone()))?;

    let provider_format = &route.format;

    let provider = config
        .providers
        .get(&route.provider)
        .ok_or_else(|| ProxyError::NoProvider(route.provider.clone()))?;

    // 3. Validate provider API key
    if provider.api_key.is_empty() || provider.api_key.starts_with("sk-your-") {
        return Err(ProxyError::NoProvider(format!(
            "provider '{}' has no valid API key configured",
            route.provider
        )));
    }

    log::info!(
        "[Proxy] {} → {} (model={}, format={:?})",
        tag,
        provider.name,
        route.model,
        provider_format
    );

    // Record request log
    let log_entry = RequestLog {
        request_model: request_model.clone(),
        tag: tag.clone(),
        provider: provider.name.clone(),
        target_model: route.model.clone(),
        timestamp: chrono_now(),
    };
    state.log_request(log_entry).await;

    // 4. Build forwarded request body based on protocol conversion
    let fwd_body = match (client_protocol, provider_format) {
        ("anthropic", ProviderFormat::Anthropic) => {
            // Anthropic → Anthropic: passthrough with fixes
            // Note: don't strip_thinking here — Anthropic-compatible providers
            // like Zhipu support thinking blocks
            let mut b = body.clone();
            b["model"] = Value::String(route.model.clone());
            normalize_roles(&mut b);
            b
        }
        ("anthropic", ProviderFormat::Openai) => {
            // Anthropic → OpenAI Chat Completions
            // Note: don't strip_thinking here — thinking blocks are converted
            // to reasoning_content in anthropic_to_openai_request
            let mut b = body.clone();
            normalize_roles(&mut b);
            convert::anthropic_to_openai_request(&b, &route.model)
        }
        ("anthropic", ProviderFormat::OpenaiResponses) => {
            // Anthropic → OpenAI Responses API
            let mut b = body.clone();
            normalize_roles(&mut b);
            strip_thinking(&mut b);
            convert::anthropic_to_responses_request(&b, &route.model)
        }
        ("openai", ProviderFormat::Openai) => {
            // OpenAI Chat → OpenAI Chat: passthrough
            let mut b = body.clone();
            b["model"] = Value::String(route.model.clone());
            b
        }
        ("openai", ProviderFormat::OpenaiResponses) => {
            // OpenAI Chat → OpenAI Responses
            convert::openai_to_responses_request(&body, &route.model)
        }
        ("openai", ProviderFormat::Anthropic) => {
            // OpenAI Chat → Anthropic Messages
            convert::openai_to_anthropic_request(&body, &route.model)
        }
        ("openai_responses", ProviderFormat::OpenaiResponses) => {
            // Responses → Responses: passthrough
            let mut b = body.clone();
            b["model"] = Value::String(route.model.clone());
            b
        }
        ("openai_responses", ProviderFormat::Openai) => {
            // Codex Responses → OpenAI Chat Completions provider
            convert::responses_to_chat_request(&body, &route.model)
        }
        ("openai_responses", ProviderFormat::Anthropic) => {
            // Codex Responses → Anthropic Messages provider
            convert::responses_to_anthropic_request(&body, &route.model)
        }
        _ => {
            let mut b = body.clone();
            b["model"] = Value::String(route.model.clone());
            normalize_roles(&mut b);
            strip_thinking(&mut b);
            b
        }
    };

    // 5. Build URL
    let url = format!(
        "{}{}",
        provider.base_url.trim_end_matches('/'),
        route.endpoint
    );
    log::info!("[Proxy] forwarding to {} (model={})", url, route.model);

    // 6. Build headers
    let mut req_builder = state.http_client.post(&url);

    // DEBUG: log the forwarded body (truncated) to help diagnose validation errors
    if let Ok(s) = serde_json::to_string(&fwd_body) {
        let truncated = if s.chars().count() > 200 {
            format!("{}...(truncated)", s.chars().take(200).collect::<String>())
        } else {
            s.clone()
        };
        log::info!("[Proxy] forwarding body: {}", truncated);
    }

    // Auth
    req_builder = req_builder.header(
        provider.auth_type.header_name(),
        provider.auth_type.header_value(&provider.api_key),
    );

    // Forward client headers (hop-by-hop and auth headers are filtered internally)
    req_builder = forward_client_headers(&headers, req_builder);

    // 7. Send
    let resp = req_builder
        .json(&fwd_body)
        .timeout(std::time::Duration::from_secs(if is_streaming { 3600 } else { 300 }))
        .send()
        .await
        .map_err(|e| ProxyError::Upstream(e.to_string()))?;

    let status = resp.status();
    if !status.is_success() {
        let status_code = status.as_u16();
        let err_body = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[Proxy] failed to read upstream error body: {}", e);
                String::new()
            }
        };
        log::warn!(
            "[Proxy] <<< {} {}: {}",
            status_code,
            status.canonical_reason().unwrap_or("?"),
            &err_body[..err_body.len().min(300)]
        );
        let axum_status =
            StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_GATEWAY);
        return Ok((
            axum_status,
            [("content-type", "application/json")],
            err_body,
        )
            .into_response());
    }

    // 8. Convert response if needed
    if is_streaming {
        match (client_protocol, provider_format) {
            ("anthropic", ProviderFormat::Openai) => {
                // Convert OpenAI Chat SSE → Anthropic SSE
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = convert::convert_openai_stream_to_anthropic(
                    Box::pin(stream),
                    request_model,
                );
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            ("anthropic", ProviderFormat::OpenaiResponses) => {
                // Convert OpenAI Responses SSE → Anthropic SSE
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = convert::convert_responses_stream_to_anthropic(
                    Box::pin(stream),
                    request_model,
                );
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            ("openai_responses", ProviderFormat::Openai) => {
                // Convert OpenAI Chat SSE → Responses SSE (for Codex)
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = convert::convert_chat_stream_to_responses(
                    Box::pin(stream),
                    &request_model,
                );
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            ("openai_responses", ProviderFormat::Anthropic) => {
                // Convert Anthropic SSE → Responses SSE (for Codex)
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = convert::convert_anthropic_stream_to_responses(
                    Box::pin(stream),
                    &request_model,
                );
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            ("openai", ProviderFormat::OpenaiResponses) => {
                // Convert OpenAI Responses SSE → OpenAI Chat SSE
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = convert::convert_responses_stream_to_chat(
                    Box::pin(stream),
                    &request_model,
                );
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            ("openai", ProviderFormat::Anthropic) => {
                // Convert Anthropic SSE → OpenAI Chat SSE
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = convert::convert_anthropic_stream_to_openai(
                    Box::pin(stream),
                    request_model.clone(),
                );
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            ("anthropic", ProviderFormat::Anthropic)
            | ("openai", ProviderFormat::Openai)
            | ("openai_responses", ProviderFormat::OpenaiResponses) => {
                // Passthrough: forward raw bytes without JSON parsing
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let body = Body::from_stream(stream);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
            _ => {
                // Passthrough streaming with model preservation
                let stream = resp.bytes_stream().map(|result| {
                    result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                });
                let converted = replace_model_in_anthropic_stream(Box::pin(stream), request_model);
                let body = Body::from_stream(converted);

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .body(body)
                    .map_err(|e| ProxyError::Upstream(format!("failed to build streaming response: {}", e)))?)
            }
        }
    } else {
        // Non-streaming
        let status_code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK);
        let resp_body = resp
            .bytes()
            .await
            .map_err(|e| ProxyError::Upstream(e.to_string()))?;

        match (client_protocol, provider_format) {
            ("anthropic", ProviderFormat::Openai) => {
                // Convert OpenAI Chat response → Anthropic response
                let openai_resp: Value = parse_upstream_json(&resp_body)?;
                let anthropic_resp = convert::openai_to_anthropic_response(&openai_resp, &request_model);
                let anthropic_bytes = serde_json::to_vec(&anthropic_resp).unwrap_or(resp_body.to_vec());
                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    anthropic_bytes,
                )
                    .into_response())
            }
            ("anthropic", ProviderFormat::OpenaiResponses) => {
                // Convert OpenAI Responses response → Anthropic response
                let responses_resp: Value = parse_upstream_json(&resp_body)?;
                let anthropic_resp = convert::responses_to_anthropic_response(&responses_resp, &request_model);
                let anthropic_bytes = serde_json::to_vec(&anthropic_resp).unwrap_or(resp_body.to_vec());
                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    anthropic_bytes,
                )
                    .into_response())
            }
            ("openai", ProviderFormat::OpenaiResponses) => {
                // Convert OpenAI Responses response → OpenAI Chat response
                let responses_resp: Value = parse_upstream_json(&resp_body)?;
                let openai_resp = convert::responses_to_openai_response(&responses_resp, &request_model);
                let openai_bytes = serde_json::to_vec(&openai_resp).unwrap_or(resp_body.to_vec());
                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    openai_bytes,
                )
                    .into_response())
            }
            ("openai", ProviderFormat::Anthropic) => {
                // Convert Anthropic response → OpenAI Chat response
                let anthropic_resp: Value = parse_upstream_json(&resp_body)?;
                let openai_resp = convert::anthropic_to_openai_response(&anthropic_resp, &request_model);
                let openai_bytes = serde_json::to_vec(&openai_resp).unwrap_or(resp_body.to_vec());
                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    openai_bytes,
                )
                    .into_response())
            }
            ("openai_responses", ProviderFormat::Openai) => {
                // Convert OpenAI Chat response → Responses API response (for Codex)
                let chat_resp: Value = parse_upstream_json(&resp_body)?;
                let responses_resp = convert::chat_to_responses_response(&chat_resp, &request_model);
                let bytes = serde_json::to_vec(&responses_resp).unwrap_or(resp_body.to_vec());
                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    bytes,
                )
                    .into_response())
            }
            ("openai_responses", ProviderFormat::Anthropic) => {
                // Convert Anthropic response → Responses API response (for Codex)
                let anthropic_resp: Value = parse_upstream_json(&resp_body)?;
                let responses_resp = convert::anthropic_to_responses_response(&anthropic_resp, &request_model);
                let bytes = serde_json::to_vec(&responses_resp).unwrap_or(resp_body.to_vec());
                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    bytes,
                )
                    .into_response())
            }
            _ => {
                // Passthrough response with model preservation
                // Parse and replace model field to prevent client feedback loop
                let mut resp_value: Value = parse_upstream_json(&resp_body)?;

                // Override the model field to prevent client from updating its model
                // to provider's model name (e.g., prevent "glm-5.1" from replacing "claude-opus-4-8")
                if let Some(obj) = resp_value.as_object_mut() {
                    obj.insert("model".to_string(), Value::String(request_model.clone()));
                }

                let resp_bytes = serde_json::to_vec(&resp_value).unwrap_or(resp_body.to_vec());

                Ok((
                    status_code,
                    [("content-type", "application/json")],
                    resp_bytes,
                )
                    .into_response())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Count tokens handler
// ---------------------------------------------------------------------------

async fn handle_count_tokens(
    client_protocol: &str,
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::Json<Value>,
) -> Result<Response, ProxyError> {
    let body = body.0;
    let request_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    log::info!("[Proxy] count_tokens request: protocol={}, model={}", client_protocol, request_model);

    let config = state.config.read().await;

    let tag = resolve_tag_from_model(&request_model)
        .unwrap_or_else(|| config.current_tag.clone());

    let route = config
        .routes
        .iter()
        .find(|r| r.tags.contains(&tag))
        .or_else(|| {
            log::warn!("[Proxy] no route for tag '{}', falling back to auto/default", tag);
            config.routes.iter().find(|r| r.tags.iter().any(|t| t == "auto"))
        })
        .or_else(|| config.routes.first())
        .ok_or_else(|| ProxyError::NoRoute(tag.clone()))?;

    let provider = config
        .providers
        .get(&route.provider)
        .ok_or_else(|| ProxyError::NoProvider(route.provider.clone()))?;

    if provider.api_key.is_empty() || provider.api_key.starts_with("sk-your-") {
        return Err(ProxyError::NoProvider(format!(
            "provider '{}' has no valid API key configured",
            route.provider
        )));
    }

    log::info!(
        "[Proxy] count_tokens: {} → {} (model={}, format={:?})",
        tag,
        provider.name,
        route.model,
        route.format
    );

    // Apply protocol conversion based on (client_protocol, provider_format)
    let fwd_body = match (client_protocol, &route.format) {
        ("anthropic", ProviderFormat::Anthropic) => {
            let mut b = body.clone();
            b["model"] = Value::String(route.model.clone());
            normalize_roles(&mut b);
            b
        }
        ("anthropic", ProviderFormat::Openai) => {
            let mut b = body.clone();
            normalize_roles(&mut b);
            convert::anthropic_to_openai_request(&b, &route.model)
        }
        ("anthropic", ProviderFormat::OpenaiResponses) => {
            let mut b = body.clone();
            normalize_roles(&mut b);
            strip_thinking(&mut b);
            convert::anthropic_to_responses_request(&b, &route.model)
        }
        _ => {
            return Err(ProxyError::Upstream(format!(
                "count_tokens not supported for protocol {} with format {:?}",
                client_protocol, route.format
            )));
        }
    };

    let url = format!(
        "{}{}/count_tokens",
        provider.base_url.trim_end_matches('/'),
        route.endpoint
    );
    log::info!("[Proxy] count_tokens forwarding to {}", url);

    let mut req_builder = state.http_client.post(&url);
    req_builder = req_builder.header(
        provider.auth_type.header_name(),
        provider.auth_type.header_value(&provider.api_key),
    );

    req_builder = forward_client_headers(&headers, req_builder);

    let resp = req_builder
        .json(&fwd_body)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| ProxyError::Upstream(e.to_string()))?;

    let status = resp.status();
    let status_code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK);
    let resp_body = resp
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(e.to_string()))?;

    if !status.is_success() {
        let body_str = String::from_utf8_lossy(&resp_body);
        log::warn!(
            "[Proxy] count_tokens <<< {} {}: {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or("?"),
            &body_str[..body_str.len().min(300)]
        );
    }

    Ok((
        status_code,
        [("content-type", "application/json")],
        resp_body.to_vec(),
    )
        .into_response())
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

use bytes::Bytes;
use futures::stream::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Stream converter that replaces the model field in SSE message_start events.
/// This prevents the client from updating its model to the provider's model name.
fn replace_model_in_anthropic_stream(
    upstream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    request_model: String,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let buffer = Arc::new(Mutex::new(String::new()));

    Box::pin(upstream.flat_map(move |chunk_result| {
        let buffer = buffer.clone();
        let request_model = request_model.clone();

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

            // Process each line
            while let Some(newline_pos) = local_buf.find('\n') {
                let line = local_buf[..newline_pos].to_string();
                local_buf = local_buf[newline_pos + 1..].to_string();

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    yield Ok(Bytes::from("\n"));
                    continue;
                }

                // Try to parse as SSE data
                let data = if let Some(stripped) = trimmed.strip_prefix("data: ") {
                    stripped.to_string()
                } else if let Some(stripped) = trimmed.strip_prefix("data:") {
                    stripped.trim().to_string()
                } else {
                    // Not a data line, pass through
                    yield Ok(Bytes::from(format!("{}\n", line)));
                    continue;
                };

                if data == "[DONE]" {
                    yield Ok(Bytes::from("data: [DONE]\n\n"));
                    continue;
                }

                let mut parsed: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => {
                        // Not valid JSON, pass through
                        yield Ok(Bytes::from(format!("data: {}\n\n", data)));
                        continue;
                    }
                };

                // Replace model in message_start events and message.message
                if parsed.get("type").and_then(|t| t.as_str()) == Some("message_start") {
                    if let Some(msg) = parsed.get_mut("message").and_then(|m| m.as_object_mut()) {
                        msg.insert("model".to_string(), Value::String(request_model.clone()));
                    }
                }
                // Also handle direct message objects in the stream
                if parsed.get("message").is_some() {
                    if let Some(msg) = parsed.get_mut("message").and_then(|m| m.as_object_mut()) {
                        if msg.get("model").is_some() {
                            msg.insert("model".to_string(), Value::String(request_model.clone()));
                        }
                    }
                }

                yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&parsed).unwrap_or_default())));
            }

            // Save remaining buffer
            *buffer.lock().await = local_buf;
        }
    }))
}

// ---------------------------------------------------------------------------
// Body transformation helpers (for Anthropic passthrough)
// ---------------------------------------------------------------------------
fn normalize_roles(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(messages) = map.get_mut("messages").and_then(|m| m.as_array_mut()) {
                for msg in messages.iter_mut() {
                    if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                        let normalized = match role {
                            "assistant" | "user" => role.to_string(),
                            "tool" => "assistant".to_string(),
                            _ => "user".to_string(),
                        };
                        if normalized != role {
                            msg["role"] = Value::String(normalized);
                        }
                    }

                    // Ensure content is not null: Anthropic requires either a string
                    // or an array of content blocks. Normalize `null` -> empty string,
                    // and convert non-string/non-array content into a text block.
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

                    normalize_roles(msg);
                }
                return;
            }
            for v in map.values_mut() {
                normalize_roles(v);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                normalize_roles(item);
            }
        }
        _ => {}
    }
}

/// Strip "thinking" blocks from assistant content arrays
fn strip_thinking(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            if let Some(messages) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
                for msg in messages.iter_mut() {
                    if msg.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                        if let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
                            content.retain(|block| {
                                block.get("type").and_then(|t| t.as_str()) != Some("thinking")
                            });
                        }
                    }
                }
            }
            for v in obj.values_mut() {
                strip_thinking(v);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                strip_thinking(item);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Test endpoint — send a test prompt through the full pipeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct TestResult {
    pub success: bool,
    pub tag: String,
    pub provider: String,
    pub model: String,
    pub format: String,
    pub latency_ms: u64,
    pub error: Option<String>,
    pub response: Option<Value>,
}

pub async fn test_route(
    state: &AppState,
    tag: &str,
    prompt: &str,
) -> TestResult {
    let config = state.config.read().await;

    // Find route by tag
    let route = match config.routes.iter().find(|r| r.tags.contains(&tag.to_string())) {
        Some(r) => r,
        None => {
            return TestResult {
                success: false,
                tag: tag.to_string(),
                provider: String::new(),
                model: String::new(),
                format: String::new(),
                latency_ms: 0,
                error: Some(format!("no route found for tag '{}'", tag)),
                response: None,
            }
        }
    };

    let provider = match config.providers.get(&route.provider) {
        Some(p) => p,
        None => {
            return TestResult {
                success: false,
                tag: tag.to_string(),
                provider: String::new(),
                model: route.model.clone(),
                format: format!("{:?}", route.format),
                latency_ms: 0,
                error: Some(format!("provider '{}' not found", route.provider)),
                response: None,
            }
        }
    };

    // Validate API key
    if provider.api_key.is_empty() || provider.api_key.starts_with("sk-your-") {
        return TestResult {
            success: false,
            tag: tag.to_string(),
            provider: provider.name.clone(),
            model: route.model.clone(),
            format: format!("{:?}", route.format),
            latency_ms: 0,
            error: Some(format!(
                "provider '{}' has no valid API key (key='{}...')",
                route.provider,
                &provider.api_key[..provider.api_key.len().min(8)]
            )),
            response: None,
        };
    }

    // Construct a minimal Anthropic Messages request
    let anthropic_body = json!({
        "model": tag,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    });

    // Convert based on provider format
    let fwd_body = match route.format {
        ProviderFormat::Anthropic => {
            // Anthropic provider: don't strip thinking, provider supports it
            let mut b = anthropic_body.clone();
            b["model"] = Value::String(route.model.clone());
            normalize_roles(&mut b);
            b
        }
        ProviderFormat::Openai => {
            // OpenAI Chat: conversion function handles thinking → reasoning_content
            let mut b = anthropic_body.clone();
            normalize_roles(&mut b);
            convert::anthropic_to_openai_request(&b, &route.model)
        }
        ProviderFormat::OpenaiResponses => {
            let mut b = anthropic_body.clone();
            normalize_roles(&mut b);
            strip_thinking(&mut b);
            convert::anthropic_to_responses_request(&b, &route.model)
        }
    };

    // Build URL
    let url = format!(
        "{}{}",
        provider.base_url.trim_end_matches('/'),
        route.endpoint
    );
    log::info!("[Test] testing route tag={} → {} {} (format={:?})", tag, url, route.model, route.format);

    if let Ok(s) = serde_json::to_string(&fwd_body) {
        let truncated = if s.len() > 500 { format!("{}...(truncated)", &s[..500]) } else { s };
        log::info!("[Test] forwarding body: {}", truncated);
    }

    // Send request
    let start = std::time::Instant::now();
    let resp = match state
        .http_client
        .post(&url)
        .header(
            provider.auth_type.header_name(),
            provider.auth_type.header_value(&provider.api_key),
        )
        .header("content-type", "application/json")
        .json(&fwd_body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return TestResult {
                success: false,
                tag: tag.to_string(),
                provider: provider.name.clone(),
                model: route.model.clone(),
                format: format!("{:?}", route.format),
                latency_ms: start.elapsed().as_millis() as u64,
                error: Some(format!("connection error: {}", e)),
                response: None,
            }
        }
    };

    let latency_ms = start.elapsed().as_millis() as u64;
    let status = resp.status();
    let resp_text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            return TestResult {
                success: false,
                tag: tag.to_string(),
                provider: provider.name.clone(),
                model: route.model.clone(),
                format: format!("{:?}", route.format),
                latency_ms,
                error: Some(format!("read response error: {}", e)),
                response: None,
            }
        }
    };

    if !status.is_success() {
        let err_preview = &resp_text[..resp_text.len().min(500)];
        log::warn!("[Test] <<< HTTP {}: {}", status, err_preview);
        return TestResult {
            success: false,
            tag: tag.to_string(),
            provider: provider.name.clone(),
            model: route.model.clone(),
            format: format!("{:?}", route.format),
            latency_ms,
            error: Some(format!("HTTP {}: {}", status, err_preview)),
            response: None,
        };
    }

    // Convert response back to Anthropic format
    let response = match route.format {
        ProviderFormat::Openai => {
            let openai_resp: Value =
                serde_json::from_str(&resp_text).map_err(|e| { log::warn!("[Test] failed to parse response as JSON: {}", e); Value::Null }).unwrap_or(Value::Null);
            convert::openai_to_anthropic_response(&openai_resp, tag)
        }
        ProviderFormat::OpenaiResponses => {
            let responses_resp: Value =
                serde_json::from_str(&resp_text).map_err(|e| { log::warn!("[Test] failed to parse response as JSON: {}", e); Value::Null }).unwrap_or(Value::Null);
            convert::responses_to_anthropic_response(&responses_resp, tag)
        }
        ProviderFormat::Anthropic => {
            serde_json::from_str(&resp_text).map_err(|e| { log::warn!("[Test] failed to parse response as JSON: {}", e); Value::Null }).unwrap_or(Value::Null)
        }
    };

    log::info!("[Test] ✓ success in {}ms", latency_ms);

    TestResult {
        success: true,
        tag: tag.to_string(),
        provider: provider.name.clone(),
        model: route.model.clone(),
        format: format!("{:?}", route.format),
        latency_ms,
        error: None,
        response: Some(response),
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("no route for tag '{0}'")]
    NoRoute(String),
    #[error("provider '{0}' not found")]
    NoProvider(String),
    #[error("upstream error: {0}")]
    Upstream(String),
}


/// Parse upstream response body as JSON, returning a proper error on failure
/// instead of silently degrading to Value::Null.
fn parse_upstream_json(resp_body: &[u8]) -> Result<Value, ProxyError> {
    serde_json::from_slice(resp_body)
        .map_err(|e| ProxyError::Upstream(format!("failed to parse upstream response: {}", e)))
}


fn chrono_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            ProxyError::NoRoute(_) | ProxyError::NoProvider(_) => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            ProxyError::Upstream(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
        };
        let body = serde_json::json!({ "error": msg });
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_normalize_roles_converts_system_to_user() {
        let mut body = json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        });
        normalize_roles(&mut body);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
    }

    #[test]
    fn test_normalize_roles_preserves_user_and_assistant() {
        let mut body = json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        });
        normalize_roles(&mut body);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
    }

    #[test]
    fn test_normalize_roles_with_tool_messages() {
        let mut body = json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user msg"},
                {"role": "assistant", "content": [{"type": "tool_use", "name": "bash"}]},
                {"role": "user", "content": [{"type": "tool_result"}]}
            ]
        });
        normalize_roles(&mut body);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user", "system should become user");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[3]["role"], "user");
    }

    #[test]
    fn test_strip_thinking_array_content() {
        let mut body = json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "inner monologue"},
                    {"type": "text", "text": "Hello"}
                ]}
            ]
        });
        strip_thinking(&mut body);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
    }

    #[test]
    fn test_strip_thinking_string_content_noop() {
        let mut body = json!({
            "messages": [
                {"role": "assistant", "content": "plain text response"}
            ]
        });
        strip_thinking(&mut body);
        assert_eq!(body["messages"][0]["content"], "plain text response");
    }
}
