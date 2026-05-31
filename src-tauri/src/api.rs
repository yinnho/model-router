use crate::config::{save_config, AppConfig, AppState, RequestLog};
use crate::proxy::{self, TestResult};
use crate::takeover::{check_takeover_status, restore_claude, take_over_claude};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct StatusResponse {
    pub current_tag: String,
    pub takeover: TakeoverInfo,
}

#[derive(Serialize)]
pub struct TakeoverInfo {
    pub active: bool,
    pub proxy_url: Option<String>,
}

fn validate_config(config: &AppConfig) -> Result<(), String> {
    if config.port == 0 {
        return Err("port cannot be 0".to_string());
    }

    for (id, provider) in &config.providers {
        if provider.base_url.trim().is_empty() {
            return Err(format!("provider '{}' has empty base_url", id));
        }
        if provider.api_key.trim().is_empty() {
            return Err(format!("provider '{}' has empty api_key", id));
        }
    }

    for (i, route) in config.routes.iter().enumerate() {
        if route.model.trim().is_empty() {
            return Err(format!("route #{} has empty model", i + 1));
        }
        if route.provider.trim().is_empty() {
            return Err(format!("route #{} has empty provider", i + 1));
        }
        if !config.providers.contains_key(&route.provider) {
            return Err(format!(
                "route #{} references unknown provider '{}'",
                i + 1,
                route.provider
            ));
        }
    }

    Ok(())
}

// GET /api/config
pub async fn get_config(State(state): State<AppState>) -> Json<AppConfig> {
    let config = state.config.read().await;
    Json(config.clone())
}

// PUT /api/config
pub async fn update_config(
    State(state): State<AppState>,
    Json(new_config): Json<AppConfig>,
) -> Result<StatusCode, ApiError> {
    validate_config(&new_config).map_err(ApiError::Validation)?;
    save_config(&new_config).map_err(ApiError::from)?;
    let mut config = state.config.write().await;
    *config = new_config;
    Ok(StatusCode::OK)
}

// PUT /api/current-tag
#[derive(Deserialize)]
pub struct SetTagRequest {
    pub tag: String,
}

pub async fn set_current_tag(
    State(state): State<AppState>,
    Json(req): Json<SetTagRequest>,
) -> Result<StatusCode, ApiError> {
    let mut config = state.config.write().await;
    config.current_tag = req.tag;
    save_config(&config).map_err(ApiError::from)?;
    Ok(StatusCode::OK)
}

// POST /api/takeover/claude
pub async fn takeover_claude_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let port = state.config.read().await.port;
    let proxy_url = take_over_claude(port).map_err(ApiError::from)?;
    Ok(Json(serde_json::json!({ "proxy_url": proxy_url })))
}

// DELETE /api/takeover/claude
pub async fn restore_claude_handler() -> Result<StatusCode, ApiError> {
    restore_claude().map_err(ApiError::from)?;
    Ok(StatusCode::OK)
}

// GET /api/status
pub async fn get_status(State(state): State<AppState>) -> Json<StatusResponse> {
    let config = state.config.read().await;
    let port = config.port;
    let takeover_status = check_takeover_status(port);
    Json(StatusResponse {
        current_tag: config.current_tag.clone(),
        takeover: TakeoverInfo {
            active: takeover_status.active,
            proxy_url: takeover_status.proxy_url,
        },
    })
}

// GET /api/logs
pub async fn get_logs(State(state): State<AppState>) -> Json<Vec<RequestLog>> {
    let logs = state.request_log.read().await;
    Json(logs.clone())
}

// POST /api/test
#[derive(Deserialize)]
pub struct TestRequest {
    pub tag: String,
    #[serde(default = "default_test_prompt")]
    pub prompt: String,
}

fn default_test_prompt() -> String {
    "Hi, reply with one word.".to_string()
}

pub async fn test_route_handler(
    State(state): State<AppState>,
    Json(req): Json<TestRequest>,
) -> Json<TestResult> {
    let result = proxy::test_route(&state, &req.tag, &req.prompt).await;
    Json(result)
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    Internal(String),
    #[error("{0}")]
    Validation(String),
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError::Internal(e.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            ApiError::Validation(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        let body = serde_json::json!({ "error": msg });
        (status, Json(body)).into_response()
    }
}
