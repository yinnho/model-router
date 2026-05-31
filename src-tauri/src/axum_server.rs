use crate::config::AppState;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{Method, StatusCode, Uri};
use axum::response::IntoResponse;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

/// Start the axum HTTP server. Returns the port once bound.
pub async fn start(state: AppState) -> u16 {
    let port = state.config.read().await.port;

    let app = axum::Router::new()
        // Anthropic client paths
        .route(
            "/anthropic/v1/messages",
            axum::routing::post(crate::proxy::handle_anthropic_messages),
        )
        .route(
            "/anthropic/v1/messages/count_tokens",
            axum::routing::post(crate::proxy::handle_anthropic_count_tokens),
        )
        // OpenAI client paths
        .route(
            "/openai/v1/chat/completions",
            axum::routing::post(crate::proxy::handle_openai_chat),
        )
        .route(
            "/openai/v1/responses",
            axum::routing::post(crate::proxy::handle_openai_responses),
        )
        // Legacy paths (backward compatible)
        .route(
            "/v1/messages",
            axum::routing::post(crate::proxy::handle_anthropic_messages),
        )
        .route(
            "/v1/messages/count_tokens",
            axum::routing::post(crate::proxy::handle_anthropic_count_tokens),
        )
        // API endpoints
        .route("/api/config", axum::routing::get(crate::api::get_config))
        .route(
            "/api/config",
            axum::routing::put(crate::api::update_config),
        )
        .route(
            "/api/current-tag",
            axum::routing::put(crate::api::set_current_tag),
        )
        .route(
            "/api/takeover/claude",
            axum::routing::post(crate::api::takeover_claude_handler),
        )
        .route(
            "/api/takeover/claude",
            axum::routing::delete(crate::api::restore_claude_handler),
        )
        .route("/api/status", axum::routing::get(crate::api::get_status))
        .route("/api/logs", axum::routing::get(crate::api::get_logs))
        .route("/api/test", axum::routing::post(crate::api::test_route_handler))
        .layer(axum::middleware::from_fn(request_log_middleware))
        .layer(CorsLayer::permissive())
        .fallback(fallback_handler)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .expect("failed to bind");

    log::info!("model-router listening on 127.0.0.1:{}", port);

    // Return port to signal readiness, then serve forever
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("server error");
    });

    port
}

/// Middleware that logs every incoming request.
async fn request_log_middleware(
    req: Request,
    next: axum::middleware::Next,
) -> impl IntoResponse {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();

    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    let api_key = req
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    let auth = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            if s.len() > 20 {
                format!("{}...", &s[..20])
            } else {
                s.to_string()
            }
        })
        .unwrap_or("-".to_string());

    log::info!(
        "[Request] {} {} | content-type={} | x-api-key={} | auth={}",
        method,
        path,
        content_type,
        api_key,
        auth
    );

    let response = next.run(req).await;
    log::info!("[Request] {} {} → {}", method, path, response.status());
    response
}

/// Fallback handler: serves static frontend files or returns 404.
async fn fallback_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    method: Method,
    uri: Uri,
) -> impl IntoResponse {
    let path = uri.path().to_string();

    if method == Method::GET || method == Method::HEAD {
        let mut serve_dir = ServeDir::new(&state.web_dist);
        let req = match Request::builder()
            .method(method)
            .uri(&uri)
            .body(Body::empty())
        {
            Ok(req) => req,
            Err(e) => {
                log::warn!("[Fallback] invalid URI '{}': {}", uri, e);
                return (
                    StatusCode::BAD_REQUEST,
                    [("content-type", "application/json")],
                    serde_json::json!({"error": format!("bad request URI: {}", uri)}).to_string(),
                )
                    .into_response();
            }
        };
        match serve_dir.try_call(req).await {
            Ok(resp) => resp.map(Body::new),
            Err(_) => (
                StatusCode::NOT_FOUND,
                [("content-type", "text/plain")],
                "Not Found",
            )
                .into_response(),
        }
    } else {
        log::warn!("[Fallback] {} {} → 404", method, path);
        (
            StatusCode::NOT_FOUND,
            [("content-type", "application/json")],
            serde_json::json!({"error": format!("not found: {} {}", method, path)}).to_string(),
        )
            .into_response()
    }
}
