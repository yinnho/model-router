use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Recent request log entry.
#[derive(Debug, Clone, Serialize)]
pub struct RequestLog {
    pub request_model: String,
    pub tag: String,
    pub provider: String,
    pub target_model: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
    #[serde(default)]
    pub routes: Vec<Route>,
    #[serde(default)]
    pub tags: Vec<Tag>,
    #[serde(default = "default_tag")]
    pub current_tag: String,
    #[serde(default = "default_management_key")]
    pub management_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    #[serde(default = "default_auth_type")]
    pub auth_type: AuthType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    Bearer,
    XApiKey,
    XGoogApiKey,
}

impl AuthType {
    pub fn header_name(&self) -> &str {
        match self {
            AuthType::Bearer => "authorization",
            AuthType::XApiKey => "x-api-key",
            AuthType::XGoogApiKey => "x-goog-api-key",
        }
    }

    pub fn header_value(&self, api_key: &str) -> String {
        match self {
            AuthType::Bearer => format!("Bearer {}", api_key),
            _ => api_key.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderFormat {
    Anthropic,
    Openai,
    #[serde(rename = "openai_responses")]
    OpenaiResponses,
}

fn default_format() -> ProviderFormat {
    ProviderFormat::Openai
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub endpoint: String,
    pub model: String,
    pub provider: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_format")]
    pub format: ProviderFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub name: String,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub is_auto: bool,
}

fn default_port() -> u16 {
    8082
}

fn default_tag() -> String {
    "auto".to_string()
}

fn default_management_key() -> String {
    "model-router-local".to_string()
}

fn default_auth_type() -> AuthType {
    AuthType::Bearer
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            providers: HashMap::new(),
            routes: Vec::new(),
            tags: Vec::new(),
            current_tag: default_tag(),
            management_key: default_management_key(),
        }
    }
}

pub fn config_path() -> Result<PathBuf> {
    // Allow override via MODEL_ROUTER_CONFIG environment variable
    if let Ok(path) = std::env::var("MODEL_ROUTER_CONFIG") {
        if !path.is_empty() {
            log::info!("[Config] using config path from MODEL_ROUTER_CONFIG: {}", path);
            return Ok(PathBuf::from(path));
        }
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    Ok(home.join(".model-router").join("config.yaml"))
}

pub fn load_config() -> Result<AppConfig> {
    let path = config_path()?;
    log::info!("[Config] loading config from {}", path.display());
    if !path.exists() {
        log::info!("[Config] config file not found, using defaults");
        return Ok(AppConfig::default());
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let config: AppConfig = serde_yaml::from_str(&content)
        .with_context(|| format!("parsing {}", path.display()))?;
    log::info!("[Config] loaded current_tag={}", config.current_tag);
    Ok(config)
}

pub fn save_config(config: &AppConfig) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = serde_yaml::to_string(config)?;

    // Atomic write: write to temp file then rename
    let tmp_path = path.with_extension("yaml.tmp");
    std::fs::write(&tmp_path, &content)
        .with_context(|| format!("writing {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("renaming {} to {}", tmp_path.display(), path.display()))?;

    Ok(())
}

const MAX_LOG_ENTRIES: usize = 50;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub http_client: reqwest::Client,
    pub request_log: Arc<RwLock<Vec<RequestLog>>>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .map_err(|e| anyhow::anyhow!("failed to create HTTP client: {}", e))?;
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            http_client,
            request_log: Arc::new(RwLock::new(Vec::new())),
        })
    }
    pub async fn log_request(&self, entry: RequestLog) {
        let mut log = self.request_log.write().await;
        log.push(entry);
        if log.len() > MAX_LOG_ENTRIES {
            log.remove(0);
        }
    }
}
/// Holds a shutdown signal for the background axum server thread.
/// When dropped or notified, the server thread's tokio runtime shuts down.
pub struct ServerShutdown {
    notify: Arc<tokio::sync::Notify>,
}

impl ServerShutdown {
    pub fn new() -> Self {
        Self { notify: Arc::new(tokio::sync::Notify::new()) }
    }

    pub fn notifier(&self) -> Arc<tokio::sync::Notify> {
        self.notify.clone()
    }

    pub fn shutdown(&self) {
        self.notify.notify_one();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_has_sensible_values() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.port, 8082);
        assert_eq!(cfg.current_tag, "auto");
        assert_eq!(cfg.management_key, "model-router-local");
        assert!(cfg.providers.is_empty());
        assert!(cfg.routes.is_empty());
    }

    #[test]
    fn test_auth_type_header_name() {
        assert_eq!(AuthType::Bearer.header_name(), "authorization");
        assert_eq!(AuthType::XApiKey.header_name(), "x-api-key");
        assert_eq!(AuthType::XGoogApiKey.header_name(), "x-goog-api-key");
    }

    #[test]
    fn test_auth_type_header_value() {
        assert_eq!(AuthType::Bearer.header_value("key123"), "Bearer key123");
        assert_eq!(AuthType::XApiKey.header_value("key123"), "key123");
        assert_eq!(AuthType::XGoogApiKey.header_value("key123"), "key123");
    }

    #[test]
    fn test_provider_format_serde() {
        let yaml = "openai";
        let fmt: ProviderFormat = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(fmt, ProviderFormat::Openai);

        let yaml = "openai_responses";
        let fmt: ProviderFormat = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(fmt, ProviderFormat::OpenaiResponses);

        let yaml = "anthropic";
        let fmt: ProviderFormat = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(fmt, ProviderFormat::Anthropic);
    }

    #[test]
    fn test_default_format_is_openai() {
        assert_eq!(default_format(), ProviderFormat::Openai);
    }

    #[test]
    fn test_request_log_ring_buffer() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = AppState::new(AppConfig::default()).unwrap();
            for i in 0..60 {
                state.log_request(RequestLog {
                    request_model: format!("model-{}", i),
                    tag: "auto".into(),
                    provider: "test".into(),
                    target_model: "target".into(),
                    timestamp: "now".into(),
                }).await;
            }
            let logs = state.request_log.read().await;
            assert_eq!(logs.len(), 50); // MAX_LOG_ENTRIES
            assert_eq!(logs[0].request_model, "model-10"); // oldest remaining
            assert_eq!(logs[49].request_model, "model-59"); // newest
        });
    }

    #[test]
    fn test_management_key_default() {
        assert_eq!(default_management_key(), "model-router-local");
    }
}
