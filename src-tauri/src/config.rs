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
        }
    }
}

pub fn config_path() -> Result<PathBuf> {
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
    pub web_dist: PathBuf,
}

impl AppState {
    pub fn new(config: AppConfig, web_dist: PathBuf) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .map_err(|e| anyhow::anyhow!("failed to create HTTP client: {}", e))?;
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            http_client,
            request_log: Arc::new(RwLock::new(Vec::new())),
            web_dist,
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
