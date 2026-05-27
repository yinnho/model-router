//! Request Classifier
//!
//! Uses a lightweight LLM to classify request complexity for auto-routing.
//! Falls back to Medium on any error.

use crate::proxy::http_client;
use crate::proxy::providers::{AuthStrategy, ClaudeAdapter, ProviderAdapter};
use crate::provider::Provider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Request complexity tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Light,
    Medium,
    Heavy,
}

impl std::fmt::Display for Complexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Complexity::Light => write!(f, "light"),
            Complexity::Medium => write!(f, "medium"),
            Complexity::Heavy => write!(f, "heavy"),
        }
    }
}

impl Complexity {
    fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "light" => Some(Complexity::Light),
            "medium" => Some(Complexity::Medium),
            "heavy" => Some(Complexity::Heavy),
            _ => None,
        }
    }
}

/// Cache entry for classification results
struct CacheEntry {
    complexity: Complexity,
    expires_at: Instant,
}

/// In-memory classification cache keyed by (session_id, message_hash)
pub struct ClassificationCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
    default_ttl: Duration,
}

impl ClassificationCache {
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            default_ttl,
        }
    }

    pub fn get(&self, session_id: &str, message_hash: u64) -> Option<Complexity> {
        let key = format!("{}:{:016x}", session_id, message_hash);
        let entries = self.entries.read().ok()?;
        let entry = entries.get(&key)?;
        if entry.expires_at > Instant::now() {
            Some(entry.complexity)
        } else {
            None
        }
    }

    pub fn set(&self, session_id: &str, message_hash: u64, complexity: Complexity) {
        let key = format!("{}:{:016x}", session_id, message_hash);
        if let Ok(mut entries) = self.entries.write() {
            entries.insert(
                key,
                CacheEntry {
                    complexity,
                    expires_at: Instant::now() + self.default_ttl,
                },
            );
        }
    }

    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
    }
}

const CLASSIFIER_PROMPT: &str = r#"Classify this AI request by complexity. Reply with exactly one word: light, medium, or heavy.
- light: simple Q&A, formatting, short answers, translations, quick lookups
- medium: code explanation, debugging, moderate analysis, feature implementation
- heavy: complex reasoning, multi-step coding, deep analysis, architecture design, large refactors"#;

const MAX_MESSAGE_CHARS: usize = 2000;
const CLASSIFIER_TIMEOUT_SECS: u64 = 2;
const MAX_OUTPUT_TOKENS: u32 = 10;

/// Extract the last user message from a request body for classification
fn extract_user_message(body: &serde_json::Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    for msg in messages.iter().rev() {
        let role = msg.get("role")?.as_str()?;
        if role == "user" {
            if let Some(content) = msg.get("content") {
                if let Some(s) = content.as_str() {
                    return Some(s.to_string());
                }
                if let Some(arr) = content.as_array() {
                    for block in arr.iter().rev() {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            return Some(text.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Simple hash for caching
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.as_bytes().iter().take(256) {
        hash = hash.wrapping_mul(33).wrapping_add(*b as u64);
    }
    hash
}

/// Classify a request by calling a lightweight LLM
pub async fn classify_request(
    body: &serde_json::Value,
    provider: &Provider,
    classifier_model: &str,
    cache: &ClassificationCache,
    session_id: &str,
) -> Complexity {
    let message = match extract_user_message(body) {
        Some(m) => m,
        None => {
            log::debug!("[Router] No user message found, defaulting to Medium");
            return Complexity::Medium;
        }
    };

    let hash = simple_hash(&message);

    // Check cache
    if let Some(cached) = cache.get(session_id, hash) {
        log::debug!("[Router] Cache hit: {} for session {}", cached, session_id);
        return cached;
    }

    let truncated: String = if message.len() > MAX_MESSAGE_CHARS {
        format!("{}...", &message[..MAX_MESSAGE_CHARS])
    } else {
        message.clone()
    };

    let request_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    let user_content = format!(
        "Request model: {}\n\nFirst user message:\n{}",
        request_model, truncated
    );

    let result = call_classifier_llm(provider, classifier_model, &user_content).await;

    let complexity = match result {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[Router] Classifier failed: {}, defaulting to Medium", e);
            Complexity::Medium
        }
    };

    cache.set(session_id, hash, complexity);
    log::info!(
        "[Router] Classified as {} (session={}, model={})",
        complexity,
        session_id,
        request_model
    );

    complexity
}

async fn call_classifier_llm(
    provider: &Provider,
    model: &str,
    user_content: &str,
) -> Result<Complexity, String> {
    let adapter = ClaudeAdapter;
    let base_url = adapter
        .extract_base_url(provider)
        .map_err(|e| format!("extract base URL: {e}"))?;
    let auth = adapter
        .extract_auth(provider)
        .ok_or_else(|| "no auth info".to_string())?;

    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));

    let request_body = serde_json::json!({
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{
            "role": "user",
            "content": format!("{}\n\n{}", CLASSIFIER_PROMPT, user_content)
        }]
    });

    let client = http_client::get();
    let mut req = client
        .post(&url)
        .timeout(Duration::from_secs(CLASSIFIER_TIMEOUT_SECS))
        .json(&request_body);

    // Set auth headers
    match auth.strategy {
        AuthStrategy::Anthropic | AuthStrategy::ClaudeAuth => {
            req = req
                .header("x-api-key", &auth.api_key)
                .header("anthropic-version", "2023-06-01");
        }
        AuthStrategy::Bearer => {
            req = req.header("Authorization", format!("Bearer {}", auth.api_key));
        }
        _ => {
            req = req.header("Authorization", format!("Bearer {}", auth.api_key));
        }
    }

    let resp = req.send().await.map_err(|e| format!("HTTP error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("status {} body: {}", status, &body[..body.len().min(200)]));
    }

    let resp_body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("parse response: {e}"))?;

    let text = resp_body
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|block| block.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    Complexity::from_str(text).ok_or_else(|| format!("unexpected classifier output: {:?}", text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_from_str() {
        assert_eq!(Complexity::from_str("light"), Some(Complexity::Light));
        assert_eq!(Complexity::from_str("medium"), Some(Complexity::Medium));
        assert_eq!(Complexity::from_str("heavy"), Some(Complexity::Heavy));
        assert_eq!(Complexity::from_str(" Light "), Some(Complexity::Light));
        assert_eq!(Complexity::from_str("unknown"), None);
    }

    #[test]
    fn test_complexity_display() {
        assert_eq!(format!("{}", Complexity::Light), "light");
        assert_eq!(format!("{}", Complexity::Medium), "medium");
        assert_eq!(format!("{}", Complexity::Heavy), "heavy");
    }

    #[test]
    fn test_extract_user_message_text() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Hello world"}
            ]
        });
        assert_eq!(extract_user_message(&body), Some("Hello world".to_string()));
    }

    #[test]
    fn test_extract_user_message_blocks() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Block content"}
                ]}
            ]
        });
        assert_eq!(
            extract_user_message(&body),
            Some("Block content".to_string())
        );
    }

    #[test]
    fn test_extract_user_message_last_user() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Second"}
            ]
        });
        assert_eq!(extract_user_message(&body), Some("Second".to_string()));
    }

    #[test]
    fn test_extract_user_message_none() {
        let body = serde_json::json!({"model": "test"});
        assert_eq!(extract_user_message(&body), None);
    }

    #[test]
    fn test_cache_basic() {
        let cache = ClassificationCache::new(Duration::from_secs(300));
        assert!(cache.get("sess1", 123).is_none());
        cache.set("sess1", 123, Complexity::Heavy);
        assert_eq!(cache.get("sess1", 123), Some(Complexity::Heavy));
        assert!(cache.get("sess1", 456).is_none());
        assert!(cache.get("sess2", 123).is_none());
    }

    #[test]
    fn test_cache_expiry() {
        let cache = ClassificationCache::new(Duration::from_millis(10));
        cache.set("sess1", 123, Complexity::Light);
        assert_eq!(cache.get("sess1", 123), Some(Complexity::Light));
        std::thread::sleep(Duration::from_millis(20));
        assert!(cache.get("sess1", 123).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = ClassificationCache::new(Duration::from_secs(300));
        cache.set("sess1", 123, Complexity::Medium);
        cache.clear();
        assert!(cache.get("sess1", 123).is_none());
    }

    #[test]
    fn test_simple_hash_deterministic() {
        let h1 = simple_hash("test message");
        let h2 = simple_hash("test message");
        assert_eq!(h1, h2);
        let h3 = simple_hash("different message");
        assert_ne!(h1, h3);
    }
}
