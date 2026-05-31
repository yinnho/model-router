use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;

fn claude_settings_path() -> PathBuf {
    let home = dirs::home_dir().expect("no home directory");
    home.join(".claude").join("settings.json")
}

fn claude_settings_backup_path() -> PathBuf {
    let home = dirs::home_dir().expect("no home directory");
    home.join(".claude").join("settings.json.model-router-backup")
}

/// Environment variable keys that would override model-router's routing.
const ROUTER_OVERRIDE_ENV_KEYS: &[&str] = &[
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
];

pub struct TakeoverStatus {
    pub active: bool,
    pub proxy_url: Option<String>,
}

pub fn take_over_claude(port: u16) -> Result<String> {
    let path = claude_settings_path();
    let backup = claude_settings_backup_path();

    let mut settings: Value = if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        serde_json::from_str(&content)?
    } else {
        serde_json::json!({})
    };

    // Backup original if not already backed up
    if path.exists() && !backup.exists() {
        std::fs::copy(&path, &backup).context("backing up settings.json")?;
    }

    let proxy_url = format!("http://127.0.0.1:{}/anthropic", port);

    // Remove old apiBaseUrl if present (we now use env vars)
    if let Some(obj) = settings.as_object_mut() {
        obj.remove("apiBaseUrl");

        // Ensure env object exists
        if obj.get("env").is_none() {
            obj.insert("env".to_string(), serde_json::json!({}));
        }

        if let Some(env) = obj.get_mut("env").and_then(|e| e.as_object_mut()) {
            // Remove old env vars that would conflict
            for key in ROUTER_OVERRIDE_ENV_KEYS {
                env.remove(*key);
            }

            // Set new env vars for model-router
            env.insert(
                "ANTHROPIC_BASE_URL".to_string(),
                Value::String(proxy_url.clone()),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_OPUS_MODEL".to_string(),
                Value::String("opus".to_string()),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_SONNET_MODEL".to_string(),
                Value::String("sonnet".to_string()),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(),
                Value::String("haiku".to_string()),
            );
            // Set default model to "auto" so Claude Code routes through auto tag
            env.insert(
                "ANTHROPIC_MODEL".to_string(),
                Value::String("auto".to_string()),
            );
            // Add a placeholder API key so Claude Code doesn't show "Not logged in"
            // Actual requests will use provider keys from proxy config
            env.insert(
                "ANTHROPIC_API_KEY".to_string(),
                Value::String("sk-ant-placeholder-model-router".to_string()),
            );
        }
    }

    let content = serde_json::to_string_pretty(&settings)?;
    std::fs::write(&path, content)?;

    Ok(proxy_url)
}

pub fn restore_claude() -> Result<()> {
    let path = claude_settings_path();
    let backup = claude_settings_backup_path();

    if backup.exists() {
        std::fs::copy(&backup, &path)?;
        std::fs::remove_file(&backup)?;
    }

    Ok(())
}

pub fn check_takeover_status(port: u16) -> TakeoverStatus {
    let path = claude_settings_path();

    if !path.exists() {
        return TakeoverStatus {
            active: false,
            proxy_url: None,
        };
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[Takeover] failed to read Claude settings file: {}", e);
            return TakeoverStatus {
                active: false,
                proxy_url: None,
            }
        }
    };

    let settings: Value = match serde_json::from_str(&content) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("[Takeover] failed to parse Claude settings JSON: {}", e);
            return TakeoverStatus {
                active: false,
                proxy_url: None,
            }
        }
    };

    let expected_url = format!("http://127.0.0.1:{}/anthropic", port);

    // Check env.ANTHROPIC_BASE_URL
    let env_base_url = settings
        .get("env")
        .and_then(|e| e.get("ANTHROPIC_BASE_URL"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Also check legacy apiBaseUrl for backward compatibility
    let legacy_url = settings
        .get("apiBaseUrl")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let active = env_base_url == expected_url || legacy_url == expected_url.trim_end_matches("/anthropic");
    let proxy_url = if active {
        Some(expected_url.clone())
    } else {
        None
    };

    TakeoverStatus { active, proxy_url }
}
