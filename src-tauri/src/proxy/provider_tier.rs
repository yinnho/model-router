//! Provider Tier Resolution
//!
//! Maps routing tiers (light/medium/heavy) and fixed targets (opus/gpt55)
//! to providers based on their `routing_tier` meta field.

use crate::proxy::request_classifier::Complexity;
use crate::provider::Provider;

/// Resolve a provider matching the given complexity tier from ALL providers.
///
/// Priority:
/// 1. Provider with `routing_tier` meta matching the tier
/// 2. Fallback: infer from model fields (ANTHROPIC_DEFAULT_*_MODEL)
/// 3. Fallback: first provider for medium, None for others
pub fn resolve_tier_provider(providers: &[Provider], tier: Complexity) -> Option<usize> {
    if providers.is_empty() {
        return None;
    }

    let tier_str = match tier {
        Complexity::Light => "light",
        Complexity::Medium => "medium",
        Complexity::Heavy => "heavy",
    };

    // 1. Match by explicit routing_tier meta
    for (i, provider) in providers.iter().enumerate() {
        if let Some(ref meta) = provider.meta {
            if meta.routing_tier.as_deref() == Some(tier_str) {
                return Some(i);
            }
        }
    }

    // 2. Fallback: infer from model fields
    for (i, provider) in providers.iter().enumerate() {
        let env = provider.settings_config.get("env");
        let matches = match tier {
            Complexity::Heavy => env
                .and_then(|e| e.get("ANTHROPIC_DEFAULT_OPUS_MODEL"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .is_some(),
            Complexity::Medium => env
                .and_then(|e| e.get("ANTHROPIC_DEFAULT_SONNET_MODEL"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .is_some()
                || env
                    .and_then(|e| e.get("ANTHROPIC_MODEL"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .is_some(),
            Complexity::Light => env
                .and_then(|e| e.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .is_some(),
        };
        if matches {
            return Some(i);
        }
    }

    // 3. Fallback: medium → first provider
    if tier == Complexity::Medium {
        return Some(0);
    }

    None
}

/// Build a routed provider list: target tier provider first, then remaining as failover.
///
/// Returns a new Vec with the matching provider at position 0.
/// If no match found, returns the original list unchanged.
pub fn build_routed_list(providers: Vec<Provider>, tier: Complexity) -> Vec<Provider> {
    if providers.is_empty() {
        return providers;
    }

    match resolve_tier_provider(&providers, tier) {
        Some(idx) if idx > 0 => {
            let mut result = Vec::with_capacity(providers.len());
            result.push(providers[idx].clone());
            for (i, p) in providers.into_iter().enumerate() {
                if i != idx {
                    result.push(p);
                }
            }
            result
        }
        Some(_) => providers,
        None => providers,
    }
}

/// Resolve a fixed routing target (e.g. "opus" or "gpt55") to a provider index.
///
/// The target string comes from RouterConfig.mode after stripping "fixed:" prefix.
/// Checks `routing_tier` meta first, then falls back to model field inference.
pub fn resolve_fixed_target(providers: &[Provider], target: &str) -> Option<usize> {
    if providers.is_empty() {
        return None;
    }

    // Map fixed target to routing_tier value
    let tier_value = match target.to_lowercase().as_str() {
        "opus" | "opus4.7" | "claude-opus" => "heavy",
        "sonnet" | "sonnet4" | "claude-sonnet" => "medium",
        "haiku" | "claude-haiku" => "light",
        "gpt55" | "gpt-5.5" => "heavy",
        _ => "",
    };

    // 1. Match by explicit routing_tier meta
    if !tier_value.is_empty() {
        for (i, provider) in providers.iter().enumerate() {
            if let Some(ref meta) = provider.meta {
                if meta.routing_tier.as_deref() == Some(tier_value) {
                    return Some(i);
                }
            }
        }
    }

    // 2. Fallback: model field inference
    for (i, provider) in providers.iter().enumerate() {
        let env = provider.settings_config.get("env");
        let matches = match target.to_lowercase().as_str() {
            "opus" | "opus4.7" | "claude-opus" => env
                .and_then(|e| e.get("ANTHROPIC_DEFAULT_OPUS_MODEL"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .is_some(),
            "sonnet" | "sonnet4" | "claude-sonnet" => env
                .and_then(|e| e.get("ANTHROPIC_DEFAULT_SONNET_MODEL"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .is_some(),
            "haiku" | "claude-haiku" => env
                .and_then(|e| e.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .is_some(),
            "gpt55" | "gpt-5.5" => env
                .and_then(|e| e.get("ANTHROPIC_MODEL"))
                .and_then(|v| v.as_str())
                .map(|m| m.to_lowercase().contains("gpt-5") || m.to_lowercase().contains("o3"))
                .unwrap_or(false),
            _ => false,
        };
        if matches {
            return Some(i);
        }
    }

    None
}

/// Build a routed provider list for a fixed target.
pub fn build_routed_list_for_fixed(providers: Vec<Provider>, target: &str) -> Vec<Provider> {
    if providers.is_empty() {
        return providers;
    }

    match resolve_fixed_target(&providers, target) {
        Some(idx) if idx > 0 => {
            let mut result = Vec::with_capacity(providers.len());
            result.push(providers[idx].clone());
            for (i, p) in providers.into_iter().enumerate() {
                if i != idx {
                    result.push(p);
                }
            }
            result
        }
        Some(_) => providers,
        None => providers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_provider(id: &str, env: serde_json::Value) -> Provider {
        Provider {
            id: id.to_string(),
            name: id.to_string(),
            settings_config: json!({ "env": env }),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: None,
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        }
    }

    fn make_tiered_provider(id: &str, tier: &str) -> Provider {
        Provider {
            id: id.to_string(),
            name: id.to_string(),
            settings_config: json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(crate::provider::ProviderMeta {
                routing_tier: Some(tier.to_string()),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        }
    }

    #[test]
    fn test_resolve_tier_by_meta() {
        let providers = vec![
            make_tiered_provider("light-p", "light"),
            make_tiered_provider("heavy-p", "heavy"),
            make_tiered_provider("medium-p", "medium"),
        ];
        assert_eq!(resolve_tier_provider(&providers, Complexity::Heavy), Some(1));
        assert_eq!(resolve_tier_provider(&providers, Complexity::Light), Some(0));
        assert_eq!(resolve_tier_provider(&providers, Complexity::Medium), Some(2));
    }

    #[test]
    fn test_resolve_tier_by_model_fallback() {
        let providers = vec![
            make_provider("p1", json!({ "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet" })),
            make_provider("p2", json!({ "ANTHROPIC_DEFAULT_OPUS_MODEL": "opus" })),
        ];
        assert_eq!(resolve_tier_provider(&providers, Complexity::Heavy), Some(1));
    }

    #[test]
    fn test_resolve_tier_medium_fallback() {
        let providers = vec![
            make_provider("p1", json!({ "ANTHROPIC_DEFAULT_OPUS_MODEL": "opus" })),
        ];
        assert_eq!(resolve_tier_provider(&providers, Complexity::Medium), Some(0));
    }

    #[test]
    fn test_meta_takes_priority_over_model() {
        let providers = vec![
            make_provider("model-based", json!({ "ANTHROPIC_DEFAULT_OPUS_MODEL": "opus" })),
            make_tiered_provider("meta-based", "heavy"),
        ];
        // Meta-based should win even though it comes second
        assert_eq!(resolve_tier_provider(&providers, Complexity::Heavy), Some(1));
    }

    #[test]
    fn test_build_routed_list() {
        let providers = vec![
            make_tiered_provider("light-p", "light"),
            make_tiered_provider("heavy-p", "heavy"),
            make_tiered_provider("medium-p", "medium"),
        ];
        let routed = build_routed_list(providers, Complexity::Heavy);
        assert_eq!(routed[0].id, "heavy-p");
        assert_eq!(routed.len(), 3);
    }

    #[test]
    fn test_resolve_fixed_by_meta() {
        let providers = vec![
            make_tiered_provider("opus-p", "heavy"),
            make_tiered_provider("haiku-p", "light"),
        ];
        assert_eq!(resolve_fixed_target(&providers, "opus"), Some(0));
        assert_eq!(resolve_fixed_target(&providers, "opus4.7"), Some(0));
    }

    #[test]
    fn test_resolve_fixed_gpt55_as_heavy() {
        let providers = vec![
            make_tiered_provider("haiku-p", "light"),
            make_tiered_provider("gpt-p", "heavy"),
        ];
        assert_eq!(resolve_fixed_target(&providers, "gpt55"), Some(1));
    }

    #[test]
    fn test_build_routed_list_for_fixed() {
        let providers = vec![
            make_tiered_provider("light-p", "light"),
            make_tiered_provider("heavy-p", "heavy"),
        ];
        let routed = build_routed_list_for_fixed(providers, "opus");
        assert_eq!(routed[0].id, "heavy-p");
    }

    #[test]
    fn test_empty_providers() {
        let providers: Vec<Provider> = vec![];
        assert_eq!(resolve_tier_provider(&providers, Complexity::Heavy), None);
        assert_eq!(resolve_fixed_target(&providers, "opus"), None);
    }
}
