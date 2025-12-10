//! SSR default values and field initialization
//!
//! Provides default SSR JSON structures and field migration helpers.

use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Create default SSR JSON value
pub fn default_ssr_value() -> Value {
    json!({
        "num_rays": 0,
        "num_hits": 0,
        "total_steps": 0,
        "num_misses": 0,
        "miss_ibl_samples": 0,
        "hit_rate": 0.0,
        "avg_steps": 0.0,
        "miss_ibl_ratio": 0.0,
        "perf_ms": {
            "trace_ms": 0.0,
            "shade_ms": 0.0,
            "fallback_ms": 0.0,
            "total_ssr_ms": 0.0
        },
        "max_delta_e_miss": 0.0,
        "min_rgb_miss": 0.0,
        "stripe_contrast": [],
        "stripe_contrast_reference": [],
        "ref_vs_ssr_mean_abs_diff": 0.0,
        "edge_streaks": {
            "num_streaks_gt1px": 0
        },
        "status": "UNINITIALIZED"
    })
}

/// Ensure SSR field exists with default value
pub fn ensure_ssr_field(ssr: &mut serde_json::Map<String, Value>, key: &str, default: Value) {
    if !ssr.contains_key(key) {
        ssr.insert(key.to_string(), default);
    }
}

/// Ensure all SSR defaults are present in meta
pub fn ensure_ssr_defaults(meta: &mut BTreeMap<String, Value>) {
    if !meta.contains_key("ssr") {
        meta.insert("ssr".to_string(), default_ssr_value());
    }

    // Migrate legacy top-level fields into the ssr object
    let legacy_contrast = meta.remove("ssr_stripe_contrast");
    let legacy_streaks = meta.remove("ssr_edge_streaks");
    let legacy_status = meta.get("ssr_status").cloned();

    let ssr_entry = meta
        .entry("ssr".to_string())
        .or_insert_with(default_ssr_value);
    if !ssr_entry.is_object() {
        *ssr_entry = default_ssr_value();
    }

    if let Some(ssr) = ssr_entry.as_object_mut() {
        ensure_ssr_field(ssr, "hit_rate", json!(0.0));
        ensure_ssr_field(ssr, "avg_steps", json!(0.0));
        ensure_ssr_field(ssr, "miss_ibl_ratio", json!(0.0));
        ensure_ssr_field(
            ssr,
            "perf_ms",
            json!({
                "trace_ms": 0.0,
                "shade_ms": 0.0,
                "fallback_ms": 0.0,
                "total_ssr_ms": 0.0
            }),
        );
        ensure_ssr_field(ssr, "max_delta_e_miss", json!(0.0));
        ensure_ssr_field(ssr, "min_rgb_miss", json!(0.0));
        ensure_ssr_field(
            ssr,
            "thickness_ablation",
            json!({
                "undershoot_before": 0.0,
                "undershoot_after": 0.0
            }),
        );

        // Handle legacy fields
        match legacy_contrast {
            Some(val) => {
                ssr.insert("stripe_contrast".to_string(), val);
            }
            None => {
                ssr.entry("stripe_contrast".to_string())
                    .or_insert_with(|| json!([]));
            }
        }
        match legacy_streaks {
            Some(val) => {
                ssr.insert("edge_streaks".to_string(), val);
            }
            None => {
                ssr.entry("edge_streaks".to_string()).or_insert_with(|| {
                    json!({
                        "num_streaks_gt1px": 0
                    })
                });
            }
        }
        match legacy_status {
            Some(val) => {
                ssr.insert("status".to_string(), val.clone());
                meta.insert("ssr_status".to_string(), val);
            }
            None => {
                ssr.entry("status".to_string())
                    .or_insert_with(|| json!("UNINITIALIZED"));
            }
        }
    }

    meta.entry("ssr_status".to_string())
        .or_insert_with(|| json!("SSR_UNINITIALIZED"));
}
