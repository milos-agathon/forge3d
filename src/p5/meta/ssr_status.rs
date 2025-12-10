//! SSR status classification and evaluation
//!
//! Provides logic for classifying SSR quality based on trace stats,
//! stripe contrast, fallback metrics, and thickness ablation.

use crate::p5::meta::constants::*;
use crate::passes::ssr::SsrStats;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Classify SSR status based on all metrics
pub fn classify_ssr_status(
    stats: Option<&SsrStats>,
    stripe_contrast: Option<&[f32; 9]>,
    stripe_contrast_reference: Option<&[f32; 9]>,
    _mean_abs_diff: f32,
    max_delta_e_miss: f32,
    min_rgb_miss: f32,
    edge_streaks_gt1px: u32,
) -> &'static str {
    // 1. Stats checks
    let stats = match stats {
        Some(stats) if stats.num_rays > 0 => stats,
        _ => return SSR_STATUS_TRACE_FAIL_NO_STATS,
    };

    if !stats.hit_rate().is_finite() || stats.hit_rate() < SSR_HIT_RATE_MIN {
        return SSR_STATUS_TRACE_FAIL_LOW_HIT_RATE;
    }

    // 2. Fallback (miss vs IBL) metrics
    if !max_delta_e_miss.is_finite() {
        return SSR_STATUS_TRACE_FAIL_INVALID;
    }
    if max_delta_e_miss > SSR_MAX_DELTA_E_MISS {
        return SSR_FALLBACK_FAIL_DELTA_E;
    }

    if !min_rgb_miss.is_finite() {
        return SSR_STATUS_TRACE_FAIL_INVALID;
    }
    if min_rgb_miss < SSR_MIN_MISS_RGB {
        return SSR_FALLBACK_FAIL_MIN_RGB;
    }

    // 3. Edge streaks
    if edge_streaks_gt1px > SSR_EDGE_STREAKS_MAX {
        return SSR_STATUS_TRACE_FAIL_EDGE_STREAKS;
    }

    // 4. Stripe contrast checks
    let contrast = match stripe_contrast {
        Some(values) => values,
        None => return SSR_STRIPE_FAIL_CONTRAST,
    };

    if contrast.len() != 9 {
        return SSR_STRIPE_FAIL_CONTRAST;
    }

    // Basic sanity: all SSR contrast values must be finite and strictly > 0
    let ssr_valid = contrast.iter().all(|&v| v.is_finite() && v > 0.0);
    if !ssr_valid {
        return SSR_STRIPE_FAIL_CONTRAST;
    }

    // Optional sanity for the reference array
    if let Some(reference) = stripe_contrast_reference {
        if reference.len() != 9 {
            return SSR_STRIPE_FAIL_CONTRAST;
        }
        let ref_valid = reference.iter().all(|&v| v.is_finite() && v > 0.0);
        if !ref_valid {
            return SSR_STRIPE_FAIL_CONTRAST;
        }
    }

    // Monotonicity: SSR contrast decreases as roughness increases
    if !is_monotonic_decreasing(contrast) {
        return SSR_STRIPE_FAIL_MONOTONIC;
    }

    SSR_STATUS_QA_OK
}

/// Check if values are monotonically non-increasing (with slack)
pub fn is_monotonic_decreasing(values: &[f32]) -> bool {
    values
        .windows(2)
        .all(|pair| pair[0] + SSR_STRIPE_MONO_SLACK >= pair[1])
}

/// Parse a 9-element stripe contrast array from JSON
pub fn parse_stripe_array(value: Option<&Value>) -> Option<[f32; 9]> {
    let arr = value?.as_array()?;
    if arr.len() != 9 {
        return None;
    }
    let mut out = [0.0f32; 9];
    for (i, v) in arr.iter().enumerate().take(9) {
        out[i] = v.as_f64()? as f32;
    }
    Some(out)
}

/// Evaluate M5 status from stored meta data
pub fn evaluate_m5_status(meta: &BTreeMap<String, Value>) -> &'static str {
    let ssr = match meta.get("ssr").and_then(|v| v.as_object()) {
        Some(obj) => obj,
        None => return SSR_STATUS_TRACE_FAIL_NO_STATS,
    };

    let mut stats = SsrStats::default();
    stats.num_rays = ssr.get("num_rays").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    stats.num_hits = ssr.get("num_hits").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    stats.total_steps = ssr.get("total_steps").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    stats.num_misses = ssr.get("num_misses").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    stats.miss_ibl_samples = ssr
        .get("miss_ibl_samples")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let stripe_analysis = ssr.get("stripe_analysis").and_then(|v| v.as_object());
    let stripe = parse_stripe_array(stripe_analysis.and_then(|o| o.get("ssr")));
    let stripe_ref = parse_stripe_array(stripe_analysis.and_then(|o| o.get("reference")));
    let mean_abs_diff = ssr
        .get("ref_vs_ssr_mean_abs_diff")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let max_delta_e_miss = ssr
        .get("max_delta_e_miss")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let min_rgb_miss = ssr
        .get("min_rgb_miss")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let edge_streaks = ssr
        .get("edge_streaks")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("num_streaks_gt1px"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let ssr_status = classify_ssr_status(
        Some(&stats),
        stripe.as_ref(),
        stripe_ref.as_ref(),
        mean_abs_diff,
        max_delta_e_miss,
        min_rgb_miss,
        edge_streaks,
    );
    if ssr_status != SSR_STATUS_QA_OK {
        return ssr_status;
    }

    // Thickness ablation check
    match ssr.get("thickness_ablation").and_then(|v| v.as_object()) {
        Some(obj) => {
            let before = obj
                .get("undershoot_before")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let after = obj
                .get("undershoot_after")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let improvement = after - before;
            if before >= 0.0 && after >= 0.0 && improvement > SSR_THICKNESS_IMPROVEMENT_FACTOR {
                SSR_STATUS_QA_OK
            } else {
                SSR_THICKNESS_ABLATION_FAIL
            }
        }
        None => SSR_THICKNESS_ABLATION_FAIL,
    }
}

/// Patch thickness ablation data into SSR object
pub fn patch_thickness_ablation(
    ssr_obj: &mut serde_json::Map<String, Value>,
    undershoot_before: f32,
    undershoot_after: f32,
) {
    let ab = ssr_obj
        .entry("thickness_ablation".to_string())
        .or_insert_with(|| {
            json!({
                "undershoot_before": 0.0,
                "undershoot_after": 0.0
            })
        });
    if let Some(map) = ab.as_object_mut() {
        map.insert("undershoot_before".to_string(), json!(undershoot_before));
        map.insert("undershoot_after".to_string(), json!(undershoot_after));
    }
}
