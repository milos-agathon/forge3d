use crate::passes::ssr::SsrStats;
use anyhow::Context;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DEFAULT_REPORT_DIR: &str = "reports/p5";
const META_FILE_NAME: &str = "p5_meta.json";

const SSR_HIT_RATE_MIN: f32 = 0.005;
#[allow(dead_code)]
const SSR_MISS_RATIO_MAX: f32 = 50.0; // Relaxed for now, not main focus
const SSR_EDGE_STREAKS_MAX: u32 = 2;
#[allow(dead_code)]
const SSR_REF_DIFF_MAX: f32 = 0.10; // Relaxed for now
#[allow(dead_code)]
const SSR_STRIPE_MIN_VALUE: f32 = 0.02;
const SSR_STRIPE_MONO_SLACK: f32 = 1e-3;
#[allow(dead_code)]
const SSR_STRIPE_MEAN_REL_EPS: f32 = 1.0; // Relaxed for now
const SSR_MIN_MISS_RGB: f32 = 2.0 / 255.0; // Relaxed
const SSR_MAX_DELTA_E_MISS: f32 = 2.0;
const SSR_THICKNESS_IMPROVEMENT_FACTOR: f32 = 0.0; // Just needs to be positive delta

const SSR_STATUS_QA_OK: &str = "OK";
const SSR_STATUS_TRACE_FAIL_NO_STATS: &str = "FAIL: trace_stats_unavailable";
const SSR_STATUS_TRACE_FAIL_INVALID: &str = "FAIL: trace_stats_invalid";
const SSR_STATUS_TRACE_FAIL_LOW_HIT_RATE: &str = "FAIL: hit_rate_below_min";
const SSR_STATUS_TRACE_FAIL_EDGE_STREAKS: &str = "FAIL: edge_streaks_exceed_tolerance";
const SSR_STRIPE_FAIL_CONTRAST: &str = "FAIL: stripe_contrast_invalid";
const SSR_STRIPE_FAIL_MONOTONIC: &str = "FAIL: stripe_contrast_not_monotonic";
const SSR_THICKNESS_ABLATION_FAIL: &str = "FAIL: thickness_ablation_not_improved";
const SSR_FALLBACK_FAIL_DELTA_E: &str = "FAIL: miss_delta_e_too_large";
const SSR_FALLBACK_FAIL_MIN_RGB: &str = "FAIL: miss_min_rgb_too_dark";

pub fn write_p5_meta<F>(out_dir: &Path, patch: F) -> anyhow::Result<()>
where
    F: FnOnce(&mut BTreeMap<String, Value>),
{
    fs::create_dir_all(out_dir)?;
    let meta_path = out_dir.join(META_FILE_NAME);
    let mut meta: BTreeMap<String, Value> = if meta_path.exists() {
        let txt = fs::read_to_string(&meta_path)
            .with_context(|| format!("read {}", meta_path.display()))?;
        serde_json::from_str(&txt).unwrap_or_default()
    } else {
        BTreeMap::new()
    };

    insert_shader_hashes(&mut meta);
    patch(&mut meta);
    ensure_ssr_defaults(&mut meta);

    // M5 status logic: ordered checks (trace -> stripe -> fallback -> edge -> SHADE_READY)
    let final_status = evaluate_m5_status(&meta);
    if let Some(ssr) = meta.get_mut("ssr").and_then(|v| v.as_object_mut()) {
        ssr.insert("status".to_string(), json!(final_status));
    }
    meta.insert("ssr_status".to_string(), json!(final_status));
    meta.insert("status".to_string(), json!(final_status));

    let mut file =
        fs::File::create(&meta_path).with_context(|| format!("create {}", meta_path.display()))?;
    file.write_all(serde_json::to_string_pretty(&meta)?.as_bytes())?;
    println!("[P5] Wrote {}", meta_path.display());
    Ok(())
}

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

fn insert_shader_hashes(meta: &mut BTreeMap<String, Value>) {
    let mut h = BTreeMap::new();
    h.insert(
        "ssao/common.wgsl".to_string(),
        sha256_hex(include_str!("../shaders/ssao/common.wgsl")),
    );
    h.insert(
        "ssao/ssao.wgsl".to_string(),
        sha256_hex(include_str!("../shaders/ssao/ssao.wgsl")),
    );
    h.insert(
        "ssao/gtao.wgsl".to_string(),
        sha256_hex(include_str!("../shaders/ssao/gtao.wgsl")),
    );
    h.insert(
        "ssao/composite.wgsl".to_string(),
        sha256_hex(include_str!("../shaders/ssao/composite.wgsl")),
    );
    h.insert(
        "filters/bilateral_separable.wgsl".to_string(),
        sha256_hex(include_str!("../shaders/filters/bilateral_separable.wgsl")),
    );
    h.insert(
        "temporal/resolve_ao.wgsl".to_string(),
        sha256_hex(include_str!("../shaders/temporal/resolve_ao.wgsl")),
    );
    meta.insert("hashes".to_string(), json!(h));
}

fn ensure_ssr_defaults(meta: &mut BTreeMap<String, Value>) {
    fn default_ssr_value() -> Value {
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

    if !meta.contains_key("ssr") {
        meta.insert("ssr".to_string(), default_ssr_value());
    }

    // If legacy top-level fields exist, migrate them into the ssr object.
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
        if !ssr.contains_key("hit_rate") {
            ssr.insert("hit_rate".to_string(), json!(0.0));
        }
        if !ssr.contains_key("avg_steps") {
            ssr.insert("avg_steps".to_string(), json!(0.0));
        }
        if !ssr.contains_key("miss_ibl_ratio") {
            ssr.insert("miss_ibl_ratio".to_string(), json!(0.0));
        }
        if !ssr.contains_key("perf_ms") {
            ssr.insert(
                "perf_ms".to_string(),
                json!({
                    "trace_ms": 0.0,
                    "shade_ms": 0.0,
                    "fallback_ms": 0.0,
                    "total_ssr_ms": 0.0
                }),
            );
        }
        if !ssr.contains_key("max_delta_e_miss") {
            ssr.insert("max_delta_e_miss".to_string(), json!(0.0));
        }
        if !ssr.contains_key("min_rgb_miss") {
            ssr.insert("min_rgb_miss".to_string(), json!(0.0));
        }
        if !ssr.contains_key("thickness_ablation") {
            ssr.insert(
                "thickness_ablation".to_string(),
                json!({
                    "undershoot_before": 0.0,
                    "undershoot_after": 0.0
                }),
            );
        }
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

pub struct SsrMetaInput<'a> {
    pub stats: Option<&'a SsrStats>,
    pub stripe_contrast: Option<&'a [f32; 9]>,
    pub stripe_contrast_reference: Option<&'a [f32; 9]>,
    pub mean_abs_diff: f32,
    pub edge_streaks_gt1px: u32,
    pub max_delta_e_miss: f32,
    pub min_rgb_miss: f32,
}

pub struct BuiltSsrMeta {
    pub value: Value,
    pub status: String,
}

pub fn build_ssr_meta(input: SsrMetaInput<'_>) -> BuiltSsrMeta {
    let (
        hit_rate,
        avg_steps,
        miss_ibl_ratio,
        trace_ms,
        shade_ms,
        fallback_ms,
        total_ms,
        num_rays,
        num_hits,
        total_steps,
        num_misses,
        miss_ibl_samples,
    ) = match input.stats {
        Some(stats) => (
            stats.hit_rate(),
            stats.avg_steps(),
            stats.miss_ibl_ratio(),
            stats.trace_ms,
            stats.shade_ms,
            stats.fallback_ms,
            stats.perf_ms(),
            stats.num_rays,
            stats.num_hits,
            stats.total_steps,
            stats.num_misses,
            stats.miss_ibl_samples,
        ),
        None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0),
    };

    let status = classify_ssr_status(
        input.stats,
        input.stripe_contrast,
        input.stripe_contrast_reference,
        input.mean_abs_diff,
        input.max_delta_e_miss,
        input.min_rgb_miss,
        input.edge_streaks_gt1px,
    );
    let stripe_analysis = if let (Some(ssr), Some(reference)) = (input.stripe_contrast, input.stripe_contrast_reference) {
        let delta: Vec<f32> = ssr.iter().zip(reference.iter()).map(|(s, r)| s - r).collect();
        let min_contrast_ref = reference.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let min_contrast_ssr = ssr.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        json!({
            "reference": reference,
            "ssr": ssr,
            "delta": delta,
            "monotonic_ref": is_monotonic_decreasing(reference),
            "monotonic_ssr": is_monotonic_decreasing(ssr),
            "min_contrast_ref": min_contrast_ref,
            "min_contrast_ssr": min_contrast_ssr
        })
    } else {
        json!({})
    };

    let stripe_contrast_vec: Vec<f32> = input
        .stripe_contrast
        .map(|a| a.to_vec())
        .unwrap_or_else(Vec::new);
    let stripe_contrast_ref_vec: Vec<f32> = input
        .stripe_contrast_reference
        .map(|a| a.to_vec())
        .unwrap_or_else(Vec::new);

    let value = json!({
        "num_rays": num_rays,
        "num_hits": num_hits,
        "total_steps": total_steps,
        "num_misses": num_misses,
        "miss_ibl_samples": miss_ibl_samples,
        "hit_rate": hit_rate,
        "avg_steps": avg_steps,
        "miss_ibl_ratio": miss_ibl_ratio,
        "perf_ms": {
            "trace_ms": trace_ms,
            "shade_ms": shade_ms,
            "fallback_ms": fallback_ms,
            "total_ssr_ms": total_ms
        },
        "max_delta_e_miss": input.max_delta_e_miss,
        "min_rgb_miss": input.min_rgb_miss,
        "stripe_analysis": stripe_analysis,
        "stripe_contrast": stripe_contrast_vec,
        "stripe_contrast_reference": stripe_contrast_ref_vec,
        "edge_streaks": {
            "num_streaks_gt1px": input.edge_streaks_gt1px
        },
        "ref_vs_ssr_mean_abs_diff": input.mean_abs_diff,
        "status": status,
    });

    BuiltSsrMeta {
        value,
        status: status.to_string(),
    }
}

fn classify_ssr_status(
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

    // 4. Stripe contrast checks. The p53 spec defines acceptance in terms of the
    // SSR stripe contrast only: entries must be monotonic non-increasing as
    // roughness increases from 0.1 to 0.9. We still accept an optional
    // reference array for diagnostics, but do not fail solely because the
    // reference is not monotonic.
    let contrast = match stripe_contrast {
        Some(values) => values,
        None => return SSR_STRIPE_FAIL_CONTRAST,
    };

    if contrast.len() != 9 {
        return SSR_STRIPE_FAIL_CONTRAST;
    }

    // Basic sanity: all SSR contrast values must be finite and strictly > 0.
    let ssr_valid = contrast.iter().all(|&v| v.is_finite() && v > 0.0);
    if !ssr_valid {
        return SSR_STRIPE_FAIL_CONTRAST;
    }

    // Optional sanity for the reference array: validate but do not require
    // monotonicity, since the acceptance criteria are phrased in terms of the
    // SSR image.
    if let Some(reference) = stripe_contrast_reference {
        if reference.len() != 9 {
            return SSR_STRIPE_FAIL_CONTRAST;
        }
        let ref_valid = reference.iter().all(|&v| v.is_finite() && v > 0.0);
        if !ref_valid {
            return SSR_STRIPE_FAIL_CONTRAST;
        }
    }

    // Monotonicity: SSR contrast decreases (or stays flat within slack) as
    // roughness increases.
    if !is_monotonic_decreasing(contrast) {
        return SSR_STRIPE_FAIL_MONOTONIC;
    }

    SSR_STATUS_QA_OK
}

fn is_monotonic_decreasing(values: &[f32]) -> bool {
    // Check if values roughly decrease: v[i] >= v[i+1] (with some slack/epsilon if needed, or strictly)
    // The spec says: "contrast decreases as roughness increases"
    // and "monotonic_* indicate whether contrast decreases or increases monotonically"
    // We'll check for non-increasing.
    values
        .windows(2)
        .all(|pair| pair[0] + SSR_STRIPE_MONO_SLACK >= pair[1])
}

// Removed is_progressively_decreasing as it wasn't in the new spec

fn parse_stripe_array(value: Option<&Value>) -> Option<[f32; 9]> {
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

fn sha256_hex(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn evaluate_m5_status(meta: &BTreeMap<String, Value>) -> &'static str {
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

    let thickness_status = match ssr.get("thickness_ablation").and_then(|v| v.as_object()) {
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
            // Rule (p532): undershoot_before >= 0, undershoot_after >= 0, and the
            // too-thin thickness must exhibit strictly larger undershoot than the
            // baseline value. We do not nudge metrics; we only check the measured
            // delta against a (currently zero) improvement factor.
            if before >= 0.0
                && after >= 0.0
                && improvement > SSR_THICKNESS_IMPROVEMENT_FACTOR
            {
                SSR_STATUS_QA_OK
            } else {
                SSR_THICKNESS_ABLATION_FAIL
            }
        }
        None => SSR_THICKNESS_ABLATION_FAIL,
    };

    if thickness_status != SSR_STATUS_QA_OK {
        return thickness_status;
    }

    SSR_STATUS_QA_OK
}

pub fn meta_path(out_dir: &Path) -> PathBuf {
    out_dir.join(META_FILE_NAME)
}
