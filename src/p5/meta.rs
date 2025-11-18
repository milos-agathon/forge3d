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
        input.mean_abs_diff,
        input.min_rgb_miss,
    );
    let stripe_values: Vec<f32> = input
        .stripe_contrast
        .map(|arr| arr.iter().copied().collect())
        .unwrap_or_default();

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
        "stripe_contrast": stripe_values,
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
    mean_abs_diff: f32,
    min_rgb_miss: f32,
) -> &'static str {
    // Require stats to be present and sane
    let stats = match stats {
        Some(stats) if stats.num_rays > 0 => stats,
        _ => return "SSR_UNINITIALIZED",
    };

    if !stats.hit_rate().is_finite()
        || stats.hit_rate() <= 0.0
        || !stats.avg_steps().is_finite()
        || stats.avg_steps() <= 0.0
        || !stats.perf_ms().is_finite()
        || stats.perf_ms() <= 0.0
        || !stats.miss_ibl_ratio().is_finite()
        || stats.miss_ibl_ratio() < 0.0
    {
        return "SSR_STATS_INVALID";
    }

    let contrast = match stripe_contrast {
        Some(values) => values,
        None => return "SSR_STRIPE_FAIL",
    };
    // Require finiteness only; absolute magnitude handled by acceptance later
    if contrast.iter().any(|v| !v.is_finite()) {
        return "SSR_STRIPE_FAIL";
    }
    // Monotonic non-increasing with a small numerical slack
    if !is_monotonic_non_increasing(contrast) {
        return "SSR_STRIPE_FAIL";
    }

    // No strict thresholds on absolute contrast or diff for acceptance; ensure inputs are finite
    if !mean_abs_diff.is_finite() {
        return "SSR_DIFF_INVALID";
    }
    if !min_rgb_miss.is_finite() {
        return "SSR_FAIL_BLACK_HOLES";
    }

    // All checks passed
    "SHADE_READY"
}

fn is_monotonic_non_increasing(values: &[f32]) -> bool {
    values.windows(2).all(|pair| pair[0] + 1e-3 >= pair[1])
}

fn sha256_hex(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn evaluate_m5_status(meta: &BTreeMap<String, Value>) -> &'static str {
    let ssr = match meta.get("ssr").and_then(|v| v.as_object()) {
        Some(obj) => obj,
        None => return "SSR_UNINITIALIZED",
    };

    // 1) Trace validity: require reasonable hit rate and hit count share
    let num_rays = ssr.get("num_rays").and_then(|v| v.as_u64()).unwrap_or(0) as f32;
    let num_hits = ssr.get("num_hits").and_then(|v| v.as_u64()).unwrap_or(0) as f32;
    let hit_rate = ssr.get("hit_rate").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    if num_rays <= 0.0 {
        return "SSR_UNINITIALIZED";
    }
    let hits_share_ok = num_hits >= 0.1 * num_rays; // >10% of rays
    let hit_rate_ok = hit_rate.is_finite() && hit_rate >= 0.2 && hit_rate <= 0.8;
    if !(hits_share_ok && hit_rate_ok) {
        return "SSR_TRACE_FAIL";
    }

    // 2) Stripe contrast invariants
    let mut stripe_ok = false;
    if let Some(arr) = ssr.get("stripe_contrast").and_then(|v| v.as_array()) {
        if arr.len() == 9 {
            let mut vals = [0f32; 9];
            let mut valid = true;
            for (i, v) in arr.iter().enumerate().take(9) {
                match v.as_f64() {
                    Some(f) if f.is_finite() && f > 0.0 => vals[i] = f as f32,
                    _ => { valid = false; break; }
                }
            }
            if valid {
                let mut mono = true;
                for i in 0..8 {
                    if !(vals[i] >= vals[i + 1] + 0.005) { mono = false; break; }
                }
                let s0_ok = vals[0] >= 0.02;
                let sep_ok = vals[8] >= 0.0 && vals[0] >= 3.0 * vals[8].max(1e-6);
                stripe_ok = mono && s0_ok && sep_ok;
            }
        }
    }
    if !stripe_ok { return "SSR_STRIPE_CONTRAST_FAIL"; }

    // 3) Fallback quality invariants
    let min_rgb_miss = ssr.get("min_rgb_miss").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let max_delta_e_miss = ssr.get("max_delta_e_miss").and_then(|v| v.as_f64()).unwrap_or(f64::INFINITY as f64) as f32;
    let fallback_ok = min_rgb_miss >= (2.0 / 255.0) && max_delta_e_miss <= 2.0;
    if !fallback_ok { return "SSR_FALLBACK_FAIL"; }

    // 4) Edge streaks threshold
    let streaks_ok = ssr
        .get("edge_streaks")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("num_streaks_gt1px"))
        .and_then(|v| v.as_u64())
        .map(|n| n <= 5)
        .unwrap_or(true);
    if !streaks_ok { return "SSR_EDGE_STREAK_FAIL"; }

    // Passed all
    "SSR_SHADE_READY"
}

pub fn meta_path(out_dir: &Path) -> PathBuf {
    out_dir.join(META_FILE_NAME)
}
