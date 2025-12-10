//! P5 metadata writing and SSR metrics
//!
//! Provides functionality for writing P5 QA metadata including SSR stats,
//! stripe contrast analysis, and shader hashes.

pub mod constants;
pub mod defaults;
pub mod ssr_status;

use crate::passes::ssr::SsrStats;
use anyhow::Context;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub use constants::{DEFAULT_REPORT_DIR, META_FILE_NAME};
pub use defaults::ensure_ssr_defaults;
pub use ssr_status::{
    classify_ssr_status, evaluate_m5_status, is_monotonic_decreasing, parse_stripe_array,
    patch_thickness_ablation,
};

/// Write P5 meta file with a patch function
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

    // M5 status logic
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

fn insert_shader_hashes(meta: &mut BTreeMap<String, Value>) {
    let mut h = BTreeMap::new();
    h.insert(
        "ssao/common.wgsl".to_string(),
        sha256_hex(include_str!("../../shaders/ssao/common.wgsl")),
    );
    h.insert(
        "ssao/ssao.wgsl".to_string(),
        sha256_hex(include_str!("../../shaders/ssao/ssao.wgsl")),
    );
    h.insert(
        "ssao/gtao.wgsl".to_string(),
        sha256_hex(include_str!("../../shaders/ssao/gtao.wgsl")),
    );
    h.insert(
        "ssao/composite.wgsl".to_string(),
        sha256_hex(include_str!("../../shaders/ssao/composite.wgsl")),
    );
    h.insert(
        "filters/bilateral_separable.wgsl".to_string(),
        sha256_hex(include_str!("../../shaders/filters/bilateral_separable.wgsl")),
    );
    h.insert(
        "temporal/resolve_ao.wgsl".to_string(),
        sha256_hex(include_str!("../../shaders/temporal/resolve_ao.wgsl")),
    );
    meta.insert("hashes".to_string(), json!(h));
}

fn sha256_hex(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Input data for building SSR meta
pub struct SsrMetaInput<'a> {
    pub stats: Option<&'a SsrStats>,
    pub stripe_contrast: Option<&'a [f32; 9]>,
    pub stripe_contrast_reference: Option<&'a [f32; 9]>,
    pub mean_abs_diff: f32,
    pub edge_streaks_gt1px: u32,
    pub max_delta_e_miss: f32,
    pub min_rgb_miss: f32,
}

/// Built SSR meta result
pub struct BuiltSsrMeta {
    pub value: Value,
    pub status: String,
}

/// Build SSR meta JSON from input data
pub fn build_ssr_meta(input: SsrMetaInput<'_>) -> BuiltSsrMeta {
    let (hit_rate, avg_steps, miss_ibl_ratio, trace_ms, shade_ms, fallback_ms, total_ms, num_rays, num_hits, total_steps, num_misses, miss_ibl_samples) =
        extract_stats_values(input.stats);

    let status = classify_ssr_status(
        input.stats,
        input.stripe_contrast,
        input.stripe_contrast_reference,
        input.mean_abs_diff,
        input.max_delta_e_miss,
        input.min_rgb_miss,
        input.edge_streaks_gt1px,
    );

    let stripe_analysis = build_stripe_analysis(input.stripe_contrast, input.stripe_contrast_reference);
    let stripe_contrast_vec: Vec<f32> = input.stripe_contrast.map(|a| a.to_vec()).unwrap_or_default();
    let stripe_ref_vec: Vec<f32> = input.stripe_contrast_reference.map(|a| a.to_vec()).unwrap_or_default();

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
        "stripe_contrast_reference": stripe_ref_vec,
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

fn extract_stats_values(stats: Option<&SsrStats>) -> (f32, f32, f32, f32, f32, f32, f32, u32, u32, u32, u32, u32) {
    match stats {
        Some(s) => (
            s.hit_rate(),
            s.avg_steps(),
            s.miss_ibl_ratio(),
            s.trace_ms,
            s.shade_ms,
            s.fallback_ms,
            s.perf_ms(),
            s.num_rays,
            s.num_hits,
            s.total_steps,
            s.num_misses,
            s.miss_ibl_samples,
        ),
        None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0),
    }
}

fn build_stripe_analysis(ssr: Option<&[f32; 9]>, reference: Option<&[f32; 9]>) -> Value {
    if let (Some(ssr), Some(reference)) = (ssr, reference) {
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
    }
}

/// Get the meta file path for a given output directory
pub fn meta_path(out_dir: &Path) -> PathBuf {
    out_dir.join(META_FILE_NAME)
}
