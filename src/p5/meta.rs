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

    let status = compute_p52_status(&meta);
    meta.insert("status".to_string(), json!(status));

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
    if !meta.contains_key("ssr") {
        meta.insert(
            "ssr".to_string(),
            json!({
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
                "min_rgb_miss": 0.0
            }),
        );
    }
    if !meta.contains_key("ssr_stripe_contrast") {
        meta.insert("ssr_stripe_contrast".to_string(), json!([]));
    }
    if !meta.contains_key("ssr_edge_streaks") {
        meta.insert(
            "ssr_edge_streaks".to_string(),
            json!({
                "num_streaks_gt1px": 0
            }),
        );
    }
    if !meta.contains_key("ssr_status") {
        meta.insert("ssr_status".to_string(), json!("UNINITIALIZED"));
    }
}

fn sha256_hex(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn compute_p52_status(meta: &BTreeMap<String, Value>) -> String {
    let ssgi = meta.get("ssgi").and_then(|v| v.as_object());
    let bounce = meta.get("ssgi_bounce").and_then(|v| v.as_object());
    let perf_ok = ssgi
        .and_then(|ssgi| ssgi.get("perf_ms"))
        .and_then(|perf| perf.as_object())
        .map(|perf| {
            perf.get("total_ssgi_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                > 0.0
        })
        .unwrap_or(false);

    let bounce_ok = bounce.map_or(false, |obj| {
        let red = obj
            .get("red_pct")
            .and_then(|v| v.as_f64())
            .unwrap_or_default();
        let green = obj
            .get("green_pct")
            .and_then(|v| v.as_f64())
            .unwrap_or_default();
        (0.05..=0.12).contains(&red) && (0.05..=0.12).contains(&green)
    });
    let delta_e_ok = ssgi
        .and_then(|obj| obj.get("max_delta_e"))
        .and_then(|v| v.as_f64())
        .map(|v| v <= 1.0)
        .unwrap_or(false);
    let ssim_ok = meta
        .get("ssgi_temporal")
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("ssim_frame8_9"))
        .and_then(|v| v.as_f64())
        .map(|ssim| ssim >= 0.95)
        .unwrap_or(false);

    if bounce_ok && delta_e_ok && ssim_ok && perf_ok {
        "OK".to_string()
    } else {
        let mut reasons = Vec::new();
        if !bounce_ok {
            let red = bounce
                .and_then(|obj| obj.get("red_pct"))
                .and_then(|v| v.as_f64())
                .unwrap_or_default();
            let green = bounce
                .and_then(|obj| obj.get("green_pct"))
                .and_then(|v| v.as_f64())
                .unwrap_or_default();
            reasons.push(format!(
                "bounce: red={:.3} green={:.3} (need 0.05-0.12)",
                red, green
            ));
        }
        if !delta_e_ok {
            let delta_e = ssgi
                .and_then(|obj| obj.get("max_delta_e"))
                .and_then(|v| v.as_f64())
                .unwrap_or_default();
            reasons.push(format!("max_delta_e: {:.3} (need <=1.0)", delta_e));
        }
        if !ssim_ok {
            let ssim = meta
                .get("ssgi_temporal")
                .and_then(|obj| obj.get("ssim_frame8_9"))
                .and_then(|v| v.as_f64())
                .unwrap_or_default();
            reasons.push(format!("ssim: {:.3} (need >=0.95)", ssim));
        }
        if !perf_ok {
            reasons.push("perf_ms must be >0".to_string());
        }
        format!("FAIL: {}", reasons.join(", "))
    }
}

pub fn meta_path(out_dir: &Path) -> PathBuf {
    out_dir.join(META_FILE_NAME)
}
