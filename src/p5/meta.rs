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
    fn default_ssr_value() -> Value {
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
            "min_rgb_miss": 0.0,
            "stripe_contrast": [],
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
    let legacy_status = meta.remove("ssr_status");

    let ssr_entry = meta.entry("ssr".to_string()).or_insert_with(default_ssr_value);
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
                ssr.insert("status".to_string(), val);
            }
            None => {
                ssr.entry("status".to_string())
                    .or_insert_with(|| json!("UNINITIALIZED"));
            }
        }
    }
}

fn sha256_hex(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn compute_p52_status(meta: &BTreeMap<String, Value>) -> String {
    let ssr = match meta.get("ssr").and_then(|v| v.as_object()) {
        Some(obj) => obj,
        None => return "SSR_UNINITIALIZED".to_string(),
    };
    let perf_total = ssr
        .get("perf_ms")
        .and_then(|v| v.as_object())
        .and_then(|perf| perf.get("total_ssr_ms"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    if perf_total <= 0.0 {
        return "SSR_UNINITIALIZED".to_string();
    }
    let raw_status = ssr
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("UNINITIALIZED");
    if raw_status.starts_with("SSR_") {
        raw_status.to_string()
    } else {
        format!("SSR_{raw_status}")
    }
}

pub fn meta_path(out_dir: &Path) -> PathBuf {
    out_dir.join(META_FILE_NAME)
}
