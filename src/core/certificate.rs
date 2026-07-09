//! CENSOR Task 9: RenderCertificate execution report.
//!
//! Assembles a deterministic, machine-readable record of the LAST completed
//! native render: the engine revision, the shader hashes that were actually
//! handed to naga, the adapter and negotiated capabilities, per-pass live GPU
//! timings, the peak allocation ledger, and every recorded degradation.
//!
//! Report assembly ONLY — the Ed25519 signing/verification lives Python-side
//! (Task 10). Determinism is load-bearing: every map is a [`BTreeMap`],
//! degradations are sorted by `(kind, name)`, and passes are kept in the order
//! they were recorded. Two reports built back-to-back from identical process
//! state serialize byte-for-byte identically except for the live `gpu_ms`
//! timing values.

use crate::core::degradation::degradations_snapshot;
use crate::core::error::RenderError;
use crate::core::resource_tracker::ledger_snapshot;
use crate::core::shader_registry::shader_hashes_snapshot;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// One timed GPU pass in the execution report.
#[derive(Clone, Debug)]
pub struct PassRecord {
    pub label: String,
    pub gpu_ms: f64,
    pub draw_calls: u32,
}

#[derive(Clone, Default)]
struct AdapterSnapshot {
    vendor: String,
    device: String,
    backend: String,
    driver_info: String,
}

/// Immutable snapshot of one completed render capture. Everything needed to
/// serialize the certificate is captured at `finish_render_capture()` time so
/// `execution_report_json()` is a pure function of this frozen state.
struct FinishedCapture {
    wgsl_module_hashes: BTreeMap<String, String>,
    adapter: AdapterSnapshot,
    requested: Vec<String>,
    granted: Vec<String>,
    limits: BTreeMap<String, u64>,
    passes: Vec<PassRecord>,
    peak_host_visible_bytes: u64,
    peak_device_local_bytes: u64,
    by_label: BTreeMap<String, u64>,
    /// (kind, name, consequence), sorted by (kind, name).
    degradations: Vec<(String, String, String)>,
}

/// Passes recorded for the render currently in flight.
static CURRENT: Mutex<Vec<PassRecord>> = Mutex::new(Vec::new());
/// The last completed capture, serialized by `execution_report_json`.
static LAST: Mutex<Option<FinishedCapture>> = Mutex::new(None);

fn lock_current() -> std::sync::MutexGuard<'static, Vec<PassRecord>> {
    CURRENT.lock().unwrap_or_else(|p| p.into_inner())
}

fn lock_last() -> std::sync::MutexGuard<'static, Option<FinishedCapture>> {
    LAST.lock().unwrap_or_else(|p| p.into_inner())
}

/// Start a render capture: clears the per-render pass list. `entry_point` names
/// the render entry for logging/debugging (not part of the serialized schema).
pub fn begin_render_capture(entry_point: &str) {
    let mut cur = lock_current();
    cur.clear();
    log::debug!("render capture begin: {entry_point}");
}

/// Record one timed GPU pass for the in-flight render, in call order.
pub fn record_pass(label: &str, gpu_ms: f64, draw_calls: u32) {
    lock_current().push(PassRecord {
        label: label.to_string(),
        gpu_ms,
        draw_calls,
    });
}

/// Finish the in-flight capture: snapshot the allocation ledger, degradations,
/// shader hashes, adapter info, and negotiated capabilities into the last
/// completed report. Adapter/capability info is read from the process GPU
/// context when one already exists; certificate assembly never forces GPU
/// initialization.
pub fn finish_render_capture() {
    let passes = lock_current().clone();
    let ledger = ledger_snapshot();

    let mut degradations: Vec<(String, String, String)> = degradations_snapshot()
        .into_iter()
        .map(|d| (d.kind, d.name, d.consequence))
        .collect();
    degradations.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let wgsl_module_hashes = shader_hashes_snapshot();

    let (adapter, requested, granted, limits) = match crate::core::gpu::ctx_if_initialized() {
        Some(ctx) => {
            let info = ctx.adapter.get_info();
            let adapter = AdapterSnapshot {
                vendor: info.vendor.to_string(),
                device: info.name.clone(),
                backend: format!("{:?}", info.backend).to_lowercase(),
                driver_info: info.driver_info.clone(),
            };
            let requested = ctx
                .capabilities
                .wanted_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            let granted = ctx
                .capabilities
                .granted_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            let l = ctx.device.limits();
            let mut limits = BTreeMap::new();
            limits.insert(
                "max_texture_dimension_2d".to_string(),
                l.max_texture_dimension_2d as u64,
            );
            limits.insert("max_buffer_size".to_string(), l.max_buffer_size);
            limits.insert("max_bind_groups".to_string(), l.max_bind_groups as u64);
            limits.insert(
                "max_storage_buffers_per_shader_stage".to_string(),
                l.max_storage_buffers_per_shader_stage as u64,
            );
            limits.insert(
                "min_uniform_buffer_offset_alignment".to_string(),
                l.min_uniform_buffer_offset_alignment as u64,
            );
            limits.insert(
                "min_storage_buffer_offset_alignment".to_string(),
                l.min_storage_buffer_offset_alignment as u64,
            );
            (adapter, requested, granted, limits)
        }
        None => (
            AdapterSnapshot::default(),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        ),
    };

    let finished = FinishedCapture {
        wgsl_module_hashes,
        adapter,
        requested,
        granted,
        limits,
        passes,
        peak_host_visible_bytes: ledger.peak_host_visible_bytes,
        peak_device_local_bytes: ledger.peak_device_local_bytes,
        by_label: ledger.by_label,
        degradations,
    };

    *lock_last() = Some(finished);
}

// ---------------------------------------------------------------------------
// Serialization (borrowed views over the frozen FinishedCapture).
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct EngineJson<'a> {
    version: &'a str,
    git_sha: &'a str,
    wgsl_module_hashes: &'a BTreeMap<String, String>,
}

#[derive(Serialize)]
struct AdapterJson<'a> {
    vendor: &'a str,
    device: &'a str,
    backend: &'a str,
    driver_info: &'a str,
}

#[derive(Serialize)]
struct CapabilitiesJson<'a> {
    requested: &'a [String],
    granted: &'a [String],
    limits: &'a BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct PassJson<'a> {
    label: &'a str,
    gpu_ms: f64,
    draw_calls: u32,
}

#[derive(Serialize)]
struct AllocationsJson<'a> {
    peak_host_visible_bytes: u64,
    peak_device_local_bytes: u64,
    by_label: &'a BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct DegradationJson<'a> {
    kind: &'a str,
    name: &'a str,
    consequence: &'a str,
}

#[derive(Serialize)]
struct ReportJson<'a> {
    schema: &'a str,
    engine: EngineJson<'a>,
    adapter: AdapterJson<'a>,
    capabilities: CapabilitiesJson<'a>,
    passes: Vec<PassJson<'a>>,
    allocations: AllocationsJson<'a>,
    degradations: Vec<DegradationJson<'a>>,
}

/// Serialize the LAST completed render capture as the canonical certificate
/// JSON. Deterministic: BTreeMap key order, recorded pass order, and
/// (kind, name)-sorted degradations. Errors when no render has completed yet.
pub fn execution_report_json() -> Result<String, RenderError> {
    let last = lock_last();
    let cap = last.as_ref().ok_or_else(|| {
        RenderError::render(
            "no completed render capture available; run a native render \
             (begin/finish_render_capture) before requesting the execution report",
        )
    })?;

    let report = ReportJson {
        schema: "forge3d.render_certificate/1",
        engine: EngineJson {
            version: env!("CARGO_PKG_VERSION"),
            git_sha: env!("FORGE3D_GIT_SHA"),
            wgsl_module_hashes: &cap.wgsl_module_hashes,
        },
        adapter: AdapterJson {
            vendor: &cap.adapter.vendor,
            device: &cap.adapter.device,
            backend: &cap.adapter.backend,
            driver_info: &cap.adapter.driver_info,
        },
        capabilities: CapabilitiesJson {
            requested: &cap.requested,
            granted: &cap.granted,
            limits: &cap.limits,
        },
        passes: cap
            .passes
            .iter()
            .map(|p| PassJson {
                label: &p.label,
                gpu_ms: p.gpu_ms,
                draw_calls: p.draw_calls,
            })
            .collect(),
        allocations: AllocationsJson {
            peak_host_visible_bytes: cap.peak_host_visible_bytes,
            peak_device_local_bytes: cap.peak_device_local_bytes,
            by_label: &cap.by_label,
        },
        degradations: cap
            .degradations
            .iter()
            .map(|(kind, name, consequence)| DegradationJson {
                kind,
                name,
                consequence,
            })
            .collect(),
    };

    serde_json::to_string(&report)
        .map_err(|e| RenderError::render(format!("certificate serialization failed: {e}")))
}

/// Honor a `certificate=` render kwarg from the Python boundary for the LAST
/// completed native render.
///
/// The value follows the render entry-point contract: `None`/`False` is off;
/// `True` assembles a signed certificate via `forge3d.diagnostics`; any other
/// value is treated as a filesystem path and the signed certificate is written
/// there via `forge3d.certificate.write_certificate`. Assembly/signing and
/// writing are delegated to the pure-Python surface so there is one signing
/// implementation.
#[cfg(feature = "extension-module")]
pub fn emit_certificate_for_kwarg(
    py: pyo3::Python<'_>,
    certificate: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyAnyMethods;
    let Some(arg) = certificate else {
        return Ok(());
    };
    if arg.is_none() {
        return Ok(());
    }
    // A real Python bool: False -> off, True -> assemble-only (no file).
    let is_path = match arg.extract::<bool>() {
        Ok(false) => return Ok(()),
        Ok(true) => false,
        Err(_) => true,
    };

    let diagnostics = py.import_bound("forge3d.diagnostics")?;
    let cert = diagnostics.call_method1("render_certificate", ())?;
    if is_path {
        let certificate_mod = py.import_bound("forge3d.certificate")?;
        certificate_mod.call_method1("write_certificate", (cert, arg))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes global state, so back-to-back building tests must not
    /// interleave. `cargo test certificate --lib` runs multi-threaded by
    /// default; serialize the whole-module state behind one lock.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn zero_gpu_ms(value: &mut serde_json::Value) {
        if let Some(passes) = value.get_mut("passes").and_then(|p| p.as_array_mut()) {
            for pass in passes {
                pass["gpu_ms"] = serde_json::json!(0.0);
            }
        }
    }

    #[test]
    fn canonical_payload_is_byte_stable_modulo_gpu_ms() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        begin_render_capture("test.render");
        record_pass("terrain.main", 1.5, 3);
        record_pass("terrain.sky", 0.25, 1);
        finish_render_capture();
        let a = execution_report_json().expect("first report must assemble");

        // Rebuild from identical state with DIFFERENT live timings.
        begin_render_capture("test.render");
        record_pass("terrain.main", 42.0, 3);
        record_pass("terrain.sky", 99.0, 1);
        finish_render_capture();
        let b = execution_report_json().expect("second report must assemble");

        // Both are valid JSON.
        let mut va: serde_json::Value = serde_json::from_str(&a).expect("report a parses");
        let mut vb: serde_json::Value = serde_json::from_str(&b).expect("report b parses");

        assert_eq!(va["schema"], "forge3d.render_certificate/1");
        // Passes are preserved in recorded order.
        assert_eq!(va["passes"][0]["label"], "terrain.main");
        assert_eq!(va["passes"][1]["label"], "terrain.sky");

        // After zeroing the only nondeterministic field, the payloads are
        // byte-identical.
        zero_gpu_ms(&mut va);
        zero_gpu_ms(&mut vb);
        assert_eq!(
            va, vb,
            "certificate must be byte-stable across renders once gpu_ms is zeroed"
        );
    }

    #[test]
    fn finish_without_begin_does_not_panic() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // No matching begin_render_capture: must not panic, and the report
        // (if any prior state exists) must still be serializable.
        finish_render_capture();
        let _ = execution_report_json();
    }
}
