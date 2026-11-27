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

use crate::core::degradation::{begin_degradation_capture, finish_degradation_capture};
use crate::core::error::RenderError;
use crate::core::resource_tracker::{
    abort_ledger_capture, begin_ledger_capture, extend_ledger_capture, finish_ledger_capture,
};
use crate::core::shader_registry::{begin_shader_render_capture, finish_shader_render_capture};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

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
#[derive(Clone)]
struct FinishedCapture {
    process_memory: Option<Box<serde_json::value::RawValue>>,
    flythrough_process_memory: Option<Box<serde_json::value::RawValue>>,
    wgsl_module_hashes: BTreeMap<String, String>,
    adapter: AdapterSnapshot,
    requested: Vec<String>,
    granted: Vec<String>,
    limits: BTreeMap<String, u64>,
    inputs: BTreeMap<String, String>,
    passes: Vec<PassRecord>,
    /// Signed model declarations are properties, not GPU pass claims.
    models: BTreeMap<String, String>,
    ledger: crate::core::resource_tracker::LedgerReport,
    /// (kind, name, consequence), sorted by (kind, name).
    degradations: Vec<(String, String, String)>,
    codec: Option<CodecSnapshot>,
    precision: Option<PrecisionEvidence>,
    jitter: Option<JitterEvidence>,
}

#[derive(Clone)]
struct CodecSnapshot {
    eps: f32,
    pages_base_quality: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrecisionEvidence {
    pub backend: String,
    pub adapter: String,
    pub operation: String,
    pub two_prod_variant: String,
    pub shader_label: String,
    pub shader_hash: String,
    pub generated_count: u64,
    pub adversarial_count: u64,
    pub mismatch_count: u64,
    pub max_err_u2: f64,
    pub cited_bound_u2: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct JitterEvidence {
    pub unit: String,
    pub backend: String,
    pub adapter: String,
    pub two_prod_variant: String,
    pub shader_label: String,
    pub shader_hash: String,
    pub frame_count: u32,
    pub camera_step_metres: f64,
    pub dd_max_error_px: f64,
    pub threshold_px: f64,
    pub raw_max_error_px: f64,
    pub raw_over_one_px: u32,
    pub dd_hash_a: String,
    pub dd_hash_b: String,
}

/// The latest process-wide capture for the explicit diagnostics getter.
static LAST: Mutex<Option<Arc<FinishedCapture>>> = Mutex::new(None);

thread_local! {
    static LAST_ON_THREAD: RefCell<Option<Arc<FinishedCapture>>> = const { RefCell::new(None) };
    static CAPTURE_GENERATION: Cell<u64> = const { Cell::new(0) };
    static CURRENT_PASSES: RefCell<Vec<PassRecord>> = const { RefCell::new(Vec::new()) };
    static CAPTURE_USES_GPU: Cell<bool> = const { Cell::new(true) };
    static EXTERNAL_CAPTURE: RefCell<Option<RenderCaptureGuard>> = const { RefCell::new(None) };
    static CAPTURE_DEPTH: Cell<usize> = const { Cell::new(0) };
    static CURRENT_CODEC: RefCell<Option<CodecSnapshot>> = const { RefCell::new(None) };
    static CURRENT_PRECISION: RefCell<Option<PrecisionEvidence>> = const { RefCell::new(None) };
    static CURRENT_JITTER: RefCell<Option<JitterEvidence>> = const { RefCell::new(None) };
    static CURRENT_INPUTS: RefCell<BTreeMap<String, String>> = const { RefCell::new(BTreeMap::new()) };
    static CURRENT_MODELS: RefCell<BTreeMap<String, String>> = const { RefCell::new(BTreeMap::new()) };
}

pub fn record_precision_evidence(evidence: PrecisionEvidence) {
    CURRENT_PRECISION.with(|slot| *slot.borrow_mut() = Some(evidence));
}

pub fn record_jitter_evidence(evidence: JitterEvidence) {
    CURRENT_JITTER.with(|slot| *slot.borrow_mut() = Some(evidence));
}

#[must_use = "a render capture must be finished or retained until the render exits"]
pub struct RenderCaptureGuard {
    active: bool,
    root: bool,
}

impl RenderCaptureGuard {
    pub fn finish(self) {
        self.finish_with_memory(None);
    }

    fn finish_with_memory(mut self, process_memory: Option<Box<serde_json::value::RawValue>>) {
        self.active = false;
        if self.root {
            finish_render_capture(process_memory);
            CAPTURE_DEPTH.with(|depth| depth.set(0));
        } else {
            CAPTURE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        }
    }
}

impl Drop for RenderCaptureGuard {
    fn drop(&mut self) {
        if self.active {
            if self.root {
                abort_render_capture();
                CAPTURE_DEPTH.with(|depth| depth.set(0));
            } else {
                CAPTURE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
            }
        }
    }
}

/// A progress callback is a caller-owned render boundary. Save only the
/// in-flight collectors. A successful callback render owns its own LAST
/// result until the outer render finishes, including when the outer aborts.
#[cfg(feature = "extension-module")]
pub(crate) fn with_render_capture_suspended<T>(callback: impl FnOnce() -> T) -> T {
    use crate::core::degradation::{swap_degradation_capture, Degradation};
    use crate::core::shader_contract_runtime::{
        swap_runtime_contract_capture, RuntimeContractObservation,
    };
    use crate::core::shader_registry::swap_shader_render_capture;

    struct Suspended {
        depth: usize,
        external: Option<RenderCaptureGuard>,
        uses_gpu: bool,
        passes: Vec<PassRecord>,
        inputs: BTreeMap<String, String>,
        models: BTreeMap<String, String>,
        codec: Option<CodecSnapshot>,
        precision: Option<PrecisionEvidence>,
        jitter: Option<JitterEvidence>,
        shaders: Option<BTreeMap<String, String>>,
        degradations: Option<Vec<Degradation>>,
        observations: Option<Vec<RuntimeContractObservation>>,
    }

    impl Drop for Suspended {
        fn drop(&mut self) {
            // If callback code abandoned a Python-owned capture, let it
            // unwind while its collectors are still current.
            abort_external_render_capture();
            EXTERNAL_CAPTURE.with(|slot| slot.replace(self.external.take()));
            CAPTURE_DEPTH.with(|slot| slot.set(self.depth));
            CAPTURE_USES_GPU.set(self.uses_gpu);
            CURRENT_PASSES.replace(std::mem::take(&mut self.passes));
            CURRENT_INPUTS.with(|slot| slot.replace(std::mem::take(&mut self.inputs)));
            CURRENT_MODELS.with(|slot| slot.replace(std::mem::take(&mut self.models)));
            CURRENT_CODEC.with(|slot| slot.replace(self.codec.take()));
            CURRENT_PRECISION.with(|slot| slot.replace(self.precision.take()));
            CURRENT_JITTER.with(|slot| slot.replace(self.jitter.take()));
            swap_shader_render_capture(self.shaders.take());
            swap_degradation_capture(self.degradations.take());
            swap_runtime_contract_capture(self.observations.take());
        }
    }

    let depth = CAPTURE_DEPTH.with(Cell::get);
    if depth == 0 {
        return callback();
    }
    let _ledger = crate::core::resource_tracker::suspend_ledger_capture();
    let _saved = Suspended {
        depth: CAPTURE_DEPTH.with(|slot| slot.replace(0)),
        external: EXTERNAL_CAPTURE.with(|slot| slot.take()),
        uses_gpu: CAPTURE_USES_GPU.replace(true),
        passes: CURRENT_PASSES.take(),
        inputs: CURRENT_INPUTS.with(|slot| slot.take()),
        models: CURRENT_MODELS.with(|slot| slot.take()),
        codec: CURRENT_CODEC.with(|slot| slot.take()),
        precision: CURRENT_PRECISION.with(|slot| slot.take()),
        jitter: CURRENT_JITTER.with(|slot| slot.take()),
        shaders: swap_shader_render_capture(None),
        degradations: swap_degradation_capture(None),
        observations: swap_runtime_contract_capture(None),
    };
    callback()
}

fn lock_last() -> std::sync::MutexGuard<'static, Option<Arc<FinishedCapture>>> {
    LAST.lock().unwrap_or_else(|p| p.into_inner())
}

type DegradationRecords = Vec<(String, String, String)>;

#[cfg(feature = "extension-module")]
fn python_capture_records(method: &str) -> DegradationRecords {
    use pyo3::prelude::PyAnyMethods;

    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
        return Vec::new();
    }
    pyo3::Python::with_gil(|py| {
        let records = py.import_bound("forge3d._degradation")
            .and_then(|module| module.call_method0(method))
            .and_then(|value| if value.is_none() { Ok(Vec::new()) } else { value.extract() });
        match records {
            Ok(records) => records,
            Err(error) => {
                log::warn!("Python degradation capture {method} failed: {error}");
                Vec::new()
            }
        }
    })
}

#[cfg(not(feature = "extension-module"))]
fn python_capture_records(_method: &str) -> DegradationRecords {
    Vec::new()
}

/// Start a render capture: clears the per-render pass list. `entry_point` names
/// the render entry for logging/debugging (not part of the serialized schema).
pub fn begin_render_capture(entry_point: &str) -> RenderCaptureGuard {
    begin_render_capture_with_resources(entry_point, &[])
}

/// Start a render capture for a native CPU renderer. The certificate uses the
/// explicit CPU adapter identity and does not inherit capabilities from an
/// unrelated, already-initialized GPU context.
pub fn begin_cpu_render_capture(entry_point: &str) -> RenderCaptureGuard {
    let root = CAPTURE_DEPTH.with(|depth| depth.get() == 0);
    let capture = begin_render_capture_with_resources(entry_point, &[]);
    if root {
        CAPTURE_USES_GPU.set(false);
    }
    capture
}

/// Start a render capture seeded with the modules owned by its renderer.
/// Start a render capture with renderer-owned shader and allocation state.
pub fn begin_render_capture_with_resources(
    entry_point: &str,
    allocation_owner_ids: &[u64],
) -> RenderCaptureGuard {
    CAPTURE_USES_GPU.set(true);
    let root = CAPTURE_DEPTH.with(|depth| {
        let current = depth.get();
        depth.set(current + 1);
        current == 0
    });
    if !root {
        extend_ledger_capture(allocation_owner_ids);
        log::debug!("nested render capture join: {entry_point}");
        return RenderCaptureGuard {
            active: true,
            root: false,
        };
    }

    CURRENT_CODEC.with(|codec| codec.borrow_mut().take());
    begin_ledger_capture(allocation_owner_ids);
    begin_shader_render_capture(&BTreeMap::new());
    crate::core::shader_contract_runtime::begin_runtime_contract_capture();
    begin_degradation_capture();
    CURRENT_PRECISION.with(|slot| slot.borrow_mut().take());
    CURRENT_JITTER.with(|slot| slot.borrow_mut().take());
    CURRENT_INPUTS.with(|inputs| inputs.borrow_mut().clear());
    CURRENT_MODELS.with(|models| models.borrow_mut().clear());
    python_capture_records("begin_capture");
    CURRENT_PASSES.with(|passes| passes.borrow_mut().clear());
    log::debug!("render capture begin: {entry_point}");
    RenderCaptureGuard {
        active: true,
        root: true,
    }
}

/// Record one timed GPU pass for the in-flight render, in call order.
pub fn record_pass(label: &str, gpu_ms: f64, draw_calls: u32) {
    CURRENT_PASSES.with(|passes| passes.borrow_mut().push(PassRecord {
        label: label.to_string(),
        gpu_ms,
        draw_calls,
    }));
}

/// Record a deterministic render input in the active certificate.
pub fn record_input(key: impl Into<String>, value: impl Into<String>) {
    if CAPTURE_DEPTH.with(|depth| depth.get() > 0) {
        CURRENT_INPUTS.with(|inputs| {
            inputs.borrow_mut().insert(key.into(), value.into());
        });
    }
}

/// Declare a rendering or physical model used by the active render.
///
/// This is deliberately separate from [`record_pass`]: a model assumption is
/// signed certificate context, not a fabricated timed GPU pass.
pub fn record_model(name: &str, description: &str) {
    let active = CAPTURE_DEPTH.with(|depth| depth.get() > 0);
    if active {
        CURRENT_MODELS.with(|models| {
            models
                .borrow_mut()
                .insert(name.to_string(), description.to_string());
        });
    }
}

/// Attach F3DZ source evidence to the render capture currently active on this
/// thread. Multiple tiles accumulate; if their epsilon differs, the
/// certificate records the largest (weakest) bound, which remains honest for
/// every contributing page.
pub fn record_f3dz_pages(eps: f32, page_count: u32, base_quality: bool) {
    let active = CAPTURE_DEPTH.with(|depth| depth.get() > 0);
    if !active || !eps.is_finite() || eps <= 0.0 {
        return;
    }
    CURRENT_CODEC.with(|slot| {
        let mut slot = slot.borrow_mut();
        match slot.as_mut() {
            Some(codec) => {
                codec.eps = codec.eps.max(eps);
                if base_quality {
                    codec.pages_base_quality = codec.pages_base_quality.saturating_add(page_count);
                }
            }
            None => {
                *slot = Some(CodecSnapshot {
                    eps,
                    pages_base_quality: if base_quality { page_count } else { 0 },
                });
            }
        }
    });
}

/// Finish the in-flight capture: snapshot the allocation ledger, degradations,
/// shader hashes, adapter info, and negotiated capabilities into the last
/// completed report. Adapter/capability info is read from the process GPU
/// context when one already exists; certificate assembly never forces GPU
/// initialization.
fn finish_render_capture(process_memory: Option<Box<serde_json::value::RawValue>>) {
    let uses_gpu = CAPTURE_USES_GPU.get();
    let passes = CURRENT_PASSES.take();
    let models = CURRENT_MODELS.with(|models| std::mem::take(&mut *models.borrow_mut()));
    let ledger = finish_ledger_capture();
    let codec = CURRENT_CODEC.with(|slot| slot.borrow_mut().take());

    let mut degradations: Vec<(String, String, String)> = finish_degradation_capture()
        .into_iter()
        .map(|d| (d.kind, d.name, d.consequence))
        .collect();

    // Re-derive capability_absent entries from the live context so a cleared
    // degradation sink cannot certify a capability gap away: the negotiated
    // CapabilitySet is the source of truth for what this device lacks.
    if uses_gpu {
        if let Some(ctx) = crate::core::gpu::ctx_if_initialized() {
            for (name, feature, consequence) in crate::core::capabilities::WANTED {
                let absent = ctx.capabilities.wanted.contains(*feature)
                    && !ctx.capabilities.granted.contains(*feature);
                if absent
                    && !degradations
                        .iter()
                        .any(|(k, n, _)| k == "capability_absent" && n == name)
                {
                    degradations.push((
                        "capability_absent".to_string(),
                        (*name).to_string(),
                        (*consequence).to_string(),
                    ));
                }
            }
        }
    }
    for entry in python_capture_records("finish_capture") {
        if !degradations.iter().any(|(kind, name, _)| *kind == entry.0 && *name == entry.1) {
            degradations.push(entry);
        }
    }
    degradations.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let wgsl_module_hashes = finish_shader_render_capture();
    crate::core::shader_contract_runtime::finish_runtime_contract_capture();
    let (adapter, requested, granted, limits) = if !uses_gpu {
        (
            AdapterSnapshot {
                vendor: "cpu".to_string(),
                device: "python".to_string(),
                backend: "cpu".to_string(),
                driver_info: String::new(),
            },
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
    } else {
        match crate::core::gpu::ctx_if_initialized() {
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
        }
    };

    let finished = FinishedCapture {
        flythrough_process_memory: None,
        process_memory,
        wgsl_module_hashes,
        adapter,
        requested,
        granted,
        limits,
        inputs: CURRENT_INPUTS.with(|inputs| std::mem::take(&mut *inputs.borrow_mut())),
        passes,
        models,
        ledger,
        degradations,
        codec,
        precision: CURRENT_PRECISION.with(|slot| slot.borrow_mut().take()),
        jitter: CURRENT_JITTER.with(|slot| slot.borrow_mut().take()),
    };

    let finished = Arc::new(finished);
    LAST_ON_THREAD.replace(Some(Arc::clone(&finished)));
    CAPTURE_GENERATION.set(CAPTURE_GENERATION.get().wrapping_add(1));
    *lock_last() = Some(finished);
}

#[cfg(feature = "extension-module")]
pub(crate) fn capture_generation() -> u64 {
    CAPTURE_GENERATION.get()
}

/// Publish measurements only on a capture completed by this job. Cache hits
/// without a new capture and newer process-wide captures keep their identity.
#[cfg(feature = "extension-module")]
pub(crate) fn attach_job_memory(
    previous: u64, memory: Box<serde_json::value::RawValue>, flythrough: bool,
) -> bool {
    if capture_generation() == previous || CAPTURE_DEPTH.get() != 0 { return false; }
    LAST_ON_THREAD.with(|slot| {
        let mut own = slot.borrow_mut();
        let Some(old) = own.as_ref() else { return false; };
        let mut next = (**old).clone();
        if flythrough { next.flythrough_process_memory = Some(memory); }
        else { next.process_memory = Some(memory); }
        let next = Arc::new(next);
        let mut latest = lock_last();
        if latest.as_ref().is_some_and(|cap| Arc::ptr_eq(cap, old)) {
            *latest = Some(next.clone());
        }
        *own = Some(next);
        true
    })
}

pub fn abort_render_capture() {
    abort_ledger_capture();
    crate::core::shader_registry::abort_shader_render_capture();
    crate::core::shader_contract_runtime::abort_runtime_contract_capture();
    crate::core::degradation::abort_degradation_capture();
    python_capture_records("abort_capture");
    CURRENT_CODEC.with(|codec| codec.borrow_mut().take());
    CURRENT_PASSES.with(|passes| passes.borrow_mut().clear());
    CURRENT_PRECISION.with(|slot| slot.borrow_mut().take());
    CURRENT_JITTER.with(|slot| slot.borrow_mut().take());
    CURRENT_INPUTS.with(|inputs| inputs.borrow_mut().clear());
    CURRENT_MODELS.with(|models| models.borrow_mut().clear());
}

/// Start a render-local capture owned by a Python renderer.
pub fn begin_external_render_capture(entry_point: &str) {
    EXTERNAL_CAPTURE.with(|slot| {
        slot.borrow_mut().take();
        *slot.borrow_mut() = Some(begin_render_capture(entry_point));
        CAPTURE_USES_GPU.set(false);
    });
}

/// Finish the active Python-owned render capture with one CPU/synthetic pass.
pub fn finish_external_render_capture(
    pass_label: &str,
    draw_calls: u32,
) -> Result<(), RenderError> {
    finish_external_render_capture_with_memory(pass_label, draw_calls, None)
}

pub(crate) fn finish_external_render_capture_with_memory(
    pass_label: &str,
    draw_calls: u32,
    process_memory: Option<Box<serde_json::value::RawValue>>,
) -> Result<(), RenderError> {
    let capture = EXTERNAL_CAPTURE
        .with(|slot| slot.borrow_mut().take())
        .ok_or_else(|| RenderError::render("no Python render capture is active"))?;
    record_pass(pass_label, 0.0, draw_calls);
    capture.finish_with_memory(process_memory);
    Ok(())
}

/// Abort the active Python-owned capture after an exception.
pub fn abort_external_render_capture() {
    EXTERNAL_CAPTURE.with(|slot| {
        slot.borrow_mut().take();
    });
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
struct CodecJson {
    codec: &'static str,
    eps: f32,
    pages_base_quality: u32,
}

#[derive(Serialize)]
struct ReportJson<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    process_memory: Option<&'a serde_json::value::RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    flythrough_process_memory: Option<&'a serde_json::value::RawValue>,
    schema: &'a str,
    engine: EngineJson<'a>,
    adapter: AdapterJson<'a>,
    capabilities: CapabilitiesJson<'a>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    inputs: &'a BTreeMap<String, String>,
    passes: Vec<PassJson<'a>>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    models: &'a BTreeMap<String, String>,
    allocations: AllocationsJson<'a>,
    degradations: Vec<DegradationJson<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    codec: Option<CodecJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precision: Option<&'a PrecisionEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    jitter: Option<&'a JitterEvidence>,
}

/// Serialize the LAST completed render capture as the canonical certificate
/// JSON. Deterministic: BTreeMap key order, recorded pass order, and
/// (kind, name)-sorted degradations. Errors when no render has completed yet.
pub fn execution_report_json() -> Result<String, RenderError> {
    let last = lock_last().clone();
    let cap = last.as_deref().ok_or_else(|| {
        RenderError::render(
            "no completed render capture available; run a native render \
             (begin/finish_render_capture) before requesting the execution report",
        )
    })?;

    serialize_capture(cap)
}

/// Serialize the calling thread's completed capture for render-owned certificates.
pub(crate) fn calling_thread_execution_report_json() -> Result<String, RenderError> {
    if CAPTURE_DEPTH.get() > 0 {
        return Err(RenderError::render("the calling thread's render capture is still active"));
    }
    LAST_ON_THREAD.with(|slot| {
        let last = slot.borrow();
        let capture = last.as_deref().ok_or_else(|| {
            RenderError::render("the calling thread has no completed render capture")
        })?;
        serialize_capture(capture)
    })
}

fn serialize_capture(cap: &FinishedCapture) -> Result<String, RenderError> {
    let report = ReportJson {
        process_memory: cap.process_memory.as_deref(),
        flythrough_process_memory: cap.flythrough_process_memory.as_deref(),
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
        inputs: &cap.inputs,
        passes: cap
            .passes
            .iter()
            .map(|p| PassJson {
                label: &p.label,
                gpu_ms: p.gpu_ms,
                draw_calls: p.draw_calls,
            })
            .collect(),
        models: &cap.models,
        allocations: AllocationsJson {
            peak_host_visible_bytes: cap.ledger.peak_host_visible_bytes,
            peak_device_local_bytes: cap.ledger.peak_device_local_bytes,
            by_label: &cap.ledger.by_label,
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
        codec: cap.codec.as_ref().map(|codec| CodecJson {
            codec: "f3dz/1",
            eps: codec.eps,
            pages_base_quality: codec.pages_base_quality,
        }),
        precision: cap.precision.as_ref(),
        jitter: cap.jitter.as_ref(),
    };

    serde_json::to_string(&report)
        .map_err(|e| RenderError::render(format!("certificate serialization failed: {e}")))
}

pub(crate) fn completed_ledger_report(
) -> Result<crate::core::resource_tracker::LedgerReport, RenderError> {
    LAST_ON_THREAD.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|capture| capture.ledger.clone())
            .ok_or_else(|| RenderError::render("no completed render allocation capture on the calling thread"))
    })
}

/// Sign the canonical RenderCertificate payload digest with Ed25519.
///
/// Canonicalization and SHA256 happen at the Python boundary so the offline
/// verifier can reproduce them without the native module. Signing itself stays
/// here so it uses the crate's pinned `ed25519-dalek` implementation.
pub fn sign_payload_digest(seed: &[u8], digest: &[u8]) -> Result<(String, String), RenderError> {
    use ed25519_dalek::{Signer, SigningKey};

    let seed: [u8; 32] = seed.try_into().map_err(|_| {
        RenderError::render("certificate signing seed must contain exactly 32 bytes")
    })?;
    let digest: [u8; 32] = digest.try_into().map_err(|_| {
        RenderError::render("certificate payload digest must contain exactly 32 bytes")
    })?;

    let signing_key = SigningKey::from_bytes(&seed);
    let mut message = Vec::with_capacity(28 + digest.len());
    message.extend_from_slice(b"forge3d.render_certificate.v1");
    message.extend_from_slice(&digest);
    let signature = signing_key.sign(&message).to_bytes();
    let public_key = signing_key.verifying_key().to_bytes();
    Ok((
        crate::core::provenance::to_hex(&signature),
        crate::core::provenance::to_hex(&public_key),
    ))
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
    let report = calling_thread_execution_report_json()?;
    let cert = diagnostics.call_method1("_certificate_from_report", (report, true))?;
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

    #[test]
    fn process_memory_belongs_to_the_finished_capture() {
        let _guard = TEST_LOCK.lock().unwrap();
        let memory = serde_json::value::RawValue::from_string(
            r#"{"measured_peak_bytes":12345}"#.to_owned(),
        ).unwrap();
        begin_cpu_render_capture("test.poster").finish_with_memory(Some(memory));
        let own = calling_thread_execution_report_json().unwrap();
        assert_eq!(own, execution_report_json().unwrap());
        let report: serde_json::Value = serde_json::from_str(&own).unwrap();
        assert_eq!(report["process_memory"]["measured_peak_bytes"], 12345);
        std::thread::spawn(|| begin_cpu_render_capture("test.unrelated").finish())
            .join().unwrap();
        let newest: serde_json::Value = serde_json::from_str(&execution_report_json().unwrap()).unwrap();
        assert!(newest.get("process_memory").is_none());
        assert_eq!(own, calling_thread_execution_report_json().unwrap());
    }

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

        let first_capture = begin_render_capture("test.render");
        record_pass("terrain.main", 1.5, 3);
        record_pass("terrain.sky", 0.25, 1);
        first_capture.finish();
        let a = execution_report_json().expect("first report must assemble");

        // Rebuild from identical state with DIFFERENT live timings.
        let second_capture = begin_render_capture("test.render");
        record_pass("terrain.main", 42.0, 3);
        record_pass("terrain.sky", 99.0, 1);
        second_capture.finish();
        let b = execution_report_json().expect("second report must assemble");

        // Both are valid JSON.
        let mut va: serde_json::Value = serde_json::from_str(&a).expect("report a parses");
        let mut vb: serde_json::Value = serde_json::from_str(&b).expect("report b parses");

        assert_eq!(va["schema"], "forge3d.render_certificate/1");
        assert!(
            va.get("models").is_none(),
            "model-free reports must preserve the pre-SIDERA schema/1 payload shape"
        );
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
    fn model_declarations_are_signed_context_not_fake_passes() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        let capture = begin_render_capture("test.models");
        record_model(
            "astro.twilight",
            "civil-to-astronomical smoothstep over solar altitude",
        );
        record_pass("astro.night.overlay", 0.5, 1);
        capture.finish();

        let report: serde_json::Value =
            serde_json::from_str(&execution_report_json().expect("report assembles"))
                .expect("report parses");
        assert_eq!(
            report["models"]["astro.twilight"],
            "civil-to-astronomical smoothstep over solar altitude"
        );
        assert_eq!(report["passes"].as_array().map(Vec::len), Some(1));
        assert_eq!(report["passes"][0]["label"], "astro.night.overlay");

        let clean = begin_render_capture("test.models.clean");
        clean.finish();
        let report: serde_json::Value =
            serde_json::from_str(&execution_report_json().expect("clean report assembles"))
                .expect("clean report parses");
        assert!(
            report.get("models").is_none(),
            "model declarations must be capture-local and absent when unused"
        );
    }

    #[test]
    fn finish_without_begin_does_not_panic() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // No matching begin_render_capture: must not panic, and the report
        // (if any prior state exists) must still be serializable.
        finish_render_capture(None);
        let _ = execution_report_json();
    }

    #[test]
    fn dropped_capture_guard_discards_failed_render_state() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        let failed = begin_render_capture("test.failed");
        record_pass("failed.pass", 1.0, 1);
        drop(failed);

        let successful = begin_render_capture("test.successful");
        record_pass("successful.pass", 2.0, 1);
        successful.finish();

        let report: serde_json::Value = serde_json::from_str(
            &execution_report_json().expect("successful capture must assemble"),
        )
        .expect("report parses");
        assert_eq!(report["passes"].as_array().map(Vec::len), Some(1));
        assert_eq!(report["passes"][0]["label"], "successful.pass");
    }

    #[test]
    fn nested_captures_join_the_outer_render_transaction() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        let outer = begin_cpu_render_capture("test.outer");
        record_pass("outer.before", 0.0, 1);
        let inner = begin_render_capture("test.inner_gpu");
        record_pass("inner.gpu", 3.0, 2);
        inner.finish();
        record_pass("outer.after", 0.0, 1);
        outer.finish();

        let report: serde_json::Value =
            serde_json::from_str(&execution_report_json().expect("nested capture must assemble"))
                .expect("report parses");
        let labels: Vec<&str> = report["passes"]
            .as_array()
            .expect("passes array")
            .iter()
            .map(|pass| pass["label"].as_str().expect("pass label"))
            .collect();
        assert_eq!(labels, ["outer.before", "inner.gpu", "outer.after"]);
    }

    #[test]
    fn f3dz_evidence_is_capture_local_and_records_the_weakest_bound() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        let capture = begin_cpu_render_capture("test.f3dz");
        record_f3dz_pages(0.1, 4, false);
        record_f3dz_pages(0.5, 3, true);
        capture.finish();
        let report: serde_json::Value =
            serde_json::from_str(&execution_report_json().expect("report assembles"))
                .expect("report parses");
        assert_eq!(report["codec"]["codec"], "f3dz/1");
        assert_eq!(report["codec"]["eps"], 0.5);
        assert_eq!(report["codec"]["pages_base_quality"], 3);

        let clean = begin_cpu_render_capture("test.no-codec");
        clean.finish();
        let report: serde_json::Value =
            serde_json::from_str(&execution_report_json().expect("report assembles"))
                .expect("report parses");
        assert!(report.get("codec").is_none());
    }

    #[test]
    fn precision_evidence_is_scoped_and_optional() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        let dd = begin_render_capture("test.dd");
        record_precision_evidence(PrecisionEvidence {
            backend: "vulkan".to_string(),
            adapter: "test-adapter".to_string(),
            operation: "add".to_string(),
            two_prod_variant: "fma".to_string(),
            shader_label: "forge3d.dd_harness.v1".to_string(),
            shader_hash: "abc123".to_string(),
            generated_count: 100_000_000,
            adversarial_count: 1_000_000,
            mismatch_count: 0,
            max_err_u2: 2.5,
            cited_bound_u2: 3.0,
        });
        dd.finish();
        let with_precision: serde_json::Value =
            serde_json::from_str(&execution_report_json().expect("DD certificate must assemble"))
                .expect("DD certificate parses");
        assert_eq!(with_precision["precision"]["operation"], "add");
        assert_eq!(with_precision["precision"]["mismatch_count"], 0);

        let ordinary = begin_render_capture("test.ordinary");
        ordinary.finish();
        let without_precision: serde_json::Value = serde_json::from_str(
            &execution_report_json().expect("ordinary certificate must assemble"),
        )
        .expect("ordinary certificate parses");
        assert!(without_precision.get("precision").is_none());
    }
}
