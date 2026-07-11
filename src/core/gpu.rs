// src/gpu.rs
// Global GPU context helpers and utilities
// Exists to share wgpu device creation across runtime and tests
// RELEVANT FILES: src/vector/polygon.rs, src/vector/point.rs, src/vector/line.rs, src/vector/gpu_extrusion.rs
use crate::core::error::{RenderError, RenderResult};
use once_cell::sync::OnceCell;
use std::sync::Arc;

pub struct GpuContext {
    pub instance: Arc<wgpu::Instance>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
    /// True when no hardware adapter was found and a software rasterizer
    /// (e.g. WARP on Windows, Mesa lavapipe on Linux) is in use instead.
    pub software_fallback: bool,
    /// Negotiated capability set: what forge3d wanted vs. what the adapter
    /// granted. Replaces the old `Features::empty()` request.
    pub capabilities: crate::core::capabilities::CapabilitySet,
}

static CTX: OnceCell<GpuContext> = OnceCell::new();

/// Set once the GPU context becomes unusable (device lost). Every subsequent
/// [`try_ctx`] call fails loudly with this reason rather than handing back a
/// dead device and letting callers hit undefined behavior.
static POISONED: OnceCell<String> = OnceCell::new();

/// Mark the global GPU context as poisoned. Called from the device-lost
/// callback; idempotent (first reason wins).
pub fn poison_context(reason: String) {
    let _ = POISONED.set(reason);
}

/// Backend name of the already-initialized GPU context, if one exists.
///
/// Returns `None` when no context has been created yet (peeks `CTX` without
/// forcing initialization), so callers can decide whether a backend pin can
/// still be honored via `WGPU_BACKENDS`.
pub fn active_backend() -> Option<String> {
    CTX.get()
        .map(|c| format!("{:?}", c.adapter.get_info().backend).to_lowercase())
}

/// Adapter that owns the initialized global render context, if any.
pub(crate) fn active_adapter_info() -> Option<(wgpu::AdapterInfo, bool)> {
    CTX.get()
        .map(|c| (c.adapter.get_info(), c.software_fallback))
}

/// TERRA-DETERMINATA: deterministic rendering mode.
///
/// When `FORGE3D_DETERMINISTIC` is set (1/true/yes), the process must pin a
/// SINGLE backend via `WGPU_BACKENDS`/`WGPU_BACKEND` before any GPU context
/// creation, and `try_ctx()` asserts the acquired adapter actually runs on that
/// backend before the device/queue exist (i.e. before any queue submission).
/// Determinism failures are loud: missing or ambiguous configuration panics.
///
/// Fast-math policy (verified against wgpu 0.19 / naga 0.19): neither
/// `wgpu::DeviceDescriptor` nor the naga backend writers expose a fast-math /
/// relaxed-precision toggle, so there is no optimization flag to gate off —
/// codegen is already strict. The one codegen degree of freedom wgpu does
/// expose, the DX12 shader compiler choice (`Dx12Compiler`), is pinned to FXC
/// under deterministic mode so HLSL lowering does not depend on which
/// dxcompiler DLLs happen to be installed. For wasm builds, RUSTFLAGS must
/// NOT include `-C target-feature=+relaxed-simd` (relaxed SIMD is
/// nondeterministic by design); the determinism CI matrix documents this.
///
/// Software rasterizer adapters (WARP, lavapipe) and hypervisor-virtualized
/// GPUs (Apple Paravirtual, VirtIO, VMware, ...) are REFUSED under
/// deterministic mode unless `FORGE3D_DETERMINISTIC_ALLOW_SOFTWARE=1` is set
/// (see [`deterministic_allow_software`]): neither is the physical hardware a
/// determinism leg claims to measure, and their hashes must never masquerade
/// as a hardware leg's.
pub fn deterministic_mode() -> bool {
    matches!(
        std::env::var("FORGE3D_DETERMINISTIC")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// TERRA-DETERMINATA escape hatch: allow a software rasterizer or
/// hypervisor-virtualized GPU adapter under deterministic mode. Off by
/// default because such adapters are a different "vendor" whose hash must
/// never masquerade as a hardware leg's; set
/// `FORGE3D_DETERMINISTIC_ALLOW_SOFTWARE=1` only for an explicitly
/// software/virtual-labelled determinism leg.
pub fn deterministic_allow_software() -> bool {
    matches!(
        std::env::var("FORGE3D_DETERMINISTIC_ALLOW_SOFTWARE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Parse `WGPU_BACKENDS`/`WGPU_BACKEND` into a single requested backend, if set.
/// Returns (raw env value, backend mask for instance creation, expected adapter backend).
///
/// When the variable is set but names no recognized backend, this returns an
/// error rather than silently falling through to the platform default — a bad
/// pin must surface as a catchable Python exception, never be ignored.
fn requested_backend_from_env() -> RenderResult<Option<(String, wgpu::Backends, wgpu::Backend)>> {
    use std::env;
    let s = match env::var("WGPU_BACKENDS").or_else(|_| env::var("WGPU_BACKEND")) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let s_l = s.to_lowercase();
    if s_l.contains("metal") {
        return Ok(Some((s, wgpu::Backends::METAL, wgpu::Backend::Metal)));
    }
    if s_l.contains("vulkan") {
        return Ok(Some((s, wgpu::Backends::VULKAN, wgpu::Backend::Vulkan)));
    }
    if s_l.contains("dx12") {
        return Ok(Some((s, wgpu::Backends::DX12, wgpu::Backend::Dx12)));
    }
    if s_l.contains("gl") {
        return Ok(Some((s, wgpu::Backends::GL, wgpu::Backend::Gl)));
    }
    if s_l.contains("webgpu") {
        return Ok(Some((
            s,
            wgpu::Backends::BROWSER_WEBGPU,
            wgpu::Backend::BrowserWebGpu,
        )));
    }
    Err(RenderError::device(format!(
        "Unrecognized WGPU_BACKENDS value '{s}'. Valid: vulkan|dx12|metal|gl|webgpu"
    )))
}

fn backends_from_env() -> RenderResult<wgpu::Backends> {
    if let Some((_, mask, _)) = requested_backend_from_env()? {
        return Ok(mask);
    }
    if deterministic_mode() {
        return Err(RenderError::device(
            "FORGE3D_DETERMINISTIC is set but WGPU_BACKENDS/WGPU_BACKEND does not name a \
             single backend (metal|vulkan|dx12|gl|webgpu). Deterministic mode requires an \
             explicit backend pin BEFORE any GPU context creation.",
        ));
    }
    #[cfg(target_os = "macos")]
    {
        Ok(wgpu::Backends::METAL)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(wgpu::Backends::all())
    }
}

/// Describe the WGPU_BACKENDS/WGPU_BACKEND environment for error messages.
fn backend_env_description() -> String {
    match std::env::var("WGPU_BACKENDS").or_else(|_| std::env::var("WGPU_BACKEND")) {
        Ok(v) => format!("WGPU_BACKENDS/WGPU_BACKEND set to '{v}'"),
        Err(_) => "WGPU_BACKENDS/WGPU_BACKEND not set".to_string(),
    }
}

/// Fallible GPU context acquisition.
///
/// Tries a hardware adapter first; when none is available it retries with
/// `force_fallback_adapter: true` so headless hosts can run on a software
/// rasterizer (WARP on Windows, Mesa lavapipe on Linux). Only when even the
/// software fallback is unavailable does this return `RenderError::Device`
/// with remediation guidance — it never panics. Python-reachable entry points
/// must use this (`?` converts to a catchable `RuntimeError`), not `ctx()`.
pub fn try_ctx() -> RenderResult<&'static GpuContext> {
    if let Some(reason) = POISONED.get() {
        return Err(RenderError::device(format!(
            "GPU context poisoned: {reason}"
        )));
    }
    CTX.get_or_try_init(|| {
        let backends = backends_from_env()?;
        let mut instance_desc = wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        };
        if deterministic_mode() {
            // Pin HLSL codegen: FXC regardless of installed DXC DLLs (see
            // deterministic_mode docs for the full fast-math policy).
            instance_desc.dx12_shader_compiler = wgpu::Dx12Compiler::Fxc;
        }
        let instance = Arc::new(wgpu::Instance::new(instance_desc));
        let hardware = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            // LowPower tends to resolve faster and avoids eGPU/discrete probing on macOS
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let (adapter, mut software_fallback) = match hardware {
            Some(adapter) => (adapter, false),
            None => {
                let fallback =
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: true,
                    }));
                match fallback {
                    Some(adapter) => (adapter, true),
                    None => {
                        return Err(RenderError::device(format!(
                            "No suitable GPU adapter found (backends tried: {backends:?}; {env}). \
                             A hardware adapter was not found and the software fallback adapter \
                             is also unavailable. Verify GPU drivers are installed, pin a backend \
                             via WGPU_BACKENDS (vulkan|dx12|metal|gl), or install a software \
                             rasterizer for headless use (Windows ships WARP; on Linux install \
                             Mesa's lavapipe).",
                            env = backend_env_description(),
                        )));
                    }
                }
            }
        };

        let adapter_info = adapter.get_info();
        if adapter_info.device_type == wgpu::DeviceType::Cpu {
            software_fallback = true;
        }
        if software_fallback {
            log::warn!(
                "forge3d: no hardware GPU adapter found; using software fallback adapter \
                 '{}' ({:?} backend). Rendering will be slow.",
                adapter_info.name,
                adapter_info.backend
            );
        }

        if deterministic_mode() {
            // A software rasterizer (WARP, lavapipe) or a hypervisor-virtualized
            // GPU (e.g. the "Apple Paravirtual device" on hosted macOS runners)
            // is effectively a different "vendor": accepting one under
            // deterministic mode would silently change what a CI leg measures
            // (a software/VM hash instead of the hardware hash the golden pins;
            // measured 2026-07-10: the paravirtual Metal hash is bit-stable but
            // systematically differs from real-hardware goldens). Refuse both
            // unless the caller explicitly opts in for a dedicated leg.
            let lowered_name = adapter_info.name.to_lowercase();
            let virtualized = ["paravirtual", "virtio", "vmware", "virtualbox", "qxl"]
                .iter()
                .any(|marker| lowered_name.contains(marker));
            if (software_fallback || virtualized) && !deterministic_allow_software() {
                let kind = if software_fallback {
                    "software rasterizer"
                } else {
                    "hypervisor-virtualized GPU"
                };
                return Err(RenderError::device(format!(
                    "FORGE3D_DETERMINISTIC: only a {kind} adapter is available \
                     ('{}', {:?} backend). Such an adapter is a different \"vendor\" and \
                     its hash would not be comparable to hardware goldens; refusing to \
                     render. Run on a host with a physical GPU for this backend, or set \
                     FORGE3D_DETERMINISTIC_ALLOW_SOFTWARE=1 to explicitly measure this \
                     adapter as its own leg.",
                    adapter_info.name, adapter_info.backend
                )));
            }
            let (raw, _, expected) = requested_backend_from_env()?.ok_or_else(|| {
                RenderError::device(
                    "FORGE3D_DETERMINISTIC requires WGPU_BACKENDS/WGPU_BACKEND to pin one backend",
                )
            })?;
            let actual = adapter_info.backend;
            if actual != expected {
                return Err(RenderError::device(format!(
                    "FORGE3D_DETERMINISTIC: acquired adapter backend {actual:?} does not match \
                     the requested backend '{raw}'. The backend must be locked before any queue \
                     submission; refusing to continue nondeterministically."
                )));
            }
        }

        // Respect the adapter's native limits instead of clamping to downlevel defaults
        // (which cap 2D textures at 2048 and break high-res renders).
        let mut limits = adapter.limits();
        let desired_storage_buffers = 8;
        limits.max_storage_buffers_per_shader_stage = limits
            .max_storage_buffers_per_shader_stage
            .max(desired_storage_buffers);

        // Negotiate the capability set against the adapter's advertised
        // features (records `capability_absent` degradations for anything the
        // adapter cannot grant). Nothing here is hard-required.
        let mut capabilities =
            crate::core::capabilities::CapabilitySet::negotiate(adapter.features());

        // Robustness: some drivers advertise features that still fail at
        // request_device time. Requesting a feature must never hard-fail the
        // context, so on error we record a degradation and retry once with an
        // empty feature set before surfacing the original error.
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: capabilities.granted,
                required_limits: limits.clone(),
                label: Some("forge3d-device"),
            },
            None,
        )) {
            Ok(pair) => pair,
            Err(first_err) => {
                crate::core::degradation::record_degradation(
                    "capability_request_failed",
                    "negotiated_features",
                    &first_err.to_string(),
                );
                capabilities = crate::core::capabilities::CapabilitySet {
                    wanted: capabilities.wanted,
                    required: wgpu::Features::empty(),
                    granted: wgpu::Features::empty(),
                };
                pollster::block_on(adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::empty(),
                        required_limits: limits,
                        label: Some("forge3d-device"),
                    },
                    None,
                ))
                .map_err(|e| {
                    RenderError::device(format!(
                        "request_device failed for adapter '{}' ({:?} backend): {e}. Try updating \
                         GPU drivers, or pin a different backend via WGPU_BACKENDS \
                         (vulkan|dx12|metal|gl).",
                        adapter_info.name, adapter_info.backend,
                    ))
                })?
            }
        };

        // Surface GPU errors that escape error scopes as recorded degradations
        // instead of wgpu's default (which logs then panics on the poll thread).
        device.on_uncaptured_error(Box::new(|e| {
            crate::core::degradation::record_degradation(
                "uncaptured_gpu_error",
                "wgpu",
                &format!("{e}"),
            );
            log::error!("wgpu uncaptured error: {e}");
        }));
        // On device loss, poison the context so later try_ctx() calls fail
        // loudly rather than returning a dead device.
        device.set_device_lost_callback(|reason, msg| {
            crate::core::degradation::record_degradation(
                "device_lost",
                &format!("{reason:?}"),
                &msg,
            );
            poison_context(format!("device lost ({reason:?}): {msg}"));
        });

        Ok(GpuContext {
            instance,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
            software_fallback,
            capabilities,
        })
    })
}

/// Align to WebGPU's required bytes-per-row for copies.
#[inline]
pub fn align_copy_bpr(unpadded: u32) -> u32 {
    let a = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + a - 1) / a) * a
}

/// Create a small wgpu device for unit tests.
///
/// Returns `None` when no GPU adapter is available (e.g. headless CI runners),
/// allowing tests to skip gracefully instead of panicking.
pub fn create_device_for_test() -> Option<wgpu::Device> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    // Keep native limits to allow larger render targets in tests as well.
    let mut limits = adapter.limits();
    let desired_storage_buffers = 8;
    limits.max_storage_buffers_per_shader_stage = limits
        .max_storage_buffers_per_shader_stage
        .max(desired_storage_buffers);

    let capabilities = crate::core::capabilities::CapabilitySet::negotiate(adapter.features());
    let device = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: capabilities.granted,
            required_limits: limits.clone(),
            label: Some("forge3d-test-device"),
        },
        None,
    )) {
        Ok((device, _queue)) => device,
        Err(_) => {
            // Some drivers advertise features that fail at request time; retry
            // once with no optional features so tests still get a device.
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    label: Some("forge3d-test-device"),
                },
                None,
            ))
            .ok()?
            .0
        }
    };
    Some(device)
}

/// Create device and queue for unit tests (P3-08).
///
/// Returns `None` when no GPU adapter is available (e.g. headless CI runners).
pub fn create_device_and_queue_for_test() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let mut limits = adapter.limits();
    let desired_storage_buffers = 8;
    limits.max_storage_buffers_per_shader_stage = limits
        .max_storage_buffers_per_shader_stage
        .max(desired_storage_buffers);

    let capabilities = crate::core::capabilities::CapabilitySet::negotiate(adapter.features());
    let (device, queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: capabilities.granted,
            required_limits: limits.clone(),
            label: Some("forge3d-test-device"),
        },
        None,
    )) {
        Ok(pair) => pair,
        Err(_) => {
            // Retry once with no optional features if a driver advertises a
            // feature that fails at request time.
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    label: Some("forge3d-test-device"),
                },
                None,
            ))
            .ok()?
        }
    };
    Some((device, queue))
}
