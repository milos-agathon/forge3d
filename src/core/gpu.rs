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

/// TERRA-DETERMINATA: deterministic rendering mode.
///
/// When `FORGE3D_DETERMINISTIC` is set (1/true/yes), the process must pin a
/// SINGLE backend via `WGPU_BACKENDS`/`WGPU_BACKEND` before any GPU context
/// creation, and `ctx()` asserts the acquired adapter actually runs on that
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
pub fn deterministic_mode() -> bool {
    matches!(
        std::env::var("FORGE3D_DETERMINISTIC")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Parse `WGPU_BACKENDS`/`WGPU_BACKEND` into a single requested backend, if set.
/// Returns (raw env value, backend mask for instance creation, expected adapter backend).
fn requested_backend_from_env() -> Option<(String, wgpu::Backends, wgpu::Backend)> {
    use std::env;
    let s = env::var("WGPU_BACKENDS")
        .or_else(|_| env::var("WGPU_BACKEND"))
        .ok()?;
    let s_l = s.to_lowercase();
    if s_l.contains("metal") {
        return Some((s, wgpu::Backends::METAL, wgpu::Backend::Metal));
    }
    if s_l.contains("vulkan") {
        return Some((s, wgpu::Backends::VULKAN, wgpu::Backend::Vulkan));
    }
    if s_l.contains("dx12") {
        return Some((s, wgpu::Backends::DX12, wgpu::Backend::Dx12));
    }
    if s_l.contains("gl") {
        return Some((s, wgpu::Backends::GL, wgpu::Backend::Gl));
    }
    if s_l.contains("webgpu") {
        return Some((
            s,
            wgpu::Backends::BROWSER_WEBGPU,
            wgpu::Backend::BrowserWebGpu,
        ));
    }
    None
}

fn backends_from_env() -> RenderResult<wgpu::Backends> {
    if let Some((_, mask, _)) = requested_backend_from_env() {
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
            let (raw, _, expected) = requested_backend_from_env().ok_or_else(|| {
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

/// Infallible GPU context accessor for internal callers that predate
/// [`try_ctx`]. Panics with the structured device-error message when
/// acquisition fails; Python-reachable code must call [`try_ctx`] instead so
/// users get a catchable `RuntimeError` rather than a `PanicException`.
pub fn ctx() -> &'static GpuContext {
    match try_ctx() {
        Ok(ctx) => ctx,
        Err(err) => panic!("{err}"),
    }
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
