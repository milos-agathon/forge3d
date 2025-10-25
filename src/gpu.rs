// src/gpu.rs
// Global GPU context helpers and utilities
// Exists to share wgpu device creation across runtime and tests
// RELEVANT FILES: src/vector/polygon.rs, src/vector/point.rs, src/vector/line.rs, src/vector/gpu_extrusion.rs
use once_cell::sync::OnceCell;
use std::sync::Arc;

pub struct GpuContext {
    pub instance: Arc<wgpu::Instance>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
}

static CTX: OnceCell<GpuContext> = OnceCell::new();

fn backends_from_env() -> wgpu::Backends {
    use std::env;
    if let Ok(s) = env::var("WGPU_BACKENDS").or_else(|_| env::var("WGPU_BACKEND")) {
        let s_l = s.to_lowercase();
        if s_l.contains("metal") {
            return wgpu::Backends::METAL;
        }
        if s_l.contains("vulkan") {
            return wgpu::Backends::VULKAN;
        }
        if s_l.contains("dx12") {
            return wgpu::Backends::DX12;
        }
        if s_l.contains("gl") {
            return wgpu::Backends::GL;
        }
        if s_l.contains("webgpu") {
            return wgpu::Backends::BROWSER_WEBGPU;
        }
    }
    #[cfg(target_os = "macos")]
    {
        wgpu::Backends::METAL
    }
    #[cfg(not(target_os = "macos"))]
    {
        wgpu::Backends::all()
    }
}

pub fn ctx() -> &'static GpuContext {
    CTX.get_or_init(|| {
        let instance = Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backends_from_env(),
            ..Default::default()
        }));
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            // LowPower tends to resolve faster and avoids eGPU/discrete probing on macOS
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter");

        let mut limits = adapter.limits();
        let baseline = wgpu::Limits::downlevel_defaults();
        limits = limits.using_resolution(baseline);
        let desired_storage_buffers = 8;
        limits.max_storage_buffers_per_shader_stage = limits
            .max_storage_buffers_per_shader_stage
            .max(desired_storage_buffers);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                label: Some("forge3d-device"),
            },
            None,
        ))
        .expect("request_device failed");

        GpuContext {
            instance,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
        }
    })
}

/// Align to WebGPU's required bytes-per-row for copies.
#[inline]
pub fn align_copy_bpr(unpadded: u32) -> u32 {
    let a = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + a - 1) / a) * a
}

/// Create a small wgpu device for unit tests.
pub fn create_device_for_test() -> wgpu::Device {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter for tests");
    let mut limits = adapter.limits();
    let baseline = wgpu::Limits::downlevel_defaults();
    limits = limits.using_resolution(baseline);
    let desired_storage_buffers = 8;
    limits.max_storage_buffers_per_shader_stage = limits
        .max_storage_buffers_per_shader_stage
        .max(desired_storage_buffers);

    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            label: Some("forge3d-test-device"),
        },
        None,
    ))
    .expect("request_device failed for tests");
    device
}
