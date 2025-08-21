use once_cell::sync::OnceCell;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

static CTX: OnceCell<GpuContext> = OnceCell::new();

pub fn ctx() -> &'static GpuContext {
    CTX.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("No suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                label: Some("forge3d-device"),
            }, None
        )).expect("request_device failed");

        GpuContext { device, queue, adapter }
    })
}

/// Align to WebGPU's required bytes-per-row for copies.
#[inline]
pub fn align_copy_bpr(unpadded: u32) -> u32 {
    let a = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + a - 1) / a) * a
}