// src/viewer/init/device_init.rs
// Device and surface initialization for the Viewer

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration};
use winit::window::Window;

/// Resources created during device initialization
pub struct DeviceResources {
    pub surface: Surface<'static>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter: Arc<Adapter>,
    pub config: SurfaceConfiguration,
    pub adapter_name: String,
}

/// Create wgpu device, queue, adapter, and surface
pub async fn create_device_and_surface(
    window: Arc<Window>,
    vsync: bool,
) -> Result<DeviceResources, Box<dyn std::error::Error>> {
    let size = window.inner_size();

    // Create wgpu instance
    let instance = Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // Create surface
    let surface = instance.create_surface(Arc::clone(&window))?;

    // Request adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to find suitable adapter")?;

    let adapter = Arc::new(adapter);
    let adapter_name = adapter.get_info().name;

    // Request device and queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Viewer Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Configure surface
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let config = SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        },
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(&device, &config);

    Ok(DeviceResources {
        surface,
        device,
        queue,
        adapter,
        config,
        adapter_name,
    })
}
