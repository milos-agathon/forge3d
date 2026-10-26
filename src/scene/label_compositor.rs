//! Private GPU compositor for cached raster frames and native label quads.
//! It uses the same input conversion and shaders as Scene, without allocating
//! terrain, SSAO, reflection, or 3D text resources that composition never uses.
use crate::core::resource_tracker::{tracked_create_texture, AllocationOwner, TrackedTexture};
use pyo3::prelude::*;

mod py_api;
mod render;

#[pyclass(module = "forge3d._forge3d", name = "_LabelCompositor")]
pub(crate) struct LabelCompositor {
    width: u32,
    height: u32,
    color: TrackedTexture,
    color_view: wgpu::TextureView,
    overlay_renderer: Option<crate::core::overlays::OverlayRenderer>,
    overlay_enabled: bool,
    text_overlay_renderer: Option<crate::core::text_overlay::TextOverlayRenderer>,
    text_overlay_enabled: bool,
    text_overlay_alpha: f32,
    text_instances: Vec<crate::core::text_overlay::TextInstance>,
    allocation_owner: AllocationOwner,
}

impl LabelCompositor {
    pub(crate) fn new(width: u32, height: u32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "label compositor dimensions must be non-zero",
            ));
        }
        let allocation_owner = AllocationOwner::new();
        let _allocation_scope = allocation_owner.activate();
        let g = crate::core::gpu::try_ctx()?;
        let color = tracked_create_texture(
            &g.device,
            &wgpu::TextureDescriptor {
                label: Some("label-compositor-color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let color_view = color.create_view(&Default::default());
        let mut overlay = crate::core::overlays::OverlayRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8Unorm,
            g.device
                .features()
                .contains(wgpu::Features::FLOAT32_FILTERABLE),
        )?;
        overlay.recreate_bind_group(&g.device, None, None, None, None)?;
        overlay.upload_uniforms(&g.queue);
        let mut text = crate::core::text_overlay::TextOverlayRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8Unorm,
        )?;
        text.set_resolution(width, height);
        text.set_alpha(1.0);
        text.set_enabled(false);
        text.upload_uniforms(&g.queue);
        Ok(Self {
            width,
            height,
            color,
            color_view,
            overlay_renderer: Some(overlay),
            overlay_enabled: false,
            text_overlay_renderer: Some(text),
            text_overlay_enabled: false,
            text_overlay_alpha: 1.0,
            text_instances: Vec::new(),
            allocation_owner,
        })
    }
}
