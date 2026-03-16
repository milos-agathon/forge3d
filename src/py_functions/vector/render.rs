use super::*;
use crate::vector::api::{PointDef, PolylineDef};
use crate::vector::{LineRenderer, PointRenderer};

pub(super) const IDENTITY_VIEW_PROJ: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(super) struct UploadedVectorScene {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) point_renderer: PointRenderer,
    pub(super) line_renderer: LineRenderer,
    pub(super) point_count: u32,
    pub(super) line_count: u32,
}

pub(super) fn vector_runtime_err<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(super) fn gpu_device_queue() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let gpu = crate::core::gpu::ctx();
    (Arc::clone(&gpu.device), Arc::clone(&gpu.queue))
}

pub(super) fn viewport_dims(width: u32, height: u32) -> [f32; 2] {
    [width as f32, height as f32]
}

pub(super) fn create_rgba_target(
    device: &wgpu::Device,
    label: &'static str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(super) fn create_pick_target(
    device: &wgpu::Device,
    label: &'static str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(super) fn upload_vector_scene(
    point_defs: &[PointDef],
    poly_defs: &[PolylineDef],
) -> PyResult<UploadedVectorScene> {
    let (device, queue) = gpu_device_queue();
    let mut point_renderer = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
        .map_err(vector_runtime_err)?;
    let mut line_renderer = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
        .map_err(vector_runtime_err)?;

    if !point_defs.is_empty() {
        let instances = point_renderer
            .pack_points(point_defs)
            .map_err(vector_runtime_err)?;
        point_renderer
            .upload_points(&device, &queue, &instances)
            .map_err(vector_runtime_err)?;
    }

    if !poly_defs.is_empty() {
        let instances = line_renderer
            .pack_polylines(poly_defs)
            .map_err(vector_runtime_err)?;
        line_renderer
            .upload_lines(&device, &instances)
            .map_err(vector_runtime_err)?;
    }

    Ok(UploadedVectorScene {
        device,
        queue,
        point_renderer,
        line_renderer,
        point_count: point_defs.len() as u32,
        line_count: poly_defs.len() as u32,
    })
}

#[cfg(feature = "weighted-oit")]
pub(super) fn render_oit_scene<'a>(
    scene: &'a mut UploadedVectorScene,
    pass: &mut wgpu::RenderPass<'a>,
    width: u32,
    height: u32,
) -> PyResult<()> {
    let viewport = viewport_dims(width, height);
    if scene.line_count > 0 {
        scene
            .line_renderer
            .render_oit(
                pass,
                &scene.queue,
                &IDENTITY_VIEW_PROJ,
                viewport,
                scene.line_count,
                crate::vector::line::LineCap::Round,
                crate::vector::line::LineJoin::Round,
                2.0,
            )
            .map_err(vector_runtime_err)?;
    }
    if scene.point_count > 0 {
        scene
            .point_renderer
            .render_oit(
                pass,
                &scene.queue,
                &IDENTITY_VIEW_PROJ,
                viewport,
                1.0,
                scene.point_count,
            )
            .map_err(vector_runtime_err)?;
    }
    Ok(())
}

pub(super) fn render_pick_scene<'a>(
    scene: &'a mut UploadedVectorScene,
    pass: &mut wgpu::RenderPass<'a>,
    width: u32,
    height: u32,
    base_pick_id: u32,
) -> PyResult<()> {
    let viewport = viewport_dims(width, height);
    let mut base = base_pick_id;
    if scene.point_count > 0 {
        scene
            .point_renderer
            .render_pick(
                pass,
                &scene.queue,
                &IDENTITY_VIEW_PROJ,
                viewport,
                1.0,
                scene.point_count,
                base,
            )
            .map_err(vector_runtime_err)?;
        base += scene.point_count;
    }
    if scene.line_count > 0 {
        scene
            .line_renderer
            .render_pick(
                pass,
                &scene.queue,
                &IDENTITY_VIEW_PROJ,
                viewport,
                scene.line_count,
                base,
            )
            .map_err(vector_runtime_err)?;
    }
    Ok(())
}
