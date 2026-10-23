use super::*;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};

mod accessors;
mod constructor;
mod controls;
mod resize;
mod runtime;

/// SSGI renderer
pub struct SsgiRenderer {
    settings: SsgiSettings,
    settings_buffer: TrackedBuffer,
    camera_buffer: TrackedBuffer,
    frame_index: u32,

    // Pipelines
    concordia_projection: bool,
    trace_pipeline: ComputePipeline,
    trace_bind_group_layout: BindGroupLayout,
    shade_pipeline: ComputePipeline,
    shade_bind_group_layout: BindGroupLayout,
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,
    upsample_pipeline: ComputePipeline,
    upsample_bind_group_layout: BindGroupLayout,
    composite_pipeline: ComputePipeline,
    composite_bind_group_layout: BindGroupLayout,

    // Output and temporal textures
    // ssgi_hit       : Rgba16Float half-res hit buffer (xy = hit UV in [0,1], z = travelled
    //                   distance in view units, w = hit mask in {0,1}).
    ssgi_hit: TrackedTexture,
    ssgi_hit_view: TextureView,
    // ssgi_texture   : Rgba16Float half-res GI radiance (rgb = diffuse bounce light in
    //                   linear HDR units, a unused/1.0).
    ssgi_texture: TrackedTexture,
    ssgi_view: TextureView,
    // ssgi_history   : Rgba16Float previous-frame GI radiance used for temporal resolve.
    ssgi_history: TrackedTexture,
    ssgi_history_view: TextureView,
    history_valid: bool,
    // ssgi_filtered  : Rgba16Float temporally filtered GI radiance (same layout as
    //                   ssgi_texture).
    ssgi_filtered: TrackedTexture,
    ssgi_filtered_view: TextureView,
    // Full-resolution upscaled output for half-res mode
    // ssgi_upscaled  : Rgba16Float full-res GI radiance after edge-aware upsample.
    ssgi_upscaled: TrackedTexture,
    ssgi_upscaled_view: TextureView,
    // Composited material (material + SSGI)
    // ssgi_composited: Rgba8Unorm material buffer + SSGI diffuse contribution, used for
    //                   P5 visualization and metrics (not the main HDR lighting buffer).
    _ssgi_composited: TrackedTexture,
    ssgi_composited_view: TextureView,
    composite_uniform: TrackedBuffer,
    scene_history: [TrackedTexture; 2],
    scene_history_views: [TextureView; 2],
    scene_history_index: usize,
    scene_history_ready: bool,
    scene_color_override: Option<TextureView>,
    linear_sampler: Sampler,

    // Env
    _env_texture: TrackedTexture,
    env_view: TextureView,
    env_sampler: Sampler,

    width: u32,
    height: u32,
    half_res: bool,

    // Timings (ms)
    last_trace_ms: f32,
    last_shade_ms: f32,
    last_temporal_ms: f32,
    last_upsample_ms: f32,
}

pub(crate) use resize::PreparedSsgiTargets;

impl SsgiRenderer {
    pub(crate) fn configure_concordia_projection(&mut self, device: &Device) -> RenderResult<()> {
        {
            let source = super::concordia_reconstruction(
                include_str!("../../shaders/ssgi/trace.wgsl"),
                "uCam",
                "reconstruct_view_position",
            )?;
            let module = crate::core::shader_registry::create_labeled_shader_module(
                device,
                "concordia.ssgi.trace",
                &source,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("concordia.ssgi.trace.layout"),
                bind_group_layouts: &[&self.trace_bind_group_layout],
                push_constant_ranges: &[],
            });
            self.trace_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("concordia.ssgi.trace.pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: "cs_trace",
                },
            )
            .map_err(crate::core::error::RenderError::Render)?;
        }
        self.concordia_projection = true;
        Ok(())
    }
}
