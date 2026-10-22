use super::*;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};

mod layouts;
mod pipelines;
pub(super) mod resources;

struct ConstructorLayouts {
    trace_bind_group_layout: BindGroupLayout,
    shade_bind_group_layout: BindGroupLayout,
    temporal_bind_group_layout: BindGroupLayout,
    upsample_bind_group_layout: BindGroupLayout,
    composite_bind_group_layout: BindGroupLayout,
}

struct ConstructorPipelines {
    trace_pipeline: ComputePipeline,
    shade_pipeline: ComputePipeline,
    temporal_pipeline: ComputePipeline,
    upsample_pipeline: ComputePipeline,
    composite_pipeline: ComputePipeline,
}

struct ConstructorBuffers {
    settings_buffer: TrackedBuffer,
    camera_buffer: TrackedBuffer,
    composite_uniform: TrackedBuffer,
}

pub(super) struct ViewportTextures {
    pub(super) ssgi_hit: TrackedTexture,
    pub(super) ssgi_hit_view: TextureView,
    pub(super) ssgi_texture: TrackedTexture,
    pub(super) ssgi_view: TextureView,
    pub(super) ssgi_history: TrackedTexture,
    pub(super) ssgi_history_view: TextureView,
    pub(super) ssgi_filtered: TrackedTexture,
    pub(super) ssgi_filtered_view: TextureView,
    pub(super) ssgi_upscaled: TrackedTexture,
    pub(super) ssgi_upscaled_view: TextureView,
    pub(super) ssgi_composited: TrackedTexture,
    pub(super) ssgi_composited_view: TextureView,
    pub(super) scene_history: [TrackedTexture; 2],
    pub(super) scene_history_views: [TextureView; 2],
}

pub(super) struct ConstructorResources {
    pub(super) viewport: ViewportTextures,
    pub(super) env_texture: TrackedTexture,
    pub(super) env_view: TextureView,
    pub(super) env_sampler: Sampler,
    pub(super) linear_sampler: Sampler,
}

impl SsgiRenderer {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        material_format: TextureFormat,
    ) -> RenderResult<Self> {
        let mut settings = SsgiSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];

        let buffers = resources::create_buffers(device)?;
        let layouts = layouts::create_layouts(device);
        let pipelines = pipelines::create_pipelines(device, &layouts);
        let textures = resources::create_textures(device, width, height, material_format, false)?;

        Ok(Self {
            concordia_projection: false,
            settings,
            settings_buffer: buffers.settings_buffer,
            camera_buffer: buffers.camera_buffer,
            frame_index: 0,
            trace_pipeline: pipelines.trace_pipeline,
            trace_bind_group_layout: layouts.trace_bind_group_layout,
            shade_pipeline: pipelines.shade_pipeline,
            shade_bind_group_layout: layouts.shade_bind_group_layout,
            temporal_pipeline: pipelines.temporal_pipeline,
            temporal_bind_group_layout: layouts.temporal_bind_group_layout,
            upsample_pipeline: pipelines.upsample_pipeline,
            upsample_bind_group_layout: layouts.upsample_bind_group_layout,
            composite_pipeline: pipelines.composite_pipeline,
            composite_bind_group_layout: layouts.composite_bind_group_layout,
            ssgi_hit: textures.viewport.ssgi_hit,
            ssgi_hit_view: textures.viewport.ssgi_hit_view,
            ssgi_texture: textures.viewport.ssgi_texture,
            ssgi_view: textures.viewport.ssgi_view,
            ssgi_history: textures.viewport.ssgi_history,
            ssgi_history_view: textures.viewport.ssgi_history_view,
            history_valid: false,
            ssgi_filtered: textures.viewport.ssgi_filtered,
            ssgi_filtered_view: textures.viewport.ssgi_filtered_view,
            ssgi_upscaled: textures.viewport.ssgi_upscaled,
            ssgi_upscaled_view: textures.viewport.ssgi_upscaled_view,
            _ssgi_composited: textures.viewport.ssgi_composited,
            ssgi_composited_view: textures.viewport.ssgi_composited_view,
            composite_uniform: buffers.composite_uniform,
            scene_history: textures.viewport.scene_history,
            scene_history_views: textures.viewport.scene_history_views,
            scene_history_index: 0,
            scene_history_ready: false,
            scene_color_override: None,
            linear_sampler: textures.linear_sampler,
            _env_texture: textures.env_texture,
            env_view: textures.env_view,
            env_sampler: textures.env_sampler,
            width,
            height,
            half_res: false,
            last_trace_ms: 0.0,
            last_shade_ms: 0.0,
            last_temporal_ms: 0.0,
            last_upsample_ms: 0.0,
        })
    }
}
