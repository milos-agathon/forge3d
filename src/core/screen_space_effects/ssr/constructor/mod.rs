use super::*;

mod layouts;
mod pipelines;
mod resources;

struct ConstructorLayouts {
    trace_bind_group_layout: BindGroupLayout,
    shade_bind_group_layout: BindGroupLayout,
    fallback_bind_group_layout: BindGroupLayout,
    temporal_bind_group_layout: BindGroupLayout,
    composite_bind_group_layout: BindGroupLayout,
}

struct ConstructorPipelines {
    trace_pipeline: ComputePipeline,
    shade_pipeline: ComputePipeline,
    fallback_pipeline: ComputePipeline,
    temporal_pipeline: ComputePipeline,
    composite_pipeline: ComputePipeline,
}

struct ConstructorBuffers {
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    counters_buffer: Buffer,
    counters_readback: Buffer,
    composite_params: Buffer,
    temporal_params: Buffer,
}

struct ConstructorResources {
    ssr_spec_texture: Texture,
    ssr_spec_view: TextureView,
    ssr_final_texture: Texture,
    ssr_final_view: TextureView,
    ssr_history_texture: Texture,
    ssr_history_view: TextureView,
    ssr_filtered_texture: Texture,
    ssr_filtered_view: TextureView,
    ssr_hit_texture: Texture,
    ssr_hit_view: TextureView,
    ssr_composited_texture: Texture,
    ssr_composited_view: TextureView,
    env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,
    linear_sampler: Sampler,
}

impl SsrRenderer {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let mut settings = SsrSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];

        let buffers = resources::create_buffers(device);
        let layouts = layouts::create_layouts(device);
        let pipelines = pipelines::create_pipelines(device, &layouts);
        let textures = resources::create_textures(device, width, height);

        Ok(Self {
            settings,
            settings_buffer: buffers.settings_buffer,
            camera_buffer: buffers.camera_buffer,
            trace_pipeline: pipelines.trace_pipeline,
            trace_bind_group_layout: layouts.trace_bind_group_layout,
            shade_pipeline: pipelines.shade_pipeline,
            shade_bind_group_layout: layouts.shade_bind_group_layout,
            fallback_pipeline: pipelines.fallback_pipeline,
            fallback_bind_group_layout: layouts.fallback_bind_group_layout,
            temporal_pipeline: pipelines.temporal_pipeline,
            temporal_bind_group_layout: layouts.temporal_bind_group_layout,
            composite_pipeline: pipelines.composite_pipeline,
            composite_bind_group_layout: layouts.composite_bind_group_layout,
            composite_params: buffers.composite_params,
            ssr_spec_texture: textures.ssr_spec_texture,
            ssr_spec_view: textures.ssr_spec_view,
            ssr_final_texture: textures.ssr_final_texture,
            ssr_final_view: textures.ssr_final_view,
            ssr_history_texture: textures.ssr_history_texture,
            ssr_history_view: textures.ssr_history_view,
            ssr_filtered_texture: textures.ssr_filtered_texture,
            ssr_filtered_view: textures.ssr_filtered_view,
            ssr_hit_texture: textures.ssr_hit_texture,
            ssr_hit_view: textures.ssr_hit_view,
            ssr_composited_texture: textures.ssr_composited_texture,
            ssr_composited_view: textures.ssr_composited_view,
            scene_color_override: None,
            env_texture: textures.env_texture,
            env_view: textures.env_view,
            env_sampler: textures.env_sampler,
            linear_sampler: textures.linear_sampler,
            width,
            height,
            counters_buffer: buffers.counters_buffer,
            counters_readback: buffers.counters_readback,
            temporal_params: buffers.temporal_params,
            last_trace_ms: 0.0,
            last_shade_ms: 0.0,
            last_fallback_ms: 0.0,
            stats_readback_pending: false,
        })
    }
}
