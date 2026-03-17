use super::*;

mod layouts;
mod pipelines;
mod resources;

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
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    composite_uniform: Buffer,
}

struct ConstructorResources {
    ssgi_hit: Texture,
    ssgi_hit_view: TextureView,
    ssgi_texture: Texture,
    ssgi_view: TextureView,
    ssgi_history: Texture,
    ssgi_history_view: TextureView,
    ssgi_filtered: Texture,
    ssgi_filtered_view: TextureView,
    ssgi_upscaled: Texture,
    ssgi_upscaled_view: TextureView,
    ssgi_composited: Texture,
    ssgi_composited_view: TextureView,
    scene_history: [Texture; 2],
    scene_history_views: [TextureView; 2],
    env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,
    linear_sampler: Sampler,
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

        let buffers = resources::create_buffers(device);
        let layouts = layouts::create_layouts(device);
        let pipelines = pipelines::create_pipelines(device, &layouts);
        let textures = resources::create_textures(device, width, height, material_format);

        Ok(Self {
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
            ssgi_hit: textures.ssgi_hit,
            ssgi_hit_view: textures.ssgi_hit_view,
            ssgi_texture: textures.ssgi_texture,
            ssgi_view: textures.ssgi_view,
            ssgi_history: textures.ssgi_history,
            ssgi_history_view: textures.ssgi_history_view,
            ssgi_filtered: textures.ssgi_filtered,
            ssgi_filtered_view: textures.ssgi_filtered_view,
            ssgi_upscaled: textures.ssgi_upscaled,
            ssgi_upscaled_view: textures.ssgi_upscaled_view,
            ssgi_composited: textures.ssgi_composited,
            ssgi_composited_view: textures.ssgi_composited_view,
            composite_uniform: buffers.composite_uniform,
            scene_history: textures.scene_history,
            scene_history_views: textures.scene_history_views,
            scene_history_index: 0,
            scene_history_ready: false,
            linear_sampler: textures.linear_sampler,
            env_texture: textures.env_texture,
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
