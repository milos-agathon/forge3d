use super::*;

mod accessors;
mod constructor;
mod controls;
mod runtime;

/// SSGI renderer
#[allow(dead_code)]
pub struct SsgiRenderer {
    settings: SsgiSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    frame_index: u32,

    // Pipelines
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
    ssgi_hit: Texture,
    ssgi_hit_view: TextureView,
    // ssgi_texture   : Rgba16Float half-res GI radiance (rgb = diffuse bounce light in
    //                   linear HDR units, a unused/1.0).
    ssgi_texture: Texture,
    ssgi_view: TextureView,
    // ssgi_history   : Rgba16Float previous-frame GI radiance used for temporal resolve.
    ssgi_history: Texture,
    ssgi_history_view: TextureView,
    // ssgi_filtered  : Rgba16Float temporally filtered GI radiance (same layout as
    //                   ssgi_texture).
    ssgi_filtered: Texture,
    ssgi_filtered_view: TextureView,
    // Full-resolution upscaled output for half-res mode
    // ssgi_upscaled  : Rgba16Float full-res GI radiance after edge-aware upsample.
    ssgi_upscaled: Texture,
    ssgi_upscaled_view: TextureView,
    // Composited material (material + SSGI)
    // ssgi_composited: Rgba8Unorm material buffer + SSGI diffuse contribution, used for
    //                   P5 visualization and metrics (not the main HDR lighting buffer).
    ssgi_composited: Texture,
    ssgi_composited_view: TextureView,
    composite_uniform: Buffer,
    scene_history: [Texture; 2],
    scene_history_views: [TextureView; 2],
    scene_history_index: usize,
    scene_history_ready: bool,
    linear_sampler: Sampler,

    // Env
    env_texture: Texture,
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
