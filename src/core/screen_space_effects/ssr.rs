use super::*;

mod accessors;
mod constructor;
mod runtime;
mod stats;

pub struct SsrRenderer {
    settings: SsrSettings,
    settings_buffer: Buffer,
    camera_buffer: Buffer,
    trace_pipeline: ComputePipeline,
    trace_bind_group_layout: BindGroupLayout,
    shade_pipeline: ComputePipeline,
    shade_bind_group_layout: BindGroupLayout,
    fallback_pipeline: ComputePipeline,
    fallback_bind_group_layout: BindGroupLayout,
    temporal_pipeline: ComputePipeline,
    temporal_bind_group_layout: BindGroupLayout,
    composite_pipeline: ComputePipeline,
    composite_bind_group_layout: BindGroupLayout,
    composite_params: Buffer,

    // ssr_spec_texture   : Rgba16Float raw SSR specular from cs_shade
    //                      (rgb = spec radiance, a = reflection weight in [0,1]).
    _ssr_spec_texture: Texture,
    ssr_spec_view: TextureView,
    // ssr_final_texture  : Rgba16Float SSR after environment fallback
    //                      (rgb = spec radiance, a > 0 for surface hits, a = 0 for
    //                      env-only misses; see fallback_env.wgsl).
    _ssr_final_texture: Texture,
    ssr_final_view: TextureView,
    // ssr_history_texture: Rgba16Float previous-frame SSR used for temporal filtering.
    ssr_history_texture: Texture,
    ssr_history_view: TextureView,
    // ssr_filtered_texture: Rgba16Float temporally filtered SSR (input to composite).
    ssr_filtered_texture: Texture,
    ssr_filtered_view: TextureView,
    // ssr_hit_texture    : Rgba16Float hit buffer from cs_trace (xy = hit UV in [0,1],
    //                      z = normalized step count, w = hit mask in {0,1}).
    ssr_hit_texture: Texture,
    ssr_hit_view: TextureView,
    // ssr_composited_texture: Rgba8Unorm view of base lighting + SSR specular after
    //                         tone mapping; used by the viewer for SSR previews.
    _ssr_composited_texture: Texture,
    ssr_composited_view: TextureView,
    scene_color_override: Option<TextureView>,

    _env_texture: Texture,
    env_view: TextureView,
    env_sampler: Sampler,
    linear_sampler: Sampler,
    width: u32,
    height: u32,

    counters_buffer: Buffer,
    counters_readback: Buffer,
    temporal_params: Buffer,

    last_trace_ms: f32,
    last_shade_ms: f32,
    last_fallback_ms: f32,
    stats_readback_pending: bool,
}
