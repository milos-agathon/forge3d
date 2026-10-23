use super::*;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};
use crate::core::temporal_history::TemporalHistoryState;

mod accessors;
mod constructor;
mod resize;
mod runtime;
mod stats;

pub struct SsrRenderer {
    settings: SsrSettings,
    settings_buffer: TrackedBuffer,
    camera_buffer: TrackedBuffer,
    concordia_projection: bool,
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
    composite_params: TrackedBuffer,

    // ssr_spec_texture   : Rgba16Float raw SSR specular from cs_shade
    //                      (rgb = spec radiance, a = reflection weight in [0,1]).
    _ssr_spec_texture: TrackedTexture,
    ssr_spec_view: TextureView,
    // ssr_final_texture  : Rgba16Float SSR after environment fallback
    //                      (rgb = spec radiance, a > 0 for surface hits, a = 0 for
    //                      env-only misses; see fallback_env.wgsl).
    _ssr_final_texture: TrackedTexture,
    ssr_final_view: TextureView,
    // ssr_history_texture: Rgba16Float previous-frame SSR used for temporal filtering.
    ssr_history_texture: TrackedTexture,
    ssr_history_view: TextureView,
    // ssr_filtered_texture: Rgba16Float temporally filtered SSR (input to composite).
    ssr_filtered_texture: TrackedTexture,
    ssr_filtered_view: TextureView,
    // ssr_hit_texture    : Rgba16Float hit buffer from cs_trace (xy = hit UV in [0,1],
    //                      z = normalized step count, w = hit mask in {0,1}).
    ssr_hit_texture: TrackedTexture,
    ssr_hit_view: TextureView,
    // ssr_composited_texture: Rgba8Unorm view of base lighting + SSR specular after
    //                         tone mapping; used by the viewer for SSR previews.
    _ssr_composited_texture: TrackedTexture,
    ssr_composited_view: TextureView,
    scene_color_override: Option<TextureView>,

    _env_texture: TrackedTexture,
    env_view: TextureView,
    env_sampler: Sampler,
    linear_sampler: Sampler,
    width: u32,
    height: u32,

    counters_buffer: TrackedBuffer,
    counters_readback: TrackedBuffer,
    temporal_params: TrackedBuffer,
    history_state: TemporalHistoryState,

    last_trace_ms: f32,
    last_shade_ms: f32,
    last_fallback_ms: f32,
    stats_readback_pending: bool,
}

pub(crate) use resize::PreparedSsrTargets;

impl SsrRenderer {
    pub(crate) fn configure_concordia_projection(&mut self, device: &Device) -> RenderResult<()> {
        {
            let source = super::concordia_reconstruction(include_str!("../../shaders/ssr/trace.wgsl"), "camera", "reconstruct_view_position")?
                .replace("normalize(-origin_vs)", "select(normalize(-origin_vs), vec3<f32>(0.0,0.0,1.0), camera.proj_matrix[3][3] != 0.0)")
                .replace("normalize(-view_pos)", "select(normalize(-view_pos), vec3<f32>(0.0,0.0,1.0), camera.proj_matrix[3][3] != 0.0)");
            let module = crate::core::shader_registry::create_labeled_shader_module(
                device,
                "concordia.ssr.trace",
                &source,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("concordia.ssr.trace.layout"),
                bind_group_layouts: &[&self.trace_bind_group_layout],
                push_constant_ranges: &[],
            });
            self.trace_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("concordia.ssr.trace.pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: "cs_trace",
                },
            )
            .map_err(crate::core::error::RenderError::Render)?;
        }
        {
            let source = super::concordia_reconstruction(include_str!("../../shaders/ssr/shade.wgsl"), "camera", "reconstruct_view_position")?
                .replace("normalize(-origin_vs)", "select(normalize(-origin_vs), vec3<f32>(0.0,0.0,1.0), camera.proj_matrix[3][3] != 0.0)")
                .replace("normalize(-view_pos)", "select(normalize(-view_pos), vec3<f32>(0.0,0.0,1.0), camera.proj_matrix[3][3] != 0.0)");
            let module = crate::core::shader_registry::create_labeled_shader_module(
                device,
                "concordia.ssr.shade",
                &source,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("concordia.ssr.shade.layout"),
                bind_group_layouts: &[&self.shade_bind_group_layout],
                push_constant_ranges: &[],
            });
            self.shade_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some("concordia.ssr.shade.pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: "cs_shade",
                },
            )
            .map_err(crate::core::error::RenderError::Render)?;
        }
        {
            let source = super::concordia_reconstruction(include_str!("../../shaders/ssr/fallback_env.wgsl"), "camera", "reconstruct_view_position")?
                .replace("normalize(-origin_vs)", "select(normalize(-origin_vs), vec3<f32>(0.0,0.0,1.0), camera.proj_matrix[3][3] != 0.0)")
                .replace("normalize(-view_pos)", "select(normalize(-view_pos), vec3<f32>(0.0,0.0,1.0), camera.proj_matrix[3][3] != 0.0)");
            let module = crate::core::shader_registry::create_labeled_shader_module(
                device,
                "concordia.ssr.fallback",
                &source,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("concordia.ssr.fallback.layout"),
                bind_group_layouts: &[&self.fallback_bind_group_layout],
                push_constant_ranges: &[],
            });
            self.fallback_pipeline =
                crate::core::shader_registry::try_create_compute_pipeline_scoped(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("concordia.ssr.fallback.pipeline"),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: "cs_fallback",
                    },
                )
                .map_err(crate::core::error::RenderError::Render)?;
        }
        self.concordia_projection = true;
        Ok(())
    }
}
