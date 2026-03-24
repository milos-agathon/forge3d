use super::*;

/// Reusable GPU terrain scene (M2).
///
/// Owns the WGPU pipeline state for the PBR+POM terrain path and is free of
/// any PyO3 attributes so it can be reused by the interactive viewer and
/// other Rust callers.
#[allow(dead_code)]
pub struct TerrainScene {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) adapter: Arc<wgpu::Adapter>,
    pub(super) pipeline: Mutex<PipelineCache>,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) ibl_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) blit_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) blit_pipeline: wgpu::RenderPipeline,
    pub(super) aov_blit_pipeline: wgpu::RenderPipeline,
    pub(super) background_blit_pipeline: wgpu::RenderPipeline,
    pub(super) normal_blit_pipeline: wgpu::RenderPipeline,
    pub(super) offline_compute: OfflineComputeResources,
    pub(super) sampler_linear: wgpu::Sampler,
    pub(super) sky_bind_group_layout0: wgpu::BindGroupLayout,
    pub(super) sky_bind_group_layout1: wgpu::BindGroupLayout,
    pub(super) sky_pipeline: wgpu::ComputePipeline,
    pub(super) sky_fallback_texture: wgpu::Texture,
    pub(super) sky_fallback_view: wgpu::TextureView,
    pub(super) height_curve_identity_texture: wgpu::Texture,
    pub(super) height_curve_identity_view: wgpu::TextureView,
    pub(super) water_mask_fallback_texture: wgpu::Texture,
    pub(super) water_mask_fallback_view: wgpu::TextureView,
    pub(super) ao_debug_fallback_texture: wgpu::Texture,
    pub(super) ao_debug_fallback_view: wgpu::TextureView,
    pub(super) ao_debug_sampler: wgpu::Sampler,
    pub(super) ao_debug_view: Option<wgpu::TextureView>,
    pub(super) coarse_ao_texture: Option<wgpu::Texture>,
    pub(super) coarse_ao_view: Option<wgpu::TextureView>,
    pub(super) detail_normal_fallback_view: wgpu::TextureView,
    pub(super) detail_normal_sampler: wgpu::Sampler,
    pub(super) height_ao_fallback_view: wgpu::TextureView,
    pub(super) height_ao_sampler: wgpu::Sampler,
    pub(super) sun_vis_fallback_view: wgpu::TextureView,
    pub(super) sun_vis_sampler: wgpu::Sampler,
    pub(super) height_ao_compute_pipeline: wgpu::ComputePipeline,
    pub(super) height_ao_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) height_ao_uniform_buffer: wgpu::Buffer,
    pub(super) height_ao_texture: Mutex<Option<wgpu::Texture>>,
    pub(super) height_ao_storage_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) height_ao_sample_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) height_ao_size: Mutex<(u32, u32)>,
    pub(super) sun_vis_compute_pipeline: wgpu::ComputePipeline,
    pub(super) sun_vis_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sun_vis_uniform_buffer: wgpu::Buffer,
    pub(super) sun_vis_texture: Mutex<Option<wgpu::Texture>>,
    pub(super) sun_vis_storage_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) sun_vis_sample_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) sun_vis_size: Mutex<(u32, u32)>,
    pub(super) height_curve_lut_sampler: wgpu::Sampler,
    pub(super) color_format: wgpu::TextureFormat,
    pub(super) light_buffer: Arc<Mutex<LightBuffer>>,
    pub(super) noop_shadow: NoopShadow,
    pub(super) csm_renderer: crate::shadows::CsmRenderer,
    pub(super) shadow_depth_pipeline: wgpu::RenderPipeline,
    pub(super) shadow_depth_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) shadow_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) shadow_pcss_radius: f32,
    pub(super) shadow_technique: u32,
    pub(super) moment_pass: Option<crate::shadows::MomentGenerationPass>,
    pub(super) fog_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) fog_uniform_buffer: wgpu::Buffer,
    pub(super) water_reflection_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) water_reflection_uniform_buffer: wgpu::Buffer,
    pub(super) water_reflection_texture: Mutex<wgpu::Texture>,
    pub(super) water_reflection_view: Mutex<wgpu::TextureView>,
    pub(super) water_reflection_sampler: wgpu::Sampler,
    pub(super) water_reflection_depth_texture: Mutex<wgpu::Texture>,
    pub(super) water_reflection_depth_view: Mutex<wgpu::TextureView>,
    pub(super) water_reflection_size: Mutex<(u32, u32)>,
    pub(super) water_reflection_fallback_view: wgpu::TextureView,
    pub(super) water_reflection_pipeline: wgpu::RenderPipeline,
    pub(super) material_layer_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) material_layer_uniform_buffer: wgpu::Buffer,
    pub(super) vt_uniform_buffer: wgpu::Buffer,
    pub(super) vt_fallback_uniform_buffer: wgpu::Buffer,
    pub(super) vt_atlas_fallback_texture: wgpu::Texture,
    pub(super) vt_atlas_fallback_view: wgpu::TextureView,
    pub(super) vt_page_table_fallback_texture: wgpu::Texture,
    pub(super) vt_page_table_fallback_view: wgpu::TextureView,
    pub(super) vt_atlas_sampler: wgpu::Sampler,
    pub(super) probe_grid_uniform_buffer: wgpu::Buffer,
    pub(super) probe_ssbo: wgpu::Buffer,
    pub(super) probe_grid_uniform_alloc_bytes: u64,
    pub(super) probe_ssbo_alloc_bytes: u64,
    pub(super) probe_grid_uniform_bytes: u64,
    pub(super) probe_ssbo_bytes: u64,
    pub(super) probe_cache_key: Option<u64>,
    pub(super) probe_cached_grid: Option<crate::terrain::probes::ProbeGridDesc>,
    pub(super) probe_cached_data: Vec<crate::terrain::probes::GpuProbeData>,
    pub(super) reflection_probe_grid_uniform_buffer: wgpu::Buffer,
    pub(super) reflection_probe_sampler: wgpu::Sampler,
    pub(super) reflection_probe_fallback_texture: wgpu::Texture,
    pub(super) reflection_probe_fallback_view: wgpu::TextureView,
    pub(super) reflection_probe_texture: Option<wgpu::Texture>,
    pub(super) reflection_probe_view: wgpu::TextureView,
    pub(super) reflection_probe_grid_uniform_alloc_bytes: u64,
    pub(super) reflection_probe_grid_uniform_bytes: u64,
    pub(super) reflection_probe_texture_alloc_bytes: u64,
    pub(super) reflection_probe_texture_bytes: u64,
    pub(super) reflection_probe_cache_key: Option<u64>,
    pub(super) reflection_probe_cached_grid: Option<crate::terrain::probes::ProbeGridDesc>,
    pub(super) reflection_probe_count: u32,
    pub(super) reflection_probe_resolution: u32,
    pub(super) reflection_probe_mip_levels: u32,
    pub(super) aov_pipeline: Mutex<Option<wgpu::RenderPipeline>>,
    pub(super) aov_pipeline_sample_count: Mutex<u32>,
    pub(super) dof_renderer: Mutex<Option<crate::core::dof::DofRenderer>>,
    pub(super) offline_state: Mutex<Option<OfflineAccumulationState>>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_renderer: crate::render::mesh_instanced::MeshInstancedRenderer,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_renderer_sample_count: u32,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_batches: Vec<crate::terrain::scatter::TerrainScatterBatch>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_last_frame_stats: crate::terrain::scatter::TerrainScatterFrameStats,
    #[cfg(feature = "enable-renderer-config")]
    pub(super) config: Arc<Mutex<crate::render::params::RendererConfig>>,
    pub(super) viewer_heightmap: Option<ViewerTerrainData>,
}

#[allow(dead_code)]
pub struct ViewerTerrainData {
    pub heightmap: Vec<f32>,
    pub dimensions: (u32, u32),
    pub domain: (f32, f32),
    pub heightmap_texture: wgpu::Texture,
    pub heightmap_view: wgpu::TextureView,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_fov_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
}

#[pyclass(module = "forge3d._forge3d", name = "TerrainRenderer")]
pub struct TerrainRenderer {
    pub(super) scene: TerrainScene,
}

pub(super) struct OfflineAccumulationState {
    pub(super) params: crate::terrain::render_params::TerrainRenderParams,
    pub(super) decoded: crate::terrain::render_params::DecodedTerrainSettings,
    pub(super) height_inputs: super::draw::UploadedHeightInputs,
    pub(super) materials: super::draw::PreparedMaterials,
    pub(super) ibl_bind_group: wgpu::BindGroup,
    pub(super) height_curve_lut_uploaded: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub(super) hdr_aov_pipeline: wgpu::RenderPipeline,
    pub(super) hdr_background_blit_pipeline: wgpu::RenderPipeline,
    pub(super) render_targets: super::draw::RenderTargets,
    pub(super) aov_targets: super::aov::TerrainAovTargets,
    pub(super) beauty_accumulation: crate::terrain::AccumulationBuffer,
    pub(super) albedo_accumulation: crate::terrain::AccumulationBuffer,
    pub(super) normal_accumulation: crate::terrain::AccumulationBuffer,
    pub(super) depth_reference_texture: wgpu::Texture,
    pub(super) depth_reference_view: wgpu::TextureView,
    pub(super) luminance_texture: wgpu::Texture,
    pub(super) luminance_view: wgpu::TextureView,
    pub(super) luminance_width: u32,
    pub(super) luminance_height: u32,
    pub(super) jitter_sequence: crate::terrain::JitterSequence,
    pub(super) total_samples: u32,
    pub(super) out_width: u32,
    pub(super) out_height: u32,
    pub(super) internal_width: u32,
    pub(super) internal_height: u32,
    pub(super) needs_scaling: bool,
    pub(super) prev_tile_means: Vec<f32>,
    pub(super) prev_tile_mean_history: Vec<Vec<f32>>,
    pub(super) prev_tile_size: u32,
}

pub(super) struct OfflineComputeResources {
    pub(super) accumulate_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) accumulate_pipeline: wgpu::ComputePipeline,
    pub(super) resolve_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) resolve_pipeline: wgpu::ComputePipeline,
    pub(super) depth_extract_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_extract_pipeline: wgpu::ComputePipeline,
    pub(super) depth_expand_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_expand_pipeline: wgpu::ComputePipeline,
    pub(super) luminance_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) luminance_pipeline: wgpu::ComputePipeline,
    pub(super) tonemap_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) tonemap_pipeline: wgpu::ComputePipeline,
}

#[allow(dead_code)]
pub(super) struct NoopShadow {
    pub(super) _csm_uniform_buffer: wgpu::Buffer,
    pub(super) _shadow_maps_texture: wgpu::Texture,
    pub(super) shadow_maps_view: wgpu::TextureView,
    pub(super) shadow_sampler: wgpu::Sampler,
    pub(super) _moment_maps_texture: wgpu::Texture,
    pub(super) moment_maps_view: wgpu::TextureView,
    pub(super) moment_sampler: wgpu::Sampler,
    pub(super) bind_group: wgpu::BindGroup,
}

pub(super) struct OverlayBinding {
    pub(super) uniform: OverlayUniforms,
    pub(super) lut: Option<Arc<crate::terrain::ColormapLUT>>,
}

pub(super) struct PipelineCache {
    pub(super) sample_count: u32,
    pub(super) pipeline: wgpu::RenderPipeline,
}

pub(super) const TERRAIN_DEFAULT_CASCADE_SPLITS: [f32; 4] = [50.0, 200.0, 800.0, 3000.0];
pub(super) const MATERIAL_LAYER_CAPACITY: usize = 4;
pub(super) const TERRAIN_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct IblUniforms {
    pub(super) intensity: f32,
    pub(super) sin_theta: f32,
    pub(super) cos_theta: f32,
    pub(super) specular_mip_count: f32,
}

impl Drop for TerrainScene {
    fn drop(&mut self) {
        let tracker = crate::core::memory_tracker::global_tracker();
        if self.probe_grid_uniform_alloc_bytes > 0 {
            tracker.free_buffer_allocation(self.probe_grid_uniform_alloc_bytes, false);
        }
        if self.probe_ssbo_alloc_bytes > 0 {
            tracker.free_buffer_allocation(self.probe_ssbo_alloc_bytes, false);
        }
        if self.reflection_probe_grid_uniform_alloc_bytes > 0 {
            tracker.free_buffer_allocation(self.reflection_probe_grid_uniform_alloc_bytes, false);
        }
    }
}
