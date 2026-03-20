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
    pub(super) accumulation_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) accumulation_pipeline: wgpu::ComputePipeline,
    pub(super) accumulation_texture: Mutex<Option<wgpu::Texture>>,
    pub(super) accumulation_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) accumulation_size: Mutex<(u32, u32)>,
    pub(super) accumulation_params_buffer: wgpu::Buffer,
    pub(super) material_layer_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) material_layer_uniform_buffer: wgpu::Buffer,
    pub(super) aov_pipeline: Mutex<Option<wgpu::RenderPipeline>>,
    pub(super) aov_pipeline_sample_count: Mutex<u32>,
    pub(super) dof_renderer: Mutex<Option<crate::core::dof::DofRenderer>>,
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
