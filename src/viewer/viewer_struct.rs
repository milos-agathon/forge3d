// src/viewer/viewer_struct.rs
// Viewer struct definition with pub(crate) fields to allow method extraction
// RELEVANT FILES: src/viewer/mod.rs

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use wgpu::{
    Adapter, BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, Queue, RenderPipeline,
    Sampler, Surface, SurfaceConfiguration, Texture, TextureView,
};
use winit::keyboard::KeyCode;
use winit::window::Window;

use crate::cli::args::GiVizMode;
use crate::core::gpu_timing::GpuTimingManager;
use crate::core::ibl::IBLRenderer;
use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::core::shadows::{CsmConfig, CsmShadowMap};
use crate::core::text_overlay::TextOverlayRenderer;
use crate::labels::LabelManager;
use crate::picking::UnifiedPickingSystem;
use crate::p5::ssr::SsrScenePreset;
use crate::passes::gi::GiPass;
use crate::render::params::SsrParams;

use super::camera_controller::CameraController;
use super::viewer_config::{FpsCounter, ViewerConfig};
use super::viewer_enums::{CaptureKind, FogMode, VizMode};

pub struct Viewer {
    pub(crate) window: Arc<Window>,
    pub(crate) surface: Surface<'static>,
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) config: SurfaceConfiguration,
    pub(crate) camera: CameraController,
    pub(crate) view_config: ViewerConfig,
    pub(crate) frame_count: u64,
    pub(crate) fps_counter: FpsCounter,
    #[cfg(feature = "extension-module")]
    pub(crate) terrain_scene: Option<crate::terrain::TerrainScene>,
    // Standalone terrain viewer (no PyO3 dependencies)
    pub(crate) terrain_viewer: Option<super::terrain::ViewerTerrainScene>,
    // Input state
    pub(crate) keys_pressed: HashSet<KeyCode>,
    pub(crate) shift_pressed: bool,
    // GI manager and toggles
    pub(crate) gi: Option<ScreenSpaceEffectsManager>,
    pub(crate) gi_pass: Option<GiPass>,
    pub(crate) ssr_params: SsrParams,
    pub(crate) gi_seed: Option<u32>,
    pub(crate) gi_timing: Option<GpuTimingManager>,
    pub(crate) gi_gpu_hzb_ms: f32,
    pub(crate) gi_gpu_ssao_ms: f32,
    pub(crate) gi_gpu_ssgi_ms: f32,
    pub(crate) gi_gpu_ssr_ms: f32,
    pub(crate) gi_gpu_composite_ms: f32,
    // Snapshot request path (processed on next frame before present)
    pub(crate) snapshot_request: Option<String>,
    // Offscreen color to read back when snapshotting this frame
    pub(crate) pending_snapshot_tex: Option<Texture>,
    // P5.1: deferred capture queue processed after rendering
    pub(crate) pending_captures: VecDeque<CaptureKind>,
    // GBuffer geometry pipeline and resources
    pub(crate) geom_bind_group_layout: Option<BindGroupLayout>,
    pub(crate) geom_pipeline: Option<RenderPipeline>,
    pub(crate) geom_camera_buffer: Option<Buffer>,
    pub(crate) geom_bind_group: Option<BindGroup>,
    pub(crate) geom_vb: Option<Buffer>,
    pub(crate) geom_ib: Option<Buffer>,
    pub(crate) geom_index_count: u32,
    // Store original mesh data for CPU-side transform (workaround for GPU buffer sync issue)
    pub(crate) original_mesh_positions: Vec<[f32; 3]>,
    pub(crate) original_mesh_normals: Vec<[f32; 3]>,
    pub(crate) original_mesh_uvs: Vec<[f32; 2]>,
    pub(crate) original_mesh_indices: Vec<u32>,
    pub(crate) z_texture: Option<Texture>,
    pub(crate) z_view: Option<TextureView>,
    // Albedo texture for geometry
    pub(crate) albedo_texture: Option<Texture>,
    pub(crate) albedo_view: Option<TextureView>,
    pub(crate) albedo_sampler: Option<Sampler>,
    pub(crate) ssr_env_texture: Option<Texture>,
    // Composite pipeline (debug show material GBuffer on screen)
    pub(crate) comp_bind_group_layout: Option<BindGroupLayout>,
    pub(crate) comp_pipeline: Option<RenderPipeline>,
    pub(crate) comp_uniform: Option<Buffer>,
    // Lit viz compute pipeline (albedo+normal shading)
    pub(crate) lit_bind_group_layout: BindGroupLayout,
    pub(crate) lit_pipeline: ComputePipeline,
    pub(crate) lit_uniform: Buffer,
    pub(crate) lit_output: Texture,
    pub(crate) lit_output_view: TextureView,
    pub(crate) gi_baseline_hdr: Texture,
    pub(crate) gi_baseline_hdr_view: TextureView,
    pub(crate) gi_baseline_diffuse_hdr: Texture,
    pub(crate) gi_baseline_diffuse_hdr_view: TextureView,
    pub(crate) gi_baseline_spec_hdr: Texture,
    pub(crate) gi_baseline_spec_hdr_view: TextureView,
    pub(crate) gi_output_hdr: Texture,
    pub(crate) gi_output_hdr_view: TextureView,
    pub(crate) gi_debug: Texture,
    pub(crate) gi_debug_view: TextureView,
    pub(crate) gi_baseline_bgl: BindGroupLayout,
    pub(crate) gi_baseline_pipeline: ComputePipeline,
    pub(crate) gi_split_bgl: BindGroupLayout,
    pub(crate) gi_split_pipeline: ComputePipeline,
    pub(crate) gi_ao_weight: f32,
    pub(crate) gi_ssgi_weight: f32,
    pub(crate) gi_ssr_weight: f32,
    // Lit params (exposed via :lit-* commands)
    pub(crate) lit_sun_intensity: f32,
    pub(crate) lit_ibl_intensity: f32,
    pub(crate) lit_use_ibl: bool,
    pub(crate) lit_ibl_rotation_deg: f32,
    // Lit BRDF selection (0=Lambert,1=Phong,4=GGX,6=Disney)
    pub(crate) lit_brdf: u32,
    // Lit roughness (used by debug modes and future shading controls)
    pub(crate) lit_roughness: f32,
    // Lit debug mode: 0=off, 1=roughness smoke test, 2=NDF-only GGX
    pub(crate) lit_debug_mode: u32,
    // Fallback pipeline to draw a solid color when GI/geometry path is unavailable
    pub(crate) fallback_pipeline: RenderPipeline,
    pub(crate) viz_mode: VizMode,
    pub(crate) gi_viz_mode: GiVizMode,
    // SSAO composite control
    pub(crate) use_ssao_composite: bool,
    pub(crate) ssao_composite_mul: f32,
    // Cached SSAO blur toggle for query commands
    pub(crate) ssao_blur_enabled: bool,
    // IBL integration
    pub(crate) ibl_renderer: Option<IBLRenderer>,
    pub(crate) ibl_env_view: Option<TextureView>,
    pub(crate) ibl_sampler: Option<Sampler>,
    pub(crate) ibl_hdr_path: Option<String>,
    pub(crate) ibl_cache_dir: Option<std::path::PathBuf>,
    pub(crate) ibl_base_resolution: Option<u32>,
    // Viz depth override
    pub(crate) viz_depth_max_override: Option<f32>,
    // Auto-snapshot support (one-time)
    pub(crate) auto_snapshot_path: Option<String>,
    pub(crate) auto_snapshot_done: bool,
    // P5 dump request
    pub(crate) dump_p5_requested: bool,
    // Adapter name for meta
    pub(crate) adapter_name: String,
    // Debug: log render gate and snapshot once
    pub(crate) debug_logged_render_gate: bool,

    // Sky rendering (P6-01)
    pub(crate) sky_bind_group_layout0: BindGroupLayout,
    pub(crate) sky_bind_group_layout1: BindGroupLayout,
    pub(crate) sky_pipeline: ComputePipeline,
    pub(crate) sky_params: Buffer,
    pub(crate) sky_camera: Buffer,
    pub(crate) sky_output: Texture,
    pub(crate) sky_output_view: TextureView,
    pub(crate) sky_enabled: bool,

    // P6: Fog rendering resources and parameters
    pub(crate) fog_enabled: bool,
    pub(crate) fog_params: Buffer,
    pub(crate) fog_camera: Buffer,
    pub(crate) fog_output: Texture,
    pub(crate) fog_output_view: TextureView,
    pub(crate) fog_history: Texture,
    pub(crate) fog_history_view: TextureView,
    pub(crate) fog_depth_sampler: Sampler,
    pub(crate) fog_history_sampler: Sampler,
    pub(crate) fog_pipeline: ComputePipeline,
    pub(crate) fog_frame_index: u32,
    // Froxelized volumetrics (Milestone 4)
    pub(crate) fog_bgl3: BindGroupLayout,
    pub(crate) froxel_tex: Texture,
    pub(crate) froxel_view: TextureView,
    pub(crate) froxel_sampler: Sampler,
    pub(crate) froxel_build_pipeline: ComputePipeline,
    pub(crate) froxel_apply_pipeline: ComputePipeline,
    // P6-10: Half-resolution fog + upsample
    pub(crate) fog_half_res_enabled: bool,
    pub(crate) fog_output_half: Texture,
    pub(crate) fog_output_half_view: TextureView,
    pub(crate) fog_history_half: Texture,
    pub(crate) fog_history_half_view: TextureView,
    pub(crate) fog_upsample_bgl: BindGroupLayout,
    pub(crate) fog_upsample_pipeline: ComputePipeline,
    pub(crate) fog_upsample_params: Buffer,
    // Bilateral upsample controls
    pub(crate) fog_bilateral: bool,
    pub(crate) fog_upsigma: f32,
    // Fog bind group layouts and shadow resources
    pub(crate) fog_bgl0: BindGroupLayout,
    pub(crate) fog_bgl1: BindGroupLayout,
    pub(crate) fog_bgl2: BindGroupLayout,
    #[allow(dead_code)]
    pub(crate) fog_shadow_map: Texture,
    pub(crate) fog_shadow_view: TextureView,
    pub(crate) fog_shadow_sampler: Sampler,
    pub(crate) fog_shadow_matrix: Buffer,
    // Fog zero fallback (1x1 RGBA16F zero) for disabled fog compositing
    #[allow(dead_code)]
    pub(crate) fog_zero_tex: Texture,
    pub(crate) fog_zero_view: TextureView,
    // Exposed toggles
    pub(crate) fog_density: f32,
    pub(crate) fog_g: f32,
    pub(crate) fog_steps: u32,
    pub(crate) fog_temporal_alpha: f32,
    pub(crate) fog_use_shadows: bool,
    pub(crate) fog_mode: FogMode,
    // Cascaded shadow maps for directional sun shadows (future fog + lighting)
    pub(crate) csm: Option<CsmShadowMap>,
    pub(crate) csm_config: CsmConfig,
    pub(crate) csm_depth_pipeline: Option<RenderPipeline>,
    pub(crate) csm_depth_camera: Option<Buffer>,
    // Sky exposed controls (runtime adjustable)
    pub(crate) sky_model_id: u32, // 0=Preetham,1=Hosek-Wilkie
    pub(crate) sky_turbidity: f32,
    pub(crate) sky_ground_albedo: f32,
    pub(crate) sky_exposure: f32,
    pub(crate) sky_sun_intensity: f32,

    // HUD overlay renderer
    pub(crate) hud_enabled: bool,
    pub(crate) hud: TextOverlayRenderer,
    // Label manager for screen-space text labels
    pub(crate) label_manager: LabelManager,
    // Unified picking system (Plan 3)
    pub(crate) unified_picking: UnifiedPickingSystem,
    // Currently selected feature for highlighting
    pub(crate) selected_feature_id: u32,
    pub(crate) selected_layer_name: String,
    pub(crate) ssr_scene_loaded: bool,
    pub(crate) ssr_scene_preset: Option<SsrScenePreset>,
    // Object transform (for IPC SetTransform command)
    pub(crate) object_translation: glam::Vec3,
    pub(crate) object_rotation: glam::Quat,
    pub(crate) object_scale: glam::Vec3,
    pub(crate) object_transform: glam::Mat4,
    // Transform version counter for IPC ack (incremented on each set_transform)
    pub(crate) transform_version: u64,
    // P0.1/M1: OIT (Order-Independent Transparency)
    pub(crate) oit_enabled: bool,
    pub(crate) oit_mode: String,
}
