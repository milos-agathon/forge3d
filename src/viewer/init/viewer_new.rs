// src/viewer/init/viewer_new.rs
// Orchestrates Viewer::new() using init/ factory functions
// Extracted from mod.rs as part of the viewer refactoring

use std::collections::VecDeque;
use std::sync::Arc;

use winit::window::Window;

use crate::core::gpu_timing::{create_default_config as create_gpu_timing_config, GpuTimingManager};
use crate::core::shadows::{CsmConfig, CsmShadowMap};
use crate::render::params::SsrParams;

use super::super::camera_controller::CameraController;
use super::super::viewer_config::{FpsCounter, ViewerConfig};
use super::super::viewer_enums::{FogMode, VizMode};
use super::super::viewer_types::SkyUniforms;
use super::super::Viewer;
use super::{
    create_device_and_surface, create_fallback_pipeline, create_fog_resources,
    create_gbuffer_resources, create_gi_baseline_resources, create_lit_resources,
    create_sky_resources,
};
use crate::cli::args::GiVizMode;

impl Viewer {
    /// Create a new Viewer instance using the init/ factory functions
    pub async fn new(
        window: Arc<Window>,
        config: ViewerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Device and surface initialization
        let dev_res = create_device_and_surface(Arc::clone(&window), config.vsync).await?;
        let surface = dev_res.surface;
        let device = dev_res.device;
        let queue = dev_res.queue;
        let adapter = dev_res.adapter;
        let surface_config = dev_res.config;
        let adapter_name = dev_res.adapter_name;

        let width = surface_config.width;
        let height = surface_config.height;

        // Optional GPU timing manager for GI profiling
        let gi_timing = match GpuTimingManager::new(
            device.clone(),
            queue.clone(),
            create_gpu_timing_config(&device),
        ) {
            Ok(mgr) if mgr.is_supported() => Some(mgr),
            Ok(_) => None,
            Err(e) => {
                eprintln!("[viewer] GPU timing manager unavailable: {e}");
                None
            }
        };

        // Initialize P5 Screen-space effects manager (optional)
        let gi = match crate::core::screen_space_effects::ScreenSpaceEffectsManager::new(
            &device, width, height,
        ) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Failed to create ScreenSpaceEffectsManager: {}", e);
                None
            }
        };

        // GBuffer resources (depends on GI manager)
        let gbuf = create_gbuffer_resources(&device, gi.as_ref(), width, height, surface_config.format);

        // CSM depth pipeline resources
        let (csm_depth_pipeline, csm_depth_camera) = if gi.is_some() {
            create_csm_depth_resources(&device)
        } else {
            (None, None)
        };

        // Lit resources
        let lit = create_lit_resources(&device, width, height);

        // GI baseline resources
        let gi_base = create_gi_baseline_resources(&device, width, height);

        // Sky resources
        let sky = create_sky_resources(&device, width, height);

        // Fog resources
        let fog = create_fog_resources(&device, width, height);

        // Fallback pipeline
        let fb_pipeline = create_fallback_pipeline(&device, surface_config.format);

        // HUD overlay renderer
        let mut hud = crate::core::text_overlay::TextOverlayRenderer::new(&device, surface_config.format);
        hud.set_enabled(true);
        hud.set_resolution(width, height);

        // Configure CSM
        let mut csm_config = CsmConfig::default();
        csm_config.camera_near = config.znear;
        csm_config.camera_far = config.zfar;
        let csm = Some(CsmShadowMap::new(device.as_ref(), csm_config.clone()));

        // Read sky params from environment
        let sky_params_init = read_sky_env_params();

        // Write initial sky params
        queue.write_buffer(&sky.sky_params, 0, bytemuck::bytes_of(&sky_params_init));

        let mut viewer = Self {
            window,
            surface,
            device,
            queue,
            adapter,
            config: surface_config,
            camera: CameraController::new(),
            view_config: config,
            frame_count: 0,
            fps_counter: FpsCounter::new(),
            #[cfg(feature = "extension-module")]
            terrain_scene: None,
            terrain_viewer: None,
            keys_pressed: std::collections::HashSet::new(),
            shift_pressed: false,
            gi,
            gi_pass: None,
            ssr_params: SsrParams::default(),
            gi_seed: None,
            gi_timing,
            gi_gpu_hzb_ms: 0.0,
            gi_gpu_ssao_ms: 0.0,
            gi_gpu_ssgi_ms: 0.0,
            gi_gpu_ssr_ms: 0.0,
            gi_gpu_composite_ms: 0.0,
            snapshot_request: None,
            pending_snapshot_tex: None,
            pending_captures: VecDeque::new(),
            // GBuffer resources
            geom_bind_group_layout: gbuf.geom_bind_group_layout,
            geom_pipeline: gbuf.geom_pipeline,
            geom_camera_buffer: gbuf.geom_camera_buffer,
            geom_bind_group: gbuf.geom_bind_group,
            geom_vb: gbuf.geom_vb,
            geom_ib: None,
            geom_index_count: 36,
            original_mesh_positions: Vec::new(),
            original_mesh_normals: Vec::new(),
            original_mesh_uvs: Vec::new(),
            original_mesh_indices: Vec::new(),
            z_texture: gbuf.z_texture,
            z_view: gbuf.z_view,
            albedo_texture: gbuf.albedo_texture,
            albedo_view: gbuf.albedo_view,
            albedo_sampler: gbuf.albedo_sampler,
            ssr_env_texture: None,
            comp_bind_group_layout: gbuf.comp_bind_group_layout,
            comp_pipeline: gbuf.comp_pipeline,
            comp_uniform: None,
            // Lit resources
            lit_bind_group_layout: lit.lit_bind_group_layout,
            lit_pipeline: lit.lit_pipeline,
            lit_uniform: lit.lit_uniform,
            lit_output: lit.lit_output,
            lit_output_view: lit.lit_output_view,
            // GI baseline resources
            gi_baseline_hdr: gi_base.gi_baseline_hdr,
            gi_baseline_hdr_view: gi_base.gi_baseline_hdr_view,
            gi_baseline_diffuse_hdr: gi_base.gi_baseline_diffuse_hdr,
            gi_baseline_diffuse_hdr_view: gi_base.gi_baseline_diffuse_hdr_view,
            gi_baseline_spec_hdr: gi_base.gi_baseline_spec_hdr,
            gi_baseline_spec_hdr_view: gi_base.gi_baseline_spec_hdr_view,
            gi_output_hdr: gi_base.gi_output_hdr,
            gi_output_hdr_view: gi_base.gi_output_hdr_view,
            gi_debug: gi_base.gi_debug,
            gi_debug_view: gi_base.gi_debug_view,
            gi_baseline_bgl: gi_base.gi_baseline_bgl,
            gi_baseline_pipeline: gi_base.gi_baseline_pipeline,
            gi_split_bgl: gi_base.gi_split_bgl,
            gi_split_pipeline: gi_base.gi_split_pipeline,
            gi_ao_weight: 1.0,
            gi_ssgi_weight: 1.0,
            gi_ssr_weight: 1.0,
            // Lit params
            lit_sun_intensity: 1.0,
            lit_ibl_intensity: 0.6,
            lit_use_ibl: true,
            lit_ibl_rotation_deg: 0.0,
            lit_brdf: 4,
            lit_roughness: 0.5,
            lit_debug_mode: 0,
            fallback_pipeline: fb_pipeline,
            viz_mode: VizMode::Material,
            gi_viz_mode: GiVizMode::None,
            use_ssao_composite: true,
            ssao_composite_mul: 1.0,
            ssao_blur_enabled: true,
            ibl_renderer: None,
            ibl_env_view: Some(lit.dummy_env_view),
            ibl_sampler: Some(lit.dummy_env_sampler),
            ibl_hdr_path: None,
            ibl_cache_dir: None,
            ibl_base_resolution: None,
            viz_depth_max_override: None,
            auto_snapshot_path: std::env::var("FORGE3D_AUTO_SNAPSHOT_PATH").ok(),
            auto_snapshot_done: false,
            dump_p5_requested: false,
            adapter_name,
            debug_logged_render_gate: false,
            // Sky resources
            sky_bind_group_layout0: sky.sky_bind_group_layout0,
            sky_bind_group_layout1: sky.sky_bind_group_layout1,
            sky_pipeline: sky.sky_pipeline,
            sky_params: sky.sky_params,
            sky_camera: sky.sky_camera,
            sky_output: sky.sky_output,
            sky_output_view: sky.sky_output_view,
            sky_enabled: true,
            // Fog resources
            fog_enabled: false,
            fog_params: fog.fog_params,
            fog_camera: fog.fog_camera,
            fog_output: fog.fog_output,
            fog_output_view: fog.fog_output_view,
            fog_history: fog.fog_history,
            fog_history_view: fog.fog_history_view,
            fog_depth_sampler: fog.fog_depth_sampler,
            fog_history_sampler: fog.fog_history_sampler,
            fog_pipeline: fog.fog_pipeline,
            fog_frame_index: 0,
            fog_bgl3: fog.fog_bgl3,
            froxel_tex: fog.froxel_tex,
            froxel_view: fog.froxel_view,
            froxel_sampler: fog.froxel_sampler,
            froxel_build_pipeline: fog.froxel_build_pipeline,
            froxel_apply_pipeline: fog.froxel_apply_pipeline,
            fog_half_res_enabled: false,
            fog_output_half: fog.fog_output_half,
            fog_output_half_view: fog.fog_output_half_view,
            fog_history_half: fog.fog_history_half,
            fog_history_half_view: fog.fog_history_half_view,
            fog_upsample_bgl: fog.fog_upsample_bgl,
            fog_upsample_pipeline: fog.fog_upsample_pipeline,
            fog_upsample_params: fog.fog_upsample_params,
            fog_bilateral: true,
            fog_upsigma: 0.02,
            fog_bgl0: fog.fog_bgl0,
            fog_bgl1: fog.fog_bgl1,
            fog_bgl2: fog.fog_bgl2,
            fog_shadow_map: fog.fog_shadow_map,
            fog_shadow_view: fog.fog_shadow_view,
            fog_shadow_sampler: fog.fog_shadow_sampler,
            fog_shadow_matrix: fog.fog_shadow_matrix,
            fog_zero_tex: fog.fog_zero_tex,
            fog_zero_view: fog.fog_zero_view,
            fog_density: 0.02,
            fog_g: 0.0,
            fog_steps: 64,
            fog_temporal_alpha: 0.2,
            fog_use_shadows: false,
            fog_mode: FogMode::Raymarch,
            csm,
            csm_config,
            csm_depth_pipeline,
            csm_depth_camera,
            // Sky controls
            sky_model_id: sky_params_init.model,
            sky_turbidity: sky_params_init.turbidity,
            sky_ground_albedo: sky_params_init.ground_albedo,
            sky_exposure: sky_params_init.exposure,
            sky_sun_intensity: sky_params_init.sun_intensity,
            // HUD
            hud_enabled: true,
            hud,
            // Label manager
            label_manager: crate::labels::LabelManager::new(width, height),
            ssr_scene_loaded: false,
            ssr_scene_preset: None,
            // Object transform
            object_translation: glam::Vec3::ZERO,
            object_rotation: glam::Quat::IDENTITY,
            object_scale: glam::Vec3::ONE,
            object_transform: glam::Mat4::IDENTITY,
            transform_version: 0,
        };

        viewer.sync_ssr_params_to_gi();

        Ok(viewer)
    }
}

/// Read sky parameters from environment variables
fn read_sky_env_params() -> SkyUniforms {
    let mut params = SkyUniforms {
        sun_direction: [0.3, 0.8, 0.5],
        turbidity: 2.5,
        ground_albedo: 0.2,
        model: 1,
        sun_intensity: 20.0,
        exposure: 1.0,
        _pad: [0.0; 4],
    };

    if let Ok(model_str) = std::env::var("FORGE3D_SKY_MODEL") {
        let key = model_str.trim().to_ascii_lowercase().replace(['-', '_', ' '], "");
        params.model = match key.as_str() {
            "preetham" => 0,
            "hosekwilkie" => 1,
            _ => 1,
        };
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_TURBIDITY") {
        if let Ok(f) = v.parse::<f32>() {
            params.turbidity = f.clamp(1.0, 10.0);
        }
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_GROUND") {
        if let Ok(f) = v.parse::<f32>() {
            params.ground_albedo = f.clamp(0.0, 1.0);
        }
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_EXPOSURE") {
        if let Ok(f) = v.parse::<f32>() {
            params.exposure = f.max(0.0);
        }
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_INTENSITY") {
        if let Ok(f) = v.parse::<f32>() {
            params.sun_intensity = f.max(0.0);
        }
    }

    params
}

/// Create CSM depth pipeline resources
fn create_csm_depth_resources(
    device: &Arc<wgpu::Device>,
) -> (Option<wgpu::RenderPipeline>, Option<wgpu::Buffer>) {
    let csm_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("viewer.csm.depth.shader"),
        source: wgpu::ShaderSource::Wgsl(CSM_DEPTH_SHADER.into()),
    });

    let csm_depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.csm.depth.bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let csm_depth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.csm.depth.pl"),
        bind_group_layouts: &[&csm_depth_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewer.csm.depth.pipeline"),
        layout: Some(&csm_depth_pl),
        vertex: wgpu::VertexState {
            module: &csm_depth_shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 40,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 32, shader_location: 3 },
                ],
            }],
        },
        fragment: None,
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewer.csm.depth.camera"),
        size: std::mem::size_of::<[f32; 16]>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    (Some(pipeline), Some(camera))
}

const CSM_DEPTH_SHADER: &str = r#"
struct CsmCamera {
    light_view_proj : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uCam : CsmCamera;

struct VSIn {
    @location(0) pos : vec3<f32>,
    @location(1) nrm : vec3<f32>,
    @location(2) uv  : vec2<f32>,
    @location(3) rough_metal : vec2<f32>,
};

@vertex
fn vs_main(inp: VSIn) -> @builtin(position) vec4<f32> {
    let pos_ws = vec4<f32>(inp.pos, 1.0);
    return uCam.light_view_proj * pos_ws;
}

@fragment
fn fs_main() { }
"#;
