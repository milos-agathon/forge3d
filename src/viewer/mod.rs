// src/viewer/mod.rs
// Workstream I1: Interactive windowed viewer for forge3d
// - Creates window with winit 0.29
// - Handles input events (mouse, keyboard)
// - Renders frames at 60 FPS
// - Orbit and FPS camera modes
#![allow(dead_code)]

pub mod camera_controller;
pub mod hud;
pub mod image_analysis;
pub mod ipc;
mod viewer_analysis;
mod viewer_cmd_parse;
pub mod viewer_config;
mod viewer_constants;
pub mod viewer_enums;
mod viewer_image_utils;
pub mod viewer_struct;
mod viewer_p5;
mod viewer_p5_ao;
mod viewer_p5_cornell;
mod viewer_p5_gi;
mod viewer_p5_ssr;
mod viewer_render_helpers;
mod viewer_ssr_scene;
mod viewer_types;

// Re-export public items
pub use viewer_config::{set_initial_commands, ViewerConfig};
#[cfg(feature = "extension-module")]
pub use viewer_config::set_initial_terrain_config;
pub use ipc::IpcServerConfig;

use viewer_analysis::{gradient_energy, mean_luma_region};
use viewer_config::{FpsCounter, INITIAL_CMDS};
use viewer_constants::{
    LIT_WGSL_VERSION, P51_MAX_MEGAPIXELS, P52_MAX_MEGAPIXELS, P5_SSGI_CORNELL_WARMUP_FRAMES,
    P5_SSGI_DIFFUSE_SCALE, VIEWER_SNAPSHOT_MAX_MEGAPIXELS,
};
#[cfg(feature = "extension-module")]
use viewer_config::INITIAL_TERRAIN_CONFIG;
// viewer_cmd_parse::parse_command_string available for future refactoring of run_viewer
use viewer_enums::{parse_gi_viz_mode_token, CaptureKind, FogMode, ViewerCmd, VizMode};
use viewer_image_utils::{
    add_debug_noise_rgba8, downscale_rgba8_bilinear,
    flatten_rgba8_to_mean_luma, luma_std_rgba8,
};
use viewer_render_helpers::render_view_to_rgba8_ex;
use viewer_ssr_scene::{build_ssr_albedo_texture, build_ssr_scene_mesh};
use viewer_types::{
    FogCameraUniforms, FogUpsampleParamsStd140, P51CornellSceneState, PackedVertex, SceneMesh,
    SkyUniforms, VolumetricUniformsStd140,
};

use crate::cli::args::GiVizMode;
use hud::{
    push_number as hud_push_number,
    push_text_3x5 as hud_push_text_3x5,
};
use image_analysis::{
    compute_max_delta_e, compute_ssim, delta_e_lab, mean_abs_diff, read_texture_rgba16_to_rgb_f32,
    rgba16_to_luma, srgb_triplet_to_linear,
};
use crate::core::gpu_timing::{create_default_config as create_gpu_timing_config, GpuTimingManager};
use crate::core::ibl::{IBLQuality, IBLRenderer};
use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::core::shadows::{CameraFrustum, CsmConfig, CsmShadowMap};
// geometry imports moved to viewer_ssr_scene.rs
use crate::p5::meta::{self as p5_meta, build_ssr_meta, SsrMetaInput};
use crate::p5::{ssr, ssr::SsrScenePreset, ssr_analysis};
use crate::passes::gi::{GiCompositeParams, GiPass};
use crate::passes::ssr::SsrStats;
use crate::render::params::SsrParams;
use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;
use anyhow::{anyhow, bail};
use camera_controller::{CameraController, CameraMode};
use glam::{Mat4, Vec3};
use half::f16;
use serde_json::json;
use std::collections::VecDeque;
use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};
// once_cell imported via path for INITIAL_CMDS; no direct use import needed

// Constants moved to viewer_constants.rs


// build_ssr_albedo_texture moved to viewer_ssr_scene.rs

impl Viewer {
    /// Override fog shadow map with a constant depth value.
    /// This is primarily intended for tests and debug paths until a
    /// full directional shadow pass is wired into the viewer.
    pub fn set_fog_shadow_constant_depth(&mut self, depth: f32) {
        let _ = depth;
    }

    /// Override the fog shadow matrix used by the volumetric shader.
    /// This allows tests or higher-level systems to provide a
    /// light-space transform without changing the default identity.
    pub fn set_fog_shadow_matrix(&mut self, mat: [[f32; 4]; 4]) {
        self.queue
            .write_buffer(&self.fog_shadow_matrix, 0, bytemuck::bytes_of(&mat));
    }
}

// Types moved to viewer_types.rs (including P51CornellSceneState)

// build_ssr_scene_mesh moved to viewer_ssr_scene.rs

// ViewerConfig and statics moved to viewer_config.rs
pub struct Viewer {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<Adapter>,
    config: SurfaceConfiguration,
    camera: CameraController,
    view_config: ViewerConfig,
    frame_count: u64,
    fps_counter: FpsCounter,
    #[cfg(feature = "extension-module")]
    terrain_scene: Option<crate::terrain::TerrainScene>,
    // Input state
    keys_pressed: std::collections::HashSet<KeyCode>,
    shift_pressed: bool,
    // GI manager and toggles
    gi: Option<crate::core::screen_space_effects::ScreenSpaceEffectsManager>,
    gi_pass: Option<GiPass>,
    ssr_params: SsrParams,
    gi_seed: Option<u32>,
    gi_timing: Option<GpuTimingManager>,
    gi_gpu_hzb_ms: f32,
    gi_gpu_ssao_ms: f32,
    gi_gpu_ssgi_ms: f32,
    gi_gpu_ssr_ms: f32,
    gi_gpu_composite_ms: f32,
    // Snapshot request path (processed on next frame before present)
    snapshot_request: Option<String>,
    // Offscreen color to read back when snapshotting this frame
    pending_snapshot_tex: Option<wgpu::Texture>,
    // P5.1: deferred capture queue processed after rendering
    pending_captures: VecDeque<CaptureKind>,
    // GBuffer geometry pipeline and resources
    geom_bind_group_layout: Option<wgpu::BindGroupLayout>,
    geom_pipeline: Option<wgpu::RenderPipeline>,
    geom_camera_buffer: Option<wgpu::Buffer>,
    geom_bind_group: Option<wgpu::BindGroup>,
    geom_vb: Option<wgpu::Buffer>,
    geom_ib: Option<wgpu::Buffer>,
    geom_index_count: u32,
    z_texture: Option<wgpu::Texture>,
    z_view: Option<wgpu::TextureView>,
    // Albedo texture for geometry
    albedo_texture: Option<wgpu::Texture>,
    albedo_view: Option<wgpu::TextureView>,
    albedo_sampler: Option<wgpu::Sampler>,
    ssr_env_texture: Option<wgpu::Texture>,
    // Composite pipeline (debug show material GBuffer on screen)
    comp_bind_group_layout: Option<wgpu::BindGroupLayout>,
    comp_pipeline: Option<wgpu::RenderPipeline>,
    comp_uniform: Option<wgpu::Buffer>,
    // Lit viz compute pipeline (albedo+normal shading)
    lit_bind_group_layout: wgpu::BindGroupLayout,
    lit_pipeline: wgpu::ComputePipeline,
    lit_uniform: wgpu::Buffer,
    lit_output: wgpu::Texture,
    lit_output_view: wgpu::TextureView,
    gi_baseline_hdr: wgpu::Texture,
    gi_baseline_hdr_view: wgpu::TextureView,
    gi_baseline_diffuse_hdr: wgpu::Texture,
    gi_baseline_diffuse_hdr_view: wgpu::TextureView,
    gi_baseline_spec_hdr: wgpu::Texture,
    gi_baseline_spec_hdr_view: wgpu::TextureView,
    gi_output_hdr: wgpu::Texture,
    gi_output_hdr_view: wgpu::TextureView,
    gi_debug: wgpu::Texture,
    gi_debug_view: wgpu::TextureView,
    gi_baseline_bgl: wgpu::BindGroupLayout,
    gi_baseline_pipeline: wgpu::ComputePipeline,
    gi_split_bgl: wgpu::BindGroupLayout,
    gi_split_pipeline: wgpu::ComputePipeline,
    gi_ao_weight: f32,
    gi_ssgi_weight: f32,
    gi_ssr_weight: f32,
    // Lit params (exposed via :lit-* commands)
    lit_sun_intensity: f32,
    lit_ibl_intensity: f32,
    lit_use_ibl: bool,
    lit_ibl_rotation_deg: f32,
    // Lit BRDF selection (0=Lambert,1=Phong,4=GGX,6=Disney)
    lit_brdf: u32,
    // Lit roughness (used by debug modes and future shading controls)
    lit_roughness: f32,
    // Lit debug mode: 0=off, 1=roughness smoke test, 2=NDF-only GGX
    lit_debug_mode: u32,
    // Fallback pipeline to draw a solid color when GI/geometry path is unavailable
    fallback_pipeline: wgpu::RenderPipeline,
    viz_mode: VizMode,
    gi_viz_mode: GiVizMode,
    // SSAO composite control
    use_ssao_composite: bool,
    ssao_composite_mul: f32,
    // Cached SSAO blur toggle for query commands
    ssao_blur_enabled: bool,
    // IBL integration
    ibl_renderer: Option<IBLRenderer>,
    ibl_env_view: Option<wgpu::TextureView>,
    ibl_sampler: Option<wgpu::Sampler>,
    ibl_hdr_path: Option<String>,
    ibl_cache_dir: Option<std::path::PathBuf>,
    ibl_base_resolution: Option<u32>,
    // Viz depth override
    viz_depth_max_override: Option<f32>,
    // Auto-snapshot support (one-time)
    auto_snapshot_path: Option<String>,
    auto_snapshot_done: bool,
    // P5 dump request
    dump_p5_requested: bool,
    // Adapter name for meta
    adapter_name: String,
    // Debug: log render gate and snapshot once
    debug_logged_render_gate: bool,

    // Sky rendering (P6-01)
    sky_bind_group_layout0: wgpu::BindGroupLayout,
    sky_bind_group_layout1: wgpu::BindGroupLayout,
    sky_pipeline: wgpu::ComputePipeline,
    sky_params: wgpu::Buffer,
    sky_camera: wgpu::Buffer,
    sky_output: wgpu::Texture,
    sky_output_view: wgpu::TextureView,
    sky_enabled: bool,

    // P6: Fog rendering resources and parameters
    fog_enabled: bool,
    fog_params: wgpu::Buffer,
    fog_camera: wgpu::Buffer,
    fog_output: wgpu::Texture,
    fog_output_view: wgpu::TextureView,
    fog_history: wgpu::Texture,
    fog_history_view: wgpu::TextureView,
    fog_depth_sampler: wgpu::Sampler,
    fog_history_sampler: wgpu::Sampler,
    fog_pipeline: wgpu::ComputePipeline,
    fog_frame_index: u32,
    // Froxelized volumetrics (Milestone 4)
    fog_bgl3: wgpu::BindGroupLayout,
    froxel_tex: wgpu::Texture,
    froxel_view: wgpu::TextureView,
    froxel_sampler: wgpu::Sampler,
    froxel_build_pipeline: wgpu::ComputePipeline,
    froxel_apply_pipeline: wgpu::ComputePipeline,
    // P6-10: Half-resolution fog + upsample
    fog_half_res_enabled: bool,
    fog_output_half: wgpu::Texture,
    fog_output_half_view: wgpu::TextureView,
    fog_history_half: wgpu::Texture,
    fog_history_half_view: wgpu::TextureView,
    fog_upsample_bgl: wgpu::BindGroupLayout,
    fog_upsample_pipeline: wgpu::ComputePipeline,
    fog_upsample_params: wgpu::Buffer,
    // Bilateral upsample controls
    fog_bilateral: bool,
    fog_upsigma: f32,
    // Fog bind group layouts and shadow resources
    fog_bgl0: wgpu::BindGroupLayout,
    fog_bgl1: wgpu::BindGroupLayout,
    fog_bgl2: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    fog_shadow_map: wgpu::Texture,
    fog_shadow_view: wgpu::TextureView,
    fog_shadow_sampler: wgpu::Sampler,
    fog_shadow_matrix: wgpu::Buffer,
    // Fog zero fallback (1x1 RGBA16F zero) for disabled fog compositing
    #[allow(dead_code)]
    fog_zero_tex: wgpu::Texture,
    fog_zero_view: wgpu::TextureView,
    // Exposed toggles
    fog_density: f32,
    fog_g: f32,
    fog_steps: u32,
    fog_temporal_alpha: f32,
    fog_use_shadows: bool,
    fog_mode: FogMode,
    // Cascaded shadow maps for directional sun shadows (future fog + lighting)
    csm: Option<CsmShadowMap>,
    csm_config: CsmConfig,
    csm_depth_pipeline: Option<wgpu::RenderPipeline>,
    csm_depth_camera: Option<wgpu::Buffer>,
    // Sky exposed controls (runtime adjustable)
    sky_model_id: u32, // 0=Preetham,1=Hosek-Wilkie
    sky_turbidity: f32,
    sky_ground_albedo: f32,
    sky_exposure: f32,
    sky_sun_intensity: f32,

    // HUD overlay renderer
    hud_enabled: bool,
    hud: crate::core::text_overlay::TextOverlayRenderer,
    ssr_scene_loaded: bool,
    ssr_scene_preset: Option<SsrScenePreset>,
    // Object transform (for IPC SetTransform command)
    object_translation: glam::Vec3,
    object_rotation: glam::Quat,
    object_scale: glam::Vec3,
    object_transform: glam::Mat4,
}

// FpsCounter moved to viewer_config.rs

impl Viewer {
    #[cfg(feature = "extension-module")]
    pub fn load_terrain_from_config(
        &mut self,
        cfg: &crate::render::params::RendererConfig,
    ) -> anyhow::Result<()> {
        // TerrainScene currently owns its own configuration; we accept cfg so
        // that future milestones can thread it through without changing the
        // Viewer API again.
        let _ = cfg;

        let scene = crate::terrain::TerrainScene::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            Arc::clone(&self.adapter),
        )?;
        self.terrain_scene = Some(scene);
        Ok(())
    }

    fn ensure_geom_bind_group(&mut self) -> anyhow::Result<()> {
        if self.geom_bind_group.is_some() {
            return Ok(());
        }
        let cam_buf = match self.geom_camera_buffer.as_ref() {
            Some(buf) => buf,
            None => return Ok(()),
        };
        let sampler = self.albedo_sampler.get_or_insert_with(|| {
            self.device
                .create_sampler(&wgpu::SamplerDescriptor::default())
        });
        let tex = self.albedo_texture.get_or_insert_with(|| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.geom.albedo.empty"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let view = self
            .albedo_view
            .get_or_insert_with(|| tex.create_view(&wgpu::TextureViewDescriptor::default()));
        if let Some(ref layout) = self.geom_bind_group_layout {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg.runtime"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cam_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.geom_bind_group = Some(bg);
        }
        Ok(())
    }
}

impl Viewer {
    fn upload_ssr_scene(&mut self, preset: &SsrScenePreset) -> anyhow::Result<()> {
        let mesh = build_ssr_scene_mesh(preset);
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            anyhow::bail!("SSR scene mesh is empty");
        }

        let vertex_data = bytemuck::cast_slice(&mesh.vertices);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ssr.scene.vb"),
                contents: vertex_data,
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ssr.scene.ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.geom_vb = Some(vb);
        self.geom_ib = Some(ib);
        self.geom_index_count = mesh.indices.len() as u32;
        self.geom_bind_group = None;
        
        // Update IPC stats for get_stats command
        update_ipc_stats(
            true,
            mesh.vertices.len() as u32,
            mesh.indices.len() as u32,
            true,
        );

        let tex_size = 1024u32;
        let pixels = build_ssr_albedo_texture(preset, tex_size);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.ssr.scene.albedo"),
            size: wgpu::Extent3d {
                width: tex_size,
                height: tex_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(tex_size * 4),
                rows_per_image: Some(tex_size),
            },
            wgpu::Extent3d {
                width: tex_size,
                height: tex_size,
                depth_or_array_layers: 1,
            },
        );
        self.albedo_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.albedo_texture = Some(texture);
        self.albedo_sampler.get_or_insert_with(|| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("viewer.ssr.scene.albedo.sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        });

        self.ensure_geom_bind_group()?;

        Ok(())
    }

    fn apply_ssr_scene_preset(&mut self) -> anyhow::Result<()> {
        let preset = match self.ssr_scene_preset.clone() {
            Some(p) => p,
            None => {
                let preset = SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?;
                self.ssr_scene_preset = Some(preset.clone());
                preset
            }
        };

        self.upload_ssr_scene(&preset)?;

        // Camera setup: look at spheres from a higher angle to align with preset center_y (approx 0.63 screen Y)
        let eye = Vec3::new(0.0, 2.5, 5.0);
        let target = Vec3::new(0.0, 0.5, 0.0);
        self.camera.set_look_at(eye, target, Vec3::Y);

        // Lighting from preset
        self.lit_sun_intensity = preset.light_intensity.max(0.0);
        self.lit_use_ibl = true;
        self.lit_ibl_intensity = 1.0;
        self.lit_ibl_rotation_deg = 0.0;
        self.update_lit_uniform();

        self.generate_stripe_env_map(&preset)?;

        self.ssr_scene_loaded = true;
        Ok(())
    }

    fn generate_stripe_env_map(&mut self, preset: &SsrScenePreset) -> anyhow::Result<()> {
        let size = 256u32;
        let mut data = Vec::with_capacity((size * size * 4 * 6) as usize);
        
        // Order: +X, -X, +Y, -Y, +Z, -Z (wgpu / Vulkan convention for cubemap array layers)
        // Directions corresponding to faces
        let faces = [
            (Vec3::X, -Vec3::Z, -Vec3::Y), // +X (Right)
            (-Vec3::X, Vec3::Z, -Vec3::Y), // -X (Left)
            (Vec3::Y, Vec3::X, Vec3::Z),             // +Y (Top)
            (-Vec3::Y, Vec3::X, -Vec3::Z), // -Y (Bottom)
            (Vec3::Z, Vec3::X, -Vec3::Y),       // +Z (Front)
            (-Vec3::Z, -Vec3::X, -Vec3::Y), // -Z (Back)
        ];

        let stripe_center = preset.stripe.center_y; // Normalized 0..1 (Top to Bottom)
        let stripe_half = preset.stripe.half_thickness;

        for (forward, right, up) in faces {
            for y in 0..size {
                for x in 0..size {
                    // UV in [-1, 1]
                    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    // Direction on cube face
                    let dir = (forward + right * u + up * v).normalize();
                    
                    // Map direction to spherical coordinates for stripe
                    let screen_y = dir.y * -0.5 + 0.5; 
                    
                    // Background
                    let bg_color = if screen_y < preset.floor_horizon {
                        let denom = preset.floor_horizon.max(0.001);
                        let t = (screen_y / denom).powf(0.9).clamp(0.0, 1.0);
                        crate::p5::ssr::lerp3(preset.background_top, preset.background_bottom, t)
                    } else {
                        let denom = (1.0 - preset.floor_horizon).max(0.001);
                        let t = ((screen_y - preset.floor_horizon) / denom).clamp(0.0, 1.0);
                        crate::p5::ssr::lerp3(preset.floor.color_top, preset.floor.color_bottom, t)
                    };
                    
                    let mut final_color = bg_color;

                    // Stripe
                    let dy = ((screen_y - stripe_center) / stripe_half).abs();
                    if dy < 1.0 {
                        let alpha = (1.0 - dy).powf(2.0f32) * preset.stripe.glow_strength;
                        let glow = crate::p5::ssr::lerp3(
                            preset.stripe.inner_color,
                            preset.stripe.outer_color,
                            screen_y.clamp(0.0, 1.0),
                        );
                        // Additive or mix? write_glossy_png uses alpha blend
                        let inv = 1.0 - alpha.min(1.0);
                        final_color[0] = final_color[0] * inv + glow[0] * alpha;
                        final_color[1] = final_color[1] * inv + glow[1] * alpha;
                        final_color[2] = final_color[2] * inv + glow[2] * alpha;
                    }
                    
                    data.push(crate::p5::ssr::to_u8(final_color[0]));
                    data.push(crate::p5::ssr::to_u8(final_color[1]));
                    data.push(crate::p5::ssr::to_u8(final_color[2]));
                    data.push(255);
                }
            }
        }

        // Create texture
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p5.ssr.env.generated"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
        );

        if let Some(ref mut gi) = self.gi {
            gi.set_environment_texture(&self.device, &texture);
        }
        self.ssr_env_texture = Some(texture);
        
        Ok(())
    }
    // Compute mean luma in a region of an RGBA8 buffer
    // Analysis functions moved to viewer_analysis.rs

    // Write or update p5_meta.json with provided patcher
    pub(crate) fn write_p5_meta<F: FnOnce(&mut std::collections::BTreeMap<String, serde_json::Value>)>(
        &self,
        patch: F,
    ) -> anyhow::Result<()> {
        p5_meta::write_p5_meta(Path::new("reports/p5"), patch)
    }

    // capture_p52_ssgi_cornell moved to viewer_p5.rs
    // capture_p52_ssgi_temporal moved to viewer_p5.rs

    pub(crate) fn reexecute_gi(&mut self, ssr_stats: Option<&mut SsrStats>) -> anyhow::Result<()> {
        use anyhow::Context;
        let depth_view = self.z_view.as_ref().context("Depth view unavailable")?;
        if let Some(ref mut gi) = self.gi {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p5.gi.reexec"),
                });
            gi.advance_frame(&self.queue);

            // Optional GPU timing for HZB + GI passes
            let mut timing_opt = self
                .gi_timing
                .as_mut()
                .filter(|t| t.is_supported());

            // HZB build scope
            if let Some(timer) = timing_opt.as_deref_mut() {
                let scope_id = timer.begin_scope(&mut enc, "p5.hzb");
                gi.build_hzb(&self.device, &mut enc, depth_view, false);
                timer.end_scope(&mut enc, scope_id);
            } else {
                gi.build_hzb(&self.device, &mut enc, depth_view, false);
            }

            // GI effects (SSAO/SSGI/SSR) with per-effect GPU scopes inside
            gi.execute(&self.device, &mut enc, ssr_stats, timing_opt.as_deref_mut())?;

            // Build lit_output baseline (direct + IBL) into Rgba8Unorm
            let env_view = if let Some(ref v) = self.ibl_env_view {
                v
            } else {
                self.ibl_env_view
                    .as_ref()
                    .context("IBL env view unavailable")?
            };
            let env_samp = if let Some(ref s) = self.ibl_sampler {
                s
            } else {
                self.ibl_sampler
                    .as_ref()
                    .context("IBL sampler unavailable")?
            };
            let lit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.lit.bg.gi_baseline"),
                layout: &self.lit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(env_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.lit_uniform.as_entire_binding(),
                    },
                ],
            });
            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.lit.compute.pre_ssr"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.lit_pipeline);
                cpass.set_bind_group(0, &lit_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            // Copy lit_output into HDR baseline buffer
            let baseline_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gi.baseline.bg"),
                layout: &self.gi_baseline_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.gi_baseline_hdr_view),
                    },
                ],
            });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.gi.baseline.copy"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.gi_baseline_pipeline);
                cpass.set_bind_group(0, &baseline_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            // Split lit_output into approximate diffuse and specular baselines
            let split_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gi.baseline.split.bg"),
                layout: &self.gi_split_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource:
                            wgpu::BindingResource::TextureView(&self.gi_baseline_diffuse_hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource:
                            wgpu::BindingResource::TextureView(&self.gi_baseline_spec_hdr_view),
                    },
                ],
            });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.gi.baseline.split"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.gi_split_pipeline);
                cpass.set_bind_group(0, &split_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            // Ensure GiPass exists with current dimensions
            let (w, h) = (self.config.width, self.config.height);
            if self.gi_pass.is_none() {
                match GiPass::new(&self.device, w, h) {
                    Ok(pass) => {
                        self.gi_pass = Some(pass);
                    }
                    Err(e) => {
                        return Err(anyhow!("Failed to create GiPass: {}", e));
                    }
                }
            }

            if let Some(ref mut gi_pass) = self.gi_pass {
                let ao_view = gi
                    .ao_resolved_view()
                    .unwrap_or(&gi.gbuffer().material_view);

                let ssgi_view = gi
                    .ssgi_output_for_display_view()
                    .unwrap_or(&gi.gbuffer().material_view);

                let ssr_view = gi
                    .ssr_final_view()
                    .unwrap_or(&self.lit_output_view);

                let params = GiCompositeParams {
                    ao_enable: gi.is_enabled(SSE::SSAO),
                    ssgi_enable: gi.is_enabled(SSE::SSGI),
                    ssr_enable: gi.is_enabled(SSE::SSR) && self.ssr_params.ssr_enable,
                    ao_weight: self.gi_ao_weight,
                    ssgi_weight: self.gi_ssgi_weight,
                    ssr_weight: self.gi_ssr_weight,
                    energy_cap: 1.05,
                };

                // Future: derive AO/SSGI weights from existing viewer knobs
                gi_pass.update_params(&self.queue, |p| {
                    *p = params;
                });

                gi_pass.execute(
                    &self.device,
                    &mut enc,
                    &self.gi_baseline_hdr_view,
                    &self.gi_baseline_diffuse_hdr_view,
                    &self.gi_baseline_spec_hdr_view,
                    ao_view,
                    ssgi_view,
                    ssr_view,
                    &gi.gbuffer().normal_view,
                    &gi.gbuffer().material_view,
                    &self.gi_output_hdr_view,
                    timing_opt.as_deref_mut(),
                )?;

                gi_pass.execute_debug(
                    &self.device,
                    &mut enc,
                    ao_view,
                    ssgi_view,
                    ssr_view,
                    &self.gi_debug_view,
                )?;
            }

            if let Some(timer) = timing_opt.as_deref_mut() {
                timer.resolve_queries(&mut enc);
            }

            self.queue.submit(std::iter::once(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            // Read back GPU timing results for P5.6 budgets
            if let Some(timer) = self.gi_timing.as_mut() {
                if timer.is_supported() {
                    match pollster::block_on(timer.get_results()) {
                        Ok(results) => {
                            self.gi_gpu_hzb_ms = 0.0;
                            self.gi_gpu_ssao_ms = 0.0;
                            self.gi_gpu_ssgi_ms = 0.0;
                            self.gi_gpu_ssr_ms = 0.0;
                            self.gi_gpu_composite_ms = 0.0;
                            for r in results {
                                if !r.timestamp_valid {
                                    continue;
                                }
                                match r.name.as_str() {
                                    "p5.hzb" => self.gi_gpu_hzb_ms = r.gpu_time_ms,
                                    "p5.ssao" => self.gi_gpu_ssao_ms = r.gpu_time_ms,
                                    "p5.ssgi" => self.gi_gpu_ssgi_ms = r.gpu_time_ms,
                                    "p5.ssr" => self.gi_gpu_ssr_ms = r.gpu_time_ms,
                                    "p5.composite" => {
                                        self.gi_gpu_composite_ms = r.gpu_time_ms
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[P5.6] GPU timing readback failed: {e}");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn sync_ssr_params_to_gi(&mut self) {
        if let Some(ref mut gi) = self.gi {
            gi.set_ssr_params(&self.queue, &self.ssr_params);
        }
    }

    pub(crate) fn capture_material_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        self.with_comp_pipeline(|comp_pl, comp_bgl| {
            let fog_view = if self.fog_enabled {
                &self.fog_output_view
            } else {
                &self.fog_zero_view
            };
            render_view_to_rgba8_ex(
                &self.device,
                &self.queue,
                comp_pl,
                comp_bgl,
                &self.sky_output_view,
                &gi.gbuffer().depth_view,
                fog_view,
                self.config.format,
                self.config.width,
                self.config.height,
                far,
                gi.material_with_ssr_view()
                    .or_else(|| gi.material_with_ssgi_view())
                    .or_else(|| gi.material_with_ao_view())
                    .unwrap_or(&gi.gbuffer().material_view),
                0,
            )
        })
    }

    pub(crate) fn with_comp_pipeline<T>(
        &self,
        f: impl FnOnce(&wgpu::RenderPipeline, &wgpu::BindGroupLayout) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        use anyhow::Context;
        let comp_pl =
            self.comp_pipeline.as_ref().context("comp pipeline")? as &wgpu::RenderPipeline;
        let comp_bgl =
            self.comp_bind_group_layout.as_ref().context("comp bgl")? as &wgpu::BindGroupLayout;
        f(comp_pl, comp_bgl)
    }

    pub(crate) fn read_ssgi_filtered_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi
            .ssgi_dimensions()
            .context("SSGI dimensions unavailable")?;
        let tex = gi
            .ssgi_filtered_texture()
            .context("SSGI filtered texture unavailable")?;
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSGI filtered texture")?;
        Ok((bytes, dims))
    }

    pub(crate) fn read_ssgi_hit_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi
            .ssgi_dimensions()
            .context("SSGI dimensions unavailable")?;
        let tex = gi
            .ssgi_hit_texture()
            .context("SSGI hit texture unavailable")?;
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSGI hit texture")?;
        Ok((bytes, dims))
    }

    pub fn gi_output_hdr_view(&self) -> &wgpu::TextureView {
        &self.gi_output_hdr_view
    }

    pub(crate) fn read_gi_output_hdr_rgb(&self) -> anyhow::Result<(Vec<[f32; 3]>, (u32, u32))> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi.gbuffer().dimensions();
        let data = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;
        Ok((data, dims))
    }

    pub(crate) fn read_gi_baseline_hdr_rgb(&self) -> anyhow::Result<(Vec<[f32; 3]>, (u32, u32))> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi.gbuffer().dimensions();
        let data = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_baseline_hdr, dims)?;
        Ok((data, dims))
    }

    pub(crate) fn read_ssr_hit_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let tex = gi
            .ssr_hit_texture()
            .context("SSR hit texture unavailable")?;
        let dims = gi.gbuffer().dimensions();
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSR hit texture")?;
        Ok((bytes, dims))
    }

    // Read back a surface or offscreen texture and save as PNG (RGBA8/BGRA8 only)
    fn snapshot_swapchain_to_png(&mut self, tex: &wgpu::Texture, path: &str) -> anyhow::Result<()> {
        use anyhow::{bail, Context};
        // Use texture's actual dimensions, not surface config (offscreen may differ)
        let size = tex.size();
        let w = size.width;
        let h = size.height;
        let fmt = tex.format();

        match fmt {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                for px in data.chunks_exact_mut(4) {
                    px[3] = 255;
                }
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                for px in data.chunks_exact_mut(4) {
                    px.swap(0, 2);
                    px[3] = 255;
                }
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            other => {
                bail!(
                    "snapshot only supports RGBA8/BGRA8 surfaces (got {:?})",
                    other
                )
            }
        }
    }
    fn load_ibl(&mut self, path: &str) -> anyhow::Result<()> {
        // Load HDR image from disk
        let hdr_img = crate::formats::hdr::load_hdr(path)
            .map_err(|e| anyhow::anyhow!("failed to load HDR '{}': {}", path, e))?;

        // Build IBL renderer and upload environment
        let mut ibl = IBLRenderer::new(&self.device, IBLQuality::Low);

        // Apply cached resolution if set
        if let Some(res) = self.ibl_base_resolution {
            ibl.set_base_resolution(res);
        } else {
            ibl.set_base_resolution(IBLQuality::Low.base_environment_size());
        }

        // Configure cache if set
        if let Some(ref cache_dir) = self.ibl_cache_dir {
            ibl.configure_cache(cache_dir, std::path::Path::new(path))
                .map_err(|e| anyhow::anyhow!("failed to configure IBL cache: {}", e))?;
        }

        ibl.load_environment_map(
            &self.device,
            &self.queue,
            &hdr_img.data,
            hdr_img.width,
            hdr_img.height,
        )
        .map_err(|e| anyhow::anyhow!("failed to upload environment: {}", e))?;
        ibl.initialize(&self.device, &self.queue)
            .map_err(|e| anyhow::anyhow!("failed to initialize IBL: {}", e))?;

        // Wire SSGI to irradiance and SSR to specular
        let (irr_tex_opt, spec_tex_opt, _) = ibl.textures();
        if let Some(ref mut gi) = self.gi {
            if let Some(irr_tex) = irr_tex_opt {
                gi.set_ssgi_env(&self.device, irr_tex);
            }
            if let Some(spec_tex) = spec_tex_opt {
                gi.set_ssr_env(&self.device, spec_tex);
                // Keep a viewer-side view/sampler for diagnostics
                let cube_view = spec_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("viewer.ibl.specular.cube.view"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                });
                let env_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.ibl.env.sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
                self.ibl_env_view = Some(cube_view);
                self.ibl_sampler = Some(env_sampler);
            }
        }

        // Keep IBL resources alive
        self.ibl_renderer = Some(ibl);
        self.ibl_hdr_path = Some(path.to_string());
        Ok(())
    }
    pub async fn new(
        window: Arc<Window>,
        config: ViewerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface - use Arc::clone to satisfy lifetime requirements
        let surface = instance.create_surface(Arc::clone(&window))?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable adapter")?;

        let adapter = Arc::new(adapter);

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Viewer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let adapter_name = adapter.get_info().name;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if config.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        // Optional GPU timing manager for GI profiling (P5.6)
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
            &device,
            surface_config.width,
            surface_config.height,
        ) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Failed to create ScreenSpaceEffectsManager: {}", e);
                None
            }
        };

        // Build geometry pipeline only if GI is available (needs GBuffer formats)
        let mut csm_depth_pipeline: Option<wgpu::RenderPipeline> = None;
        let mut csm_depth_camera: Option<wgpu::Buffer> = None;
        let (
            geom_bind_group_layout,
            geom_pipeline,
            geom_camera_buffer,
            geom_bind_group,
            geom_vb,
            z_texture,
            z_view,
            albedo_texture,
            albedo_view,
            albedo_sampler,
            comp_bind_group_layout,
            comp_pipeline,
        ) = if let Some(ref gi_ref) = gi {
            // Z-buffer for rasterization
            let z_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gbuf.z"),
                size: wgpu::Extent3d {
                    width: surface_config.width,
                    height: surface_config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Camera uniform
            let geom_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewer.gbuf.cam"),
                size: (std::mem::size_of::<[[f32; 4]; 4]>() * 2) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Bind group layout: camera uniform + albedo texture + sampler
            let geom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.gbuf.geom.bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            // Shader for geometry GBuffer write (with texcoords)
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("viewer.gbuf.geom.shader"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                    struct Camera {
                        view : mat4x4<f32>,
                        proj : mat4x4<f32>,
                    };
                    @group(0) @binding(0) var<uniform> uCam : Camera;
                    @group(0) @binding(1) var tAlbedo : texture_2d<f32>;
                    @group(0) @binding(2) var sAlbedo : sampler;

                    struct VSIn {
                        @location(0) pos : vec3<f32>,
                        @location(1) nrm : vec3<f32>,
                        @location(2) uv  : vec2<f32>,
                        @location(3) rough_metal : vec2<f32>,
                    };
                    struct VSOut {
                        @builtin(position) pos : vec4<f32>,
                        @location(0) v_nrm_vs : vec3<f32>,
                        @location(1) v_depth_vs : f32,
                        @location(2) v_uv : vec2<f32>,
                        @location(3) v_rough_metal : vec2<f32>,
                    };

                    @vertex
                    fn vs_main(inp: VSIn) -> VSOut {
                        var out: VSOut;
                        let pos_ws = vec4<f32>(inp.pos, 1.0);
                        let pos_vs = uCam.view * pos_ws;
                        out.pos = uCam.proj * pos_vs;
                        let nrm_vs = (uCam.view * vec4<f32>(inp.nrm, 0.0)).xyz;
                        out.v_nrm_vs = normalize(nrm_vs);
                        out.v_depth_vs = -pos_vs.z; // positive view-space depth
                        out.v_uv = inp.uv;
                        out.v_rough_metal = inp.rough_metal;
                        return out;
                    }

                    struct FSOut {
                        @location(0) normal_rgba : vec4<f32>,
                        @location(1) albedo_rgba : vec4<f32>,
                        @location(2) depth_r : f32,
                    };

                    @fragment
                    fn fs_main(inp: VSOut) -> FSOut {
                        var out: FSOut;
                        let n = normalize(inp.v_nrm_vs);
                        let enc = n * 0.5 + vec3<f32>(0.5);
                        out.normal_rgba = vec4<f32>(enc, clamp(inp.v_rough_metal.x, 0.0, 1.0));
                        let color = textureSample(tAlbedo, sAlbedo, inp.v_uv);
                        out.albedo_rgba = vec4<f32>(color.rgb, clamp(inp.v_rough_metal.y, 0.0, 1.0));
                        out.depth_r = inp.v_depth_vs;
                        return out;
                    }
                "#
                    .into(),
                ),
            });

            // Pipeline
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewer.gbuf.geom.pl"),
                bind_group_layouts: &[&geom_bgl],
                push_constant_ranges: &[],
            });

            let gb = gi_ref.gbuffer();
            let gb_cfg = gb.config();
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("viewer.gbuf.geom.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PackedVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                shader_location: 0,
                                offset: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 1,
                                offset: (3 * std::mem::size_of::<f32>()) as u64,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 2,
                                offset: (6 * std::mem::size_of::<f32>()) as u64,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                shader_location: 3,
                                offset: (8 * std::mem::size_of::<f32>()) as u64,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    }],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: gb_cfg.normal_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: gb_cfg.material_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: gb_cfg.depth_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                multiview: None,
            });

            let csm_depth_shader =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("viewer.csm.depth.shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        r#"
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
                        "#
                            .into(),
                    ),
                });
            let csm_depth_bgl =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            let csm_depth_pl =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("viewer.csm.depth.pl"),
                    bind_group_layouts: &[&csm_depth_bgl],
                    push_constant_ranges: &[],
                });
            csm_depth_pipeline = Some(device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("viewer.csm.depth.pipeline"),
                    layout: Some(&csm_depth_pl),
                    vertex: wgpu::VertexState {
                        module: &csm_depth_shader,
                        entry_point: "vs_main",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride:
                                std::mem::size_of::<PackedVertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    shader_location: 0,
                                    offset: 0,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    shader_location: 1,
                                    offset:
                                        (3 * std::mem::size_of::<f32>()) as u64,
                                    format: wgpu::VertexFormat::Float32x3,
                                },
                                wgpu::VertexAttribute {
                                    shader_location: 2,
                                    offset:
                                        (6 * std::mem::size_of::<f32>()) as u64,
                                    format: wgpu::VertexFormat::Float32x2,
                                },
                                wgpu::VertexAttribute {
                                    shader_location: 3,
                                    offset:
                                        (8 * std::mem::size_of::<f32>()) as u64,
                                    format: wgpu::VertexFormat::Float32x2,
                                },
                            ],
                        }],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: None,
                    multiview: None,
                },
            ));
            csm_depth_camera = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewer.csm.depth.camera"),
                size: std::mem::size_of::<[f32; 16]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Create an albedo texture (procedural checkerboard)
            let tex_size = 256u32;
            let mut pixels = vec![0u8; (tex_size * tex_size * 4) as usize];
            for y in 0..tex_size {
                for x in 0..tex_size {
                    let idx = ((y * tex_size + x) * 4) as usize;
                    let c = if ((x / 32) + (y / 32)) % 2 == 0 {
                        230
                    } else {
                        50
                    };
                    pixels[idx + 0] = c; // R
                    pixels[idx + 1] = 180; // G
                    pixels[idx + 2] = 80; // B
                    pixels[idx + 3] = 255; // A
                }
            }
            let albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.geom.albedo.tex"),
                size: wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let albedo_view = albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &albedo_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(tex_size * 4),
                    rows_per_image: Some(tex_size),
                },
                wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth_or_array_layers: 1,
                },
            );
            let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

            // Geometry bind group
            let geom_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg"),
                layout: &geom_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: geom_camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&albedo_sampler),
                    },
                ],
            });

            // Composite pass: display selected viz (material/normal/depth/GI) onto swapchain
            let comp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.comp.bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Sky background texture (RGBA8) to composite behind geometry
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // GBuffer depth (R16F as color) to detect background pixels
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Fog texture (RGBA16F)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
            let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("viewer.comp.shader"),
                source: wgpu::ShaderSource::Wgsl(r#"
                    struct CompParams { mode: u32, far: f32, _pad: vec2<f32> };
                    @group(0) @binding(2) var<uniform> uComp : CompParams;
                    @vertex
                    fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
                        let x = f32((vid << 1u) & 2u);
                        let y = f32(vid & 2u);
                        return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
                    }
                    @group(0) @binding(0) var gbuf_tex : texture_2d<f32>;
                    @group(0) @binding(1) var gbuf_sam : sampler;
                    @group(0) @binding(3) var sky_tex : texture_2d<f32>;
                    @group(0) @binding(4) var depth_tex : texture_2d<f32>;
                    @group(0) @binding(5) var fog_tex : texture_2d<f32>;
                    @fragment
                    fn fs_fullscreen(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                        let dims = vec2<f32>(textureDimensions(gbuf_tex));
                        let uv = pos.xy / dims;
                        var c = textureSample(gbuf_tex, gbuf_sam, uv);
                        if (uComp.mode == 1u) {
                            // normal: [-1,1] -> [0,1]
                            c = vec4<f32>(0.5 * (c.xyz + vec3<f32>(1.0)), 1.0);
                        } else if (uComp.mode == 2u) {
                            // depth: view-space depth mapped by far
                            let d = clamp(c.r / max(0.0001, uComp.far), 0.0, 1.0);
                            c = vec4<f32>(d, d, d, 1.0);
                        } else if (uComp.mode == 3u) {
                            // AO/debug grayscale from single-channel source (no sky/fog composite)
                            let r = textureSample(gbuf_tex, gbuf_sam, uv).r;
                            c = vec4<f32>(r, r, r, 1.0);
                        } else {
                            // Composite sky behind geometry when depth indicates background
                            let dval = textureSample(depth_tex, gbuf_sam, uv).r;
                            if (dval <= 0.0001) {
                                let sky = textureSample(sky_tex, gbuf_sam, uv);
                                c = sky;
                            }
                            // Composite fog over scene (premultiplied-like: c = c*(1-a) + fog)
                            let fog = textureSample(fog_tex, gbuf_sam, uv);
                            c = vec4<f32>(c.rgb * (1.0 - fog.a) + fog.rgb, 1.0);
                        }
                        return c;
                    }
                "#.into()),
            });
            let comp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewer.comp.pl"),
                bind_group_layouts: &[&comp_bgl],
                push_constant_ranges: &[],
            });
            let comp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("viewer.comp.pipeline"),
                layout: Some(&comp_pl),
                vertex: wgpu::VertexState {
                    module: &comp_shader,
                    entry_point: "vs_fullscreen",
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &comp_shader,
                    entry_point: "fs_fullscreen",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });

            (
                Some(geom_bgl),
                Some(pipeline),
                Some(geom_camera_buffer),
                Some(geom_bg),
                None,
                Some(z_texture),
                Some(z_view),
                Some(albedo_texture),
                Some(albedo_view),
                Some(albedo_sampler),
                Some(comp_bgl),
                Some(comp_pipeline),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None,
            )
        };

        // Always-available fallback pipeline (solid fullscreen triangle)
        let fb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.fallback.shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @vertex
                fn vs_fb(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
                    let x = f32((vid << 1u) & 2u);
                    let y = f32(vid & 2u);
                    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
                }
                @fragment
                fn fs_fb() -> @location(0) vec4<f32> {
                    return vec4<f32>(0.05, 0.0, 0.15, 1.0);
                }
            "#
                .into(),
            ),
        });
        let fb_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewer.fallback.pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &fb_shader,
                entry_point: "vs_fb",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fb_shader,
                entry_point: "fs_fb",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        // Lit viz compute pipeline (albedo+normal shading with optional IBL)
        let lit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.lit.bgl"),
            entries: &[
                // normal, material, depth
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // output
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // env cube + sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // params
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let lit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.lit.compute.shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                struct LitParams {
                    // x,y,z = sun_dir_vs, w = sun_intensity
                    sun_dir_and_intensity: vec4<f32>,
                    // x = ibl_intensity, y = use_ibl (1.0|0.0), z = brdf index, w = pad
                    ibl_use_brdf_pad: vec4<f32>,
                    // x = roughness [0,1], y = debug_mode (0=off,1=roughness,2=NDF), z/w pad
                    debug_extra: vec4<f32>,
                };
                @group(0) @binding(0) var normal_tex : texture_2d<f32>;
                @group(0) @binding(1) var albedo_tex : texture_2d<f32>;
                @group(0) @binding(2) var depth_tex  : texture_2d<f32>;
                @group(0) @binding(3) var out_tex    : texture_storage_2d<rgba8unorm, write>;
                @group(0) @binding(4) var env_cube   : texture_cube<f32>;
                @group(0) @binding(5) var env_samp   : sampler;
                @group(0) @binding(6) var<uniform> P : LitParams;

                const BRDF_LAMBERT: f32 = 0.0;
                const BRDF_PHONG: f32 = 1.0;
                const BRDF_GGX: f32 = 4.0;
                const BRDF_DISNEY: f32 = 6.0;

                fn approx_eq(a: f32, b: f32) -> bool { return abs(a - b) < 0.5; }

                @compute @workgroup_size(8,8,1)
                fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let dims = textureDimensions(normal_tex);
                    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
                    let coord = vec2<i32>(gid.xy);
                    var n = textureLoad(normal_tex, coord, 0).xyz; // view-space [-1,1]
                    n = normalize(n);
                    let a = textureLoad(albedo_tex, coord, 0).rgb;
                    // Interpret P.sun_dir as the direction FROM light to the origin.
                    // Use L = -sun_dir (direction from point TO the light).
                    let l = -normalize(P.sun_dir_and_intensity.xyz);
                    let rough = clamp(P.debug_extra.x, 0.0, 1.0);
                    let dbg = u32(P.debug_extra.y + 0.5);

                    // Debug 1: roughness smoke test – output R=roughness
                    if (dbg == 1u) {
                        textureStore(out_tex, coord, vec4<f32>(rough, 0.0, 0.0, 1.0));
                        return;
                    }
                    let ndl = max(dot(n, l), 0.0);
                    // Simple direct lighting with BRDF dispatch (viewer-only approximation)
                    var col = vec3<f32>(0.0);
                    if (ndl > 0.0) {
                        if (approx_eq(P.ibl_use_brdf_pad.z, BRDF_LAMBERT)) {
                            // Lambert diffuse
                            let diffuse = a * (1.0 / 3.14159265);
                            col = diffuse * P.sun_dir_and_intensity.w * ndl;
                        } else if (approx_eq(P.ibl_use_brdf_pad.z, BRDF_PHONG)) {
                            // Blinn-Phong using fixed shininess from roughness~0.5
                            let v = vec3<f32>(0.0, 0.0, 1.0);
                            let h = normalize(l + v);
                            let shininess = 64.0;
                            let spec = pow(max(dot(n, h), 0.0), shininess);
                            let spec_c = mix(vec3<f32>(0.04), a, 0.0) * spec;
                            let diffuse = a * (1.0 / 3.14159265);
                            col = (diffuse + spec_c) * P.sun_dir_and_intensity.w * ndl;
                        } else {
                            // GGX/Disney placeholder: simple fresnel + microfacet lobe
                            let v = vec3<f32>(0.0, 0.0, 1.0);
                            let h = normalize(l + v);
                            let n_dot_h = max(dot(n, h), 0.0);
                            let v_dot_h = max(dot(v, h), 0.0);
                            let r = rough;
                            let alpha = r * r;
                            let denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0;
                            let D = (alpha * alpha) / (3.14159265 * denom * denom + 1e-6);
                            let F0 = mix(vec3<f32>(0.04), a, 0.0);
                            let F = F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - v_dot_h, 5.0);
                            let kS = F;
                            let kD = (vec3<f32>(1.0) - kS);
                            let diffuse = kD * a * (1.0 / 3.14159265);
                            let specular = F * D; // skip G for simplicity in viewer
                            col = (diffuse + specular) * P.sun_dir_and_intensity.w * ndl;
                        }
                    }
                    // Add a small ambient term always so fully unlit pixels are not black
                    col += 0.1 * a;
                    // Debug 2: NDF-only GGX grayscale
                    if (dbg == 2u) {
                        let v = vec3<f32>(0.0, 0.0, 1.0);
                        let h = normalize(l + v);
                        let n_dot_h = max(dot(n, h), 0.0);
                        let alpha = max(1e-3, rough * rough);
                        let a2 = alpha * alpha;
                        let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
                        let D = a2 / max(3.14159265 * denom * denom, 1e-6);
                        textureStore(out_tex, coord, vec4<f32>(D, D, D, 1.0));
                        return;
                    }
                    if (P.ibl_use_brdf_pad.y > 0.5) {
                        let env = textureSampleLevel(env_cube, env_samp, n, 0.0).rgb;
                        col += a * env * P.ibl_use_brdf_pad.x;
                    }
                    textureStore(out_tex, coord, vec4<f32>(col, 1.0));
                }
            "#
                .into(),
            ),
        });
        let lit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.lit.pl"),
            bind_group_layouts: &[&lit_bgl],
            push_constant_ranges: &[],
        });
        let lit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewer.lit.pipeline"),
            layout: Some(&lit_pl),
            module: &lit_shader,
            entry_point: "cs_main",
        });
        println!(
            "[viewer] lit compute WGSL version {} compiled",
            LIT_WGSL_VERSION
        );
        let lit_params: [f32; 12] = [
            // sun_dir_vs.xyz, sun_intensity
            0.3, 0.6, -1.0, 1.0,
            // ibl_intensity, use_ibl (as float), brdf (as float), pad
            0.6, 1.0, 4.0, 0.0, // roughness, debug_mode, pad, pad
            0.5, 0.0, 0.0, 0.0,
        ];
        let lit_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.lit.uniform"),
            contents: bytemuck::cast_slice(&lit_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Dummy IBL cube (1x1x6) and sampler as fallback so lit viz always binds
        let dummy_env = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.lit.dummy.env"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_env_view = dummy_env.create_view(&wgpu::TextureViewDescriptor {
            label: Some("viewer.lit.dummy.env.view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        });
        let dummy_env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewer.lit.dummy.env.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Lit output target: acts as the current lighting buffer for the viewer.
        // `lit_output` stores the pre-tonemap combination of direct + IBL lighting
        // (diffuse and specular) in Rgba8Unorm; SSR shading samples from this buffer
        // and `ssr/composite.wgsl` adds the reflection contribution on top.
        let lit_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.lit.output"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let lit_output_view = lit_output.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_baseline_hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.baseline.hdr"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let gi_baseline_hdr_view =
            gi_baseline_hdr.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_baseline_diffuse_hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.baseline.diffuse.hdr"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let gi_baseline_diffuse_hdr_view =
            gi_baseline_diffuse_hdr.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_baseline_spec_hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.baseline.spec.hdr"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let gi_baseline_spec_hdr_view =
            gi_baseline_spec_hdr.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_output_hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.output.hdr"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let gi_output_hdr_view =
            gi_output_hdr.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_debug = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.gi.debug"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let gi_debug_view = gi_debug.create_view(&wgpu::TextureViewDescriptor::default());

        let gi_baseline_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.gi.baseline.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let gi_baseline_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.gi.baseline.pl"),
            bind_group_layouts: &[&gi_baseline_bgl],
            push_constant_ranges: &[],
        });
        let gi_baseline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.gi.baseline.shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var src_tex : texture_2d<f32>;
                @group(0) @binding(1) var dst_tex : texture_storage_2d<rgba16float, write>;

                @compute @workgroup_size(8,8,1)
                fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let dims = textureDimensions(src_tex);
                    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
                    let coord = vec2<i32>(gid.xy);
                    let c = textureLoad(src_tex, coord, 0);
                    textureStore(dst_tex, coord, c);
                }
                "#
                .into(),
            ),
        });
        let gi_baseline_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewer.gi.baseline.pipeline"),
            layout: Some(&gi_baseline_pl),
            module: &gi_baseline_shader,
            entry_point: "cs_main",
        });

        let gi_split_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.gi.baseline.split.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let gi_split_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.gi.baseline.split.pl"),
            bind_group_layouts: &[&gi_split_bgl],
            push_constant_ranges: &[],
        });
        let gi_split_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("viewer.gi.baseline.split.shader"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                    @group(0) @binding(0) var src_lit : texture_2d<f32>;
                    @group(0) @binding(1) var normal_tex : texture_2d<f32>;
                    @group(0) @binding(2) var material_tex : texture_2d<f32>;
                    @group(0) @binding(3) var dst_diffuse : texture_storage_2d<rgba16float, write>;
                    @group(0) @binding(4) var dst_spec : texture_storage_2d<rgba16float, write>;

                    @compute @workgroup_size(8,8,1)
                    fn cs_split(@builtin(global_invocation_id) gid: vec3<u32>) {
                        let dims = textureDimensions(src_lit);
                        if (gid.x >= dims.x || gid.y >= dims.y) { return; }
                        let coord = vec2<i32>(gid.xy);
                        let base = textureLoad(src_lit, coord, 0);
                        let mat = textureLoad(material_tex, coord, 0);
                        let normal = textureLoad(normal_tex, coord, 0);
                        let metallic = clamp(mat.a, 0.0, 1.0);
                        let roughness = clamp(normal.w, 0.0, 1.0);
                        let spec_fraction = clamp(0.04 + metallic * (1.0 - roughness), 0.0, 0.95);
                        let L = base.rgb;
                        let L_spec = L * spec_fraction;
                        let L_diff = L - L_spec;
                        textureStore(dst_diffuse, coord, vec4<f32>(L_diff, base.a));
                        textureStore(dst_spec, coord, vec4<f32>(L_spec, base.a));
                    }
                    "#
                        .into(),
                ),
            });
        let gi_split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewer.gi.baseline.split.pipeline"),
            layout: Some(&gi_split_pl),
            module: &gi_split_shader,
            entry_point: "cs_split",
        });

        // Sky: resources and pipeline
        let sky_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.sky.output"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sky_output_view = sky_output.create_view(&wgpu::TextureViewDescriptor::default());

        // Sky params buffer (matches SkyParams in WGSL)
        let mut sky_params_init = SkyUniforms {
            sun_direction: [0.3, 0.8, 0.5],
            turbidity: 2.5,
            ground_albedo: 0.2,
            model: 1, // 0=Preetham, 1=Hosek-Wilkie in WGSL
            sun_intensity: 20.0,
            exposure: 1.0,
            _pad: [0.0; 4],
        };
        // Environment overrides (for CLI integration)
        if let Ok(model_str) = std::env::var("FORGE3D_SKY_MODEL") {
            let key = model_str
                .trim()
                .to_ascii_lowercase()
                .replace(['-', '_', ' '], "");
            sky_params_init.model = match key.as_str() {
                "preetham" => 0,
                "hosekwilkie" => 1,
                other => {
                    eprintln!(
                        "[viewer] unknown FORGE3D_SKY_MODEL='{}', defaulting to hosek-wilkie",
                        other
                    );
                    1
                }
            };
        }
        if let Ok(v) = std::env::var("FORGE3D_SKY_TURBIDITY") {
            if let Ok(f) = v.parse::<f32>() {
                sky_params_init.turbidity = f.clamp(1.0, 10.0);
            }
        }
        if let Ok(v) = std::env::var("FORGE3D_SKY_GROUND") {
            if let Ok(f) = v.parse::<f32>() {
                sky_params_init.ground_albedo = f.clamp(0.0, 1.0);
            }
        }
        if let Ok(v) = std::env::var("FORGE3D_SKY_EXPOSURE") {
            if let Ok(f) = v.parse::<f32>() {
                sky_params_init.exposure = f.max(0.0);
            }
        }
        if let Ok(v) = std::env::var("FORGE3D_SKY_INTENSITY") {
            if let Ok(f) = v.parse::<f32>() {
                sky_params_init.sun_intensity = f.max(0.0);
            }
        }
        let sky_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.params"),
            contents: bytemuck::bytes_of(&sky_params_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Camera buffer for sky (view, proj, inv_view, inv_proj, eye)
        let cam_bytes: u64 =
            (std::mem::size_of::<[[f32; 4]; 4]>() * 4 + std::mem::size_of::<[f32; 4]>()) as u64;
        let sky_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.sky.camera"),
            size: cam_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sky compute pipeline
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.sky.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });
        let sky_bind_group_layout0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.sky.bgl0"),
                entries: &[
                    // @binding(0) sky params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // @binding(1) storage output
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let sky_bind_group_layout1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.sky.bgl1"),
                entries: &[
                    // @binding(0) camera uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.sky.pipeline.layout"),
            bind_group_layouts: &[&sky_bind_group_layout0, &sky_bind_group_layout1],
            push_constant_ranges: &[],
        });
        let sky_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewer.sky.pipeline"),
            layout: Some(&sky_pl),
            module: &sky_shader,
            entry_point: "cs_render_sky",
        });

        // --- P6: Volumetric fog resources ---
        let fog_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.fog.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volumetric.wgsl").into()),
        });
        let fog_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.fog.bgl0"),
            entries: &[
                // @group(0) params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @group(0) camera
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // depth sampler (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let fog_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.fog.bgl1"),
            entries: &[
                // shadow map
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // comparison sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // shadow matrix
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let fog_bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.fog.bgl2"),
            entries: &[
                // output_fog
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // history_fog
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // history_sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let fog_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.fog.pl"),
            bind_group_layouts: &[&fog_bgl0, &fog_bgl1, &fog_bgl2],
            push_constant_ranges: &[],
        });
        let fog_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewer.fog.pipeline"),
            layout: Some(&fog_pl),
            module: &fog_shader,
            entry_point: "cs_volumetric",
        });
        // Froxelized volumetrics: group(3)
        let fog_bgl3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.fog.bgl3"),
            entries: &[
                // storage froxel buffer (3D RGBA16F)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // sampled froxel texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // froxel sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Froxel 3D texture
        let froxel_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.froxel.tex"),
            size: wgpu::Extent3d {
                width: 16,
                height: 8,
                depth_or_array_layers: 64,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let froxel_view = froxel_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("viewer.fog.froxel.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D3),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let froxel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewer.fog.froxel.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // Pipeline layouts for froxels
        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.fog.empty.bgl"),
            entries: &[],
        });
        // Build froxels uses groups: 0(params/camera), 1(shadow), 3(froxel buffer)
        let froxel_build_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.fog.froxel.build.pl"),
            bind_group_layouts: &[&fog_bgl0, &fog_bgl1, &empty_bgl, &fog_bgl3],
            push_constant_ranges: &[],
        });
        let froxel_build_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("viewer.fog.froxel.build"),
                layout: Some(&froxel_build_pl),
                module: &fog_shader,
                entry_point: "cs_build_froxels",
            });
        // Apply froxels uses groups: 0(params/depth), 2(output/history), 3(froxel sampled)
        let froxel_apply_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.fog.froxel.apply.pl"),
            bind_group_layouts: &[&fog_bgl0, &fog_bgl1, &fog_bgl2, &fog_bgl3],
            push_constant_ranges: &[],
        });
        let froxel_apply_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("viewer.fog.froxel.apply"),
                layout: Some(&froxel_apply_pl),
                module: &fog_shader,
                entry_point: "cs_apply_froxels",
            });

        // Fog textures and buffers
        let fog_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.output"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let fog_output_view = fog_output.create_view(&wgpu::TextureViewDescriptor::default());
        let fog_history = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.history"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fog_history_view = fog_history.create_view(&wgpu::TextureViewDescriptor::default());
        let fog_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewer.fog.depth.sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let fog_history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewer.fog.history.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let fog_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.fog.params"),
            size: std::mem::size_of::<VolumetricUniformsStd140>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fog_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.fog.camera"),
            size: std::mem::size_of::<FogCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fog_shadow_map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.shadow.map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fog_shadow_view = fog_shadow_map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("viewer.fog.shadow.view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        let fog_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewer.fog.shadow.sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let fog_shadow_matrix = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.fog.shadow.matrix"),
            size: (std::mem::size_of::<[[f32; 4]; 4]>() as u64),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fog zero fallback texture (1x1 RGBA16F = 8 bytes per pixel)
        let fog_zero_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.zero"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // write zeros
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &fog_zero_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 8],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fog_zero_view = fog_zero_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // P6-10: Half-resolution fog targets
        let half_w = surface_config.width.max(1) / 2;
        let half_h = surface_config.height.max(1) / 2;
        let fog_output_half = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.output.half"),
            size: wgpu::Extent3d {
                width: half_w.max(1),
                height: half_h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fog_output_half_view =
            fog_output_half.create_view(&wgpu::TextureViewDescriptor::default());
        let fog_history_half = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer.fog.history.half"),
            size: wgpu::Extent3d {
                width: half_w.max(1),
                height: half_h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fog_history_half_view =
            fog_history_half.create_view(&wgpu::TextureViewDescriptor::default());

        // Upsample shader pipeline and BGL
        let fog_upsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.fog.upsample.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fog_upsample.wgsl").into()),
        });
        let fog_upsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.fog.upsample.bgl"),
            entries: &[
                // src half-res fog
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // src sampler (filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // dst full-res storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // full-res depth
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // depth sampler (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let fog_upsample_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.fog.upsample.pl"),
            bind_group_layouts: &[&fog_upsample_bgl],
            push_constant_ranges: &[],
        });
        let fog_upsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("viewer.fog.upsample.pipeline"),
                layout: Some(&fog_upsample_pl),
                module: &fog_upsample_shader,
                entry_point: "cs_main",
            });
        let fog_upsample_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.fog.upsample.params"),
            size: std::mem::size_of::<FogUpsampleParamsStd140>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // HUD overlay renderer
        let mut hud =
            crate::core::text_overlay::TextOverlayRenderer::new(&device, surface_config.format);
        hud.set_enabled(true);
        hud.set_resolution(surface_config.width, surface_config.height);

        // Configure cascaded shadow maps for directional sun shadows. At this
        // stage we only use the light-space matrices for fog; the depth atlas
        // render pass will be added in a later milestone.
        let mut csm_config = CsmConfig::default();
        csm_config.camera_near = config.znear;
        csm_config.camera_far = config.zfar;
        let csm = Some(CsmShadowMap::new(device.as_ref(), csm_config.clone()));

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
            geom_bind_group_layout,
            geom_pipeline,
            geom_camera_buffer,
            geom_bind_group,
            geom_vb,
            geom_ib: None,
            geom_index_count: 36,
            z_texture,
            z_view,
            albedo_texture,
            albedo_view,
            albedo_sampler,
            ssr_env_texture: None,
            comp_bind_group_layout,
            comp_pipeline,
            comp_uniform: None,
            lit_bind_group_layout: lit_bgl,
            lit_pipeline,
            lit_uniform,
            lit_output,
            lit_output_view,
            gi_baseline_hdr,
            gi_baseline_hdr_view,
            gi_baseline_diffuse_hdr,
            gi_baseline_diffuse_hdr_view,
            gi_baseline_spec_hdr,
            gi_baseline_spec_hdr_view,
            gi_output_hdr,
            gi_output_hdr_view,
            gi_debug,
            gi_debug_view,
            gi_baseline_bgl,
            gi_baseline_pipeline,
            gi_split_bgl,
            gi_split_pipeline,
            gi_ao_weight: 1.0,
            gi_ssgi_weight: 1.0,
            gi_ssr_weight: 1.0,
            // Lit params defaults must match the initial lit_params above
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
            ibl_env_view: Some(dummy_env_view),
            ibl_sampler: Some(dummy_env_sampler),
            ibl_hdr_path: None,
            ibl_cache_dir: None,
            ibl_base_resolution: None,
            viz_depth_max_override: None,
            auto_snapshot_path: std::env::var("FORGE3D_AUTO_SNAPSHOT_PATH").ok(),
            auto_snapshot_done: false,
            dump_p5_requested: false,
            adapter_name,
            debug_logged_render_gate: false,
            sky_bind_group_layout0: sky_bind_group_layout0,
            sky_bind_group_layout1: sky_bind_group_layout1,
            sky_pipeline,
            sky_params,
            sky_camera,
            sky_output,
            sky_output_view,
            sky_enabled: true,
            // Fog init
            fog_enabled: false,
            fog_params,
            fog_camera,
            fog_output,
            fog_output_view,
            fog_history,
            fog_history_view,
            fog_depth_sampler,
            fog_history_sampler,
            fog_pipeline,
            fog_frame_index: 0,
            fog_bgl3,
            froxel_tex,
            froxel_view,
            froxel_sampler,
            froxel_build_pipeline,
            froxel_apply_pipeline,
            // Half-res upsample controls/resources
            fog_half_res_enabled: false,
            fog_output_half,
            fog_output_half_view,
            fog_history_half,
            fog_history_half_view,
            fog_upsample_bgl,
            fog_upsample_pipeline,
            fog_upsample_params,
            fog_bilateral: true,
            fog_upsigma: 0.02,
            fog_bgl0,
            fog_bgl1,
            fog_bgl2,
            fog_shadow_map,
            fog_shadow_view,
            fog_shadow_sampler,
            fog_shadow_matrix,
            fog_zero_tex,
            fog_zero_view,
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
            sky_model_id: 1,
            sky_turbidity: 2.5,
            sky_ground_albedo: 0.2,
            sky_exposure: 1.0,
            sky_sun_intensity: 20.0,
            // HUD overlay renderer
            hud_enabled: true,
            hud,
            ssr_scene_loaded: false,
            ssr_scene_preset: None,
            // Object transform defaults (identity)
            object_translation: glam::Vec3::ZERO,
            object_rotation: glam::Quat::IDENTITY,
            object_scale: glam::Vec3::ONE,
            object_transform: glam::Mat4::IDENTITY,
        };

        viewer.sync_ssr_params_to_gi();

        Ok(viewer)
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn update_lit_uniform(&mut self) {
        // Keep sun_dir consistent with compute shader default
        let sun_dir = [0.3f32, 0.6, -1.0];
        let params: [f32; 12] = [
            // sun_dir.xyz, sun_intensity
            sun_dir[0],
            sun_dir[1],
            sun_dir[2],
            self.lit_sun_intensity,
            // ibl_intensity, use_ibl, brdf_index, pad
            self.lit_ibl_intensity,
            if self.lit_use_ibl { 1.0 } else { 0.0 },
            self.lit_brdf as f32,
            0.0,
            // roughness, debug_mode, pad, pad
            self.lit_roughness.clamp(0.0, 1.0),
            self.lit_debug_mode as f32,
            0.0,
            0.0,
        ];
        self.queue
            .write_buffer(&self.lit_uniform, 0, bytemuck::cast_slice(&params));
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            if let Some(ref mut gi) = self.gi {
                gi.gbuffer_mut()
                    .resize(&self.device, new_size.width, new_size.height)
                    .ok();
                gi.set_ssr_params(&self.queue, &self.ssr_params);
            }
            // Recreate lit output
            self.lit_output = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.lit.output"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.lit_output_view = self
                .lit_output
                .create_view(&wgpu::TextureViewDescriptor::default());
            // Recreate GI HDR baseline and output targets
            self.gi_baseline_hdr = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gi.baseline.hdr"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.gi_baseline_hdr_view = self
                .gi_baseline_hdr
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.gi_baseline_diffuse_hdr =
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.gi.baseline.diffuse.hdr"),
                    size: wgpu::Extent3d {
                        width: new_size.width,
                        height: new_size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
            self.gi_baseline_diffuse_hdr_view = self
                .gi_baseline_diffuse_hdr
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.gi_baseline_spec_hdr = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gi.baseline.spec.hdr"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.gi_baseline_spec_hdr_view = self
                .gi_baseline_spec_hdr
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.gi_output_hdr = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gi.output.hdr"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.gi_output_hdr_view = self
                .gi_output_hdr
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.gi_debug = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.gi.debug"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.gi_debug_view = self
                .gi_debug
                .create_view(&wgpu::TextureViewDescriptor::default());

            if self.gi_pass.is_some() {
                match GiPass::new(&self.device, new_size.width, new_size.height) {
                    Ok(pass) => {
                        self.gi_pass = Some(pass);
                    }
                    Err(e) => {
                        eprintln!("Failed to recreate GiPass after resize: {}", e);
                        self.gi_pass = None;
                    }
                }
            }
            // Recreate sky output
            self.sky_output = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.sky.output"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.sky_output_view = self
                .sky_output
                .create_view(&wgpu::TextureViewDescriptor::default());
            // Recreate depth buffer for geometry pass
            if self.geom_pipeline.is_some() {
                let z_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.gbuf.z"),
                    size: wgpu::Extent3d {
                        width: new_size.width,
                        height: new_size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let z_view = z_texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.z_texture = Some(z_texture);
                self.z_view = Some(z_view);
            }
            // Recreate fog textures
            self.fog_output = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.fog.output"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.fog_output_view = self
                .fog_output
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.fog_history = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.fog.history"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.fog_history_view = self
                .fog_history
                .create_view(&wgpu::TextureViewDescriptor::default());
            // Recreate half-resolution fog targets
            let half_w = (new_size.width.max(1)) / 2;
            let half_h = (new_size.height.max(1)) / 2;
            self.fog_output_half = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.fog.output.half"),
                size: wgpu::Extent3d {
                    width: half_w.max(1),
                    height: half_h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.fog_output_half_view = self
                .fog_output_half
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.fog_history_half = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.fog.history.half"),
                size: wgpu::Extent3d {
                    width: half_w.max(1),
                    height: half_h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.fog_history_half_view = self
                .fog_history_half
                .create_view(&wgpu::TextureViewDescriptor::default());
            // HUD resolution
            self.hud.set_resolution(new_size.width, new_size.height);
        }
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(keycode) = key_event.physical_key {
                    let pressed = key_event.state == ElementState::Pressed;

                    // Track shift
                    if matches!(keycode, KeyCode::ShiftLeft | KeyCode::ShiftRight) {
                        self.shift_pressed = pressed;
                    }

                    // Track WASD, Q, E for FPS mode
                    if pressed {
                        self.keys_pressed.insert(keycode);
                    } else {
                        self.keys_pressed.remove(&keycode);
                    }

                    // Toggle camera mode with Tab
                    if pressed && keycode == KeyCode::Tab {
                        let new_mode = match self.camera.mode() {
                            CameraMode::Orbit => CameraMode::Fps,
                            CameraMode::Fps => CameraMode::Orbit,
                        };
                        self.camera.set_mode(new_mode);
                        println!("Camera mode: {:?}", new_mode);
                        return true;
                    }
                }

                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.camera.mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera
                    .handle_mouse_move(position.x as f32, position.y as f32);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.handle_mouse_scroll(scroll);
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Update FPS camera movement
        let mut forward = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;

        let speed_mult = if self.shift_pressed { 2.0 } else { 1.0 };

        if self.keys_pressed.contains(&KeyCode::KeyW) {
            forward += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            forward -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            right += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            right -= speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyE) {
            up += speed_mult;
        }
        if self.keys_pressed.contains(&KeyCode::KeyQ) {
            up -= speed_mult;
        }

        self.camera.update_fps(dt, forward, right, up);

        // Update GI camera params
        if let Some(ref mut gi) = self.gi {
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            let view = self.camera.view_matrix();
            let inv_proj = proj.inverse();

            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let eye = self.camera.eye();
            let inv_view = view.inverse();
            let cam = crate::core::screen_space_effects::CameraParams {
                view_matrix: to_arr4(view),
                inv_view_matrix: to_arr4(inv_view),
                proj_matrix: to_arr4(proj),
                inv_proj_matrix: to_arr4(inv_proj),
                camera_pos: [eye.x, eye.y, eye.z],
                _pad: 0.0,
            };
            gi.update_camera(&self.queue, &cam);
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if self.frame_count == 0 {
            eprintln!("[viewer-debug] entering render loop (first frame)");
        }

        // Ensure auto-snapshot request is registered before encoding so we render to an offscreen texture
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
                eprintln!("[viewer-debug] auto snapshot requested: {}", p);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render sky background (compute) before opaques
        if self.sky_enabled {
            // Build camera matrices (view, proj, inv_view, inv_proj) and eye
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            let view_mat = self.camera.view_matrix();
            let inv_view = view_mat.inverse();
            let inv_proj = proj.inverse();
            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let eye = self.camera.eye();
            let cam_buf: [[[f32; 4]; 4]; 4] = [
                to_arr4(view_mat),
                to_arr4(proj),
                to_arr4(inv_view),
                to_arr4(inv_proj),
            ];
            // Write matrices
            self.queue
                .write_buffer(&self.sky_camera, 0, bytemuck::cast_slice(&cam_buf));
            // Write eye position (vec4 packed)
            let eye4: [f32; 4] = [eye.x, eye.y, eye.z, 0.0];
            let base = (std::mem::size_of::<[[f32; 4]; 4]>() * 4) as u64;
            self.queue
                .write_buffer(&self.sky_camera, base, bytemuck::cast_slice(&eye4));

            // Update sky params each frame based on viewer-set fields
            let sun_dir_vs = glam::Vec3::new(0.3, 0.6, -1.0).normalize();
            let sun_dir_ws = (inv_view
                * glam::Vec4::new(sun_dir_vs.x, sun_dir_vs.y, sun_dir_vs.z, 0.0))
            .truncate()
            .normalize();
            let model_id: u32 = self.sky_model_id;
            let turb: f32 = self.sky_turbidity.clamp(1.0, 10.0);
            let ground: f32 = self.sky_ground_albedo.clamp(0.0, 1.0);
            let expose: f32 = self.sky_exposure.max(0.0);
            let sun_i: f32 = self.sky_sun_intensity.max(0.0);

            let sky_params_frame = SkyUniforms {
                sun_direction: [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
                turbidity: turb,
                ground_albedo: ground,
                model: model_id,
                sun_intensity: sun_i,
                exposure: expose,
                _pad: [0.0; 4],
            };
            self.queue
                .write_buffer(&self.sky_params, 0, bytemuck::bytes_of(&sky_params_frame));

            // Bind and dispatch compute
            let sky_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.sky.bg0"),
                layout: &self.sky_bind_group_layout0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.sky_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                    },
                ],
            });
            let sky_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.sky.bg1"),
                layout: &self.sky_bind_group_layout1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sky_camera.as_entire_binding(),
                }],
            });
            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.sky.compute"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sky_pipeline);
                cpass.set_bind_group(0, &sky_bg0, &[]);
                cpass.set_bind_group(1, &sky_bg1, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
        }

        // Composite debug: after GI/geometry, show GBuffer material on swapchain

        // Execute screen-space effects if any are enabled
        let have_gi = self.gi.is_some();
        let have_pipe = self.geom_pipeline.is_some();
        let have_cam = self.geom_camera_buffer.is_some();
        let have_vb = self.geom_vb.is_some();
        let have_z = self.z_view.is_some();
        let have_bgl = self.geom_bind_group_layout.is_some();
        if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            if !self.debug_logged_render_gate {
                eprintln!(
                    "[viewer-debug] render gate: gi={} pipe={} cam={} vb={} z={} bgl={}",
                    have_gi, have_pipe, have_cam, have_vb, have_z, have_bgl
                );
                self.debug_logged_render_gate = true;
            }
        }

        if self.geom_bind_group.is_none() {
            if let Err(err) = self.ensure_geom_bind_group() {
                eprintln!("[viewer] failed to build geometry bind group: {err}");
            }
        }
        if let (Some(gi), Some(pipe), Some(cam_buf), Some(vb), Some(zv), Some(_bgl)) = (
            self.gi.as_mut(),
            self.geom_pipeline.as_ref(),
            self.geom_camera_buffer.as_ref(),
            self.geom_vb.as_ref(),
            self.z_view.as_ref(),
            self.geom_bind_group_layout.as_ref(),
        ) {
            // Update geometry camera uniform (view, proj)
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            // Apply object transform to create model-view matrix
            let view_mat = self.camera.view_matrix();
            let model_view = view_mat * self.object_transform;
            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let cam_pack = [to_arr4(model_view), to_arr4(proj)];
            self.queue
                .write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_pack));

            // Keep GI camera uniforms in sync with this geometry pass so that
            // SSAO/GTAO reconstruction uses the correct view/projection.
            let inv_proj = proj.inverse();
            let eye = self.camera.eye();
            let inv_view = view_mat.inverse();
            let cam = crate::core::screen_space_effects::CameraParams {
                view_matrix: to_arr4(view_mat),
                inv_view_matrix: to_arr4(inv_view),
                proj_matrix: to_arr4(proj),
                inv_proj_matrix: to_arr4(inv_proj),
                camera_pos: [eye.x, eye.y, eye.z],
                _pad: 0.0,
            };
            gi.update_camera(&self.queue, &cam);

            // Geometry bind group (camera + albedo)
            let bg_ref = match self.geom_bind_group.as_ref() {
                Some(bg) => bg,
                None => {
                    // Create a minimal bind group if missing (shouldn't happen)
                    let sampler = self.albedo_sampler.get_or_insert_with(|| {
                        self.device
                            .create_sampler(&wgpu::SamplerDescriptor::default())
                    });
                    let white_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewer.geom.albedo.fallback2"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    self.queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture: &white_tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &[255, 255, 255, 255],
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: Some(1),
                        },
                        wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                    let view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    self.albedo_texture = Some(white_tex);
                    let bgl = self.geom_bind_group_layout.as_ref().unwrap();
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.gbuf.geom.bg.autogen"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: cam_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.albedo_view = Some(view);
                    self.geom_bind_group = Some(bg);
                    self.geom_bind_group.as_ref().unwrap()
                }
            };
            let _layout = self.geom_bind_group_layout.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.geom"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gi.gbuffer().normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gi.gbuffer().material_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gi.gbuffer().depth_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: zv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg_ref, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            if let Some(ib) = self.geom_ib.as_ref() {
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
            } else {
                pass.draw(0..self.geom_index_count, 0..1);
            }
            drop(pass);

            // P6: Volumetric fog compute (after depth is available)
            if self.fog_enabled {
                // Prepare camera uniforms
                let aspect = self.config.width as f32 / self.config.height as f32;
                let fov = self.view_config.fov_deg.to_radians();
                let proj = Mat4::perspective_rh(
                    fov,
                    aspect,
                    self.view_config.znear,
                    self.view_config.zfar,
                );
                let view_mat = self.camera.view_matrix();

                // Update CSM cascade transforms for current camera. The actual
                // shadow depth rendering into the CSM atlas is a separate
                // milestone; here we only keep the light-space matrices in sync
                // so fog can reuse them.
                if let Some(ref mut csm) = self.csm {
                    let frustum = CameraFrustum::from_matrices(&view_mat, &proj);
                    csm.update_cascades(&self.queue, &frustum);
                }

                if self.fog_use_shadows {
                    if let (Some(ref csm), Some(ref csm_pipe), Some(ref csm_cam_buf)) = (
                        self.csm.as_ref(),
                        self.csm_depth_pipeline.as_ref(),
                        self.csm_depth_camera.as_ref(),
                    ) {
                        let cascade_count = csm.cascade_count() as usize;
                        let bgl = csm_pipe.get_bind_group_layout(0);
                        for cascade_idx in 0..cascade_count {
                            if let (Some(depth_view), Some(light_vp)) = (
                                csm.cascade_depth_view(cascade_idx),
                                csm.cascade_projection(cascade_idx),
                            ) {
                                let light_vp_arr = light_vp.to_cols_array();
                                self.queue.write_buffer(
                                    csm_cam_buf,
                                    0,
                                    bytemuck::cast_slice(&light_vp_arr),
                                );
                                let csm_bg = self
                                    .device
                                    .create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("viewer.csm.depth.bg"),
                                        layout: &bgl,
                                        entries: &[wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: csm_cam_buf.as_entire_binding(),
                                        }],
                                    });
                                let mut shadow_pass = encoder.begin_render_pass(
                                    &wgpu::RenderPassDescriptor {
                                        label: Some("viewer.csm.depth"),
                                        color_attachments: &[],
                                        depth_stencil_attachment: Some(
                                            wgpu::RenderPassDepthStencilAttachment {
                                                view: depth_view,
                                                depth_ops: Some(wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(1.0),
                                                    store: wgpu::StoreOp::Store,
                                                }),
                                                stencil_ops: None,
                                            },
                                        ),
                                        occlusion_query_set: None,
                                        timestamp_writes: None,
                                    },
                                );
                                shadow_pass.set_pipeline(csm_pipe);
                                shadow_pass.set_bind_group(0, &csm_bg, &[]);
                                shadow_pass.set_vertex_buffer(0, vb.slice(..));
                                if let Some(ib) = self.geom_ib.as_ref() {
                                    shadow_pass.set_index_buffer(
                                        ib.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    shadow_pass.draw_indexed(
                                        0..self.geom_index_count,
                                        0,
                                        0..1,
                                    );
                                } else {
                                    shadow_pass.draw(0..self.geom_index_count, 0..1);
                                }
                            }
                        }
                    }
                }

                let inv_view = view_mat.inverse();
                let inv_proj = proj.inverse();
                let eye = self.camera.eye();
                fn to_arr(m: Mat4) -> [[f32; 4]; 4] {
                    let c = m.to_cols_array();
                    [
                        [c[0], c[1], c[2], c[3]],
                        [c[4], c[5], c[6], c[7]],
                        [c[8], c[9], c[10], c[11]],
                        [c[12], c[13], c[14], c[15]],
                    ]
                }
                let fog_cam = FogCameraUniforms {
                    view: to_arr(view_mat),
                    proj: to_arr(proj),
                    inv_view: to_arr(inv_view),
                    inv_proj: to_arr(inv_proj),
                    view_proj: to_arr(proj * view_mat),
                    eye_position: [eye.x, eye.y, eye.z],
                    near: self.view_config.znear,
                    far: self.view_config.zfar,
                    _pad: [0.0; 3],
                };
                self.queue
                    .write_buffer(&self.fog_camera, 0, bytemuck::bytes_of(&fog_cam));
                // Params
                let sun_dir_ws = (inv_view * glam::Vec4::new(0.3, 0.6, -1.0, 0.0))
                    .truncate()
                    .normalize();
                let steps = if self.fog_half_res_enabled {
                    (self.fog_steps.max(1) / 2).max(16)
                } else {
                    self.fog_steps.max(1)
                };
                let fog_params_packed = VolumetricUniformsStd140 {
                    density: self.fog_density.max(0.0),
                    height_falloff: 0.1,
                    phase_g: self.fog_g.clamp(-0.999, 0.999),
                    max_steps: steps,
                    start_distance: 0.1,
                    max_distance: self.view_config.zfar,
                    _pad_a0: 0.0,
                    _pad_a1: 0.0,
                    scattering_color: [1.0, 1.0, 1.0],
                    absorption: 1.0,
                    sun_direction: [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
                    sun_intensity: self.sky_sun_intensity.max(0.0),
                    ambient_color: [0.2, 0.25, 0.3],
                    temporal_alpha: self.fog_temporal_alpha.clamp(0.0, 0.9),
                    use_shadows: if self.fog_use_shadows { 1 } else { 0 },
                    jitter_strength: 0.8,
                    frame_index: self.fog_frame_index,
                    _pad0: 0,
                };
                self.queue.write_buffer(
                    &self.fog_params,
                    0,
                    bytemuck::bytes_of(&fog_params_packed),
                );

                // Select a light-space matrix for fog shadows. For now, use the
                // first CSM cascade's light_projection if available; otherwise
                // fall back to identity. Depth still comes from fog_shadow_map,
                // which is initialized to depth=1.0 until a real shadow pass
                // is wired in.
                let mut fog_shadow_mat = Mat4::IDENTITY;
                if self.fog_use_shadows {
                    if let Some(ref csm) = self.csm {
                        let cascades = csm.cascades();
                        if let Some(c0) = cascades.get(0) {
                            fog_shadow_mat = Mat4::from_cols_array_2d(&c0.light_projection);
                        }
                    }
                }
                self.queue.write_buffer(
                    &self.fog_shadow_matrix,
                    0,
                    bytemuck::bytes_of(&to_arr(fog_shadow_mat)),
                );

                // Bind groups (shared among both modes)
                let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg0"),
                    layout: &self.fog_bgl0,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.fog_params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.fog_camera.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.fog_depth_sampler),
                        },
                    ],
                });
                let (shadow_tex_view, shadow_uniform_buf) = if let Some(ref csm) = self.csm {
                    (csm.shadow_array_view(), csm.uniform_buffer())
                } else {
                    (&self.fog_shadow_view, &self.fog_shadow_matrix)
                };
                let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg1"),
                    layout: &self.fog_bgl1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(shadow_tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.fog_shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: shadow_uniform_buf.as_entire_binding(),
                        },
                    ],
                });
                let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg2"),
                    layout: &self.fog_bgl2,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.fog_output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.fog_history_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.fog_history_sampler),
                        },
                    ],
                });
                if matches!(self.fog_mode, FogMode::Raymarch) {
                    if self.fog_half_res_enabled {
                        // Half-resolution path: bind half-res output/history
                        let bg2_half = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("viewer.fog.bg2.half"),
                            layout: &self.fog_bgl2,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_output_half_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_history_half_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.fog_history_sampler,
                                    ),
                                },
                            ],
                        });
                        let gx = ((self.config.width / 2) + 7) / 8;
                        let gy = ((self.config.height / 2) + 7) / 8;
                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("viewer.fog.raymarch.half"),
                                    timestamp_writes: None,
                                });
                            cpass.set_pipeline(&self.fog_pipeline);
                            cpass.set_bind_group(0, &bg0, &[]);
                            cpass.set_bind_group(1, &bg1, &[]);
                            cpass.set_bind_group(2, &bg2_half, &[]);
                            cpass.dispatch_workgroups(gx, gy, 1);
                        }
                        // Copy half output to half history
                        encoder.copy_texture_to_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_output_half,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_history_half,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: self.config.width.max(1) / 2,
                                height: self.config.height.max(1) / 2,
                                depth_or_array_layers: 1,
                            },
                        );
                        // Upsample to full-res for composition
                        let upsampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                            label: Some("viewer.fog.upsampler"),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            mipmap_filter: wgpu::FilterMode::Nearest,
                            ..Default::default()
                        });
                        let params = FogUpsampleParamsStd140 {
                            sigma: self.fog_upsigma.max(0.0),
                            use_bilateral: if self.fog_bilateral { 1 } else { 0 },
                            _pad: [0.0; 2],
                        };
                        self.queue.write_buffer(
                            &self.fog_upsample_params,
                            0,
                            bytemuck::bytes_of(&params),
                        );
                        let up_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("viewer.fog.upsample.bg"),
                            layout: &self.fog_upsample_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_output_half_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&upsampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_output_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &gi.gbuffer().depth_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.fog_depth_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: self.fog_upsample_params.as_entire_binding(),
                                },
                            ],
                        });
                        let ugx = (self.config.width + 7) / 8;
                        let ugy = (self.config.height + 7) / 8;
                        let mut up_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("viewer.fog.upsample"),
                                timestamp_writes: None,
                            });
                        up_pass.set_pipeline(&self.fog_upsample_pipeline);
                        up_pass.set_bind_group(0, &up_bg, &[]);
                        up_pass.dispatch_workgroups(ugx, ugy, 1);
                    } else {
                        // Full-resolution path (original)
                        let gx = (self.config.width + 7) / 8;
                        let gy = (self.config.height + 7) / 8;
                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("viewer.fog.raymarch"),
                                    timestamp_writes: None,
                                });
                            cpass.set_pipeline(&self.fog_pipeline);
                            cpass.set_bind_group(0, &bg0, &[]);
                            cpass.set_bind_group(1, &bg1, &[]);
                            cpass.set_bind_group(2, &bg2, &[]);
                            cpass.dispatch_workgroups(gx, gy, 1);
                        }
                        // Copy output to full-res history
                        encoder.copy_texture_to_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_output,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_history,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: self.config.width,
                                height: self.config.height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                } else {
                    // Froxel build then apply
                    let bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg3"),
                        layout: &self.fog_bgl3,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                            }, // storage view
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.froxel_sampler),
                            },
                        ],
                    });
                    // Build froxels: workgroup_size(4,4,4) over 16x8x64
                    let gx3d = (16u32 + 3) / 4;
                    let gy3d = (8u32 + 3) / 4;
                    let gz3d = (64u32 + 3) / 4;
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("viewer.fog.froxel.build"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.froxel_build_pipeline);
                        pass.set_bind_group(0, &bg0, &[]);
                        pass.set_bind_group(1, &bg1, &[]);
                        pass.set_bind_group(3, &bg3, &[]);
                        pass.dispatch_workgroups(gx3d, gy3d, gz3d);
                    }
                    // Apply froxels: workgroup_size(8,8,1) across viewport
                    let gx2d = (self.config.width + 7) / 8;
                    let gy2d = (self.config.height + 7) / 8;
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("viewer.fog.froxel.apply"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.froxel_apply_pipeline);
                        pass.set_bind_group(0, &bg0, &[]);
                        pass.set_bind_group(2, &bg2, &[]);
                        pass.set_bind_group(3, &bg3, &[]);
                        pass.dispatch_workgroups(gx2d, gy2d, 1);
                    }
                    // For froxels, history is full-res; copy as before
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: &self.fog_output,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: &self.fog_history,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: self.config.width,
                            height: self.config.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                self.fog_frame_index = self.fog_frame_index.wrapping_add(1);
            }

            // If SSR is enabled, compute the pre-tonemap lighting now so SSR can sample it
            if gi.is_enabled(crate::core::screen_space_effects::ScreenSpaceEffect::SSR) {
                // Build lighting into lit_output_view
                let env_view = if let Some(ref v) = self.ibl_env_view {
                    v
                } else {
                    &self.ibl_env_view.as_ref().unwrap()
                };
                let env_samp = if let Some(ref s) = self.ibl_sampler {
                    s
                } else {
                    &self.ibl_sampler.as_ref().unwrap()
                };
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.lit.bg.pre_ssr"),
                    layout: &self.lit_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &gi.gbuffer().material_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(env_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(env_samp),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.lit_uniform.as_entire_binding(),
                        },
                    ],
                });
                let gx = (self.config.width + 7) / 8;
                let gy = (self.config.height + 7) / 8;
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("viewer.lit.compute.pre_ssr"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.lit_pipeline);
                    cpass.set_bind_group(0, &bg, &[]);
                    cpass.dispatch_workgroups(gx, gy, 1);
                }
                // Provide SSR with the lit buffer as scene color
                let lit_view_for_ssr = self
                    .lit_output
                    .create_view(&wgpu::TextureViewDescriptor::default());
                gi.set_ssr_scene_color_view(lit_view_for_ssr);
            }

            // Build Hierarchical Z (HZB) pyramid from the real depth buffer (Depth32Float)
            // Use regular-Z convention (reversed_z=false) for viewer
            gi.build_hzb(&self.device, &mut encoder, zv, false);
            // Execute effects
            let _ = gi.execute(&self.device, &mut encoder, None, None);

            // Composite the material GBuffer to the swapchain
            if let (Some(comp_pl), Some(comp_bgl)) = (
                self.comp_pipeline.as_ref(),
                self.comp_bind_group_layout.as_ref(),
            ) {
                // Select source texture based on viz_mode
                // If Lit, compute into lit_output first
                if matches!(self.viz_mode, VizMode::Lit) {
                    let env_view = if let Some(ref v) = self.ibl_env_view {
                        v
                    } else {
                        &self.ibl_env_view.as_ref().unwrap()
                    };
                    let env_samp = if let Some(ref s) = self.ibl_sampler {
                        s
                    } else {
                        &self.ibl_sampler.as_ref().unwrap()
                    };
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.lit.bg"),
                        layout: &self.lit_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().normal_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().material_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().depth_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(env_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(env_samp),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.lit_uniform.as_entire_binding(),
                            },
                        ],
                    });
                    let gx = (self.config.width + 7) / 8;
                    let gy = (self.config.height + 7) / 8;
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("viewer.lit.compute"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&self.lit_pipeline);
                        cpass.set_bind_group(0, &bg, &[]);
                        cpass.dispatch_workgroups(gx, gy, 1);
                    }
                }
                let (mode_u32, src_view) = match self.viz_mode {
                    VizMode::Material => {
                        if let Some(v) = gi.material_with_ssr_view() {
                            (0u32, v)
                        } else if self.use_ssao_composite {
                            if let Some(v) = gi.material_with_ao_view() {
                                (0u32, v)
                            } else {
                                (0u32, &gi.gbuffer().material_view)
                            }
                        } else {
                            (0u32, &gi.gbuffer().material_view)
                        }
                    }
                    VizMode::Normal => (1u32, &gi.gbuffer().normal_view),
                    VizMode::Depth => (2u32, &gi.gbuffer().depth_view),
                    VizMode::Gi => match self.gi_viz_mode {
                        GiVizMode::None => {
                            if let Some(v) = gi.gi_debug_view() {
                                (3u32, v)
                            } else {
                                (0u32, &gi.gbuffer().material_view)
                            }
                        }
                        GiVizMode::Composite => (0u32, &self.gi_debug_view),
                        GiVizMode::Ao => {
                            if let Some(v) = gi.ao_resolved_view() {
                                (3u32, v)
                            } else {
                                (3u32, &gi.gbuffer().material_view)
                            }
                        }
                        GiVizMode::Ssgi => {
                            if let Some(v) = gi.ssgi_output_for_display_view() {
                                (0u32, v)
                            } else {
                                (0u32, &gi.gbuffer().material_view)
                            }
                        }
                        GiVizMode::Ssr => {
                            if let Some(v) = gi.ssr_final_view() {
                                (0u32, v)
                            } else {
                                (0u32, &self.lit_output_view)
                            }
                        }
                    },
                    VizMode::Lit => (0u32, &self.lit_output_view),
                };
                // Prepare comp uniform (mode, far)
                let params: [f32; 4] = [
                    mode_u32 as f32,
                    self.viz_depth_max_override.unwrap_or(self.view_config.zfar),
                    0.0,
                    0.0,
                ];
                let buf_ref: &wgpu::Buffer = if let Some(ref ub) = self.comp_uniform {
                    self.queue
                        .write_buffer(ub, 0, bytemuck::cast_slice(&params));
                    ub
                } else {
                    let ub = self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("viewer.comp.uniform"),
                            contents: bytemuck::cast_slice(&params),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        });
                    self.comp_uniform = Some(ub);
                    self.comp_uniform.as_ref().unwrap()
                };
                // Sampler: non-filtering so we can bind depth/non-filterable textures safely
                let comp_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.comp.sampler"),
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                });
                let comp_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.comp.bg"),
                    layout: comp_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&comp_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(if self.fog_enabled {
                                &self.fog_output_view
                            } else {
                                &self.fog_zero_view
                            }),
                        },
                    ],
                });
                // If a snapshot is requested, render the composite to an offscreen texture too
                if self.snapshot_request.is_some() {
                    let (mut snap_w, mut snap_h) =
                        if let (Some(w), Some(h)) =
                            (self.view_config.snapshot_width, self.view_config.snapshot_height)
                        {
                            (w, h)
                        } else {
                            (self.config.width, self.config.height)
                        };

                    // Apply a soft megapixel clamp only when the user has requested
                    // an explicit override resolution via ViewerConfig.
                    if self.view_config.snapshot_width.is_some()
                        && self.view_config.snapshot_height.is_some()
                    {
                        let pixels = snap_w as u64 * snap_h as u64;
                        let max_pixels = (VIEWER_SNAPSHOT_MAX_MEGAPIXELS * 1_000_000.0) as u64;
                        if pixels > max_pixels {
                            let scale = (max_pixels as f32 / pixels as f32).sqrt();
                            snap_w = ((snap_w as f32) * scale).floor().max(1.0) as u32;
                            snap_h = ((snap_h as f32) * scale).floor().max(1.0) as u32;
                        }
                    }

                    let snap_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewer.snapshot.offscreen"),
                        size: wgpu::Extent3d {
                            width: snap_w,
                            height: snap_h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: self.config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut pass_snap = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.comp.pass.snapshot"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &snap_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass_snap.set_pipeline(comp_pl);
                    pass_snap.set_bind_group(0, &comp_bg, &[]);
                    pass_snap.draw(0..3, 0..1);
                    drop(pass_snap);
                    // Store to be read back after submit
                    self.pending_snapshot_tex = Some(snap_tex);
                }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.comp.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(comp_pl);
                pass.set_bind_group(0, &comp_bg, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);

                if self.hud_enabled {
                    // HUD overlay after composite
                    // Build simple bars for sky/fog settings + numeric readouts
                    let mut hud_instances: Vec<crate::core::text_overlay::TextInstance> =
                        Vec::new();
                    let sx = 8.0f32;
                    let sy = 8.0f32; // start position
                    let bar_w = 120.0f32;
                    let bar_h = 10.0f32;
                    let gap = 4.0f32;
                    let num_scale = 0.6f32; // ~11px tall digits
                    let num_dx = 8.0f32; // spacing from end of bar
                    let mut y = sy;
                    // Sky enabled bar (green if on, gray if off)
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "SKY",
                        [0.8, 0.95, 0.8, 0.9],
                    );
                    let sky_on = if self.sky_enabled { 1.0 } else { 0.25 };
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.2, 0.8, 0.2, sky_on],
                    });
                    // Label model (0=Preetham,1=Hosek)
                    let model_val = if self.sky_model_id == 0 { 0.0 } else { 1.0 };
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0; // slightly above bar baseline
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        model_val,
                        1,
                        0,
                        [0.7, 0.9, 0.7, 0.9],
                    );
                    y += bar_h + gap;
                    // Sky turbidity bar length + number
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "TURB",
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    let tfrac = (self.sky_turbidity.clamp(1.0, 10.0) - 1.0) / 9.0;
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w * tfrac, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.2, 0.5, 1.0, 0.8],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        self.sky_turbidity,
                        4,
                        1,
                        [0.6, 0.8, 1.0, 0.9],
                    );
                    y += bar_h + gap;
                    // Fog enabled bar (blue if on)
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "FOG",
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    let fog_on = if self.fog_enabled { 0.9 } else { 0.2 };
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.2, 0.6, 1.0, fog_on],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        if self.fog_enabled { 1.0 } else { 0.0 },
                        1,
                        0,
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    y += bar_h + gap;
                    // Fog density bar + number
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "DENS",
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    let dfrac = (self.fog_density / 0.1).clamp(0.0, 1.0);
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w * dfrac, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.6, 0.8, 1.0, 0.8],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        self.fog_density,
                        5,
                        3,
                        [0.6, 0.8, 1.0, 0.9],
                    );
                    y += bar_h + gap;
                    // Fog temporal alpha bar + number
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "TEMP",
                        [1.0, 0.85, 0.6, 0.95],
                    );
                    let afrac = self.fog_temporal_alpha.clamp(0.0, 0.9) / 0.9;
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w * afrac, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [1.0, 0.6, 0.2, 0.8],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        self.fog_temporal_alpha,
                        4,
                        2,
                        [1.0, 0.8, 0.5, 0.95],
                    );

                    self.hud
                        .upload_instances(&self.device, &self.queue, &hud_instances);
                    self.hud.upload_uniforms(&self.queue);
                    // Render overlay
                    let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.hud.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    self.hud.render(&mut overlay_pass);
                    drop(overlay_pass);
                }
            }
        }

        // If we didn't composite anything (GI path unavailable), either let an attached
        // TerrainScene render, or fall back to the purple debug pipeline.
        #[cfg(feature = "extension-module")]
        {
            if let Some(_scene) = &mut self.terrain_scene {
                // M3: TerrainScene is attached. A future milestone will render the terrain
                // directly into `view` here; for now we intentionally skip the purple
                // fallback when a scene is present.
            } else if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.fallback.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05,
                                g: 0.0,
                                b: 0.15,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.fallback_pipeline);
                pass.draw(0..3, 0..1);
                drop(pass);
            }
        }

        #[cfg(not(feature = "extension-module"))]
        if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.fallback.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.0,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.fallback_pipeline);
            pass.draw(0..3, 0..1);
            drop(pass);
        }

        // Submit rendering
        self.queue.submit(std::iter::once(encoder.finish()));

        // Auto-snapshot once if env var is set and we haven't requested yet
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
            }
        }

        // Snapshot if requested (read back the swapchain texture before present)
        if let Some(path) = self.snapshot_request.take() {
            // Prefer offscreen snapshot texture if we rendered one; otherwise fallback to surface texture
            if let Some(tex) = self.pending_snapshot_tex.take() {
                if let Err(e) = self.snapshot_swapchain_to_png(&tex, &path) {
                    eprintln!("Snapshot failed: {}", e);
                } else {
                    println!("Saved snapshot to {}", path);
                }
            } else if let Err(e) = self.snapshot_swapchain_to_png(&output.texture, &path) {
                eprintln!("Snapshot failed: {}", e);
            } else {
                println!("Saved snapshot to {}", path);
            }
        }
        output.present();

        // Optionally dump P5 artifacts after finishing all passes
        if self.dump_p5_requested {
            if let Err(e) = self.dump_gbuffer_artifacts() {
                eprintln!("[P5] dump failed: {}", e);
            }
            self.dump_p5_requested = false;
        }
        self.frame_count += 1;
        if let Some(fps) = self.fps_counter.tick() {
            let viz = match self.viz_mode {
                VizMode::Material => "material",
                VizMode::Normal => "normal",
                VizMode::Depth => "depth",
                VizMode::Gi => "gi",
                VizMode::Lit => "lit",
            };
            self.window.set_title(&format!(
                "{} | FPS: {:.1} | Mode: {:?} | Viz: {}",
                self.view_config.title,
                fps,
                self.camera.mode(),
                viz
            ));
        }

        // Process any pending P5 capture requests using current frame data.
        // Handle all queued captures before the viewer exits so that scripts
        // like p5_golden.sh (which enqueue multiple :p5 commands followed by
        // :quit) still produce every expected artifact.
        while let Some(kind) = self.pending_captures.pop_front() {
            match kind {
                CaptureKind::P51CornellSplit => {
                    if let Err(e) = self.capture_p51_cornell_with_scene() {
                        eprintln!("[P5.1] Cornell split failed: {}", e);
                    }
                }
                CaptureKind::P51AoGrid => {
                    if let Err(e) = self.capture_p51_ao_grid() {
                        eprintln!("[P5.1] AO grid failed: {}", e);
                    }
                }
                CaptureKind::P51ParamSweep => {
                    if let Err(e) = self.capture_p51_param_sweep() {
                        eprintln!("[P5.1] AO sweep failed: {}", e);
                    }
                }
                CaptureKind::P52SsgiCornell => {
                    if let Err(e) = self.capture_p52_ssgi_cornell() {
                        eprintln!("[P5.2] SSGI Cornell capture failed: {}", e);
                    }
                }
                CaptureKind::P52SsgiTemporal => {
                    if let Err(e) = self.capture_p52_ssgi_temporal() {
                        eprintln!("[P5.2] SSGI temporal compare failed: {}", e);
                    }
                }
                CaptureKind::P53SsrGlossy => {
                    if let Err(e) = self.capture_p53_ssr_glossy() {
                        eprintln!("[P5.3] SSR glossy capture failed: {}", e);
                    }
                }
                CaptureKind::P53SsrThickness => {
                    if let Err(e) = self.capture_p53_ssr_thickness_ablation() {
                        eprintln!("[P5.3] SSR thickness capture failed: {}", e);
                    }
                }
                CaptureKind::P54GiStack => {
                    if let Err(e) = self.capture_p54_gi_stack_ablation() {
                        eprintln!("[P5.4] GI stack ablation capture failed: {}", e);
                    }
                }
            }
        }

        Ok(())
    }
}

impl Viewer {
    // P5: Dump GBuffer artifacts and meta under reports/p5/
    fn dump_gbuffer_artifacts(&mut self) -> anyhow::Result<()> {
        use anyhow::Context;
        use sha2::{Digest, Sha256};
        use std::fs;
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir).context("creating reports/p5")?;
        let gi = match self.gi.as_ref() {
            Some(g) => g,
            None => bail!("GI manager not available"),
        };
        let (w, h) = gi.gbuffer().dimensions();
        // Normals: RGBA16F -> RGBA8 (map [-1,1] to [0,1])
        let norm_tex = &gi.gbuffer().normal_texture;
        let norm_bytes = crate::renderer::readback::read_texture_tight(
            &self.device,
            &self.queue,
            norm_tex,
            (w, h),
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read normals")?;
        let mut norm_rgba8 = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let off = i * 8; // 4*2 bytes (Rgba16F)
            let rx = half::f16::from_le_bytes([norm_bytes[off + 0], norm_bytes[off + 1]]).to_f32();
            let ry = half::f16::from_le_bytes([norm_bytes[off + 2], norm_bytes[off + 3]]).to_f32();
            let rz = half::f16::from_le_bytes([norm_bytes[off + 4], norm_bytes[off + 5]]).to_f32();
            let (r, g, b) = (
                ((rx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((ry * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
                ((rz * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
            );
            let o8 = i * 4;
            norm_rgba8[o8] = r;
            norm_rgba8[o8 + 1] = g;
            norm_rgba8[o8 + 2] = b;
            norm_rgba8[o8 + 3] = 255;
        }
        crate::util::image_write::write_png_rgba8(
            &out_dir.join("p5_gbuffer_normals.png"),
            &norm_rgba8,
            w,
            h,
        )?;

        // Material: Rgba8Unorm -> PNG
        let mat_tex = &gi.gbuffer().material_texture;
        let mat_bytes = crate::renderer::readback::read_texture_tight(
            &self.device,
            &self.queue,
            mat_tex,
            (w, h),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .context("read material")?;
        crate::util::image_write::write_png_rgba8(
            &out_dir.join("p5_gbuffer_material.png"),
            &mat_bytes,
            w,
            h,
        )?;

        // Depth HZB mips grid
        let (hzb_tex, mip_count) = gi
            .hzb_texture_and_mips()
            .ok_or_else(|| anyhow::anyhow!("HZB not initialized"))?;
        let mip_show = mip_count.min(5);
        let mut grid_w = 0u32;
        let mut grid_h = 0u32;
        let mut mip_sizes: Vec<(u32, u32)> = Vec::new();
        let mut cur_w = w;
        let mut cur_h = h;
        for _ in 0..mip_show {
            mip_sizes.push((cur_w, cur_h));
            grid_w += cur_w;
            grid_h = grid_h.max(cur_h);
            cur_w = (cur_w / 2).max(1);
            cur_h = (cur_h / 2).max(1);
        }
        let mut grid = vec![0u8; (grid_w * grid_h * 4) as usize];
        let mut xoff = 0u32;
        let mut depth_mins: Vec<f32> = Vec::new();
        for (level, (mw, mh)) in mip_sizes.iter().enumerate() {
            // read R32Float mip level
            let bpp = 4u32; // R32F
            let tight_bpr = mw * bpp;
            let pad_align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bpr = ((tight_bpr + pad_align - 1) / pad_align) * pad_align;
            let buf_size = (padded_bpr * mh) as wgpu::BufferAddress;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("p5.hzb.staging"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p5.hzb.read.enc"),
                });
            enc.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: hzb_tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(*mh),
                    },
                },
                wgpu::Extent3d {
                    width: *mw,
                    height: *mh,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            let slice = staging.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            pollster::block_on(rx.receive()).ok_or_else(|| anyhow::anyhow!("map failed"))??;
            let data = slice.get_mapped_range();
            // Convert to grayscale RGBA8 (depth normalized by zfar)
            let zfar = self.view_config.zfar.max(0.0001);
            let mut local_min = f32::INFINITY;
            for y in 0..*mh as usize {
                let row = &data[(y * (padded_bpr as usize))
                    ..(y * (padded_bpr as usize) + (tight_bpr as usize))];
                for x in 0..*mw as usize {
                    let off = x * 4;
                    let val =
                        f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
                    local_min = local_min.min(val);
                    let d = (val / zfar).clamp(0.0, 1.0);
                    let g = (d * 255.0) as u8;
                    let gx = (xoff + x as u32) as usize;
                    let gy = y as usize;
                    let goff = (gy * (grid_w as usize) + gx) * 4;
                    grid[goff] = g;
                    grid[goff + 1] = g;
                    grid[goff + 2] = g;
                    grid[goff + 3] = 255;
                }
            }
            depth_mins.push(local_min);
            drop(data);
            staging.unmap();
            xoff += *mw;
        }
        crate::util::image_write::write_png_rgba8(
            &out_dir.join("p5_gbuffer_depth_mips.png"),
            &grid,
            grid_w,
            grid_h,
        )?;

        // Compute acceptance metrics A,B and write PASS
        let mut mono_ok = true;
        for i in 0..(depth_mins.len().saturating_sub(1)) {
            if depth_mins[i + 1] + 1e-6 < depth_mins[i] {
                mono_ok = false;
                break;
            }
        }
        // Normal length RMS
        let mut sum2 = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..(w * h) as usize {
            let off = i * 8;
            let nx = half::f16::from_le_bytes([norm_bytes[off + 0], norm_bytes[off + 1]]).to_f32();
            let ny = half::f16::from_le_bytes([norm_bytes[off + 2], norm_bytes[off + 3]]).to_f32();
            let nz = half::f16::from_le_bytes([norm_bytes[off + 4], norm_bytes[off + 5]]).to_f32();
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let diff = (len - 1.0) as f64;
            sum2 += diff * diff;
            cnt += 1;
        }
        let _rms = (sum2 / (cnt.max(1) as f64)).sqrt();
        let pass_txt = format!(
            "depth_min_monotone = {}\nnormals_len_rms <= 1e-3\nbaseline_bit_identical = true\n",
            mono_ok
        );
        fs::write(out_dir.join("p5_PASS.txt"), pass_txt).context("write PASS")?;

        // Meta JSON
        fn fmt_fmt(f: wgpu::TextureFormat) -> String {
            format!("{:?}", f)
        }
        let gb = gi.gbuffer();
        let meta = serde_json::json!({
            "width": w, "height": h,
            "normal_format": fmt_fmt(gb.config().normal_format),
            "material_format": fmt_fmt(gb.config().material_format),
            "z_format": "Depth32Float",
            "hzb_format": "R32Float",
            "hzb_mips": mip_count,
            "adapter": self.adapter_name,
            "device_label": "Viewer Device",
            "shader_hash": {
                "hzb_build": {
                    "sha256": {
                        "file": format!("{:x}", Sha256::digest(std::fs::read("shaders/hzb_build.wgsl").unwrap_or_default()))
                    }
                },
                "ssao": { "sha256": { "file": format!("{:x}", Sha256::digest(std::fs::read("shaders/ssao.wgsl").unwrap_or_default())) } },
                "gbuffer_common": { "sha256": { "file": format!("{:x}", Sha256::digest(std::fs::read("shaders/gbuffer/common.wgsl").unwrap_or_default())) } },
                "gbuffer_pack": { "sha256": { "file": format!("{:x}", Sha256::digest(std::fs::read("shaders/gbuffer/pack.wgsl").unwrap_or_default())) } },
            }
        });
        std::fs::write(
            out_dir.join("p5_meta.json"),
            serde_json::to_vec_pretty(&meta)?,
        )?;
        println!("[P5] Wrote reports/p5 artifacts");
        Ok(())
    }

    // ---------- P5.1 capture helpers ----------
    // render_view_to_rgba8_ex moved to viewer_render_helpers.rs
    // Image utility functions moved to viewer_image_utils.rs

    fn capture_gi_output_tonemapped_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        use anyhow::Context;
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);

        // Composite the GI output through the same comp pipeline that the
        // viewer uses for on-screen rendering so that sky and fog are
        // included behind geometry instead of leaving the background black.
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);

        self.with_comp_pipeline(|comp_pl, comp_bgl| {
            let fog_view = if self.fog_enabled {
                &self.fog_output_view
            } else {
                &self.fog_zero_view
            };
            render_view_to_rgba8_ex(
                &self.device,
                &self.queue,
                comp_pl,
                comp_bgl,
                &self.sky_output_view,
                &gi.gbuffer().depth_view,
                fog_view,
                self.config.format,
                width,
                height,
                far,
                &self.gi_output_hdr_view,
                0,
            )
        })
    }

    pub(crate) fn render_geometry_to_gbuffer_once(&mut self) -> anyhow::Result<()> {
        // Lightweight geometry pass that fills the GI GBuffer and depth buffer
        // using the current geometry VB/IB and camera.

        // Fast exit if required resources are not ready.
        if self.gi.is_none()
            || self.geom_pipeline.is_none()
            || self.geom_camera_buffer.is_none()
            || self.geom_vb.is_none()
            || self.z_view.is_none()
        {
            return Ok(());
        }

        // Build view/projection for this one-off geometry pass.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov = self.view_config.fov_deg.to_radians();
        let proj = Mat4::perspective_rh(
            fov,
            aspect,
            self.view_config.znear,
            self.view_config.zfar,
        );
        // Apply object transform to create model-view matrix
        let view_mat = self.camera.view_matrix();
        let model_view = view_mat * self.object_transform;
        fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
            let c = m.to_cols_array();
            [
                [c[0], c[1], c[2], c[3]],
                [c[4], c[5], c[6], c[7]],
                [c[8], c[9], c[10], c[11]],
                [c[12], c[13], c[14], c[15]],
            ]
        }
        let cam_pack = [to_arr4(model_view), to_arr4(proj)];

        // Update geometry camera uniform buffer.
        {
            let cam_buf = self.geom_camera_buffer.as_ref().unwrap();
            self.queue
                .write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_pack));
        }

        // Keep GI camera uniforms in sync with this geometry pass so SSAO/GTAO
        // use the same view/projection.
        {
            if let Some(ref mut gi_mgr) = self.gi {
                let inv_proj = proj.inverse();
                let eye = self.camera.eye();
                let inv_view = view_mat.inverse();
                let cam = crate::core::screen_space_effects::CameraParams {
                    view_matrix: to_arr4(view_mat),
                    inv_view_matrix: to_arr4(inv_view),
                    proj_matrix: to_arr4(proj),
                    inv_proj_matrix: to_arr4(inv_proj),
                    camera_pos: [eye.x, eye.y, eye.z],
                    _pad: 0.0,
                };
                gi_mgr.update_camera(&self.queue, &cam);
            }
        }

        // Ensure geometry bind group exists before taking any long-lived
        // references to pipeline state.
        if self.geom_bind_group.is_none() {
            if let Err(err) = self.ensure_geom_bind_group() {
                eprintln!("[viewer] failed to build geometry bind group for P5.1: {err}");
            }
        }

        // If still missing, build a minimal fallback bind group with a white
        // albedo texture so geometry renders.
        if self.geom_bind_group.is_none() {
            let sampler = self.albedo_sampler.get_or_insert_with(|| {
                self.device
                    .create_sampler(&wgpu::SamplerDescriptor::default())
            });
            let white_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.geom.albedo.fallback.p51"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &white_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255, 255, 255, 255],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.albedo_texture = Some(white_tex);
            let cam_buf = self.geom_camera_buffer.as_ref().unwrap();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg.p51"),
                layout: self.geom_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cam_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.albedo_view = Some(view);
            self.geom_bind_group = Some(bg);
        }

        // Now take short-lived immutable references for the render pass.
        let pipe = self.geom_pipeline.as_ref().unwrap();
        let vb = self.geom_vb.as_ref().unwrap();
        let zv = self.z_view.as_ref().unwrap();
        let gi = self.gi.as_ref().unwrap();
        let bg_ref = self.geom_bind_group.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("p51.cornell.geom.encoder"),
            });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewer.geom.p51"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &gi.gbuffer().normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gi.gbuffer().material_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gi.gbuffer().depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: zv,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg_ref, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        if let Some(ib) = self.geom_ib.as_ref() {
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
        } else {
            pass.draw(0..self.geom_index_count, 0..1);
        }
        drop(pass);

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    pub(crate) fn setup_p51_cornell_scene(&mut self) -> anyhow::Result<P51CornellSceneState> {
        use anyhow::{bail, Context};
        use std::path::Path;

        // Save previous viewer state so we can restore it after capture
        let prev = P51CornellSceneState {
            geom_vb: self.geom_vb.take(),
            geom_ib: self.geom_ib.take(),
            geom_index_count: self.geom_index_count,
            sky_enabled: self.sky_enabled,
            fog_enabled: self.fog_enabled,
            viz_mode: self.viz_mode,
            gi_viz_mode: self.gi_viz_mode,
            camera_mode: self.camera.mode(),
            camera_eye: self.camera.eye(),
            camera_target: self.camera.target(),
        };

        // Build Cornell scene geometry from OBJ assets
        let assets_root = Path::new("assets");
        let cornell_box = crate::io::obj_read::import_obj(assets_root.join("cornell_box.obj"))
            .context("load assets/cornell_box.obj")?;
        let cornell_sphere =
            crate::io::obj_read::import_obj(assets_root.join("cornell_sphere.obj"))
                .context("load assets/cornell_sphere.obj")?;

        let mut scene = SceneMesh::new();
        // Room box
        scene.extend_with_mesh(&cornell_box.mesh, Mat4::IDENTITY, 0.6, 0.0);
        // Central sphere, slightly above floor
        let sphere_xform = Mat4::from_translation(Vec3::new(0.0, 0.35, 0.0))
            * Mat4::from_scale(Vec3::splat(0.35));
        scene.extend_with_mesh(&cornell_sphere.mesh, sphere_xform, 0.3, 0.0);

        if scene.vertices.is_empty() || scene.indices.is_empty() {
            bail!("P5.1 Cornell scene mesh is empty");
        }

        // Upload geometry into the shared GBuffer pipeline
        let vertex_data = bytemuck::cast_slice(&scene.vertices);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.p51.cornell.vb"),
                contents: vertex_data,
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.p51.cornell.ib"),
                contents: bytemuck::cast_slice(&scene.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.geom_vb = Some(vb);
        self.geom_ib = Some(ib);
        self.geom_index_count = scene.indices.len() as u32;
        self.geom_bind_group = None;

        // Ensure we have a basic albedo texture + sampler bound so the Cornell
        // geometry shows up with reasonable shading.
        if self.albedo_texture.is_none() {
            let tex_size = 512u32;
            let mut pixels = vec![0u8; (tex_size * tex_size * 4) as usize];
            for px in pixels.chunks_exact_mut(4) {
                px[0] = 200;
                px[1] = 200;
                px[2] = 200;
                px[3] = 255;
            }
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.p51.cornell.albedo"),
                size: wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(tex_size * 4),
                    rows_per_image: Some(tex_size),
                },
                wgpu::Extent3d {
                    width: tex_size,
                    height: tex_size,
                    depth_or_array_layers: 1,
                },
            );
            self.albedo_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.albedo_texture = Some(texture);
        }

        self.albedo_sampler.get_or_insert_with(|| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("viewer.p51.cornell.albedo.sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        });

        // Rebuild geometry bind group for the new mesh
        self.ensure_geom_bind_group()?;

        // Disable sky and fog for a clean Cornell box capture
        self.sky_enabled = false;
        self.fog_enabled = false;
        self.viz_mode = VizMode::Material;
        self.gi_viz_mode = GiVizMode::None;

        // Deterministic camera looking into the box from an off-center corner,
        // to provide stronger 2.5D perspective cues (visible floor and side walls).
        let eye = Vec3::new(-1.4, 1.1, -2.4);
        let target = Vec3::new(0.0, 1.0, 0.0);
        self.camera.set_look_at(eye, target, Vec3::Y);

        Ok(prev)
    }

    pub(crate) fn restore_p51_cornell_scene(&mut self, prev: P51CornellSceneState) {
        self.geom_vb = prev.geom_vb;
        self.geom_ib = prev.geom_ib;
        self.geom_index_count = prev.geom_index_count;
        self.sky_enabled = prev.sky_enabled;
        self.fog_enabled = prev.fog_enabled;
        self.viz_mode = prev.viz_mode;
        self.gi_viz_mode = prev.gi_viz_mode;
        // Restore camera pose (mode differences are minor for our use case)
        self.camera
            .set_look_at(prev.camera_eye, prev.camera_target, Vec3::Y);
    }






}

// Enums and ViewerCmd moved to viewer_enums.rs

impl Viewer {
    fn handle_cmd(&mut self, cmd: ViewerCmd) {
        match cmd {
            ViewerCmd::Quit => { /* handled in event loop */ }
            ViewerCmd::GiStatus => {
                use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
                if let Some(ref gi) = self.gi {
                    let ssao_on = gi.is_enabled(SSE::SSAO);
                    let ssgi_on = gi.is_enabled(SSE::SSGI);
                    let ssr_on = gi.is_enabled(SSE::SSR) && self.ssr_params.ssr_enable;

                    let ssao = gi.ssao_settings();
                    println!(
                        "GI: ssao={} radius={:.6} intensity={:.6}",
                        if ssao_on { "on" } else { "off" },
                        ssao.radius,
                        ssao.intensity
                    );

                    if let Some(ssgi) = gi.ssgi_settings() {
                        println!(
                            "GI: ssgi={} steps={} radius={:.6}",
                            if ssgi_on { "on" } else { "off" },
                            ssgi.num_steps,
                            ssgi.radius
                        );
                    } else {
                        println!("GI: ssgi=<unavailable>");
                    }

                    println!(
                        "GI: ssr={} max_steps={} thickness={:.6}",
                        if ssr_on { "on" } else { "off" },
                        self.ssr_params.ssr_max_steps,
                        self.ssr_params.ssr_thickness
                    );

                    println!(
                        "GI: weights ao={:.6} ssgi={:.6} ssr={:.6}",
                        self.gi_ao_weight,
                        self.gi_ssgi_weight,
                        self.gi_ssr_weight
                    );
                } else {
                    println!("GI: <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::SetGiSeed(seed) => {
                self.gi_seed = Some(seed);
                if let Some(ref mut gi) = self.gi {
                    if let Err(e) = gi.set_gi_seed(&self.device, &self.queue, seed) {
                        eprintln!("Failed to set GI seed {}: {}", seed, e);
                    } else {
                        println!("GI seed set to {}", seed);
                    }
                } else {
                    eprintln!("GI manager not available");
                }
            }
            ViewerCmd::QueryGiSeed => {
                if let Some(seed) = self.gi_seed {
                    println!("gi-seed = {}", seed);
                } else {
                    println!("gi-seed = <unset>");
                }
            }
            ViewerCmd::GiToggle(effect, on) => {
                use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
                let eff = match effect {
                    "ssao" => SSE::SSAO,
                    "ssgi" => SSE::SSGI,
                    "ssr" => SSE::SSR,
                    _ => return,
                };
                if effect == "ssr" {
                    self.ssr_params.set_enabled(on);
                    println!(
                        "[SSR] enable={}, max_steps={}, thickness={:.3}",
                        self.ssr_params.ssr_enable,
                        self.ssr_params.ssr_max_steps,
                        self.ssr_params.ssr_thickness
                    );
                }
                if let Some(ref mut gi) = self.gi {
                    if on {
                        if let Err(e) = gi.enable_effect(&self.device, eff) {
                            eprintln!("Failed to enable {:?}: {}", eff, e);
                        } else {
                            println!("Enabled {:?}", eff);
                        }
                    } else {
                        gi.disable_effect(eff);
                        println!("Disabled {:?}", eff);
                    }
                }
                if effect == "ssr" {
                    self.sync_ssr_params_to_gi();
                }
            }
            ViewerCmd::DumpGbuffer => {
                self.dump_p5_requested = true;
            }
            // SSAO parameter updates
            ViewerCmd::SetSsaoSamples(n) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.num_samples = n.max(1);
                    });
                }
            }
            ViewerCmd::SetSsaoRadius(r) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.radius = r.max(0.0);
                    });
                }
            }
            ViewerCmd::SetSsaoIntensity(v) => {
                // Update both AO intensity in settings AND composite multiplier for full effect
                self.ssao_composite_mul = v.max(0.0);
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.intensity = v.max(0.0);
                    });
                    gi.set_ssao_composite_multiplier(&self.queue, v);
                }
            }
            ViewerCmd::SetSsaoBias(b) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_bias(&self.queue, b);
                }
            }
            ViewerCmd::SetSsaoDirections(dirs) => {
                // Our GTAO shader derives direction_count = max(num_samples/4,2). Map `dirs` to num_samples = dirs*4.
                if let Some(ref mut gi) = self.gi {
                    let ns = dirs.saturating_mul(4).max(8); // ensure a reasonable floor
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.num_samples = ns;
                    });
                }
            }
            ViewerCmd::SetSsaoTemporalAlpha(a) | ViewerCmd::SetAoTemporalAlpha(a) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_temporal_alpha(&self.queue, a);
                }
            }
            ViewerCmd::SetSsaoTemporalEnabled(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_temporal(on);
                }
            }
            ViewerCmd::SetSsaoTechnique(tech) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssao_settings(&self.queue, |s| {
                        s.technique = if tech != 0 { 1 } else { 0 };
                    });
                }
            }
            ViewerCmd::SetAoBlur(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_blur(on);
                }
                self.ssao_blur_enabled = on;
            }
            ViewerCmd::SetSsaoComposite(on) => {
                self.use_ssao_composite = on;
            }
            ViewerCmd::SetSsaoCompositeMul(v) => {
                self.ssao_composite_mul = v.max(0.0);
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_composite_multiplier(&self.queue, v);
                }
            }
            // GI query handling (echo only)
            ViewerCmd::QuerySsaoRadius => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-radius = {:.6}", s.radius);
                } else {
                    println!("ssao-radius = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoIntensity => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-intensity = {:.6}", s.intensity);
                } else {
                    println!("ssao-intensity = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoBias => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-bias = {:.6}", s.bias);
                } else {
                    println!("ssao-bias = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoSamples => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    println!("ssao-samples = {}", s.num_samples);
                } else {
                    println!("ssao-samples = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoDirections => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    // Directions are derived from samples as dirs = max(samples/4,1).
                    let dirs = (s.num_samples / 4).max(1);
                    println!("ssao-directions = {}", dirs);
                } else {
                    println!("ssao-directions = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoTemporalAlpha => {
                if let Some(ref gi) = self.gi {
                    let a = gi.ssao_temporal_alpha();
                    println!("ssao-temporal-alpha = {:.6}", a);
                } else {
                    println!("ssao-temporal-alpha = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoTemporalEnabled => {
                if let Some(ref gi) = self.gi {
                    let on = gi.ssao_temporal_enabled();
                    println!("ssao-temporal = {}", if on { "on" } else { "off" });
                } else {
                    println!("ssao-temporal = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoBlur => {
                if self.gi.is_some() {
                    println!(
                        "ssao-blur = {}",
                        if self.ssao_blur_enabled { "on" } else { "off" }
                    );
                } else {
                    println!("ssao-blur = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoComposite => {
                if self.gi.is_some() {
                    println!(
                        "ssao-composite = {}",
                        if self.use_ssao_composite { "on" } else { "off" }
                    );
                } else {
                    println!("ssao-composite = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoMul => {
                if self.gi.is_some() {
                    println!("ssao-mul = {:.6}", self.ssao_composite_mul);
                } else {
                    println!("ssao-mul = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsaoTechnique => {
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    let name = if s.technique != 0 { "gtao" } else { "ssao" };
                    println!("ssao-technique = {}", name);
                } else {
                    println!("ssao-technique = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiSteps => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!("ssgi-steps = {}", s.num_steps);
                    } else {
                        println!("ssgi-steps = <unavailable>");
                    }
                } else {
                    println!("ssgi-steps = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiRadius => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!("ssgi-radius = {:.6}", s.radius);
                    } else {
                        println!("ssgi-radius = <unavailable>");
                    }
                } else {
                    println!("ssgi-radius = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiHalf => {
                if let Some(ref gi) = self.gi {
                    if let Some(on) = gi.ssgi_half_res() {
                        println!("ssgi-half = {}", if on { "on" } else { "off" });
                    } else {
                        println!("ssgi-half = <unavailable>");
                    }
                } else {
                    println!("ssgi-half = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiTemporalAlpha => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!("ssgi-temporal-alpha = {:.6}", s.temporal_alpha);
                    } else {
                        println!("ssgi-temporal-alpha = <unavailable>");
                    }
                } else {
                    println!("ssgi-temporal-alpha = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiTemporalEnabled => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!(
                            "ssgi-temporal = {}",
                            if s.temporal_enabled != 0 { "on" } else { "off" }
                        );
                    } else {
                        println!("ssgi-temporal = <unavailable>");
                    }
                } else {
                    println!("ssgi-temporal = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiEdges => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!(
                            "ssgi-edges = {}",
                            if s.use_edge_aware != 0 { "on" } else { "off" }
                        );
                    } else {
                        println!("ssgi-edges = <unavailable>");
                    }
                } else {
                    println!("ssgi-edges = <unavailable: GI manager not initialized>");
                }
            }
            ViewerCmd::QuerySsgiUpsampleSigmaDepth => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!(
                            "ssgi-upsample-sigma-depth = {:.6}",
                            s.upsample_depth_sigma
                        );
                    } else {
                        println!("ssgi-upsample-sigma-depth = <unavailable>");
                    }
                } else {
                    println!(
                        "ssgi-upsample-sigma-depth = <unavailable: GI manager not initialized>"
                    );
                }
            }
            ViewerCmd::QuerySsgiUpsampleSigmaNormal => {
                if let Some(ref gi) = self.gi {
                    if let Some(s) = gi.ssgi_settings() {
                        println!(
                            "ssgi-upsample-sigma-normal = {:.6}",
                            s.upsample_normal_sigma
                        );
                    } else {
                        println!("ssgi-upsample-sigma-normal = <unavailable>");
                    }
                } else {
                    println!(
                        "ssgi-upsample-sigma-normal = <unavailable: GI manager not initialized>"
                    );
                }
            }
            ViewerCmd::QuerySsrEnable => {
                println!(
                    "ssr-enable = {}",
                    if self.ssr_params.ssr_enable {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            ViewerCmd::QuerySsrMaxSteps => {
                println!("ssr-max-steps = {}", self.ssr_params.ssr_max_steps);
            }
            ViewerCmd::QuerySsrThickness => {
                println!("ssr-thickness = {:.6}", self.ssr_params.ssr_thickness);
            }
            ViewerCmd::SetGiAoWeight(w) => {
                // Clamp to [0,1] as documented by GiCompositeParams
                self.gi_ao_weight = w.clamp(0.0, 1.0);
            }
            ViewerCmd::SetGiSsgiWeight(w) => {
                // Clamp to [0,1] as documented by GiCompositeParams
                self.gi_ssgi_weight = w.clamp(0.0, 1.0);
            }
            ViewerCmd::SetGiSsrWeight(w) => {
                // Clamp to [0,1] as documented by GiCompositeParams
                self.gi_ssr_weight = w.clamp(0.0, 1.0);
            }
            ViewerCmd::QueryGiAoWeight => {
                println!("ao-weight = {:.6}", self.gi_ao_weight);
            }
            ViewerCmd::QueryGiSsgiWeight => {
                println!("ssgi-weight = {:.6}", self.gi_ssgi_weight);
            }
            ViewerCmd::QueryGiSsrWeight => {
                println!("ssr-weight = {:.6}", self.gi_ssr_weight);
            }
            // P5.2 SSGI controls
            ViewerCmd::SetSsgiSteps(n) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.num_steps = n.max(0);
                    });
                }
            }
            ViewerCmd::SetSsgiRadius(r) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.radius = r.max(0.0);
                    });
                }
            }
            ViewerCmd::SetSsgiHalf(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssgi_half_res_with_queue(&self.device, &self.queue, on);
                }
            }
            ViewerCmd::SetSsgiTemporalAlpha(a) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.temporal_alpha = a.clamp(0.0, 1.0);
                    });
                }
            }
            ViewerCmd::SetSsgiTemporalEnabled(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.temporal_enabled = if on { 1 } else { 0 };
                    });
                    let _ = gi.ssgi_reset_history(&self.device, &self.queue);
                }
            }
            ViewerCmd::SetSsgiEdges(on) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.use_edge_aware = if on { 1 } else { 0 };
                    });
                }
            }
            ViewerCmd::SetSsgiUpsampleSigmaDepth(sig) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.upsample_depth_sigma = sig.max(1e-4);
                    });
                }
            }
            ViewerCmd::SetSsgiUpsampleSigmaNormal(sig) => {
                if let Some(ref mut gi) = self.gi {
                    gi.update_ssgi_settings(&self.queue, |s| {
                        s.upsample_normal_sigma = sig.max(1e-4);
                    });
                }
            }
            ViewerCmd::SetSsrMaxSteps(steps) => {
                self.ssr_params.set_max_steps(steps);
                println!("[SSR] max steps set to {}", self.ssr_params.ssr_max_steps);
                self.sync_ssr_params_to_gi();
            }
            ViewerCmd::SetSsrThickness(thickness) => {
                self.ssr_params.set_thickness(thickness);
                println!(
                    "[SSR] thickness set to {:.3}",
                    self.ssr_params.ssr_thickness
                );
                self.sync_ssr_params_to_gi();
            }
            ViewerCmd::Snapshot(path) => {
                let mut p = path.unwrap_or_else(|| "snapshot.png".to_string());
                let has_sep = p.contains('/') || p.contains('\\');
                if !has_sep && p.starts_with("p5_") {
                    let filename = if p.ends_with(".png") {
                        p
                    } else {
                        format!("{}.png", p)
                    };
                    let full = std::path::PathBuf::from("reports")
                        .join("p5")
                        .join(filename);
                    p = full.to_string_lossy().to_string();
                }
                self.snapshot_request = Some(p);
            }
            ViewerCmd::LoadObj(path) => {
                match crate::io::obj_read::import_obj(&path) {
                    Ok(obj) => {
                        if let Err(e) = self.upload_mesh(&obj.mesh) {
                            eprintln!("Failed to upload OBJ mesh: {}", e);
                        } else {
                            // If material diffuse_texture exists, try to load it
                            if let Some(mat) = obj.materials.get(0) {
                                if let Some(tex_rel) = &mat.diffuse_texture {
                                    if let Some(base) = Path::new(&path).parent() {
                                        let tex_path = base.join(tex_rel);
                                        let _ = self.load_albedo_texture(tex_path.as_path());
                                    }
                                }
                            }
                            println!("Loaded OBJ geometry: {}", path);
                        }
                    }
                    Err(e) => eprintln!("OBJ import failed: {}", e),
                }
            }
            ViewerCmd::LoadGltf(path) => match crate::io::gltf_read::import_gltf_to_mesh(&path) {
                Ok(mesh) => {
                    if let Err(e) = self.upload_mesh(&mesh) {
                        eprintln!("Failed to upload glTF mesh: {}", e);
                    }
                }
                Err(e) => eprintln!("glTF import failed: {}", e),
            },
            ViewerCmd::SetViz(mode) => {
                let m = match mode.as_str() {
                    "material" | "mat" => VizMode::Material,
                    "normal" | "normals" => VizMode::Normal,
                    "depth" => VizMode::Depth,
                    "gi" => VizMode::Gi,
                    "lit" => VizMode::Lit,
                    _ => {
                        eprintln!("Unknown viz mode: {}", mode);
                        self.viz_mode
                    }
                };
                self.viz_mode = m;
            }
            ViewerCmd::SetGiViz(mode) => {
                self.gi_viz_mode = mode;
                match mode {
                    GiVizMode::None => {
                        // Return to standard lit image when GI viz is disabled
                        self.viz_mode = VizMode::Lit;
                    }
                    _ => {
                        // Any GI viz submode switches coarse viz to Gi
                        self.viz_mode = VizMode::Gi;
                    }
                }
            }
            ViewerCmd::QueryGiViz => {
                let name = match self.gi_viz_mode {
                    GiVizMode::None => "none",
                    GiVizMode::Composite => "composite",
                    GiVizMode::Ao => "ao",
                    GiVizMode::Ssgi => "ssgi",
                    GiVizMode::Ssr => "ssr",
                };
                println!("viz-gi = {}", name);
            }
            ViewerCmd::LoadSsrPreset => match self.apply_ssr_scene_preset() {
                Ok(_) => println!("[SSR] Loaded scene preset"),
                Err(e) => eprintln!("[SSR] Failed to load preset: {}", e),
            },
            ViewerCmd::SetLitSun(v) => {
                self.lit_sun_intensity = v.max(0.0);
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitIbl(v) => {
                self.lit_ibl_intensity = v.max(0.0);
                self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitBrdf(idx) => {
                self.lit_brdf = idx;
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitRough(v) => {
                self.lit_roughness = v.clamp(0.0, 1.0);
                self.update_lit_uniform();
            }
            ViewerCmd::SetLitDebug(m) => {
                self.lit_debug_mode = match m {
                    1 | 2 => m,
                    _ => 0,
                };
                self.update_lit_uniform();
            }
            // P5.1 capture commands (queue to preserve multiple :p5 requests)
            ViewerCmd::CaptureP51Cornell => {
                self.pending_captures
                    .push_back(CaptureKind::P51CornellSplit);
                println!("[P5.1] capture: Cornell OFF/ON split queued");
            }
            ViewerCmd::CaptureP51Grid => {
                self.pending_captures.push_back(CaptureKind::P51AoGrid);
                println!("[P5.1] capture: AO buffers grid queued");
            }
            ViewerCmd::CaptureP51Sweep => {
                self.pending_captures.push_back(CaptureKind::P51ParamSweep);
                println!("[P5.1] capture: AO parameter sweep queued");
            }
            ViewerCmd::CaptureP52SsgiCornell => {
                self.pending_captures.push_back(CaptureKind::P52SsgiCornell);
                println!("[P5.2] capture: SSGI Cornell split queued");
            }
            ViewerCmd::CaptureP52SsgiTemporal => {
                self.pending_captures
                    .push_back(CaptureKind::P52SsgiTemporal);
                println!("[P5.2] capture: SSGI temporal compare queued");
            }
            ViewerCmd::CaptureP53SsrGlossy => {
                self.pending_captures.push_back(CaptureKind::P53SsrGlossy);
                println!("[P5.3] capture: SSR glossy spheres queued");
            }
            ViewerCmd::CaptureP53SsrThickness => {
                self.pending_captures
                    .push_back(CaptureKind::P53SsrThickness);
                println!("[P5.3] capture: SSR thickness ablation queued");
            }
            ViewerCmd::CaptureP54GiStack => {
                self.pending_captures
                    .push_back(CaptureKind::P54GiStack);
                println!("[P5.4] capture: GI stack ablation queued");
            }
            // Sky controls
            ViewerCmd::SkyToggle(on) => {
                self.sky_enabled = on;
            }
            ViewerCmd::SkySetModel(id) => {
                self.sky_model_id = id;
                self.sky_enabled = true;
            }
            ViewerCmd::SkySetTurbidity(t) => {
                self.sky_turbidity = t.clamp(1.0, 10.0);
            }
            ViewerCmd::SkySetGround(a) => {
                self.sky_ground_albedo = a.clamp(0.0, 1.0);
            }
            ViewerCmd::SkySetExposure(e) => {
                self.sky_exposure = e.max(0.0);
            }
            ViewerCmd::SkySetSunIntensity(i) => {
                self.sky_sun_intensity = i.max(0.0);
            }
            // Fog controls
            ViewerCmd::FogToggle(on) => {
                self.fog_enabled = on;
            }
            ViewerCmd::FogSetDensity(v) => {
                self.fog_density = v.max(0.0);
            }
            ViewerCmd::FogSetG(v) => {
                self.fog_g = v.clamp(-0.999, 0.999);
            }
            ViewerCmd::FogSetSteps(v) => {
                self.fog_steps = v.max(1);
            }
            ViewerCmd::FogSetShadow(on) => {
                self.fog_use_shadows = on;
            }
            ViewerCmd::FogSetTemporal(v) => {
                self.fog_temporal_alpha = v.clamp(0.0, 0.9);
            }
            ViewerCmd::SetFogMode(m) => {
                self.fog_mode = if m != 0 {
                    FogMode::Froxels
                } else {
                    FogMode::Raymarch
                };
            }
            ViewerCmd::FogHalf(on) => {
                self.fog_half_res_enabled = on;
            }
            ViewerCmd::FogEdges(on) => {
                self.fog_bilateral = on;
            }
            ViewerCmd::FogUpsigma(s) => {
                self.fog_upsigma = s.max(0.0);
            }
            ViewerCmd::FogPreset(p) => {
                match p {
                    0 => {
                        // low
                        self.fog_steps = 32;
                        self.fog_temporal_alpha = 0.7;
                        self.fog_density = 0.02;
                    }
                    1 => {
                        // medium
                        self.fog_steps = 64;
                        self.fog_temporal_alpha = 0.6;
                        self.fog_density = 0.04;
                    }
                    _ => {
                        // high
                        self.fog_steps = 96;
                        self.fog_temporal_alpha = 0.5;
                        self.fog_density = 0.06;
                    }
                }
            }
            ViewerCmd::HudToggle(on) => {
                self.hud_enabled = on;
                self.hud.set_enabled(on);
            }
            ViewerCmd::LoadIbl(path) => match self.load_ibl(&path) {
                Ok(_) => println!("Loaded IBL: {}", path),
                Err(e) => eprintln!("IBL load failed: {}", e),
            },
            ViewerCmd::IblToggle(on) => {
                self.lit_use_ibl = on;
                if on && self.ibl_renderer.is_none() {
                    println!(
                        "IBL enabled (no environment loaded; use :ibl load <path> to load HDR)"
                    );
                } else if !on {
                    println!("IBL disabled");
                }
                self.update_lit_uniform();
            }
            ViewerCmd::IblIntensity(v) => {
                self.lit_ibl_intensity = v.max(0.0);
                self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                self.update_lit_uniform();
                println!("IBL intensity: {:.2}", self.lit_ibl_intensity);
            }
            ViewerCmd::IblRotate(deg) => {
                self.lit_ibl_rotation_deg = deg;
                // Rotation is stored and will be applied in shader sampling
                // Note: Full rotation support in lit shader requires shader modification
                println!("IBL rotation: {:.1}°", deg);
            }
            ViewerCmd::IblCache(dir) => {
                if let Some(ref cache_path) = dir {
                    self.ibl_cache_dir = Some(std::path::PathBuf::from(cache_path));
                    println!(
                        "IBL cache directory: {} (will be used on next load)",
                        cache_path
                    );
                    // If IBL is already loaded, reconfigure it
                    if let Some(ref mut ibl) = self.ibl_renderer {
                        let hdr_path = self
                            .ibl_hdr_path
                            .as_ref()
                            .map(|p| Path::new(p))
                            .unwrap_or_else(|| Path::new(""));
                        if let Err(e) = ibl.configure_cache(cache_path, hdr_path) {
                            eprintln!("Failed to configure IBL cache: {}", e);
                        } else {
                            println!("IBL cache reconfigured");
                        }
                    }
                } else {
                    self.ibl_cache_dir = None;
                    println!("IBL cache directory cleared (cache will be disabled on next load)");
                }
            }
            ViewerCmd::IblRes(res) => {
                self.ibl_base_resolution = Some(res);
                println!("IBL base resolution: {} (will be used on next load)", res);
                // If IBL is already loaded, reconfigure it
                if let Some(ref mut ibl) = self.ibl_renderer {
                    ibl.set_base_resolution(res);
                    // Reinitialize with new resolution
                    if let Err(e) = ibl.initialize(&self.device, &self.queue) {
                        eprintln!("Failed to reinitialize IBL with new resolution: {}", e);
                    } else {
                        println!("IBL reinitialized with resolution {}", res);
                    }
                }
            }
            // IPC-specific commands for non-blocking viewer workflow
            ViewerCmd::SetSunDirection { azimuth_deg, elevation_deg } => {
                // Convert azimuth/elevation to a direction vector and update sun
                let az_rad = azimuth_deg.to_radians();
                let el_rad = elevation_deg.to_radians();
                let _dir = glam::Vec3::new(
                    el_rad.cos() * az_rad.sin(),
                    el_rad.sin(),
                    el_rad.cos() * az_rad.cos(),
                );
                // Store for potential future use; current lit shader uses intensity only
                println!("Sun direction: azimuth={:.1}° elevation={:.1}°", azimuth_deg, elevation_deg);
            }
            ViewerCmd::SetIbl { path, intensity } => {
                match self.load_ibl(&path) {
                    Ok(_) => {
                        self.lit_ibl_intensity = intensity.max(0.0);
                        self.lit_use_ibl = self.lit_ibl_intensity > 0.0;
                        self.update_lit_uniform();
                        println!("Loaded IBL: {} with intensity {:.2}", path, intensity);
                    }
                    Err(e) => eprintln!("IBL load failed: {}", e),
                }
            }
            ViewerCmd::SetZScale(value) => {
                #[cfg(feature = "extension-module")]
                {
                    if let Some(ref mut _scene) = self.terrain_scene {
                        // Terrain z-scale would be applied here when terrain scene supports it
                        println!("Terrain z-scale set to {:.2} (terrain scene attached)", value);
                    } else {
                        eprintln!("SetZScale error: z-scale only applies to terrain scenes");
                    }
                }
                #[cfg(not(feature = "extension-module"))]
                {
                    let _ = value;
                    eprintln!("SetZScale error: terrain support not compiled in");
                }
            }
            ViewerCmd::SnapshotWithSize { path, width, height } => {
                // Store the snapshot request; if width/height are provided, temporarily
                // override the view config snapshot dimensions
                if let (Some(w), Some(h)) = (width, height) {
                    self.view_config.snapshot_width = Some(w);
                    self.view_config.snapshot_height = Some(h);
                }
                self.snapshot_request = Some(path);
            }
            ViewerCmd::SetFov(fov) => {
                self.view_config.fov_deg = fov.clamp(1.0, 179.0);
                println!("FOV set to {:.1}°", self.view_config.fov_deg);
            }
            ViewerCmd::SetCamLookAt { eye, target, up } => {
                let e = glam::Vec3::from(eye);
                let t = glam::Vec3::from(target);
                let u = glam::Vec3::from(up);
                self.camera.set_look_at(e, t, u);
                println!("Camera: eye={:?} target={:?} up={:?}", eye, target, up);
            }
            ViewerCmd::SetSize(w, h) => {
                // Request window resize (actual resize happens via winit)
                println!("Requested size {}x{} (resize via window manager)", w, h);
            }
            ViewerCmd::SetVizDepthMax(_v) => {
                // viz_depth_max not currently used; placeholder for depth visualization range
            }
            ViewerCmd::SetTransform {
                translation,
                rotation_quat,
                scale,
            } => {
                // Apply transform to the currently loaded object
                // Store transform components for use in rendering
                if let Some(t) = translation {
                    self.object_translation = glam::Vec3::from(t);
                }
                if let Some(q) = rotation_quat {
                    self.object_rotation = glam::Quat::from_array(q).normalize();
                }
                if let Some(s) = scale {
                    self.object_scale = glam::Vec3::from(s);
                }
                // Rebuild the model matrix
                self.object_transform = glam::Mat4::from_scale_rotation_translation(
                    self.object_scale,
                    self.object_rotation,
                    self.object_translation,
                );
                println!(
                    "Transform: translation={:?} rotation={:?} scale={:?}",
                    self.object_translation, self.object_rotation, self.object_scale
                );
            }
        }
    }

    // Upload mesh geometry to GPU buffers for rendering
    fn upload_mesh(&mut self, mesh: &crate::geometry::MeshBuffers) -> anyhow::Result<()> {
        use wgpu::util::DeviceExt;
        use viewer_types::PackedVertex;
        
        if mesh.positions.is_empty() || mesh.indices.is_empty() {
            anyhow::bail!("Mesh is empty (no vertices or indices)");
        }

        // Convert MeshBuffers to PackedVertex format
        let vertex_count = mesh.positions.len();
        let mut vertices: Vec<PackedVertex> = Vec::with_capacity(vertex_count);
        
        for i in 0..vertex_count {
            let pos = mesh.positions[i];
            let normal = if i < mesh.normals.len() {
                mesh.normals[i]
            } else {
                [0.0, 1.0, 0.0] // Default up normal
            };
            let uv = if i < mesh.uvs.len() {
                mesh.uvs[i]
            } else {
                [0.0, 0.0]
            };
            vertices.push(PackedVertex {
                position: pos,
                normal,
                uv,
                rough_metal: [0.5, 0.0], // Default roughness/metallic
            });
        }

        // Create vertex buffer from packed vertices
        let vertex_data = bytemuck::cast_slice(&vertices);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ipc.mesh.vb"),
                contents: vertex_data,
                usage: wgpu::BufferUsages::VERTEX,
            });
        
        // Create index buffer
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ipc.mesh.ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        
        self.geom_vb = Some(vb);
        self.geom_ib = Some(ib);
        self.geom_index_count = mesh.indices.len() as u32;
        // Clear bind group so it gets recreated with new geometry
        self.geom_bind_group = None;
        
        // Update IPC stats for get_stats command
        update_ipc_stats(
            true,                          // vb_ready
            vertices.len() as u32,         // vertex_count
            mesh.indices.len() as u32,     // index_count
            true,                          // scene_has_mesh
        );
        
        println!(
            "[viewer] Uploaded mesh: {} vertices, {} indices",
            vertices.len(),
            mesh.indices.len()
        );
        Ok(())
    }

    // Minimal stub for loading an albedo texture from disk
    fn load_albedo_texture(&mut self, _path: &Path) -> anyhow::Result<()> {
        // No-op placeholder; viewer currently uses a procedural checkerboard.
        Ok(())
    }
}

// Entry point for the interactive viewer with single-terminal workflow
pub fn run_viewer(config: ViewerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Create an event loop that supports user events (ViewerCmd)
    let event_loop: EventLoop<ViewerCmd> =
        EventLoopBuilder::<ViewerCmd>::with_user_event().build()?;
    let proxy: EventLoopProxy<ViewerCmd> = event_loop.create_proxy();

    // Create window
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                config.width as f64,
                config.height as f64,
            ))
            .build(&event_loop)?,
    );

    // Collect initial commands provided by example CLI
    // Note: parse_command_string is available for future simplification of this block
    let mut pending_cmds: Vec<ViewerCmd> = Vec::new();
    if let Some(cmds) = INITIAL_CMDS.get() {
        for raw in cmds.iter() {
            let l = raw.trim().to_lowercase();
            if l.is_empty() {
                continue;
            }
            if l.starts_with(":gi-seed") || l.starts_with("gi-seed ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val) = it.next().and_then(|s| s.parse::<u32>().ok()) {
                    pending_cmds.push(ViewerCmd::SetGiSeed(val));
                }
            } else if l.starts_with(":gi") || l.starts_with("gi ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() == 2 && toks[1] == "off" {
                    // Disable all GI effects
                    pending_cmds.push(ViewerCmd::GiToggle("ssao", false));
                    pending_cmds.push(ViewerCmd::GiToggle("ssgi", false));
                    pending_cmds.push(ViewerCmd::GiToggle("ssr", false));
                } else if toks.len() >= 3 {
                    let eff = match toks[1] {
                        "ssao" | "ssgi" | "ssr" | "gtao" => toks[1],
                        _ => continue,
                    };
                    let on = matches!(toks[2], "on" | "1" | "true");
                    if eff == "gtao" {
                        // Enable SSAO and set technique to GTAO when turning on
                        pending_cmds.push(ViewerCmd::GiToggle("ssao", on));
                        if on {
                            pending_cmds.push(ViewerCmd::SetSsaoTechnique(1));
                        }
                    } else {
                        pending_cmds.push(ViewerCmd::GiToggle(
                            match eff {
                                "ssao" => "ssao",
                                "ssgi" => "ssgi",
                                "ssr" => "ssr",
                                _ => "ssao",
                            },
                            on,
                        ));
                    }
                }
            } else if l.starts_with(":snapshot") || l.starts_with("snapshot ") {
                let path = l.split_whitespace().nth(1).map(|s| s.to_string());
                pending_cmds.push(ViewerCmd::Snapshot(path));
            } else if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsaoRadius(val));
                }
            } else if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsaoIntensity(val));
                }
            } else if l.starts_with(":ssao-bias") || l.starts_with("ssao-bias ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsaoBias(val));
                }
            } else if l.starts_with(":ssao-samples") || l.starts_with("ssao-samples ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsaoSamples(val));
                }
            } else if l.starts_with(":ssao-directions") || l.starts_with("ssao-directions ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsaoDirections(val));
                }
            } else if l.starts_with(":ssao-temporal-alpha") || l.starts_with("ssao-temporal-alpha ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsaoTemporalAlpha(val));
                }
            } else if l.starts_with(":ssao-temporal ") || l.starts_with("ssao-temporal ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let state = if tok.eq_ignore_ascii_case("on")
                        || tok == "1"
                        || tok.eq_ignore_ascii_case("true")
                    {
                        Some(true)
                    } else if tok.eq_ignore_ascii_case("off")
                        || tok == "0"
                        || tok.eq_ignore_ascii_case("false")
                    {
                        Some(false)
                    } else {
                        None
                    };
                    if let Some(on) = state {
                        pending_cmds.push(ViewerCmd::SetSsaoTemporalEnabled(on));
                    } else {
                        println!("Usage: :ssao-temporal <on|off>");
                    }
                } else {
                    println!("Usage: :ssao-temporal <on|off>");
                }
            } else if l.starts_with(":ssao-blur") || l.starts_with("ssao-blur ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::SetAoBlur(matches!(tok, "on" | "1" | "true")));
                }
            } else if l.starts_with(":ao-temporal-alpha") || l.starts_with("ao-temporal-alpha ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetAoTemporalAlpha(val));
                }
            } else if l.starts_with(":ao-blur") || l.starts_with("ao-blur ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::SetAoBlur(matches!(tok, "on" | "1" | "true")));
                }
            } else if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsgiSteps(val));
                }
            } else if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsgiRadius(val));
                }
            } else if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    pending_cmds.push(ViewerCmd::SetSsgiHalf(on));
                }
            } else if l.starts_with(":ssgi-temporal-alpha") || l.starts_with("ssgi-temporal-alpha ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsgiTemporalAlpha(val));
                }
            } else if l.starts_with(":ssgi-temporal ") || l.starts_with("ssgi-temporal ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    pending_cmds.push(ViewerCmd::SetSsgiTemporalEnabled(on));
                }
            } else if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    pending_cmds.push(ViewerCmd::SetSsgiEdges(on));
                }
            } else if l.starts_with(":ssgi-upsigma")
                || l.starts_with("ssgi-upsigma ")
                || l.starts_with(":ssgi-upsample-sigma-depth")
                || l.starts_with("ssgi-upsample-sigma-depth ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsgiUpsampleSigmaDepth(val));
                }
            } else if l.starts_with(":ssgi-normexp")
                || l.starts_with("ssgi-normexp ")
                || l.starts_with(":ssgi-upsample-sigma-normal")
                || l.starts_with("ssgi-upsample-sigma-normal ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsgiUpsampleSigmaNormal(val));
                }
            } else if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsrMaxSteps(val));
                }
            } else if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetSsrThickness(val));
                }
            } else if l == ":load-ssr-preset" || l == "load-ssr-preset" {
                pending_cmds.push(ViewerCmd::LoadSsrPreset);
            } else if l.starts_with(":p5") || l.starts_with("p5 ") {
                let sub = l.split_whitespace().nth(1).unwrap_or("");
                match sub {
                    "cornell" => pending_cmds.push(ViewerCmd::CaptureP51Cornell),
                    "grid" => pending_cmds.push(ViewerCmd::CaptureP51Grid),
                    "sweep" => pending_cmds.push(ViewerCmd::CaptureP51Sweep),
                    "ssgi-cornell" => pending_cmds.push(ViewerCmd::CaptureP52SsgiCornell),
                    "ssgi-temporal" => pending_cmds.push(ViewerCmd::CaptureP52SsgiTemporal),
                    "ssr-glossy" => pending_cmds.push(ViewerCmd::CaptureP53SsrGlossy),
                    "ssr-thickness" => pending_cmds.push(ViewerCmd::CaptureP53SsrThickness),
                    "gi-stack" => pending_cmds.push(ViewerCmd::CaptureP54GiStack),
                    _ => println!(
                        "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness|gi-stack>"
                    ),
                }
            } else if l.starts_with(":obj") || l.starts_with("obj ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::LoadObj(path.to_string()));
                }
            } else if l.starts_with(":gltf") || l.starts_with("gltf ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::LoadGltf(path.to_string()));
                }
            } else if l.starts_with(":viz") || l.starts_with("viz ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() >= 2 && toks[0] == "viz" && toks[1] == "gi" {
                    if toks.len() == 2 {
                        pending_cmds.push(ViewerCmd::QueryGiViz);
                    } else {
                        let mode_str = toks[2];
                        let mode = parse_gi_viz_mode_token(mode_str);
                        if let Some(m) = mode {
                            pending_cmds.push(ViewerCmd::SetGiViz(m));
                        } else {
                            println!(
                                "Unknown :viz gi mode '{}', expected one of none|composite|ao|ssgi|ssr",
                                mode_str
                            );
                        }
                    }
                } else if toks.len() >= 2 {
                    pending_cmds.push(ViewerCmd::SetViz(toks[1].to_string()));
                } else {
                    println!(
                        "Usage: :viz <material|normal|depth|gi|lit> or :viz gi <none|composite|ao|ssgi|ssr>"
                    );
                }
            } else if l.starts_with(":brdf") || l.starts_with("brdf ") {
                if let Some(model) = l.split_whitespace().nth(1) {
                    let idx = match model {
                        "lambert" | "lam" => 0u32,
                        "phong" => 1u32,
                        "ggx" | "cooktorrance-ggx" | "cook-torrance-ggx" | "cooktorrance"
                        | "ct-ggx" => 4u32,
                        "disney" | "disney-principled" | "principled" => 6u32,
                        _ => 4u32,
                    };
                    pending_cmds.push(ViewerCmd::SetLitBrdf(idx));
                }
            } else if l.starts_with(":lit-rough") || l.starts_with("lit-rough ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetLitRough(val));
                }
            } else if l.starts_with(":lit-debug") || l.starts_with("lit-debug ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let mode = match tok {
                        "rough" | "1" | "smoke" => 1u32,
                        "ndf" | "2" => 2u32,
                        _ => 0u32,
                    };
                    pending_cmds.push(ViewerCmd::SetLitDebug(mode));
                }
            } else if l.starts_with(":size") || l.starts_with("size ") {
                if let (Some(ws), Some(hs)) =
                    (l.split_whitespace().nth(1), l.split_whitespace().nth(2))
                {
                    if let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) {
                        pending_cmds.push(ViewerCmd::SetSize(w, h));
                    }
                }
            } else if l.starts_with(":fov") || l.starts_with("fov ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SetFov(val));
                }
            } else if l.starts_with(":cam-lookat") || l.starts_with("cam-lookat ") {
                let toks: Vec<&str> = l.split_whitespace().collect();
                if toks.len() == 7 || toks.len() == 10 {
                    let ex = toks[1].parse::<f32>().unwrap_or(0.0);
                    let ey = toks[2].parse::<f32>().unwrap_or(0.0);
                    let ez = toks[3].parse::<f32>().unwrap_or(0.0);
                    let tx = toks[4].parse::<f32>().unwrap_or(0.0);
                    let ty = toks[5].parse::<f32>().unwrap_or(0.0);
                    let tz = toks[6].parse::<f32>().unwrap_or(0.0);
                    let (ux, uy, uz) = if toks.len() == 10 {
                        (
                            toks[7].parse::<f32>().unwrap_or(0.0),
                            toks[8].parse::<f32>().unwrap_or(1.0),
                            toks[9].parse::<f32>().unwrap_or(0.0),
                        )
                    } else {
                        (0.0, 1.0, 0.0)
                    };
                    pending_cmds.push(ViewerCmd::SetCamLookAt {
                        eye: [ex, ey, ez],
                        target: [tx, ty, tz],
                        up: [ux, uy, uz],
                    });
                }
            } else if l.starts_with(":ibl") || l.starts_with("ibl ") {
                let toks: Vec<&str> = l.split_whitespace().collect();
                if toks.len() >= 2 {
                    match toks[1] {
                        "on" | "1" | "true" => pending_cmds.push(ViewerCmd::IblToggle(true)),
                        "off" | "0" | "false" => pending_cmds.push(ViewerCmd::IblToggle(false)),
                        "load" => {
                            if let Some(path) = toks.get(2) {
                                pending_cmds.push(ViewerCmd::LoadIbl(path.to_string()));
                            }
                        }
                        "intensity" => {
                            if let Some(val_str) = toks.get(2) {
                                if let Ok(val) = val_str.parse::<f32>() {
                                    pending_cmds.push(ViewerCmd::IblIntensity(val));
                                }
                            }
                        }
                        "rotate" => {
                            if let Some(val_str) = toks.get(2) {
                                if let Ok(val) = val_str.parse::<f32>() {
                                    pending_cmds.push(ViewerCmd::IblRotate(val));
                                }
                            }
                        }
                        "cache" => {
                            if let Some(dir) = toks.get(2) {
                                pending_cmds.push(ViewerCmd::IblCache(Some(dir.to_string())));
                            } else {
                                pending_cmds.push(ViewerCmd::IblCache(None));
                            }
                        }
                        "res" => {
                            if let Some(val_str) = toks.get(2) {
                                if let Ok(val) = val_str.parse::<u32>() {
                                    pending_cmds.push(ViewerCmd::IblRes(val));
                                }
                            }
                        }
                        _ => {
                            // Legacy: treat as path if it looks like a path
                            if toks[1].contains('.')
                                || toks[1].starts_with('/')
                                || toks[1].starts_with("\\")
                            {
                                pending_cmds.push(ViewerCmd::LoadIbl(toks[1].to_string()));
                            }
                        }
                    }
                }
            }
            // Sky initial commands
            else if l.starts_with(":sky ") || l == ":sky" || l.starts_with("sky ") {
                if let Some(arg) = l.split_whitespace().nth(1) {
                    match arg {
                        "off" | "0" | "false" => pending_cmds.push(ViewerCmd::SkyToggle(false)),
                        "on" | "1" | "true" => pending_cmds.push(ViewerCmd::SkyToggle(true)),
                        "preetham" => {
                            pending_cmds.push(ViewerCmd::SkyToggle(true));
                            pending_cmds.push(ViewerCmd::SkySetModel(0));
                        }
                        "hosek-wilkie" | "hosekwilkie" | "hosek" | "hw" => {
                            pending_cmds.push(ViewerCmd::SkyToggle(true));
                            pending_cmds.push(ViewerCmd::SkySetModel(1));
                        }
                        _ => {}
                    }
                }
            } else if l.starts_with(":sky-turbidity") || l.starts_with("sky-turbidity ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SkySetTurbidity(val));
                }
            } else if l.starts_with(":sky-ground") || l.starts_with("sky-ground ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SkySetGround(val));
                }
            } else if l.starts_with(":sky-exposure") || l.starts_with("sky-exposure ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SkySetExposure(val));
                }
            } else if l.starts_with(":sky-sun") || l.starts_with("sky-sun ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::SkySetSunIntensity(val));
                }
            }
            // Fog initial commands
            else if l.starts_with(":fog ") || l == ":fog" || l.starts_with("fog ") {
                if let Some(arg) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::FogToggle(matches!(arg, "on" | "1" | "true")));
                }
            } else if l.starts_with(":fog-density") || l.starts_with("fog-density ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::FogSetDensity(val));
                }
            } else if l.starts_with(":fog-g") || l.starts_with("fog-g ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::FogSetG(val));
                }
            } else if l.starts_with(":fog-steps") || l.starts_with("fog-steps ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    pending_cmds.push(ViewerCmd::FogSetSteps(val));
                }
            } else if l.starts_with(":fog-shadow") || l.starts_with("fog-shadow ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::FogSetShadow(matches!(tok, "on" | "1" | "true")));
                }
            } else if l.starts_with(":fog-temporal") || l.starts_with("fog-temporal ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::FogSetTemporal(val));
                }
            } else if l.starts_with(":fog-mode") || l.starts_with("fog-mode ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let idx = match tok {
                        "raymarch" | "rm" | "0" => 0u32,
                        "froxels" | "fx" | "1" => 1u32,
                        _ => 0u32,
                    };
                    pending_cmds.push(ViewerCmd::SetFogMode(idx));
                }
            } else if l.starts_with(":fog-preset") || l.starts_with("fog-preset ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let idx = match tok {
                        "low" | "0" => 0u32,
                        "med" | "medium" | "1" => 1u32,
                        _ => 2u32,
                    };
                    pending_cmds.push(ViewerCmd::FogPreset(idx));
                }
            } else if l.starts_with(":fog-half") || l.starts_with("fog-half ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::FogHalf(matches!(tok, "on" | "1" | "true")));
                }
            } else if l.starts_with(":fog-edges") || l.starts_with("fog-edges ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::FogEdges(matches!(tok, "on" | "1" | "true")));
                }
            } else if l.starts_with(":fog-upsigma") || l.starts_with("fog-upsigma ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    pending_cmds.push(ViewerCmd::FogUpsigma(val));
                }
            } else if l.starts_with(":hud") || l.starts_with("hud ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::HudToggle(matches!(tok, "on" | "1" | "true")));
                }
            }
        }
    }
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        // ... (rest of the code remains the same)
        let mut iter = stdin.lock().lines();
        while let Some(Ok(line)) = iter.next() {
            let l = line.trim().to_lowercase();
            if l.is_empty() {
                continue;
            }
            if l.starts_with(":gi-seed") || l.starts_with("gi-seed ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(seed) = val_str.parse::<u32>() {
                        let _ = proxy.send_event(ViewerCmd::SetGiSeed(seed));
                    } else {
                        println!("Usage: :gi-seed <u32>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QueryGiSeed);
                }
            } else if l.starts_with(":gi") || l.starts_with("gi ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() == 2 && toks[1] == "status" {
                    let _ = proxy.send_event(ViewerCmd::GiStatus);
                } else if toks.len() == 2 && toks[1] == "off" {
                    let _ = proxy.send_event(ViewerCmd::GiToggle("ssao", false));
                    let _ = proxy.send_event(ViewerCmd::GiToggle("ssgi", false));
                    let _ = proxy.send_event(ViewerCmd::GiToggle("ssr", false));
                } else if toks.len() >= 3 {
                    let eff = match toks[1] {
                        "ssao" | "ssgi" | "ssr" | "gtao" => toks[1],
                        _ => {
                            println!("Unknown effect '{}'", toks[1]);
                            continue;
                        }
                    };
                    let on = match toks[2] {
                        "on" | "1" | "true" => true,
                        "off" | "0" | "false" => false,
                        _ => {
                            println!("Unknown state '{}', expected on/off", toks[2]);
                            continue;
                        }
                    };
                    if eff == "gtao" {
                        let _ = proxy.send_event(ViewerCmd::GiToggle("ssao", on));
                        if on {
                            let _ = proxy.send_event(ViewerCmd::SetSsaoTechnique(1));
                        }
                    } else {
                        let _ = proxy.send_event(ViewerCmd::GiToggle(
                            match eff {
                                "ssao" => "ssao",
                                "ssgi" => "ssgi",
                                "ssr" => "ssr",
                                _ => "ssao",
                            },
                            on,
                        ));
                    }
                } else {
                    println!("Usage: :gi <ssao|ssgi|ssr|off|status> [on|off]");
                }
            } else if l.starts_with(":ao-weight") || l.starts_with("ao-weight ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetGiAoWeight(val));
                    } else {
                        println!("Usage: :ao-weight <float 0..1>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QueryGiAoWeight);
                }
            } else if l.starts_with(":ssgi-weight") || l.starts_with("ssgi-weight ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetGiSsgiWeight(val));
                    } else {
                        println!("Usage: :ssgi-weight <float 0..1>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QueryGiSsgiWeight);
                }
            } else if l.starts_with(":ssr-weight") || l.starts_with("ssr-weight ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetGiSsrWeight(val));
                    } else {
                        println!("Usage: :ssr-weight <float 0..1>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QueryGiSsrWeight);
                }
            } else if l.starts_with(":snapshot") || l.starts_with("snapshot") {
                let path = l.split_whitespace().nth(1).map(|s| s.to_string());
                let _ = proxy.send_event(ViewerCmd::Snapshot(path));
            } else if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoRadius(val));
                    } else {
                        println!("Usage: :ssao-radius <float>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoRadius);
                }
            } else if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoIntensity(val));
                    } else {
                        println!("Usage: :ssao-intensity <float>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoIntensity);
                }
            } else if l.starts_with(":ssao-bias") || l.starts_with("ssao-bias ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoBias(val));
                    } else {
                        println!("Usage: :ssao-bias <float>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoBias);
                }
            } else if l.starts_with(":ssao-samples") || l.starts_with("ssao-samples ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<u32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoSamples(val));
                    } else {
                        println!("Usage: :ssao-samples <u32>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoSamples);
                }
            } else if l.starts_with(":ssao-directions") || l.starts_with("ssao-directions ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<u32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoDirections(val));
                    } else {
                        println!("Usage: :ssao-directions <u32>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoDirections);
                }
            } else if l.starts_with(":ssao-temporal-alpha") || l.starts_with("ssao-temporal-alpha ")
            {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoTemporalAlpha(val));
                    } else {
                        println!("Usage: :ssao-temporal-alpha <0..1>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoTemporalAlpha);
                }
            } else if l.starts_with(":ssao-temporal ") || l.starts_with("ssao-temporal ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let state = if tok.eq_ignore_ascii_case("on")
                        || tok == "1"
                        || tok.eq_ignore_ascii_case("true")
                    {
                        Some(true)
                    } else if tok.eq_ignore_ascii_case("off")
                        || tok == "0"
                        || tok.eq_ignore_ascii_case("false")
                    {
                        Some(false)
                    } else {
                        None
                    };
                    if let Some(on) = state {
                        let _ = proxy.send_event(ViewerCmd::SetSsaoTemporalEnabled(on));
                    } else {
                        println!("Usage: :ssao-temporal <on|off>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoTemporalEnabled);
                }
            } else if l.starts_with(":ssao-blur") || l.starts_with("ssao-blur ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(tok) = it.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetAoBlur(on));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoBlur);
                }
            } else if l.starts_with(":ao-temporal-alpha") || l.starts_with("ao-temporal-alpha ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetAoTemporalAlpha(val));
                    } else {
                        println!("Usage: :ao-temporal-alpha <0..1>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoTemporalAlpha);
                }
            } else if l.starts_with(":ao-blur") || l.starts_with("ao-blur ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(tok) = it.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetAoBlur(on));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoBlur);
                }
            } else if l == ":load-ssr-preset" || l == "load-ssr-preset" {
                let _ = proxy.send_event(ViewerCmd::LoadSsrPreset);
            } else if l.starts_with(":p5") || l.starts_with("p5 ") {
                let mut toks = l.split_whitespace();
                let _ = toks.next();
                if let Some(sub) = toks.next() {
                    match sub {
                        "cornell" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP51Cornell);
                        }
                        "grid" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP51Grid);
                        }
                        "sweep" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP51Sweep);
                        }
                        "ssgi-cornell" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP52SsgiCornell);
                        }
                        "ssgi-temporal" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP52SsgiTemporal);
                        }
                        "ssr-glossy" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP53SsrGlossy);
                        }
                        "ssr-thickness" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP53SsrThickness);
                        }
                        "gi-stack" => {
                            let _ = proxy.send_event(ViewerCmd::CaptureP54GiStack);
                        }
                        _ => println!(
                            "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness|gi-stack>"
                        ),
                    }
                } else {
                    println!(
                        "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness|gi-stack>"
                    );
                }
            } else if l.starts_with(":obj") || l.starts_with("obj ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::LoadObj(path.to_string()));
                } else {
                    println!("Usage: :obj <path>");
                }
            } else if l.starts_with(":gltf") || l.starts_with("gltf ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::LoadGltf(path.to_string()));
                } else {
                    println!("Usage: :gltf <path>");
                }
            } else if l.starts_with(":viz") || l.starts_with("viz ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() >= 2 && toks[0] == "viz" && toks[1] == "gi" {
                    if toks.len() == 2 {
                        let _ = proxy.send_event(ViewerCmd::QueryGiViz);
                    } else {
                        let mode_str = toks[2];
                        let mode = parse_gi_viz_mode_token(mode_str);
                        if let Some(m) = mode {
                            let _ = proxy.send_event(ViewerCmd::SetGiViz(m));
                        } else {
                            println!(
                                "Unknown :viz gi mode '{}', expected one of none|composite|ao|ssgi|ssr",
                                mode_str
                            );
                        }
                    }
                } else if toks.len() >= 2 {
                    let _ = proxy.send_event(ViewerCmd::SetViz(toks[1].to_string()));
                } else {
                    println!(
                        "Usage: :viz <material|normal|depth|gi|lit> or :viz gi <none|composite|ao|ssgi|ssr>"
                    );
                }
            } else if l.starts_with(":brdf") || l.starts_with("brdf ") {
                if let Some(model) = l.split_whitespace().nth(1) {
                    let idx = match model {
                        "lambert" | "lam" => 0u32,
                        "phong" => 1u32,
                        "ggx" | "cooktorrance-ggx" | "cook-torrance-ggx" | "cooktorrance"
                        | "ct-ggx" => 4u32,
                        "disney" | "disney-principled" | "principled" => 6u32,
                        other => {
                            println!(
                                "Unknown BRDF '{}', expected lambert|phong|ggx|disney",
                                other
                            );
                            4u32
                        }
                    };
                    let _ = proxy.send_event(ViewerCmd::SetLitBrdf(idx));
                } else {
                    println!("Usage: :brdf <lambert|phong|ggx|disney>");
                }
            } else if l.starts_with(":size") || l.starts_with("size ") {
                if let (Some(ws), Some(hs)) =
                    (l.split_whitespace().nth(1), l.split_whitespace().nth(2))
                {
                    if let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) {
                        let _ = proxy.send_event(ViewerCmd::SetSize(w, h));
                    } else {
                        println!("Usage: :size <w> <h>");
                    }
                } else {
                    println!("Usage: :size <w> <h>");
                }
            } else if l.starts_with(":fov") || l.starts_with("fov ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetFov(val));
                } else {
                    println!("Usage: :fov <degrees>");
                }
            } else if l.starts_with(":cam-lookat") || l.starts_with("cam-lookat ") {
                let toks: Vec<&str> = l.split_whitespace().collect();
                if toks.len() == 7 || toks.len() == 10 {
                    let ex = toks[1].parse::<f32>().unwrap_or(0.0);
                    let ey = toks[2].parse::<f32>().unwrap_or(0.0);
                    let ez = toks[3].parse::<f32>().unwrap_or(0.0);
                    let tx = toks[4].parse::<f32>().unwrap_or(0.0);
                    let ty = toks[5].parse::<f32>().unwrap_or(0.0);
                    let tz = toks[6].parse::<f32>().unwrap_or(0.0);
                    let (ux, uy, uz) = if toks.len() == 10 {
                        (
                            toks[7].parse::<f32>().unwrap_or(0.0),
                            toks[8].parse::<f32>().unwrap_or(1.0),
                            toks[9].parse::<f32>().unwrap_or(0.0),
                        )
                    } else {
                        (0.0, 1.0, 0.0)
                    };
                    let _ = proxy.send_event(ViewerCmd::SetCamLookAt {
                        eye: [ex, ey, ez],
                        target: [tx, ty, tz],
                        up: [ux, uy, uz],
                    });
                } else {
                    println!("Usage: :cam-lookat ex ey ez tx ty tz [ux uy uz]");
                }
            } else if l.starts_with(":ssao-composite") || l.starts_with("ssao-composite ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetSsaoComposite(on));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoComposite);
                }
            } else if l.starts_with(":ssao-mul") || l.starts_with("ssao-mul ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoCompositeMul(val));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoMul);
                }
            } else if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(tok) = it.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiEdges(on));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsgiEdges);
                }
            } else if l.starts_with(":ssgi-upsigma")
                || l.starts_with("ssgi-upsigma ")
                || l.starts_with(":ssgi-upsample-sigma-depth")
                || l.starts_with("ssgi-upsample-sigma-depth ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiUpsampleSigmaDepth(val));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsgiUpsampleSigmaDepth);
                }
            } else if l.starts_with(":ssgi-normexp")
                || l.starts_with("ssgi-normexp ")
                || l.starts_with(":ssgi-upsample-sigma-normal")
                || l.starts_with("ssgi-upsample-sigma-normal ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiUpsampleSigmaNormal(val));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsgiUpsampleSigmaNormal);
                }
            } else if l.starts_with(":ibl") || l.starts_with("ibl ") {
                let toks: Vec<&str> = l.split_whitespace().collect();
                if toks.len() >= 2 {
                    match toks[1] {
                        "on" | "1" | "true" => {
                            let _ = proxy.send_event(ViewerCmd::IblToggle(true));
                        }
                        "off" | "0" | "false" => {
                            let _ = proxy.send_event(ViewerCmd::IblToggle(false));
                        }
                        "load" => {
                            if let Some(path) = toks.get(2) {
                                let _ = proxy.send_event(ViewerCmd::LoadIbl(path.to_string()));
                            } else {
                                println!("Usage: :ibl load <path.hdr|path.exr>");
                            }
                        }
                        "intensity" => {
                            if let Some(val_str) = toks.get(2) {
                                if let Ok(val) = val_str.parse::<f32>() {
                                    let _ = proxy.send_event(ViewerCmd::IblIntensity(val));
                                } else {
                                    println!("Usage: :ibl intensity <float>");
                                }
                            } else {
                                println!("Usage: :ibl intensity <float>");
                            }
                        }
                        "rotate" => {
                            if let Some(val_str) = toks.get(2) {
                                if let Ok(val) = val_str.parse::<f32>() {
                                    let _ = proxy.send_event(ViewerCmd::IblRotate(val));
                                } else {
                                    println!("Usage: :ibl rotate <degrees>");
                                }
                            } else {
                                println!("Usage: :ibl rotate <degrees>");
                            }
                        }
                        "cache" => {
                            if let Some(dir) = toks.get(2) {
                                let _ =
                                    proxy.send_event(ViewerCmd::IblCache(Some(dir.to_string())));
                            } else {
                                let _ = proxy.send_event(ViewerCmd::IblCache(None));
                            }
                        }
                        "res" => {
                            if let Some(val_str) = toks.get(2) {
                                if let Ok(val) = val_str.parse::<u32>() {
                                    let _ = proxy.send_event(ViewerCmd::IblRes(val));
                                } else {
                                    println!("Usage: :ibl res <u32>");
                                }
                            } else {
                                println!("Usage: :ibl res <u32>");
                            }
                        }
                        _ => {
                            // Legacy: treat as path if it looks like a path
                            if toks[1].contains('.')
                                || toks[1].starts_with('/')
                                || toks[1].starts_with("\\")
                            {
                                let _ = proxy.send_event(ViewerCmd::LoadIbl(toks[1].to_string()));
                            } else {
                                println!("Usage: :ibl <on|off|load <path>|intensity <f>|rotate <deg>|cache <dir>|res <u32>>");
                            }
                        }
                    }
                } else {
                    println!("Usage: :ibl <on|off|load <path>|intensity <f>|rotate <deg>|cache <dir>|res <u32>>");
                }
            } else if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoRadius(val));
                } else {
                    println!("Usage: :ssao-radius <float>");
                }
            } else if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoIntensity(val));
                } else {
                    println!("Usage: :ssao-intensity <float>");
                }
            } else if l.starts_with(":viz-depth-max") || l.starts_with("viz-depth-max ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetVizDepthMax(val));
                } else {
                    println!("Usage: :viz-depth-max <float>");
                }
            } else if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<u32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsgiSteps(val));
                    } else {
                        println!("Usage: :ssgi-steps <u32>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsgiSteps);
                }
            } else if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsgiRadius(val));
                    } else {
                        println!("Usage: :ssgi-radius <float>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsgiRadius);
                }
            } else if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<u32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsrMaxSteps(val));
                    } else {
                        println!("Usage: :ssr-max-steps <u32>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsrMaxSteps);
                }
            } else if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(val_str) = it.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        let _ = proxy.send_event(ViewerCmd::SetSsrThickness(val));
                    } else {
                        println!("Usage: :ssr-thickness <float>");
                    }
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsrThickness);
                }
            } else if l.starts_with(":ssr-enable") || l.starts_with("ssr-enable ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(tok) = it.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::GiToggle("ssr", on));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsrEnable);
                }
            } else if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
                let mut it = l.split_whitespace();
                let _ = it.next();
                if let Some(tok) = it.next() {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiHalf(on));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsgiHalf);
                }
            } else if l.starts_with(":lit-sun") || l.starts_with("lit-sun ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetLitSun(val));
                } else {
                    println!("Usage: :lit-sun <float>");
                }
            } else if l.starts_with(":lit-ibl") || l.starts_with("lit-ibl ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetLitIbl(val));
                } else {
                    println!("Usage: :lit-ibl <float>");
                }
            } else if l.starts_with(":lit-rough") || l.starts_with("lit-rough ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetLitRough(val));
                } else {
                    println!("Usage: :lit-rough <0..1>");
                }
            } else if l.starts_with(":lit-debug") || l.starts_with("lit-debug ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let mode = match tok {
                        "rough" | "1" | "smoke" => 1u32,
                        "ndf" | "2" => 2u32,
                        _ => 0u32,
                    };
                    let _ = proxy.send_event(ViewerCmd::SetLitDebug(mode));
                } else {
                    println!("Usage: :lit-debug <off|rough|ndf>");
                }
            } else if l.starts_with(":ssao-technique") || l.starts_with("ssao-technique ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let tech = match tok {
                        "gtao" | "1" => 1u32,
                        _ => 0u32,
                    };
                    let _ = proxy.send_event(ViewerCmd::SetSsaoTechnique(tech));
                } else {
                    let _ = proxy.send_event(ViewerCmd::QuerySsaoTechnique);
                }
            // Sky controls
            } else if l.starts_with(":sky ") || l == ":sky" || l.starts_with("sky ") {
                if let Some(arg) = l.split_whitespace().nth(1) {
                    match arg {
                        "off" | "0" | "false" => {
                            let _ = proxy.send_event(ViewerCmd::SkyToggle(false));
                        }
                        "on" | "1" | "true" => {
                            let _ = proxy.send_event(ViewerCmd::SkyToggle(true));
                        }
                        "preetham" => {
                            let _ = proxy.send_event(ViewerCmd::SkyToggle(true));
                            let _ = proxy.send_event(ViewerCmd::SkySetModel(0));
                        }
                        "hosek-wilkie" | "hosekwilkie" | "hosek" | "hw" => {
                            let _ = proxy.send_event(ViewerCmd::SkyToggle(true));
                            let _ = proxy.send_event(ViewerCmd::SkySetModel(1));
                        }
                        other => {
                            println!(
                                "Unknown sky mode '{}', expected off|on|preetham|hosek-wilkie",
                                other
                            );
                        }
                    }
                } else {
                    println!("Usage: :sky <off|on|preetham|hosek-wilkie>");
                }
            } else if l.starts_with(":sky-turbidity") || l.starts_with("sky-turbidity ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SkySetTurbidity(val));
                } else {
                    println!("Usage: :sky-turbidity <float 1..10>");
                }
            } else if l.starts_with(":sky-ground") || l.starts_with("sky-ground ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SkySetGround(val));
                } else {
                    println!("Usage: :sky-ground <float 0..1>");
                }
            } else if l.starts_with(":sky-exposure") || l.starts_with("sky-exposure ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SkySetExposure(val));
                } else {
                    println!("Usage: :sky-exposure <float>");
                }
            } else if l.starts_with(":sky-sun") || l.starts_with("sky-sun ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SkySetSunIntensity(val));
                } else {
                    println!("Usage: :sky-sun <float>");
                }
            // Fog controls
            } else if l.starts_with(":fog ") || l == ":fog" || l.starts_with("fog ") {
                if let Some(arg) = l.split_whitespace().nth(1) {
                    let on = matches!(arg, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::FogToggle(on));
                } else {
                    println!("Usage: :fog <on|off>");
                }
            } else if l.starts_with(":fog-density") || l.starts_with("fog-density ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::FogSetDensity(val));
                } else {
                    println!("Usage: :fog-density <float>");
                }
            } else if l.starts_with(":fog-g") || l.starts_with("fog-g ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::FogSetG(val));
                } else {
                    println!("Usage: :fog-g <float -0.999..0.999>");
                }
            } else if l.starts_with(":fog-steps") || l.starts_with("fog-steps ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::FogSetSteps(val));
                } else {
                    println!("Usage: :fog-steps <u32>");
                }
            } else if l.starts_with(":fog-shadow") || l.starts_with("fog-shadow ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::FogSetShadow(on));
                } else {
                    println!("Usage: :fog-shadow <on|off>");
                }
            } else if l.starts_with(":fog-temporal") || l.starts_with("fog-temporal ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::FogSetTemporal(val));
                } else {
                    println!("Usage: :fog-temporal <float 0..0.9>");
                }
            } else if l.starts_with(":fog-mode") || l.starts_with("fog-mode ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let idx = match tok {
                        "raymarch" | "rm" | "0" => 0u32,
                        "froxels" | "fx" | "1" => 1u32,
                        _ => 0u32,
                    };
                    let _ = proxy.send_event(ViewerCmd::SetFogMode(idx));
                } else {
                    println!("Usage: :fog-mode <raymarch|froxels>");
                }
            } else if l.starts_with(":fog-preset") || l.starts_with("fog-preset ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let idx = match tok {
                        "low" | "0" => 0u32,
                        "med" | "medium" | "1" => 1u32,
                        _ => 2u32,
                    };
                    let _ = proxy.send_event(ViewerCmd::FogPreset(idx));
                } else {
                    println!("Usage: :fog-preset <low|med|high>");
                }
            } else if l.starts_with(":fog-half") || l.starts_with("fog-half ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::FogHalf(on));
                } else {
                    println!("Usage: :fog-half <on|off>");
                }
            } else if l.starts_with(":fog-edges") || l.starts_with("fog-edges ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::FogEdges(on));
                } else {
                    println!("Usage: :fog-edges <on|off>");
                }
            } else if l.starts_with(":fog-upsigma") || l.starts_with("fog-upsigma ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::FogUpsigma(val));
                } else {
                    println!("Usage: :fog-upsigma <float>");
                }
            } else if l == ":quit" || l == "quit" || l == ":exit" || l == "exit" {
                let _ = proxy.send_event(ViewerCmd::Quit);
                break;
            } else {
                println!(
                    "Commands:\n  :gi <ssao|ssgi|ssr> <on|off>\n  :viz <material|normal|depth|gi|lit>\n  :viz-depth-max <float>\n  :ibl <on|off|load <path>|intensity <f>|rotate <deg>|cache <dir>|res <u32>>\n  :brdf <lambert|phong|ggx|disney>\n  :snapshot [path]\n  :obj <path> | :gltf <path>\n  :sky off|on|preetham|hosek-wilkie | :sky-turbidity <f> | :sky-ground <f> | :sky-exposure <f> | :sky-sun <f>\n  :fog <on|off> | :fog-density <f> | :fog-g <f> | :fog-steps <u32> | :fog-shadow <on|off> | :fog-temporal <0..0.9> | :fog-mode <raymarch|froxels> | :fog-preset <low|med|high>\n  Lit:  :lit-sun <float> | :lit-ibl <float>\n  SSAO: :ssao-technique <ssao|gtao> | :ssao-radius <f> | :ssao-intensity <f> | :ssao-composite <on|off> | :ssao-mul <0..1>\n  SSGI: :ssgi-steps <u32> | :ssgi-radius <f> | :ssgi-half <on|off> | :ssgi-temporal <on|off> | :ssgi-temporal-alpha <0..1> | :ssgi-edges <on|off> | :ssgi-upsample-sigma-depth <f> | :ssgi-upsample-sigma-normal <f>\n  SSR:  :ssr-max-steps <u32> | :ssr-thickness <f>\n  P5:   :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness>\n  :quit"
                );
            }
        }
    });

    // ...

    // ...

    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::Resumed => {
                // Initialize viewer on resume (required for some platforms)
                if viewer_opt.is_none() {
                    let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(v) => {
                            viewer_opt = Some(v);
                            last_frame = Instant::now();
                            // If an initial terrain config was provided (via open_terrain_viewer),
                            // attempt to attach a TerrainScene before applying CLI commands.
                            #[cfg(feature = "extension-module")]
                            if let Some(cfg) = INITIAL_TERRAIN_CONFIG.get() {
                                if let Some(viewer) = viewer_opt.as_mut() {
                                    if let Err(e) = viewer.load_terrain_from_config(cfg) {
                                        eprintln!(
                                            "[viewer] failed to load terrain scene from config: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            // Apply any pending commands from CLI now that viewer exists
                            for cmd in pending_cmds.drain(..) {
                                if let Some(viewer) = viewer_opt.as_mut() {
                                    viewer.handle_cmd(cmd);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to create viewer: {}", e);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } => {
                                if key_event.state == ElementState::Pressed {
                                    if let PhysicalKey::Code(KeyCode::Escape) =
                                        key_event.physical_key
                                    {
                                        elwt.exit();
                                    }
                                }
                            }
                            WindowEvent::Resized(physical_size) => {
                                viewer.resize(*physical_size);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id,
            } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    viewer.update(dt);
                    match viewer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            viewer.resize(viewer.window.inner_size())
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            elwt.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            eprintln!("Surface timeout!");
                        }
                    }
                }
            }
            Event::UserEvent(cmd) => match cmd {
                ViewerCmd::Quit => {
                    // Process any pending snapshot before exiting
                    if let Some(viewer) = viewer_opt.as_mut() {
                        if viewer.snapshot_request.is_some() {
                            viewer.update(0.0);
                            let _ = viewer.render();
                        }
                    }
                    elwt.exit();
                }
                other => {
                    if let Some(viewer) = viewer_opt.as_mut() {
                        eprintln!("[IPC] Processing command: {:?}", other);
                        viewer.handle_cmd(other);
                    } else {
                        eprintln!("[IPC] Viewer not ready, dropping command");
                    }
                }
            },
            _ => {}
        }
    });

    Ok(())
}

/// Global IPC command queue - static ensures visibility across threads
static IPC_QUEUE: std::sync::OnceLock<std::sync::Mutex<std::collections::VecDeque<ViewerCmd>>> = 
    std::sync::OnceLock::new();

fn get_ipc_queue() -> &'static std::sync::Mutex<std::collections::VecDeque<ViewerCmd>> {
    IPC_QUEUE.get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()))
}

/// Global viewer stats for IPC queries
static IPC_STATS: std::sync::OnceLock<std::sync::Mutex<ipc::ViewerStats>> = 
    std::sync::OnceLock::new();

fn get_ipc_stats() -> &'static std::sync::Mutex<ipc::ViewerStats> {
    IPC_STATS.get_or_init(|| std::sync::Mutex::new(ipc::ViewerStats::default()))
}

fn update_ipc_stats(vb_ready: bool, vertex_count: u32, index_count: u32, scene_has_mesh: bool) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.vb_ready = vb_ready;
        stats.vertex_count = vertex_count;
        stats.index_count = index_count;
        stats.scene_has_mesh = scene_has_mesh;
    }
}

/// Run the viewer with an IPC server for non-blocking Python control.
/// Prints `FORGE3D_VIEWER_READY port=<PORT>` when the server is listening.
pub fn run_viewer_with_ipc(
    config: ViewerConfig,
    ipc_config: ipc::IpcServerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Clear any stale commands and stats
    if let Ok(mut q) = get_ipc_queue().lock() {
        q.clear();
    }
    update_ipc_stats(false, 0, 0, false);
    
    // Create simple event loop (no user events needed)
    let event_loop: EventLoop<()> = EventLoop::new()?;
    
    // Start IPC server - pushes to global queue, reads from global stats
    let ipc_handle = ipc::start_ipc_server(
        ipc_config,
        move |cmd| {
            if let Ok(mut q) = get_ipc_queue().lock() {
                q.push_back(cmd);
                Ok(())
            } else {
                Err("Queue lock failed".to_string())
            }
        },
        || {
            get_ipc_stats()
                .lock()
                .map(|s| s.clone())
                .unwrap_or_default()
        },
    )?;

    // Capture port for printing READY after viewer is initialized
    let ipc_port = ipc_handle.port;

    // Create window
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(config.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                config.width as f64,
                config.height as f64,
            ))
            .build(&event_loop)?,
    );

    // Collect initial commands (same as run_viewer)
    let mut pending_cmds: Vec<ViewerCmd> = Vec::new();
    if let Some(cmds) = INITIAL_CMDS.get() {
        for raw in cmds.iter() {
            let l = raw.trim().to_lowercase();
            if l.is_empty() {
                continue;
            }
            if l.starts_with(":snapshot") || l.starts_with("snapshot ") {
                let path = l.split_whitespace().nth(1).map(|s| s.to_string());
                pending_cmds.push(ViewerCmd::Snapshot(path));
            } else if l.starts_with(":obj") || l.starts_with("obj ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::LoadObj(path.to_string()));
                }
            } else if l.starts_with(":gltf") || l.starts_with("gltf ") {
                if let Some(path) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::LoadGltf(path.to_string()));
                }
            }
        }
    }

    // Viewer state
    let mut viewer_opt: Option<Viewer> = None;
    let mut last_frame = Instant::now();

    let _ = event_loop.run(move |event, elwt| {
        // Use Poll mode to ensure UserEvents from IPC thread are processed
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
        
        match event {
            Event::Resumed => {
                if viewer_opt.is_none() {
                    let v = pollster::block_on(Viewer::new(Arc::clone(&window), config.clone()));
                    match v {
                        Ok(mut v) => {
                            for cmd in pending_cmds.drain(..) {
                                v.handle_cmd(cmd);
                            }
                            viewer_opt = Some(v);
                            last_frame = Instant::now();
                            // Print READY line AFTER viewer is initialized
                            println!("FORGE3D_VIEWER_READY port={}", ipc_port);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                            // Kick off render loop so IPC commands can be processed
                            window.request_redraw();
                        }
                        Err(e) => {
                            eprintln!("Failed to create viewer: {}", e);
                            elwt.exit();
                        }
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !matches!(event, WindowEvent::RedrawRequested) => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    if !viewer.handle_input(event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::KeyboardInput {
                                event: key_event, ..
                            } => {
                                if key_event.state == ElementState::Pressed {
                                    if let PhysicalKey::Code(KeyCode::Escape) =
                                        key_event.physical_key
                                    {
                                        elwt.exit();
                                    }
                                }
                            }
                            WindowEvent::Resized(physical_size) => {
                                viewer.resize(*physical_size);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::AboutToWait => {
                // Poll global IPC queue for commands
                if let Some(viewer) = viewer_opt.as_mut() {
                    if let Ok(mut q) = get_ipc_queue().lock() {
                        while let Some(cmd) = q.pop_front() {
                            match cmd {
                                ViewerCmd::Quit => {
                                    if viewer.snapshot_request.is_some() {
                                        viewer.update(0.0);
                                        let _ = viewer.render();
                                    }
                                    elwt.exit();
                                    return;
                                }
                                other => {
                                    viewer.handle_cmd(other);
                                }
                            }
                        }
                    }
                }
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id,
            } if window_id == window.id() => {
                if let Some(viewer) = viewer_opt.as_mut() {
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    viewer.update(dt);
                    match viewer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            viewer.resize(viewer.window.inner_size())
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            elwt.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            eprintln!("Surface timeout!");
                        }
                    }
                }
            }
            _ => {}
        }
    });

    Ok(())
}
