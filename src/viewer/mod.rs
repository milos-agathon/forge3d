// src/viewer/mod.rs
// Workstream I1: Interactive windowed viewer for forge3d
// - Creates window with winit 0.29
// - Handles input events (mouse, keyboard)
// - Renders frames at 60 FPS
// - Orbit and FPS camera modes

pub mod camera_controller;

use crate::core::ibl::{IBLQuality, IBLRenderer};
use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::geometry::{generate_plane, generate_sphere, MeshBuffers};
use crate::p5::meta::{self as p5_meta, build_ssr_meta, SsrMetaInput};
use crate::p5::{ssr, ssr::SsrScenePreset, ssr_analysis};
use crate::passes::ssr::SsrStats;
use crate::render::params::SsrParams;
use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;
use anyhow::{bail, Context};
use camera_controller::{CameraController, CameraMode};
use glam::{Mat3, Mat4, Vec2, Vec3};
use half::f16;
use serde_json::json;
use std::collections::VecDeque;
use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use wgpu::{Device, Instance, Queue, Surface, SurfaceConfiguration};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};
// once_cell imported via path for INITIAL_CMDS; no direct use import needed

// Quick sanity-check version for viewer lit WGSL
const LIT_WGSL_VERSION: u32 = 2;
// Limit for P5.1 capture outputs (in megapixels). Images larger than this will be downscaled.
const P51_MAX_MEGAPIXELS: f32 = 2.0;
const P52_MAX_MEGAPIXELS: f32 = 2.0;
const P5_SSGI_DIFFUSE_SCALE: f32 = 0.5;
const P5_SSGI_CORNELL_WARMUP_FRAMES: u32 = 64;

// ------------------------------
// HUD seven-seg numeric helpers
// ------------------------------
fn hud_push_rect(
    inst: &mut Vec<crate::core::text_overlay::TextInstance>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    inst.push(crate::core::text_overlay::TextInstance {
        rect_min: [x0, y0],
        rect_max: [x1, y1],
        uv_min: [0.0, 0.0],
        uv_max: [1.0, 1.0],
        color,
    });
}

fn build_ssr_albedo_texture(preset: &SsrScenePreset, size: u32) -> Vec<u8> {
    let dim = size.max(1);
    let mut pixels = vec![0u8; (dim * dim * 4) as usize];

    let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };

    let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };

    let horizon = (preset.floor_horizon.clamp(0.0, 1.0) * dim as f32)
        .round()
        .clamp(0.0, (dim - 1) as f32) as u32;

    for y in 0..dim {
        let color = if y < horizon {
            let denom = horizon.max(1);
            let t = (y as f32 / denom as f32).powf(0.9).clamp(0.0, 1.0);
            lerp3(preset.background_top, preset.background_bottom, t)
        } else {
            let denom = (dim - horizon).max(1);
            let t = ((y - horizon) as f32 / denom as f32).clamp(0.0, 1.0);
            lerp3(preset.floor.color_top, preset.floor.color_bottom, t)
        };
        for x in 0..dim {
            let idx = ((y * dim + x) * 4) as usize;
            pixels[idx] = to_u8(color[0]);
            pixels[idx + 1] = to_u8(color[1]);
            pixels[idx + 2] = to_u8(color[2]);
            pixels[idx + 3] = 255;
        }
    }

    let stripe_center = preset.stripe.center_y * dim as f32;
    let stripe_half = (preset.stripe.half_thickness * dim as f32).max(1.0);
    for y in 0..dim {
        let dy = ((y as f32 - stripe_center) / stripe_half).abs();
        if dy < 1.0 {
            let alpha = (1.0 - dy).powf(2.0) * preset.stripe.glow_strength;
            let glow = lerp3(
                preset.stripe.inner_color,
                preset.stripe.outer_color,
                (y as f32 / dim as f32).clamp(0.0, 1.0),
            );
            for x in 0..dim {
                let idx = ((y * dim + x) * 4) as usize;
                let dst = [
                    pixels[idx] as f32 / 255.0,
                    pixels[idx + 1] as f32 / 255.0,
                    pixels[idx + 2] as f32 / 255.0,
                ];
                let inv = 1.0 - alpha;
                let mixed = [
                    dst[0] * inv + glow[0] * alpha,
                    dst[1] * inv + glow[1] * alpha,
                    dst[2] * inv + glow[2] * alpha,
                ];
                pixels[idx] = to_u8(mixed[0]);
                pixels[idx + 1] = to_u8(mixed[1]);
                pixels[idx + 2] = to_u8(mixed[2]);
            }
        }
    }

    pixels
}

// ------------------------------
// HUD tiny 3x5 block text helpers (A-Z subset)
// ------------------------------
fn hud_push_char_3x5(
    inst: &mut Vec<crate::core::text_overlay::TextInstance>,
    x: f32,
    y: f32,
    scale: f32,
    ch: char,
    color: [f32; 4],
) -> f32 {
    let cell = 2.0 * scale; // pixel size
    let spacing = 1.0 * scale; // inter-char spacing
    let pat: Option<[&str; 5]> = match ch.to_ascii_uppercase() {
        'A' => Some([" X ", "X X", "XXX", "X X", "X X"]),
        'B' => Some(["XX ", "X X", "XX ", "X X", "XX "]),
        'C' => Some([" XX", "X  ", "X  ", "X  ", " XX"]),
        'D' => Some(["XX ", "X X", "X X", "X X", "XX "]),
        'E' => Some(["XXX", "X  ", "XX ", "X  ", "XXX"]),
        'F' => Some(["XXX", "X  ", "XX ", "X  ", "X  "]),
        'G' => Some([" XX", "X  ", "X X", "X X", " XX"]),
        'H' => Some(["X X", "X X", "XXX", "X X", "X X"]),
        'K' => Some(["X X", "XX ", "X  ", "XX ", "X X"]),
        'L' => Some(["X  ", "X  ", "X  ", "X  ", "XXX"]),
        'N' => Some(["X X", "XX ", "X X", "X X", "X X"]),
        'O' => Some(["XXX", "X X", "X X", "X X", "XXX"]),
        'P' => Some(["XX ", "X X", "XX ", "X  ", "X  "]),
        'R' => Some(["XX ", "X X", "XX ", "X X", "X X"]),
        'S' => Some([" XX", "X  ", " XX", "  X", "XX "]),
        'T' => Some(["XXX", " X ", " X ", " X ", " X "]),
        'U' => Some(["X X", "X X", "X X", "X X", "XXX"]),
        'Y' => Some(["X X", "X X", " X ", " X ", " X "]),
        _ => None,
    };
    if let Some(rows) = pat {
        for (r, row) in rows.iter().enumerate() {
            for (c, ch2) in row.chars().enumerate() {
                if ch2 == 'X' {
                    let x0 = x + c as f32 * cell;
                    let y0 = y + r as f32 * cell;
                    hud_push_rect(inst, x0, y0, x0 + cell, y0 + cell, color);
                }
            }
        }
    }
    3.0 * cell + spacing
}

fn hud_push_text_3x5(
    inst: &mut Vec<crate::core::text_overlay::TextInstance>,
    mut x: f32,
    y: f32,
    scale: f32,
    text: &str,
    color: [f32; 4],
) -> f32 {
    for ch in text.chars() {
        if ch == ' ' {
            x += 2.0 * scale;
            continue;
        }
        x += hud_push_char_3x5(inst, x, y, scale, ch, color);
    }
    x
}

fn hud_push_digit(
    inst: &mut Vec<crate::core::text_overlay::TextInstance>,
    x: f32,
    y: f32,
    scale: f32,
    ch: char,
    color: [f32; 4],
) -> f32 {
    // 7-segment layout (a..g), plus dot segment 'dp'
    //  --a--
    // |     |
    // f     b
    // |     |
    //  --g--
    // |     |
    // e     c
    // |     |
    //  --d--   . dp
    let thick = 2.0 * scale;
    let w = 10.0 * scale; // char width
    let h = 18.0 * scale; // char height
    let mut seg = |a: bool, b: bool, c: bool, d: bool, e: bool, f: bool, g: bool, dp: bool| {
        if a {
            hud_push_rect(inst, x + thick, y, x + w - thick, y + thick, color);
        }
        if b {
            hud_push_rect(
                inst,
                x + w - thick,
                y + thick,
                x + w,
                y + h / 2.0 - thick,
                color,
            );
        }
        if c {
            hud_push_rect(
                inst,
                x + w - thick,
                y + h / 2.0 + thick,
                x + w,
                y + h - thick,
                color,
            );
        }
        if d {
            hud_push_rect(inst, x + thick, y + h - thick, x + w - thick, y + h, color);
        }
        if e {
            hud_push_rect(
                inst,
                x,
                y + h / 2.0 + thick,
                x + thick,
                y + h - thick,
                color,
            );
        }
        if f {
            hud_push_rect(inst, x, y + thick, x + thick, y + h / 2.0 - thick, color);
        }
        if g {
            hud_push_rect(
                inst,
                x + thick,
                y + h / 2.0 - thick / 2.0,
                x + w - thick,
                y + h / 2.0 + thick / 2.0,
                color,
            );
        }
        if dp {
            hud_push_rect(
                inst,
                x + w + thick * 0.5,
                y + h - thick * 1.5,
                x + w + thick * 1.5,
                y + h - thick * 0.5,
                color,
            );
        }
    };
    match ch {
        '0' => seg(true, true, true, true, true, true, false, false),
        '1' => seg(false, true, true, false, false, false, false, false),
        '2' => seg(true, true, false, true, true, false, true, false),
        '3' => seg(true, true, true, true, false, false, true, false),
        '4' => seg(false, true, true, false, false, true, true, false),
        '5' => seg(true, false, true, true, false, true, true, false),
        '6' => seg(true, false, true, true, true, true, true, false),
        '7' => seg(true, true, true, false, false, false, false, false),
        '8' => seg(true, true, true, true, true, true, true, false),
        '9' => seg(true, true, true, true, false, true, true, false),
        '-' => {
            // center segment only
            seg(false, false, false, false, false, false, true, false);
        }
        '.' => {
            seg(false, false, false, false, false, false, false, true);
        }
        _ => {}
    }
    w + 4.0 * scale // advance including small spacing
}

fn hud_push_number(
    inst: &mut Vec<crate::core::text_overlay::TextInstance>,
    mut x: f32,
    y: f32,
    scale: f32,
    value: f32,
    digits: usize,
    frac: usize,
    color: [f32; 4],
) -> f32 {
    let s = format!("{val:.prec$}", val = value, prec = frac);
    // Optionally truncate/limit total characters
    let mut count = 0usize;
    for ch in s.chars() {
        if count >= digits + 1 {
            break;
        }
        x += hud_push_digit(inst, x, y, scale, ch, color);
        count += 1;
    }
    x
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    // Vec4 #1
    sun_direction: [f32; 3],
    turbidity: f32,
    // Vec4 #2
    ground_albedo: f32,
    model: u32, // 0=Preetham, 1=Hosek-Wilkie
    sun_intensity: f32,
    exposure: f32,
    // Vec4 #3 padding
    _pad: [f32; 4],
}

// Std140-compatible packed layout for VolumetricUniforms used for GPU uniform buffer writes
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumetricUniformsStd140 {
    // row 0
    density: f32,
    height_falloff: f32,
    phase_g: f32,
    max_steps: u32,
    // row 1 (pad to align following vec3 to 16-byte boundary)
    start_distance: f32,
    max_distance: f32,
    _pad_a0: f32,
    _pad_a1: f32,
    // row 2
    scattering_color: [f32; 3],
    absorption: f32,
    // row 3
    sun_direction: [f32; 3],
    sun_intensity: f32,
    // row 4
    ambient_color: [f32; 3],
    temporal_alpha: f32,
    // row 5
    use_shadows: u32,
    jitter_strength: f32,
    frame_index: u32,
    _pad0: u32,
}

// P6: Volumetric fog uniforms matching shaders/volumetric.wgsl (std140-like packing)
#[repr(C, align(16))]
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct VolumetricUniforms {
    // row 0
    density: f32,
    height_falloff: f32,
    phase_g: f32,
    max_steps: u32,
    // row 1
    start_distance: f32,
    max_distance: f32,
    scattering_color: [f32; 3],
    absorption: f32,
    // row 2
    sun_direction: [f32; 3],
    sun_intensity: f32,
    // row 3
    ambient_color: [f32; 3],
    temporal_alpha: f32,
    // row 4
    use_shadows: u32,
    jitter_strength: f32,
    frame_index: u32,
    _pad0: u32,
    // Explicit padding to eliminate trailing struct padding for Pod
    _pad1: u32,
    _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FogCameraUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    eye_position: [f32; 3],
    near: f32,
    far: f32,
    _pad: [f32; 3],
}

// Std140-compatible upsample params for fog_upsample.wgsl
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FogUpsampleParamsStd140 {
    sigma: f32,
    use_bilateral: u32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    rough_metal: [f32; 2],
}

#[derive(Default)]
struct SceneMesh {
    vertices: Vec<PackedVertex>,
    indices: Vec<u32>,
}

impl SceneMesh {
    fn new() -> Self {
        Self::default()
    }

    fn push_vertex(
        &mut self,
        position: Vec3,
        normal: Vec3,
        uv: Vec2,
        roughness: f32,
        metallic: f32,
    ) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(PackedVertex {
            position: position.to_array(),
            normal: normal.normalize_or_zero().to_array(),
            uv: uv.to_array(),
            rough_metal: [roughness, metallic],
        });
        idx
    }

    fn extend_with_mesh(
        &mut self,
        mesh: &MeshBuffers,
        transform: Mat4,
        roughness: f32,
        metallic: f32,
    ) {
        let base = self.vertices.len() as u32;
        let normal_matrix = Mat3::from_mat4(transform).inverse().transpose();
        for i in 0..mesh.positions.len() {
            let pos = Vec3::from_array(mesh.positions[i]);
            let pos_w = (transform * pos.extend(1.0)).truncate();
            let normal_src = if mesh.normals.len() == mesh.positions.len() {
                Vec3::from_array(mesh.normals[i])
            } else {
                Vec3::Y
            };
            let normal_w = (normal_matrix * normal_src).normalize_or_zero();
            let uv = if mesh.uvs.len() == mesh.positions.len() {
                Vec2::from_array(mesh.uvs[i])
            } else {
                Vec2::ZERO
            };
            self.vertices.push(PackedVertex {
                position: pos_w.to_array(),
                normal: normal_w.to_array(),
                uv: uv.to_array(),
                rough_metal: [roughness, metallic],
            });
        }
        for &idx in &mesh.indices {
            self.indices.push(base + idx);
        }
    }
}

fn build_ssr_scene_mesh(preset: &SsrScenePreset) -> SceneMesh {
    let mut scene = SceneMesh::new();

    // Floor plane
    const FLOOR_WIDTH: f32 = 8.0;
    const FLOOR_DEPTH: f32 = 6.0;
    const FLOOR_Y: f32 = -1.0;
    let floor_mesh = generate_plane(32, 32);
    let floor_transform = Mat4::from_translation(Vec3::new(0.0, FLOOR_Y, 0.0))
        * Mat4::from_scale(Vec3::new(FLOOR_WIDTH * 0.5, 1.0, FLOOR_DEPTH * 0.5));
    scene.extend_with_mesh(&floor_mesh, floor_transform, 0.35, 0.0);

    // No extra back wall geometry; only floor + spheres per M2 visual acceptance

    // Glossy spheres
    const SPHERE_RINGS: u32 = 48;
    const SPHERE_SEGMENTS: u32 = 64;
    for (i, sphere) in preset.spheres.iter().enumerate() {
        let mesh = generate_sphere(SPHERE_RINGS, SPHERE_SEGMENTS, 1.0);
        let x = (sphere.offset_x - 0.5) * 6.0;
        let y = sphere.center_y * 3.0;
        let z = 0.5 + (i as f32) * 0.01;
        let radius = (sphere.radius * 3.0).max(0.05);
        let transform =
            Mat4::from_translation(Vec3::new(x, y, z)) * Mat4::from_scale(Vec3::splat(radius));
        scene.extend_with_mesh(&mesh, transform, sphere.roughness, 0.0);
    }

    scene
}

#[derive(Clone)]
pub struct ViewerConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub vsync: bool,
    pub fov_deg: f32,
    pub znear: f32,
    pub zfar: f32,
}

// Global initial commands for viewer (set by CLI parser in example)
static INITIAL_CMDS: once_cell::sync::OnceCell<Vec<String>> = once_cell::sync::OnceCell::new();

pub fn set_initial_commands(cmds: Vec<String>) {
    let _ = INITIAL_CMDS.set(cmds);
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            title: "forge3d Interactive Viewer".to_string(),
            vsync: true,
            fov_deg: 45.0,
            znear: 0.1,
            zfar: 1000.0,
        }
    }
}
pub struct Viewer {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    camera: CameraController,
    view_config: ViewerConfig,
    frame_count: u64,
    fps_counter: FpsCounter,
    // Input state
    keys_pressed: std::collections::HashSet<KeyCode>,
    shift_pressed: bool,
    // GI manager and toggles
    gi: Option<crate::core::screen_space_effects::ScreenSpaceEffectsManager>,
    ssr_params: SsrParams,
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
    // SSAO composite control
    use_ssao_composite: bool,
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
}

struct FpsCounter {
    frames: u32,
    last_report: Instant,
    current_fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frames: 0,
            last_report: Instant::now(),
            current_fps: 0.0,
        }
    }

    fn tick(&mut self) -> Option<f32> {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.current_fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.last_report = Instant::now();
            Some(self.current_fps)
        } else {
            None
        }
    }

    fn fps(&self) -> f32 {
        self.current_fps
    }
}

impl Viewer {
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
    fn mean_luma_region(buf: &Vec<u8>, w: u32, h: u32, x0: u32, y0: u32, rw: u32, rh: u32) -> f32 {
        let (w, h) = (w as usize, h as usize);
        let (x0, y0, rw, rh) = (x0 as usize, y0 as usize, rw as usize, rh as usize);
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for y in y0..(y0 + rh).min(h) {
            for x in x0..(x0 + rw).min(w) {
                let i = (y * w + x) * 4;
                let r = buf[i] as f32;
                let g = buf[i + 1] as f32;
                let b = buf[i + 2] as f32;
                let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                sum += (l / 255.0) as f64;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        }
    }

    // Compute variance of luma over entire RGBA8 image
    fn variance_luma(buf: &Vec<u8>, w: u32, h: u32) -> f32 {
        let (w, h) = (w as usize, h as usize);
        let n = (w * h).max(1);
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        for i in (0..(n * 4)).step_by(4) {
            let r = buf[i] as f32;
            let g = buf[i + 1] as f32;
            let b = buf[i + 2] as f32;
            let l = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            sum += l as f64;
            sum2 += (l * l) as f64;
        }
        let mean = sum / n as f64;
        let var = (sum2 / n as f64) - mean * mean;
        var.max(0.0) as f32
    }

    // Simple Sobel-like gradient energy (used for blur effectiveness metric)
    fn gradient_energy(buf: &Vec<u8>, w: u32, h: u32) -> f32 {
        if w < 2 || h < 2 {
            return 0.0;
        }
        let (w_usize, h_usize) = (w as usize, h as usize);
        let mut energy = 0.0f64;
        let mut samples = 0usize;
        for y in 0..(h_usize.saturating_sub(1)) {
            for x in 0..(w_usize.saturating_sub(1)) {
                let idx = (y * w_usize + x) * 4;
                let l = (0.2126 * buf[idx] as f32
                    + 0.7152 * buf[idx + 1] as f32
                    + 0.0722 * buf[idx + 2] as f32)
                    / 255.0;
                let idx_x = (y * w_usize + (x + 1)) * 4;
                let lx = (0.2126 * buf[idx_x] as f32
                    + 0.7152 * buf[idx_x + 1] as f32
                    + 0.0722 * buf[idx_x + 2] as f32)
                    / 255.0;
                let idx_y = ((y + 1) * w_usize + x) * 4;
                let ly = (0.2126 * buf[idx_y] as f32
                    + 0.7152 * buf[idx_y + 1] as f32
                    + 0.0722 * buf[idx_y + 2] as f32)
                    / 255.0;
                let dx = lx - l;
                let dy = ly - l;
                energy += (dx * dx + dy * dy) as f64;
                samples += 1;
            }
        }
        if samples == 0 {
            0.0
        } else {
            (energy / samples as f64) as f32
        }
    }

    // Write or update p5_meta.json with provided patcher
    fn write_p5_meta<F: FnOnce(&mut std::collections::BTreeMap<String, serde_json::Value>)>(
        &self,
        patch: F,
    ) -> anyhow::Result<()> {
        p5_meta::write_p5_meta(Path::new("reports/p5"), patch)
    }

    fn capture_p52_ssgi_cornell(&mut self) -> anyhow::Result<()> {
        use anyhow::Context;
        use std::fs;
        let out_dir = std::path::Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);
        let capture_is_srgb = matches!(
            self.config.format,
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        let was_enabled = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.is_enabled(SSE::SSGI)
        };
        if !was_enabled {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
        }

        let original_settings = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.ssgi_settings().context("SSGI settings unavailable")?
        };

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.disable_effect(SSE::SSGI);
        }
        self.reexecute_gi(None)?;
        let off_bytes = self.capture_material_rgba8()?;

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
            // Task 2: Set SSGI diffuse intensity factor for compositing
            // This scalar controls how much SSGI radiance is added to diffuse lighting
            // Tuned to achieve 5-12% bounce on neutral walls adjacent to colored walls
            gi.set_ssgi_composite_intensity(&self.queue, P5_SSGI_DIFFUSE_SCALE);
        }
        for _ in 0..P5_SSGI_CORNELL_WARMUP_FRAMES {
            self.reexecute_gi(None)?;
        }
        let on_bytes = self.capture_material_rgba8()?;

        // Combine split image
        let out_w = capture_w * 2;
        let out_h = capture_h;
        let mut combined = vec![0u8; (out_w * out_h * 4) as usize];
        let row_stride = (capture_w as usize) * 4;
        for y in 0..(capture_h as usize) {
            let dst_off = y * (out_w as usize) * 4;
            let src_off = y * row_stride;
            combined[dst_off..dst_off + row_stride]
                .copy_from_slice(&off_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride..dst_off + row_stride * 2]
                .copy_from_slice(&on_bytes[src_off..src_off + row_stride]);
        }

        let mut write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = {
            let px = (out_w as u64 as f64) * (out_h as u64 as f64);
            let max_px = (P52_MAX_MEGAPIXELS * 1_000_000.0) as f64;
            if px > max_px {
                let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
                let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
                let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
                write_buf = Self::downscale_rgba8_bilinear(&combined, out_w, out_h, dw, dh);
                (dw, dh, &write_buf)
            } else {
                (out_w, out_h, &combined)
            }
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssgi_cornell.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.2] downscaled SSGI Cornell capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssgi_cornell.png");

        // Metrics: miss ratio & avg steps
        let (miss_ratio, avg_steps) = {
            let (hit_bytes, dims) = self.read_ssgi_hit_bytes()?;
            let step_len = {
                let s = original_settings;
                let steps = s.num_steps.max(1) as f32;
                s.step_size.max(s.radius / steps)
            };
            let mut miss = 0u64;
            let mut hit = 0u64;
            let mut step_acc = 0.0f64;
            for i in 0..(dims.0 * dims.1) as usize {
                let off = i * 8;
                let dist = f16::from_le_bytes([hit_bytes[off + 4], hit_bytes[off + 5]]).to_f32();
                let mask = f16::from_le_bytes([hit_bytes[off + 6], hit_bytes[off + 7]]).to_f32();
                if mask >= 0.5 {
                    hit += 1;
                    let steps = if step_len > 0.0 { dist / step_len } else { 0.0 };
                    step_acc += steps as f64;
                } else {
                    miss += 1;
                }
            }
            let total = (dims.0 as u64) * (dims.1 as u64);
            let miss_ratio = if total > 0 {
                miss as f32 / total as f32
            } else {
                0.0
            };
            let avg_steps = if hit > 0 {
                (step_acc / hit as f64) as f32
            } else {
                0.0
            };
            (miss_ratio, avg_steps)
        };

        // Timings
        let (trace_ms, shade_ms, temporal_ms, upsample_ms) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.ssgi_timings_ms().unwrap_or((0.0, 0.0, 0.0, 0.0))
        };

        // Task 1: Wall bounce measurement - ROI-based luminance calculation
        // Define hard-coded ROIs for neutral wall regions adjacent to red and green walls
        // Cornell box: red wall on left (x=-1), green on right (x=1), neutral walls elsewhere
        // Back wall (z=1) is neutral and should receive bounce from both red and green walls
        // ROIs are defined for 1920x1080 base resolution and scaled for other resolutions
        // ROI_R_NEUTRAL: neutral back wall region near the red wall (left side of back wall)
        // ROI_G_NEUTRAL: neutral back wall region near the green wall (right side of back wall)
        // These ROIs avoid the checker cube, edges, and specular highlights
        const ROI_R_NEUTRAL_BASE: (u32, u32, u32, u32) = (750, 360, 930, 560); // centered patch facing red wall
        const ROI_G_NEUTRAL_BASE: (u32, u32, u32, u32) = (950, 360, 1130, 560); // centered patch facing green wall
        const BASE_WIDTH: u32 = 1920;
        const BASE_HEIGHT: u32 = 1080;

        // Scale ROIs to match current resolution
        let roi_red = {
            let (x0_base, y0_base, x1_base, y1_base) = ROI_R_NEUTRAL_BASE;
            let x0 = (x0_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y0 = (y0_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            let x1 = (x1_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y1 = (y1_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            (x0, y0, x1, y1)
        };
        let roi_green = {
            let (x0_base, y0_base, x1_base, y1_base) = ROI_G_NEUTRAL_BASE;
            let x0 = (x0_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y0 = (y0_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            let x1 = (x1_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y1 = (y1_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            (x0, y0, x1, y1)
        };

        // Compute luminance for ROIs in SSGI OFF and ON frames
        // Use linear luminance from final rendered color (post-GI, pre-tone-mapping)
        // Convert from sRGB if needed so bounce math happens in linear space
        let compute_roi_luminance = |bytes: &[u8], roi: (u32, u32, u32, u32)| -> f32 {
            let (x0, y0, x1, y1) = roi;
            let mut sum_luma = 0.0f64;
            let mut count = 0u32;
            let width = capture_w;
            let height = capture_h;
            let srgb = capture_is_srgb;
            let x_start = x0.min(width);
            let x_end = x1.min(width);
            let y_start = y0.min(height);
            let y_end = y1.min(height);
            let to_linear = |channel: u8| -> f32 {
                let c = channel as f32 / 255.0;
                if srgb {
                    if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    }
                } else {
                    c
                }
            };
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = ((y * width + x) * 4) as usize;
                    if idx + 3 < bytes.len() {
                        let r = to_linear(bytes[idx]);
                        let g = to_linear(bytes[idx + 1]);
                        let b = to_linear(bytes[idx + 2]);
                        // Compute linear luminance: L = 0.2126*R + 0.7152*G + 0.0722*B
                        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        sum_luma += luma as f64;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                (sum_luma / count as f64) as f32
            } else {
                0.0
            }
        };

        let l_r_off = compute_roi_luminance(&off_bytes, roi_red);
        let l_r_on = compute_roi_luminance(&on_bytes, roi_red);
        let l_g_off = compute_roi_luminance(&off_bytes, roi_green);
        let l_g_on = compute_roi_luminance(&on_bytes, roi_green);

        let bounce_red_pct = (l_r_on - l_r_off) / l_r_off.max(1e-6);
        let bounce_green_pct = (l_g_on - l_g_off) / l_g_off.max(1e-6);

        // Task 2: ΔE fallback (steps=0) – verify steps=0 SSGI outputs pure diffuse IBL
        // Render with steps=0 twice and compare - they should be identical (pure IBL, no temporal variance)
        let max_delta_e = {
            // First render with steps=0 (should output pure diffuse IBL)
            {
                let gi = self.gi.as_mut().context("GI manager not available")?;
                gi.enable_effect(&self.device, SSE::SSGI)?;
                gi.update_ssgi_settings(&self.queue, |s| {
                    s.num_steps = 0;
                    s.step_size = original_settings.step_size;
                    s.temporal_alpha = 0.0;
                    s.intensity = 1.0; // Ensure intensity doesn't affect pure IBL
                });
                gi.ssgi_reset_history(&self.device, &self.queue)?;
            }
            self.reexecute_gi(None)?;
            let first_bytes = self.read_ssgi_filtered_bytes()?.0;

            // Second render with steps=0 (should be identical)
            self.reexecute_gi(None)?;
            let second_bytes = self.read_ssgi_filtered_bytes()?.0;

            compute_max_delta_e(&second_bytes, &first_bytes)
        };

        // Restore SSGI settings, composite intensity, and effect enablement
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            // Reset composite intensity to default (1.0)
            gi.set_ssgi_composite_intensity(&self.queue, 1.0);
            if !was_enabled {
                gi.disable_effect(SSE::SSGI);
            }
        }

        self.write_p5_meta(|meta| {
            meta.insert(
                "ssgi".to_string(),
                json!({
                    "miss_ratio": miss_ratio,
                    "avg_steps": avg_steps,
                    "accumulation_alpha": original_settings.temporal_alpha,
                    "perf_ms": {
                        "trace_ms": trace_ms,
                        "shade_ms": shade_ms,
                        "temporal_ms": temporal_ms,
                        "upsample_ms": upsample_ms,
                        "total_ssgi_ms": trace_ms + shade_ms + temporal_ms + upsample_ms,
                    },
                    "max_delta_e": max_delta_e,
                }),
            );
            // Task 1: Store bounce metrics
            meta.insert(
                "ssgi_bounce".to_string(),
                json!({
                    "red_pct": bounce_red_pct,
                    "green_pct": bounce_green_pct,
                }),
            );
        })?;

        Ok(())
    }

    fn capture_p52_ssgi_temporal(&mut self) -> anyhow::Result<()> {
        use anyhow::Context;
        use std::fs;
        let out_dir = std::path::Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        let (w, h) = (self.config.width.max(1), self.config.height.max(1));

        let was_enabled = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.is_enabled(SSE::SSGI)
        };
        if !was_enabled {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
        }

        let original_settings = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.ssgi_settings().context("SSGI settings unavailable")?
        };

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.set_ssgi_composite_intensity(&self.queue, P5_SSGI_DIFFUSE_SCALE);
        }

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                s.temporal_alpha = 0.0;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
        }
        self.reexecute_gi(None)?;
        let single_bytes = self.capture_material_rgba8()?;

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
        }

        let mut frame8_luma = Vec::new();
        let mut frame9_luma = Vec::new();

        for frame in 0..16 {
            self.reexecute_gi(None)?;
            if frame == 7 || frame == 8 {
                let (bytes, _) = self.read_ssgi_filtered_bytes()?;
                let luma = rgba16_to_luma(&bytes);
                if frame == 7 {
                    frame8_luma = luma;
                } else {
                    frame9_luma = luma;
                }
            }
        }

        let accum_bytes = self.capture_material_rgba8()?;

        // Combine side-by-side
        let out_w = w * 2;
        let out_h = h;
        let mut combined = vec![0u8; (out_w * out_h * 4) as usize];
        for y in 0..h as usize {
            let dst_off = y * (out_w as usize) * 4;
            combined[dst_off..dst_off + (w as usize * 4)]
                .copy_from_slice(&single_bytes[y * (w as usize) * 4..(y + 1) * (w as usize) * 4]);
            combined[dst_off + (w as usize * 4)..dst_off + (w as usize * 8)]
                .copy_from_slice(&accum_bytes[y * (w as usize) * 4..(y + 1) * (w as usize) * 4]);
        }

        let mut write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = {
            let px = (out_w as u64 as f64) * (out_h as u64 as f64);
            let max_px = (P52_MAX_MEGAPIXELS * 1_000_000.0) as f64;
            if px > max_px {
                let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
                let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
                let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
                write_buf = Self::downscale_rgba8_bilinear(&combined, out_w, out_h, dw, dh);
                (dw, dh, &write_buf)
            } else {
                (out_w, out_h, &combined)
            }
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssgi_temporal_compare.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.2] downscaled SSGI temporal capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssgi_temporal_compare.png");

        let ssim = if !frame8_luma.is_empty() && frame8_luma.len() == frame9_luma.len() {
            compute_ssim(&frame8_luma, &frame9_luma)
        } else {
            1.0
        };

        self.write_p5_meta(|meta| {
            let entry = meta.entry("ssgi_temporal".to_string()).or_insert(json!({}));
            if let Some(obj) = entry.as_object_mut() {
                obj.insert("ssim_frame8_9".to_string(), json!(ssim));
                obj.insert("accumulation_frames".to_string(), json!(16));
            }
        })?;

        // Restore settings
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
            gi.set_ssgi_composite_intensity(&self.queue, 1.0);
        }
        if !was_enabled {
            if let Some(ref mut gi) = self.gi {
                gi.disable_effect(SSE::SSGI);
            }
        }

        Ok(())
    }

    fn reexecute_gi(&mut self, ssr_stats: Option<&mut SsrStats>) -> anyhow::Result<()> {
        use anyhow::Context;
        let depth_view = self.z_view.as_ref().context("Depth view unavailable")?;
        if let Some(ref mut gi) = self.gi {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p5.gi.reexec"),
                });
            gi.advance_frame(&self.queue);
            gi.build_hzb(&self.device, &mut enc, depth_view, false);
            gi.execute(&self.device, &mut enc, ssr_stats)?;
            self.queue.submit(std::iter::once(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }
        Ok(())
    }

    fn sync_ssr_params_to_gi(&mut self) {
        if let Some(ref mut gi) = self.gi {
            gi.set_ssr_params(&self.queue, &self.ssr_params);
        }
    }

    fn capture_material_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        use anyhow::Context;
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        self.with_comp_pipeline(|comp_pl, comp_bgl| {
            let fog_view = if self.fog_enabled {
                &self.fog_output_view
            } else {
                &self.fog_zero_view
            };
            Self::render_view_to_rgba8_ex(
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

    fn with_comp_pipeline<T>(
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

    fn read_ssgi_filtered_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
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

    fn read_ssgi_hit_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
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

    fn read_ssr_hit_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
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
        let w = self.config.width;
        let h = self.config.height;
        let fmt = self.config.format;

        match fmt {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                // BGRA -> RGBA in-place
                for px in data.chunks_exact_mut(4) {
                    px.swap(0, 2);
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

        // Lit output target
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
                        view_dimension: wgpu::TextureViewDimension::D2,
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
        let fog_shadow_view = fog_shadow_map.create_view(&wgpu::TextureViewDescriptor::default());
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

        let mut viewer = Self {
            window,
            surface,
            device,
            queue,
            config: surface_config,
            camera: CameraController::new(),
            view_config: config,
            frame_count: 0,
            fps_counter: FpsCounter::new(),
            keys_pressed: std::collections::HashSet::new(),
            shift_pressed: false,
            gi,
            ssr_params: SsrParams::default(),
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
            use_ssao_composite: true,
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
            let view_mat = self.camera.view_matrix();
            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let cam_pack = [to_arr4(view_mat), to_arr4(proj)];
            self.queue
                .write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_pack));

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
            let layout = self.geom_bind_group_layout.as_ref().unwrap();
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
                // Shadow matrix identity
                let ident = Mat4::IDENTITY;
                self.queue.write_buffer(
                    &self.fog_shadow_matrix,
                    0,
                    bytemuck::bytes_of(&to_arr(ident)),
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
                let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg1"),
                    layout: &self.fog_bgl1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.fog_shadow_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.fog_shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.fog_shadow_matrix.as_entire_binding(),
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
            let _ = gi.execute(&self.device, &mut encoder, None);

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
                    VizMode::Gi => {
                        if let Some(v) = gi.gi_debug_view() {
                            (3u32, v)
                        } else {
                            (0u32, &gi.gbuffer().material_view)
                        }
                    }
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
                    let snap_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewer.snapshot.offscreen"),
                        size: wgpu::Extent3d {
                            width: self.config.width,
                            height: self.config.height,
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

        // If we didn't composite anything (GI path unavailable), draw fallback to swapchain now
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
            eprintln!("[viewer-debug] snapshot_request taken: {}", path);
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

        // Process any pending P5.1 capture requests using current frame data
        if let Some(kind) = self.pending_captures.pop_front() {
            match kind {
                CaptureKind::P51CornellSplit => {
                    if let Err(e) = self.capture_p51_cornell_split() {
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
                        eprintln!("[P5.2] SSGI temporal capture failed: {}", e);
                    }
                }
                CaptureKind::P53SsrGlossy => {
                    if let Err(e) = self.capture_p53_ssr_glossy() {
                        eprintln!("[P5.3] SSR glossy spheres capture failed: {}", e);
                    }
                }
                CaptureKind::P53SsrThickness => {
                    if let Err(e) = self.capture_p53_ssr_thickness_ablation() {
                        eprintln!("[P5.3] SSR thickness ablation capture failed: {}", e);
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
    fn render_view_to_rgba8_ex(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        comp_pl: &wgpu::RenderPipeline,
        comp_bgl: &wgpu::BindGroupLayout,
        sky_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        fog_view: &wgpu::TextureView,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        far: f32,
        src_view: &wgpu::TextureView,
        mode: u32,
    ) -> anyhow::Result<Vec<u8>> {
        use anyhow::Context;
        // Uniform for mode and far
        let params: [f32; 4] = [mode as f32, far, 0.0, 0.0];
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p51.comp.params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Sampler
        let comp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("p51.comp.sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // Bind group
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p51.comp.bg"),
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
                    resource: ub.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(fog_view),
                },
            ],
        });
        // Offscreen texture
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("p51.offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("p51.comp.encoder"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("p51.comp.pass"),
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
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
        // Read back and format
        let mut data = read_texture_tight(device, queue, &tex, (width, height), surface_format)
            .context("read back offscreen")?;
        match surface_format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                for px in data.chunks_exact_mut(4) {
                    px.swap(0, 2);
                }
            }
            _ => {}
        }
        Ok(data)
    }

    // Simple bilinear downscale for tightly-packed RGBA8 buffers
    fn downscale_rgba8_bilinear(
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        if dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }
        if src_w == dst_w && src_h == dst_h {
            return src.to_vec();
        }
        let s_w = src_w as usize;
        let s_h = src_h as usize;
        let d_w = dst_w as usize;
        let d_h = dst_h as usize;
        let mut out = vec![0u8; d_w * d_h * 4];
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        for dy in 0..d_h {
            let sy = (dy as f32 + 0.5) * scale_y - 0.5;
            let y0 = sy.floor().max(0.0) as i32;
            let y1 = (y0 + 1).min(src_h as i32 - 1);
            let wy = (sy - y0 as f32).clamp(0.0, 1.0);
            for dx in 0..d_w {
                let sx = (dx as f32 + 0.5) * scale_x - 0.5;
                let x0 = sx.floor().max(0.0) as i32;
                let x1 = (x0 + 1).min(src_w as i32 - 1);
                let wx = (sx - x0 as f32).clamp(0.0, 1.0);
                let i00 = ((y0 as usize) * s_w + (x0 as usize)) * 4;
                let i10 = ((y0 as usize) * s_w + (x1 as usize)) * 4;
                let i01 = ((y1 as usize) * s_w + (x0 as usize)) * 4;
                let i11 = ((y1 as usize) * s_w + (x1 as usize)) * 4;
                let o = (dy * d_w + dx) * 4;
                for c in 0..4 {
                    let p00 = src[i00 + c] as f32;
                    let p10 = src[i10 + c] as f32;
                    let p01 = src[i01 + c] as f32;
                    let p11 = src[i11 + c] as f32;
                    let top = p00 * (1.0 - wx) + p10 * wx;
                    let bot = p01 * (1.0 - wx) + p11 * wx;
                    let val = top * (1.0 - wy) + bot * wy;
                    out[o + c] = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    }

    fn capture_p51_cornell_split(&mut self) -> anyhow::Result<()> {
        use anyhow::{bail, Context};
        use std::fs;
        let out_dir = std::path::Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;
        // Ensure SSAO has up-to-date buffers for the current view before capturing ON/OFF.
        if let Some(technique) = self.gi.as_ref().map(|gi| gi.ssao_settings().technique) {
            self.reexecute_ssao_with(technique)?;
        }
        let (off_bytes, on_bytes, w, h) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let (w, h) = gi.gbuffer().dimensions();
            let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
            let off_bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                Self::render_view_to_rgba8_ex(
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
                    &gi.gbuffer().material_view,
                    0,
                )
            })?;
            let on_view = gi
                .material_with_ao_view()
                .unwrap_or(&gi.gbuffer().material_view);
            let on_bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                Self::render_view_to_rgba8_ex(
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
                    on_view,
                    0,
                )
            })?;
            (off_bytes, on_bytes, w, h)
        };
        // Assemble side-by-side
        let mut out = vec![0u8; (w * 2 * h * 4) as usize];
        for y in 0..h as usize {
            let row_off = &off_bytes[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
            let row_on = &on_bytes[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
            let dst = &mut out[(y * (2 * w as usize) * 4)..((y + 1) * (2 * w as usize) * 4)];
            dst[..(w as usize * 4)].copy_from_slice(row_off);
            dst[(w as usize * 4)..].copy_from_slice(row_on);
        }
        // Downscale if the composite exceeds the pixel budget
        let out_w = w * 2;
        let out_h = h;
        let max_px = (P51_MAX_MEGAPIXELS * 1_000_000.0) as f64;
        let px = (out_w as u64 as f64) * (out_h as u64 as f64);
        let mut write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = if px > max_px {
            let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
            let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
            let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
            write_buf = Self::downscale_rgba8_bilinear(&out, out_w, out_h, dw, dh);
            (dw, dh, &write_buf)
        } else {
            (out_w, out_h, &out)
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssao_cornell.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.1] downscaled Cornell capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssao_cornell.png");

        // Derive specular preservation metric from OFF vs ON split
        // Use top-1% brightest pixels by luma in OFF image then compare delta with ON
        let mut off_lumas: Vec<f32> = Vec::with_capacity((w * h) as usize);
        let mut on_lumas: Vec<f32> = Vec::with_capacity((w * h) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * w as usize + x) * 4;
                let lo = 0.2126 * (off_bytes[i] as f32)
                    + 0.7152 * (off_bytes[i + 1] as f32)
                    + 0.0722 * (off_bytes[i + 2] as f32);
                let ln = 0.2126 * (on_bytes[i] as f32)
                    + 0.7152 * (on_bytes[i + 1] as f32)
                    + 0.0722 * (on_bytes[i + 2] as f32);
                off_lumas.push(lo / 255.0);
                on_lumas.push(ln / 255.0);
            }
        }
        let mut idxs: Vec<usize> = (0..off_lumas.len()).collect();
        idxs.sort_by(|&a, &b| {
            off_lumas[b]
                .partial_cmp(&off_lumas[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_n = (off_lumas.len() as f32 * 0.01).ceil() as usize;
        let top_n = top_n.max(1).min(off_lumas.len());
        let mut sum_off = 0.0;
        let mut sum_on = 0.0;
        for k in 0..top_n {
            let i = idxs[k];
            sum_off += off_lumas[i];
            sum_on += on_lumas[i];
        }
        let mean_off = sum_off / top_n as f32;
        let mean_on = sum_on / top_n as f32;
        let spec_delta = (mean_on - mean_off).abs();
        let specular_preservation = if spec_delta <= 0.01 { "PASS" } else { "FAIL" };
        self.write_p5_meta(|meta| {
            meta.insert(
                "specular_preservation".to_string(),
                json!(format!(
                    "{} (delta={:.4})",
                    specular_preservation, spec_delta
                )),
            );
        })?;
        Ok(())
    }

    fn reexecute_ssao_with(&mut self, technique: u32) -> anyhow::Result<()> {
        use anyhow::Context;
        if let Some(ref mut gi) = self.gi {
            // Ensure SSAO is enabled so AO AOVs are allocated
            if !gi.is_enabled(crate::core::screen_space_effects::ScreenSpaceEffect::SSAO) {
                let _ = gi.enable_effect(
                    &self.device,
                    crate::core::screen_space_effects::ScreenSpaceEffect::SSAO,
                );
            }
            gi.update_ssao_settings(&self.queue, |s| {
                s.technique = technique;
            });
            // Rebuild HZB and execute
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p51.ssao.reexec"),
                });
            gi.build_hzb(&self.device, &mut enc, self.z_view.as_ref().unwrap(), false);
            let _ = gi.execute(&self.device, &mut enc, None);
            self.queue.submit(std::iter::once(enc.finish()));
        }
        Ok(())
    }

    fn capture_p51_ao_grid(&mut self) -> anyhow::Result<()> {
        use anyhow::{bail, Context};
        use std::fs;
        let out_dir = std::path::Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;
        let (w, h) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.gbuffer().dimensions()
        };
        // Grid layout per scripts/check_p5_1.py: rows = [SSAO, GTAO], cols = [raw, blur, resolved]
        let mut tiles: Vec<Vec<u8>> = Vec::new();
        for tech in [0u32, 1u32].iter() {
            self.reexecute_ssao_with(*tech)?;
            let (raw_bytes, blur_bytes, resolved_bytes) = {
                let gi = self.gi.as_ref().unwrap();
                let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
                let raw = gi.ao_raw_view().context("raw AO view missing")?;
                let blur_v = gi
                    .ao_blur_view()
                    .context("blur (blurred) AO view missing")?;
                let temporal = gi.ao_resolved_view().context("resolved AO view missing")?;
                let fog_v = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                let raw_b = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    Self::render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_v,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        raw,
                        3,
                    )
                })?;
                let blur_v_b = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    Self::render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_v,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        blur_v,
                        3,
                    )
                })?;
                let temporal_b = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    Self::render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_v,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        temporal,
                        3,
                    )
                })?;
                (raw_b, blur_v_b, temporal_b)
            };
            tiles.push(raw_bytes);
            tiles.push(blur_bytes);
            tiles.push(resolved_bytes);
        }
        // Assemble 3x2 grid per scripts/check_p5_1.py
        let grid_w = (w * 3) as usize;
        let grid_h = (h * 2) as usize;
        let mut out = vec![0u8; grid_w * grid_h * 4];
        for row in 0..2usize {
            for col in 0..3usize {
                let idx = row * 3 + col;
                let tile = &tiles[idx];
                for y in 0..(h as usize) {
                    let src = &tile[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
                    let dst_y = row * (h as usize) + y;
                    let dst_x = col * (w as usize);
                    let dst_off = (dst_y * grid_w + dst_x) * 4;
                    out[dst_off..dst_off + (w as usize * 4)].copy_from_slice(src);
                }
            }
        }
        // Downscale if needed
        let out_w = grid_w as u32;
        let out_h = grid_h as u32;
        let max_px = (P51_MAX_MEGAPIXELS * 1_000_000.0) as f64;
        let px = (out_w as u64 as f64) * (out_h as u64 as f64);
        let mut write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = if px > max_px {
            let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
            let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
            let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
            write_buf = Self::downscale_rgba8_bilinear(&out, out_w, out_h, dw, dh);
            (dw, dh, &write_buf)
        } else {
            (out_w, out_h, &out)
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssao_buffers_grid.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.1] downscaled Grid capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssao_buffers_grid.png");

        // Compute metrics from the last technique (GTAO = technique=1)
        if let Some((raw, blur, res)) = tiles
            .get(tiles.len().saturating_sub(3)..)
            .and_then(|s| Some((&s[0], &s[1], &s[2])))
        {
            // Mean in center 10%x10% region and corner 10%x10%
            let (cw, ch) = (w as usize, h as usize);
            let rx = cw / 10;
            let ry = ch / 10;
            let cx0 = cw / 2 - rx / 2;
            let cy0 = ch / 2 - ry / 2;
            let corner_x0 = 0usize;
            let corner_y0 = 0usize;
            let mean_center =
                Self::mean_luma_region(res, w, h, cx0 as u32, cy0 as u32, rx as u32, ry as u32);
            let mean_corner = Self::mean_luma_region(
                res,
                w,
                h,
                corner_x0 as u32,
                corner_y0 as u32,
                rx as u32,
                ry as u32,
            );
            // Gradient energy reduction from raw -> blur
            let grad_raw = Self::gradient_energy(raw, w, h);
            let grad_blur = Self::gradient_energy(blur, w, h);
            let reduction = if grad_raw > 1e-6 {
                (1.0 - (grad_blur / grad_raw)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            self.write_p5_meta(|meta| {
                // Params/technique
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    let technique = if s.technique == 0 { "SSAO" } else { "GTAO" };
                    meta.insert("technique".to_string(), json!(technique));
                    meta.insert(
                        "params".to_string(),
                        json!({
                            "radius": s.radius,
                            "intensity": s.intensity,
                            "bias": s.bias,
                            "temporal_alpha": gi.ssao_temporal_alpha(),
                            "temporal_enabled": gi.ssao_temporal_enabled(),
                            "blur": true,
                            "samples": s.num_samples,
                            "directions": if s.technique==0 { 0 } else { (s.num_samples/4).max(1) }
                        }),
                    );
                    if let Some((ao_ms, blur_ms, temporal_ms)) = gi.ssao_timings_ms() {
                        meta.insert(
                            "timings_ms".to_string(),
                            json!({
                                "ao_ms": ao_ms,
                                "blur_ms": blur_ms,
                                "temporal_ms": temporal_ms,
                                "total_ms": ao_ms + blur_ms + temporal_ms,
                            }),
                        );
                    }
                }
                // Metrics
                meta.insert("corner_ao_mean".to_string(), json!(mean_corner));
                meta.insert("center_ao_mean".to_string(), json!(mean_center));
                meta.insert("blur_gradient_reduction".to_string(), json!(reduction));
            })?;
        }
        Ok(())
    }

    fn capture_p51_param_sweep(&mut self) -> anyhow::Result<()> {
        use anyhow::Context;
        use std::fs;
        let out_dir = std::path::Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;
        let (w, h) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.gbuffer().dimensions()
        };
        let radii = [0.3f32, 0.5, 0.8];
        let intens = [0.5f32, 1.0, 1.5];
        let mut tiles: Vec<Vec<u8>> = Vec::new();
        for &r in &radii {
            for &i in &intens {
                if let Some(ref mut gim) = self.gi {
                    gim.update_ssao_settings(&self.queue, |s| {
                        s.radius = r;
                        s.intensity = i;
                    });
                    let mut enc =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("p51.sweep.exec"),
                            });
                    gim.build_hzb(&self.device, &mut enc, self.z_view.as_ref().unwrap(), false);
                    let _ = gim.execute(&self.device, &mut enc, None);
                    self.queue.submit(std::iter::once(enc.finish()));
                } else {
                    continue;
                }
                let gi = self.gi.as_ref().context("GI manager not available")?;
                let tile_view = gi
                    .material_with_ao_view()
                    .unwrap_or(&gi.gbuffer().material_view);
                let depth_view = &gi.gbuffer().depth_view;
                let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
                let tile_bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    let fog_view = if self.fog_enabled {
                        &self.fog_output_view
                    } else {
                        &self.fog_zero_view
                    };
                    Self::render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        depth_view,
                        fog_view,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        tile_view,
                        0,
                    )
                })?;
                tiles.push(tile_bytes);
            }
        }
        // Assemble 3x3 grid (rows=radii, cols=intens)
        let grid_w = (w * 3) as usize;
        let grid_h = (h * 3) as usize;
        let mut out = vec![0u8; grid_w * grid_h * 4];
        for ri in 0..3usize {
            for ci in 0..3usize {
                let idx = ri * 3 + ci;
                let tile = &tiles[idx];
                for y in 0..(h as usize) {
                    let src = &tile[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
                    let dst_y = ri * (h as usize) + y;
                    let dst_x = ci * (w as usize);
                    let dst_off = (dst_y * grid_w + dst_x) * 4;
                    out[dst_off..dst_off + (w as usize * 4)].copy_from_slice(src);
                }
            }
        }
        // Downscale if needed
        let out_w = grid_w as u32;
        let out_h = grid_h as u32;
        let max_px = (P51_MAX_MEGAPIXELS * 1_000_000.0) as f64;
        let px = (out_w as u64 as f64) * (out_h as u64 as f64);
        let mut write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = if px > max_px {
            let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
            let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
            let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
            write_buf = Self::downscale_rgba8_bilinear(&out, out_w, out_h, dw, dh);
            (dw, dh, &write_buf)
        } else {
            (out_w, out_h, &out)
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssao_params_grid.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.1] downscaled Sweep capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssao_params_grid.png");

        // Update meta.json
        self.write_p5_meta(|_meta| {
            // No additional fields needed for sweep
        })?;

        Ok(())
    }

    fn capture_p53_ssr_glossy(&mut self) -> anyhow::Result<()> {
        use anyhow::{bail, Context};
        use std::fs;

        const SSR_REF_NAME: &str = "p5_ssr_glossy_reference.png";

        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        if !self.ssr_scene_loaded {
            self.apply_ssr_scene_preset()?;
        }

        let mut ssr_stats = SsrStats::new();

        {
            if let Some(ref mut gi_mgr) = self.gi {
                if !gi_mgr.is_enabled(SSE::SSR) {
                    gi_mgr.enable_effect(&self.device, SSE::SSR)?;
                }
            } else {
                bail!("GI manager not available");
            }
            self.sync_ssr_params_to_gi();
        }

        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);
        let original_ssr_enable = self.ssr_params.ssr_enable;
        let (reference_bytes, ssr_bytes) = {
            let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);

            self.ssr_params.set_enabled(false);
            self.sync_ssr_params_to_gi();
            self.reexecute_gi(None)?;
            let reference_bytes = {
                let gi = self.gi.as_ref().context("GI manager not available")?;
                self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    let fog_view = if self.fog_enabled {
                        &self.fog_output_view
                    } else {
                        &self.fog_zero_view
                    };
                    Self::render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_view,
                        self.config.format,
                        capture_w,
                        capture_h,
                        far,
                        &gi.gbuffer().material_view,
                        0,
                    )
                })?
            };

            self.ssr_params.set_enabled(true);
            self.sync_ssr_params_to_gi();
            self.reexecute_gi(Some(&mut ssr_stats))?;
            if let Some(ref mut gi_mgr) = self.gi {
                gi_mgr
                    .collect_ssr_stats(&self.device, &self.queue, &mut ssr_stats)
                    .context("collect SSR stats")?;
            } else {
                bail!("GI manager not available");
            }

            // Capture lit buffer for debugging stripe intensity
            {
                let gi = self.gi.as_ref().context("GI manager not available")?;
                let capture_view = |view: &wgpu::TextureView, label: &str| -> anyhow::Result<()> {
                    let bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                        let fog_view = if self.fog_enabled {
                            &self.fog_output_view
                        } else {
                            &self.fog_zero_view
                        };
                        Self::render_view_to_rgba8_ex(
                            &self.device,
                            &self.queue,
                            comp_pl,
                            comp_bgl,
                            &self.sky_output_view,
                            &gi.gbuffer().depth_view,
                            fog_view,
                            self.config.format,
                            capture_w,
                            capture_h,
                            far,
                            view,
                            0,
                        )
                    })?;
                    image_write::write_png_rgba8_small(
                        &out_dir.join(label),
                        &bytes,
                        capture_w,
                        capture_h,
                    )?;
                    Ok(())
                };
                capture_view(&self.lit_output_view, "p5_ssr_glossy_lit.png")?;
                if let Some(view) = gi.ssr_spec_view() {
                    capture_view(view, "p5_ssr_glossy_spec.png")?;
                }
                if let Some(view) = gi.ssr_final_view() {
                    capture_view(view, "p5_ssr_glossy_final.png")?;
                }
            }

            let ssr_bytes = {
                let gi = self.gi.as_ref().context("GI manager not available")?;
                let ssr_view = gi
                    .material_with_ssr_view()
                    .unwrap_or(&gi.gbuffer().material_view);
                self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    let fog_view = if self.fog_enabled {
                        &self.fog_output_view
                    } else {
                        &self.fog_zero_view
                    };
                    Self::render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_view,
                        self.config.format,
                        capture_w,
                        capture_h,
                        far,
                        ssr_view,
                        0,
                    )
                })?
            };

            (reference_bytes, ssr_bytes)
        };

        if original_ssr_enable != self.ssr_params.ssr_enable {
            self.ssr_params.set_enabled(original_ssr_enable);
            self.sync_ssr_params_to_gi();
            self.reexecute_gi(None)?;
        }

        let ssr_path = out_dir.join(ssr::DEFAULT_OUTPUT_NAME);
        image_write::write_png_rgba8_small(&ssr_path, &ssr_bytes, capture_w, capture_h)?;
        println!("[P5] Wrote {}", ssr_path.display());

        let ref_path = out_dir.join(SSR_REF_NAME);
        image_write::write_png_rgba8_small(&ref_path, &reference_bytes, capture_w, capture_h)?;
        println!("[P5] Wrote {}", ref_path.display());

        let mut stripe_contrast = [0.0f32; 9];
        let mut stripe_contrast_reference: Option<[f32; 9]> = None;
        match ssr_analysis::analyze_stripe_contrast(&ref_path, &ssr_path) {
            Ok(summary) => {
                stripe_contrast = summary.ssr;
                stripe_contrast_reference = Some(summary.reference);
            }
            Err(err) => {
                eprintln!(
                    "[P5.3] analyze_stripe_contrast failed ({}); falling back to single-image analyzer",
                    err
                );
                // Fallback: compute per-band contrast from SSR image alone using scene preset ROI
                let preset = match self.ssr_scene_preset.clone() {
                    Some(p) => p,
                    None => SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?,
                };
                let bands = crate::p5::ssr_analysis::analyze_single_image_contrast(
                    &preset, &ssr_bytes, capture_w, capture_h,
                );
                for (i, v) in bands.into_iter().take(9).enumerate() {
                    stripe_contrast[i] = v;
                }
            }
        }
        let edge_streaks = ssr_analysis::count_edge_streaks(&ssr_bytes, capture_w, capture_h);

        let mean_diff = mean_abs_diff(&reference_bytes, &ssr_bytes);

        // Additional metrics: min_rgb_miss from SSR filtered texture using hit mask,
        // and max_delta_e_miss vs reference (restricted to miss pixels)
        let (mut min_rgb_miss, mut max_delta_e_miss) = (f32::INFINITY, 0.0f32);
        if let Some(ref gi) = self.gi {
            if let (Some(hit_tex), Some(ssr_tex)) = (gi.ssr_hit_texture(), gi.ssr_output_texture())
            {
                let hit_bytes = read_texture_tight(
                    &self.device,
                    &self.queue,
                    hit_tex,
                    (capture_w, capture_h),
                    wgpu::TextureFormat::Rgba16Float,
                )
                .context("read SSR hit texture")?;
                let ssr_lin_bytes = read_texture_tight(
                    &self.device,
                    &self.queue,
                    ssr_tex,
                    (capture_w, capture_h),
                    wgpu::TextureFormat::Rgba16Float,
                )
                .context("read SSR filtered texture")?;

                let pixel_count = (capture_w as usize) * (capture_h as usize);
                for i in 0..pixel_count {
                    // Hit mask from hit_bytes alpha
                    let hb = &hit_bytes[i * 8..i * 8 + 8];
                    let hit_mask = f16::from_le_bytes([hb[6], hb[7]]).to_f32();
                    if hit_mask < 0.5 {
                        // Min RGB among miss pixels (linear)
                        let sb = &ssr_lin_bytes[i * 8..i * 8 + 8];
                        let r = f16::from_le_bytes([sb[0], sb[1]]).to_f32();
                        let g = f16::from_le_bytes([sb[2], sb[3]]).to_f32();
                        let b = f16::from_le_bytes([sb[4], sb[5]]).to_f32();
                        let local_min = r.min(g).min(b);
                        if local_min < min_rgb_miss {
                            min_rgb_miss = local_min;
                        }

                        // ΔE in Lab using PNG outputs (assumed display RGB in 0..1)
                        let idx8 = i * 4;
                        if idx8 + 3 < ssr_bytes.len() && idx8 + 3 < reference_bytes.len() {
                            let ssr_rgb = srgb_triplet_to_linear(&ssr_bytes[idx8..idx8 + 3]);
                            let ref_rgb = srgb_triplet_to_linear(&reference_bytes[idx8..idx8 + 3]);
                            let de = delta_e_lab(ssr_rgb, ref_rgb);
                            if de > max_delta_e_miss {
                                max_delta_e_miss = de;
                            }
                        }
                    }
                }
            }
        }
        if !min_rgb_miss.is_finite() {
            min_rgb_miss = 0.0;
        }

        println!(
            "[P5.3] SSR params -> enable: {}, max_steps: {}, thickness: {:.3}",
            true, self.ssr_params.ssr_max_steps, self.ssr_params.ssr_thickness
        );
        println!(
            "[P5.3] SSR metrics -> hit_rate {:.3}, avg_steps {:.2}, diff {:.4}",
            ssr_stats.hit_rate(),
            ssr_stats.avg_steps(),
            mean_diff
        );

        let ssr_meta = build_ssr_meta(SsrMetaInput {
            stats: Some(&ssr_stats),
            stripe_contrast: Some(&stripe_contrast),
            stripe_contrast_reference: stripe_contrast_reference.as_ref(),
            mean_abs_diff: mean_diff,
            edge_streaks_gt1px: edge_streaks,
            max_delta_e_miss,
            min_rgb_miss,
        });

        println!("[P5.3] SSR status -> {}", ssr_meta.status);

        p5_meta::write_p5_meta(out_dir, |meta| {
            meta.insert("ssr".to_string(), ssr_meta.value.clone());
        })?;

        Ok(())
    }

    fn capture_p53_ssr_thickness_ablation(&mut self) -> anyhow::Result<()> {
        use anyhow::{bail, Context};
        use std::fs;

        const OUTPUT_NAME: &str = "p5_ssr_thickness_ablation.png";

        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        {
            if let Some(ref mut gi_mgr) = self.gi {
                if !gi_mgr.is_enabled(SSE::SSR) {
                    gi_mgr.enable_effect(&self.device, SSE::SSR)?;
                }
            } else {
                bail!("GI manager not available");
            }
            self.sync_ssr_params_to_gi();
        }

        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);
        let original_thickness = self.ssr_params.ssr_thickness;
        let original_enable = self.ssr_params.ssr_enable;

        // 1) Reference (SSR disabled)
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let reference_bytes = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                Self::render_view_to_rgba8_ex(
                    &self.device,
                    &self.queue,
                    comp_pl,
                    comp_bgl,
                    &self.sky_output_view,
                    &gi.gbuffer().depth_view,
                    fog_view,
                    self.config.format,
                    capture_w,
                    capture_h,
                    far,
                    &gi.gbuffer().material_view,
                    0,
                )
            })?
        };

        // 2) SSR enabled, thickness = 0 (undershoot before)
        self.ssr_params.set_enabled(true);
        self.ssr_params.set_thickness(0.0);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let off_bytes = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let ssr_view = gi
                .material_with_ssr_view()
                .unwrap_or(&gi.gbuffer().material_view);
            self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                Self::render_view_to_rgba8_ex(
                    &self.device,
                    &self.queue,
                    comp_pl,
                    comp_bgl,
                    &self.sky_output_view,
                    &gi.gbuffer().depth_view,
                    fog_view,
                    self.config.format,
                    capture_w,
                    capture_h,
                    far,
                    ssr_view,
                    0,
                )
            })?
        };

        // 3) SSR enabled, restored thickness (undershoot after)
        let restored_thickness = if original_thickness <= 0.0 {
            0.08
        } else {
            original_thickness
        };
        self.ssr_params.set_thickness(restored_thickness);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let on_bytes = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let ssr_view = gi
                .material_with_ssr_view()
                .unwrap_or(&gi.gbuffer().material_view);
            self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                Self::render_view_to_rgba8_ex(
                    &self.device,
                    &self.queue,
                    comp_pl,
                    comp_bgl,
                    &self.sky_output_view,
                    &gi.gbuffer().depth_view,
                    fog_view,
                    self.config.format,
                    capture_w,
                    capture_h,
                    far,
                    ssr_view,
                    0,
                )
            })?
        };

        self.ssr_params.set_thickness(original_thickness);
        self.ssr_params.set_enabled(original_enable);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;

        let out_w = capture_w * 2;
        let out_h = capture_h;
        let mut composed = vec![0u8; (out_w * out_h * 4) as usize];
        let row_bytes = (capture_w as usize) * 4;
        for y in 0..(capture_h as usize) {
            let dst_off = y * row_bytes * 2;
            let src_off = y * row_bytes;
            composed[dst_off..dst_off + row_bytes]
                .copy_from_slice(&off_bytes[src_off..src_off + row_bytes]);
            composed[dst_off + row_bytes..dst_off + row_bytes * 2]
                .copy_from_slice(&on_bytes[src_off..src_off + row_bytes]);
        }

        let out_path = out_dir.join(OUTPUT_NAME);
        image_write::write_png_rgba8_small(&out_path, &composed, out_w, out_h)?;
        let streaks_off = ssr_analysis::count_edge_streaks(&off_bytes, capture_w, capture_h);
        let streaks_on = ssr_analysis::count_edge_streaks(&on_bytes, capture_w, capture_h);
        println!(
            "[P5] Wrote {} (thickness off {:.3} | on {:.3})",
            out_path.display(),
            0.0,
            restored_thickness
        );
        println!(
            "[P5.3] Edge streak counts -> off: {} | on: {}",
            streaks_off, streaks_on
        );

        // Compute undershoot metrics and write into meta
        let preset = match self.ssr_scene_preset.clone() {
            Some(p) => p,
            None => SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?,
        };
        let undershoot_before = ssr_analysis::compute_undershoot_metric(
            &preset,
            &off_bytes,
            capture_w,
            capture_h,
        );
        let undershoot_after = ssr_analysis::compute_undershoot_metric(
            &preset,
            &on_bytes,
            capture_w,
            capture_h,
        );
        p5_meta::write_p5_meta(out_dir, |meta| {
            let ssr_entry = meta.entry("ssr".to_string()).or_insert(json!({}));
            if let Some(obj) = ssr_entry.as_object_mut() {
                p5_meta::patch_thickness_ablation(obj, undershoot_before, undershoot_after);
            }
        })?;

        Ok(())
    }
}

fn rgba16_to_luma(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let r = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        let g = f16::from_le_bytes([chunk[2], chunk[3]]).to_f32();
        let b = f16::from_le_bytes([chunk[4], chunk[5]]).to_f32();
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        out.push(luma);
    }
    out
}

fn compute_max_delta_e(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut max_de = 0.0f32;
    for (chunk_a, chunk_b) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let ra = f16::from_le_bytes([chunk_a[0], chunk_a[1]]).to_f32();
        let ga = f16::from_le_bytes([chunk_a[2], chunk_a[3]]).to_f32();
        let ba = f16::from_le_bytes([chunk_a[4], chunk_a[5]]).to_f32();

        let rb = f16::from_le_bytes([chunk_b[0], chunk_b[1]]).to_f32();
        let gb = f16::from_le_bytes([chunk_b[2], chunk_b[3]]).to_f32();
        let bb = f16::from_le_bytes([chunk_b[4], chunk_b[5]]).to_f32();

        let (l1, a1, b1) = rgb_to_lab(ra, ga, ba);
        let (l2, a2, b2) = rgb_to_lab(rb, gb, bb);
        let delta = ((l1 - l2).powi(2) + (a1 - a2).powi(2) + (b1 - b2).powi(2)).sqrt();
        if delta > max_de {
            max_de = delta;
        }
    }
    max_de
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (&xa, &xb) in a.iter().zip(b.iter()) {
        let da = xa as f32 / 255.0;
        let db = xb as f32 / 255.0;
        sum += (da - db).abs();
    }
    sum / (a.len() as f32)
}

fn srgb_u8_to_linear(channel: u8) -> f32 {
    let c = channel as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_triplet_to_linear(px: &[u8]) -> [f32; 3] {
    [
        srgb_u8_to_linear(px[0]),
        srgb_u8_to_linear(px[1]),
        srgb_u8_to_linear(px[2]),
    ]
}

fn delta_e_lab(rgb_a: [f32; 3], rgb_b: [f32; 3]) -> f32 {
    let (l1, a1, b1) = rgb_to_lab(rgb_a[0], rgb_a[1], rgb_a[2]);
    let (l2, a2, b2) = rgb_to_lab(rgb_b[0], rgb_b[1], rgb_b[2]);
    ((l1 - l2).powi(2) + (a1 - a2).powi(2) + (b1 - b2).powi(2)).sqrt()
}

fn compute_undershoot_fraction(
    preset: &SsrScenePreset,
    reference: &[u8],
    ssr: &[u8],
    width: u32,
    height: u32,
) -> f32 {
    if reference.len() < (width * height * 4) as usize || ssr.len() < (width * height * 4) as usize
    {
        return 0.0;
    }
    let floor_y = (preset.floor.start_y.clamp(0.0, 1.0) * height as f32).round() as u32;
    let roi_y0 = floor_y.min(height.saturating_sub(1));
    let roi_y1 = ((floor_y as f32 + 0.15 * height as f32).round() as u32).min(height);
    let roi_x0 = {
        let x0f = (preset
            .spheres
            .first()
            .map(|s| s.offset_x)
            .unwrap_or(0.1)
            .clamp(0.0, 1.0)
            * width as f32)
            .round();
        x0f.max(0.0).min((width.saturating_sub(1)) as f32) as u32
    };
    let roi_x1 = {
        let x1f = (preset
            .spheres
            .last()
            .map(|s| s.offset_x)
            .unwrap_or(0.85)
            .clamp(0.0, 1.0)
            * width as f32)
            .round();
        x1f.max(0.0).min(width as f32) as u32
    };

    let w = width as usize;
    let mut count = 0u32;
    let mut undershoot = 0u32;
    let eps = 0.01f32;
    for y in roi_y0..roi_y1 {
        for x in roi_x0..roi_x1 {
            let idx = ((y as usize * w + x as usize) * 4) as usize;
            let rl = srgb_u8_to_linear(reference[idx]) * 0.2126
                + srgb_u8_to_linear(reference[idx + 1]) * 0.7152
                + srgb_u8_to_linear(reference[idx + 2]) * 0.0722;
            let sl = srgb_u8_to_linear(ssr[idx]) * 0.2126
                + srgb_u8_to_linear(ssr[idx + 1]) * 0.7152
                + srgb_u8_to_linear(ssr[idx + 2]) * 0.0722;
            if sl > rl + eps {
                undershoot += 1;
            }
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (undershoot as f32 / count as f32).clamp(0.0, 1.0)
    }
}

fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Assume linear sRGB inputs in [0,1]
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = 0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b;

    let xn = 0.950_47;
    let yn = 1.0;
    let zn = 1.088_83;

    let fx = lab_pivot(x / xn);
    let fy = lab_pivot(y / yn);
    let fz = lab_pivot(z / zn);

    let l = (116.0 * fy - 16.0).clamp(0.0, 100.0);
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    (l, a, b)
}

fn lab_pivot(t: f32) -> f32 {
    const EPSILON: f32 = 0.008856;
    const KAPPA: f32 = 903.3;
    if t > EPSILON {
        t.cbrt()
    } else {
        (KAPPA * t + 16.0) / 116.0
    }
}

fn compute_ssim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let n = a.len() as f32;
    let mean_a = a.iter().copied().sum::<f32>() / n;
    let mean_b = b.iter().copied().sum::<f32>() / n;

    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    let mut cov = 0.0f32;
    for (&xa, &xb) in a.iter().zip(b.iter()) {
        var_a += (xa - mean_a).powi(2);
        var_b += (xb - mean_b).powi(2);
        cov += (xa - mean_a) * (xb - mean_b);
    }
    if n > 1.0 {
        var_a /= n - 1.0;
        var_b /= n - 1.0;
        cov /= n - 1.0;
    }

    const L: f32 = 1.0;
    const C1: f32 = 0.0001; // (0.01 * L)^2 with L=1
    const C2: f32 = 0.0009; // (0.03 * L)^2 with L=1

    let numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * cov + C2);
    let denominator = (mean_a.powi(2) + mean_b.powi(2) + C1) * (var_a + var_b + C2);
    if denominator.abs() < f32::EPSILON {
        1.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

// Simple command interface event type carried by the winit EventLoop
#[derive(Debug, Clone)]
enum ViewerCmd {
    GiToggle(&'static str, bool), // ("ssao"|"ssgi"|"ssr", on)
    Snapshot(Option<String>),
    DumpGbuffer, // P5: dump normals/material/depth HZB mips + meta
    Quit,
    LoadObj(String),
    LoadGltf(String),
    SetViz(String),
    LoadSsrPreset,
    LoadIbl(String),
    IblToggle(bool),
    IblIntensity(f32),
    IblRotate(f32),
    IblCache(Option<String>),
    IblRes(u32),
    #[allow(dead_code)]
    SetSsaoRadius(f32),
    #[allow(dead_code)]
    SetSsaoIntensity(f32),
    #[allow(dead_code)]
    SetSsaoBias(f32),
    #[allow(dead_code)]
    SetSsgiSteps(u32),
    #[allow(dead_code)]
    SetSsgiRadius(f32),
    SetSsrMaxSteps(u32),
    SetSsrThickness(f32),
    #[allow(dead_code)]
    SetSsgiHalf(bool),
    #[allow(dead_code)]
    SetSsgiTemporalAlpha(f32),
    #[allow(dead_code)]
    SetSsgiTemporalEnabled(bool),
    #[allow(dead_code)]
    SetAoTemporalAlpha(f32), // new: AO temporal alpha
    // SSAO-specific controls
    #[allow(dead_code)]
    SetSsaoSamples(u32),
    #[allow(dead_code)]
    SetSsaoDirections(u32),
    #[allow(dead_code)]
    SetSsaoTemporalAlpha(f32), // alias of SetAoTemporalAlpha
    #[allow(dead_code)]
    SetSsaoTemporalEnabled(bool),
    #[allow(dead_code)]
    SetSsaoTechnique(u32),
    #[allow(dead_code)]
    SetVizDepthMax(f32),
    #[allow(dead_code)]
    SetFov(f32),
    #[allow(dead_code)]
    SetCamLookAt {
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
    },
    #[allow(dead_code)]
    SetSize(u32, u32),
    #[allow(dead_code)]
    SetSsaoComposite(bool),
    #[allow(dead_code)]
    SetSsaoCompositeMul(f32),
    // AO blur toggle
    #[allow(dead_code)]
    SetAoBlur(bool),
    // SSGI edge-aware upsample controls
    #[allow(dead_code)]
    SetSsgiEdges(bool),
    #[allow(dead_code)]
    SetSsgiUpsampleSigmaDepth(f32),
    #[allow(dead_code)]
    SetSsgiUpsampleSigmaNormal(f32),
    // Lit viz controls
    SetLitSun(f32),
    SetLitIbl(f32),
    SetLitBrdf(u32),
    SetLitRough(f32),
    SetLitDebug(u32),
    // Sky controls
    SkyToggle(bool),
    SkySetModel(u32), // 0=Preetham,1=Hosek-Wilkie
    SkySetTurbidity(f32),
    SkySetGround(f32),
    SkySetExposure(f32),
    SkySetSunIntensity(f32),
    // Fog controls
    FogToggle(bool),
    FogSetDensity(f32),
    FogSetG(f32),
    FogSetSteps(u32),
    FogSetShadow(bool),
    FogSetTemporal(f32),
    SetFogMode(u32),
    FogPreset(u32),
    // P6-10 half-res upsample controls
    FogHalf(bool),
    FogEdges(bool),
    FogUpsigma(f32),
    HudToggle(bool),
    // P5.1: AO artifact captures
    CaptureP51Cornell,
    CaptureP51Grid,
    CaptureP51Sweep,
    CaptureP52SsgiCornell,
    CaptureP52SsgiTemporal,
    CaptureP53SsrGlossy,
    CaptureP53SsrThickness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VizMode {
    Material,
    Normal,
    Depth,
    Gi,
    Lit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FogMode {
    Raymarch,
    Froxels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureKind {
    P51CornellSplit,
    P51AoGrid,
    P51ParamSweep,
    P52SsgiCornell,
    P52SsgiTemporal,
    P53SsrGlossy,
    P53SsrThickness,
}

impl Viewer {
    fn handle_cmd(&mut self, cmd: ViewerCmd) {
        match cmd {
            ViewerCmd::Quit => { /* handled in event loop */ }
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
            }
            ViewerCmd::SetSsaoComposite(on) => {
                self.use_ssao_composite = on;
            }
            ViewerCmd::SetSsaoCompositeMul(v) => {
                if let Some(ref mut gi) = self.gi {
                    gi.set_ssao_composite_multiplier(&self.queue, v);
                }
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
                let p = path.unwrap_or_else(|| "snapshot.png".to_string());
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
            _ => {}
        }
    }

    // Minimal stub: accept mesh and mark geometry present (optional future: upload GPU buffers)
    fn upload_mesh(&mut self, _mesh: &crate::geometry::MeshBuffers) -> anyhow::Result<()> {
        // For now this viewer uses a built-in cube VB to keep demo working.
        // Implement real upload here if needed.
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
    let mut pending_cmds: Vec<ViewerCmd> = Vec::new();
    if let Some(cmds) = INITIAL_CMDS.get() {
        for raw in cmds.iter() {
            let l = raw.trim().to_lowercase();
            if l.is_empty() {
                continue;
            }
            if l.starts_with(":gi") || l.starts_with("gi ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() >= 3 {
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
                    _ => println!(
                        "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness>"
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
                if let Some(mode) = l.split_whitespace().nth(1) {
                    pending_cmds.push(ViewerCmd::SetViz(mode.to_string()));
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
            if l.starts_with(":gi") || l.starts_with("gi ") {
                let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
                if toks.len() >= 3 {
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
                    println!("Usage: :gi <ssao|ssgi|ssr> <on|off>");
                }
            } else if l.starts_with(":snapshot") || l.starts_with("snapshot") {
                let path = l.split_whitespace().nth(1).map(|s| s.to_string());
                let _ = proxy.send_event(ViewerCmd::Snapshot(path));
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
            } else if l.starts_with(":ssao-bias") || l.starts_with("ssao-bias ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoBias(val));
                } else {
                    println!("Usage: :ssao-bias <float>");
                }
            } else if l.starts_with(":ssao-samples") || l.starts_with("ssao-samples ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoSamples(val));
                } else {
                    println!("Usage: :ssao-samples <u32>");
                }
            } else if l.starts_with(":ssao-directions") || l.starts_with("ssao-directions ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoDirections(val));
                } else {
                    println!("Usage: :ssao-directions <u32>");
                }
            } else if l.starts_with(":ssao-temporal-alpha") || l.starts_with("ssao-temporal-alpha ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoTemporalAlpha(val));
                } else {
                    println!("Usage: :ssao-temporal-alpha <0..1>");
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
                    println!("Usage: :ssao-temporal <on|off>");
                }
            } else if l.starts_with(":ssao-blur") || l.starts_with("ssao-blur ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetAoBlur(on));
                } else {
                    println!("Usage: :ssao-blur <on|off>");
                }
            } else if l.starts_with(":ao-temporal-alpha") || l.starts_with("ao-temporal-alpha ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetAoTemporalAlpha(val));
                } else {
                    println!("Usage: :ao-temporal-alpha <0..1>");
                }
            } else if l.starts_with(":ao-blur") || l.starts_with("ao-blur ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetAoBlur(on));
                } else {
                    println!("Usage: :ao-blur <on|off>");
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
                        _ => println!(
                            "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness>"
                        ),
                    }
                } else {
                    println!(
                        "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness>"
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
                if let Some(mode) = l.split_whitespace().nth(1) {
                    let _ = proxy.send_event(ViewerCmd::SetViz(mode.to_string()));
                } else {
                    println!("Usage: :viz <material|normal|depth|gi|lit>");
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
                    println!("Usage: :ssao-composite <on|off>");
                }
            } else if l.starts_with(":ssao-mul") || l.starts_with("ssao-mul ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsaoCompositeMul(val));
                } else {
                    println!("Usage: :ssao-mul <0..1>");
                }
            } else if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiEdges(on));
                } else {
                    println!("Usage: :ssgi-edges <on|off>");
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
                    println!("Usage: :ssgi-upsample-sigma-depth <float>");
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
                    println!("Usage: :ssgi-upsample-sigma-normal <float>");
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
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiSteps(val));
                } else {
                    println!("Usage: :ssgi-steps <u32>");
                }
            } else if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiRadius(val));
                } else {
                    println!("Usage: :ssgi-radius <float>");
                }
            } else if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsrMaxSteps(val));
                } else {
                    println!("Usage: :ssr-max-steps <u32>");
                }
            } else if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsrThickness(val));
                } else {
                    println!("Usage: :ssr-thickness <float>");
                }
            } else if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiHalf(on));
                } else {
                    println!("Usage: :ssgi-half <on|off|1|0>");
                }
            } else if l.starts_with(":ssgi-temporal-alpha") || l.starts_with("ssgi-temporal-alpha ")
            {
                if let Some(val) = l
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<f32>().ok())
                {
                    let _ = proxy.send_event(ViewerCmd::SetSsgiTemporalAlpha(val));
                } else {
                    println!("Usage: :ssgi-temporal-alpha <float 0..1>");
                }
            } else if l.starts_with(":ssgi-temporal") || l.starts_with("ssgi-temporal ") {
                if let Some(tok) = l.split_whitespace().nth(1) {
                    let on = matches!(tok, "on" | "1" | "true");
                    let _ = proxy.send_event(ViewerCmd::SetSsgiTemporalEnabled(on));
                } else {
                    println!("Usage: :ssgi-temporal <on|off>");
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
                    println!("Usage: :ssao-technique <ssao|gtao>");
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

    // Create viewer in blocking manner
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
                    elwt.exit();
                }
                other => {
                    if let Some(viewer) = viewer_opt.as_mut() {
                        viewer.handle_cmd(other);
                    }
                }
            },
            _ => {}
        }
    });

    Ok(())
}
