// src/viewer/terrain/scene.rs
// Terrain scene management for the interactive viewer

use super::denoise::DenoisePass;
use super::render::TerrainUniforms;
use super::shader::TERRAIN_SHADER;
use super::vector_overlay::{drape_vertices, VectorOverlayLayer, VectorOverlayStack, VectorVertex};
use crate::shadows::{CsmConfig, CsmRenderer};
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// P0.1/M1: WBOIT compose shader for final compositing of accumulation buffers
const WBOIT_COMPOSE_SHADER: &str = r#"
// WBOIT Compose Shader - composites accumulation buffers to final output

@group(0) @binding(0)
var color_accumulation: texture_2d<f32>;

@group(0) @binding(1)
var reveal_accumulation: texture_2d<f32>;

@group(0) @binding(2)
var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, -y) * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let accum_color = textureSample(color_accumulation, tex_sampler, in.uv);
    let reveal = textureSample(reveal_accumulation, tex_sampler, in.uv).r;
    
    let epsilon = 1e-5;
    var final_color: vec3<f32>;
    if (accum_color.a > epsilon) {
        final_color = accum_color.rgb / accum_color.a;
    } else {
        final_color = vec3<f32>(0.0);
    }
    
    let final_alpha = 1.0 - reveal;
    return vec4<f32>(final_color, final_alpha);
}
"#;

/// Stored terrain data for interactive viewer rendering
pub struct ViewerTerrainData {
    pub heightmap: Vec<f32>,
    pub dimensions: (u32, u32),
    pub domain: (f32, f32),
    pub heightmap_texture: wgpu::Texture,
    pub heightmap_view: wgpu::TextureView,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    // Camera
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_fov_deg: f32,
    // Sun/lighting
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub ambient: f32,
    // Terrain rendering
    pub z_scale: f32,
    pub shadow_intensity: f32,
    pub background_color: [f32; 3],
    pub water_level: f32,
    pub water_color: [f32; 3],
}

impl ViewerTerrainData {
    /// Set camera state from animation keyframe values
    pub fn set_camera_state(&mut self, phi_deg: f32, theta_deg: f32, radius: f32, fov_deg: f32) {
        self.cam_phi_deg = phi_deg;
        self.cam_theta_deg = theta_deg;
        self.cam_radius = radius;
        self.cam_fov_deg = fov_deg;
    }

    /// Get current camera state as tuple (phi, theta, radius, fov)
    pub fn get_camera_state(&self) -> (f32, f32, f32, f32) {
        (
            self.cam_phi_deg,
            self.cam_theta_deg,
            self.cam_radius,
            self.cam_fov_deg,
        )
    }
}

/// Simple terrain scene for interactive viewer
pub struct ViewerTerrainScene {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_texture: Option<wgpu::Texture>,
    pub(super) depth_view: Option<wgpu::TextureView>,
    pub(super) depth_size: (u32, u32),
    pub terrain: Option<ViewerTerrainData>,
    /// PBR+POM rendering configuration (opt-in, default off)
    pub pbr_config: super::pbr_renderer::ViewerTerrainPbrConfig,
    /// PBR pipeline (created lazily when PBR mode enabled)
    pub pbr_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) pbr_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) pbr_uniform_buffer: Option<wgpu::Buffer>,
    pub(super) pbr_bind_group: Option<wgpu::BindGroup>,
    // Heightfield compute pipelines for AO and sun visibility
    pub(super) height_ao_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) height_ao_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) height_ao_texture: Option<wgpu::Texture>,
    pub(super) height_ao_view: Option<wgpu::TextureView>,
    pub(super) height_ao_uniform_buffer: Option<wgpu::Buffer>,
    pub(super) sun_vis_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) sun_vis_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) sun_vis_texture: Option<wgpu::Texture>,
    pub(super) sun_vis_view: Option<wgpu::TextureView>,
    pub(super) sun_vis_uniform_buffer: Option<wgpu::Buffer>,
    pub(super) sampler_nearest: Option<wgpu::Sampler>,
    // Fallback 1x1 white texture for when AO/sun_vis are disabled
    pub(super) fallback_texture: Option<wgpu::Texture>,
    pub(super) fallback_texture_view: Option<wgpu::TextureView>,
    // Post-process pass for lens effects (distortion, CA, vignette)
    pub(super) post_process: Option<super::post_process::PostProcessPass>,
    // Depth of Field pass
    pub(super) dof_pass: Option<super::dof::DofPass>,
    // Motion blur accumulator
    pub(super) motion_blur_pass: Option<super::motion_blur::MotionBlurAccumulator>,
    // P5: Volumetrics pass
    pub(super) volumetrics_pass: Option<super::volumetrics::VolumetricsPass>,
    // M5: Denoise pass
    pub(super) denoise_pass: Option<DenoisePass>,
    pub(super) surface_format: wgpu::TextureFormat,
    // Overlay layer stack for lit draped overlays
    pub overlay_stack: Option<super::overlay::OverlayStack>,
    // Option B: Vector overlay geometry stack
    pub vector_overlay_stack: Option<VectorOverlayStack>,
    // P0.1/M1: OIT mode for transparent overlay rendering
    pub oit_enabled: bool,
    pub oit_mode: String,
    // P0.1/M1: WBOIT accumulation resources
    pub(super) wboit_color_texture: Option<wgpu::Texture>,
    pub(super) wboit_color_view: Option<wgpu::TextureView>,
    pub(super) wboit_reveal_texture: Option<wgpu::Texture>,
    pub(super) wboit_reveal_view: Option<wgpu::TextureView>,
    pub(super) wboit_compose_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) wboit_compose_bind_group: Option<wgpu::BindGroup>,
    pub(super) wboit_compose_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) wboit_sampler: Option<wgpu::Sampler>,
    pub(super) wboit_size: (u32, u32),
    // P6.2: Shadow mapping support
    pub(super) csm_renderer: Option<crate::shadows::CsmRenderer>,
    pub(super) moment_pass: Option<crate::shadows::MomentGenerationPass>,
    pub(super) csm_uniform_buffer: Option<wgpu::Buffer>,

    // Shadow rendering resources
    pub(super) shadow_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) shadow_uniform_buffers: Vec<wgpu::Buffer>, // One per cascade
    pub(super) shadow_bind_groups: Vec<wgpu::BindGroup>,  // One per cascade

    // P1.4: TAA support for terrain viewer
    pub(super) taa_renderer: Option<crate::core::taa::TaaRenderer>,
    pub(super) taa_jitter: crate::core::jitter::JitterState,
    pub(super) prev_view_proj: glam::Mat4,
    pub(super) velocity_texture: Option<wgpu::Texture>,
    pub(super) velocity_view: Option<wgpu::TextureView>,
    pub(super) velocity_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) velocity_bind_group_layout: Option<wgpu::BindGroupLayout>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_renderer: crate::render::mesh_instanced::MeshInstancedRenderer,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_batches: Vec<crate::terrain::scatter::TerrainScatterBatch>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_last_frame_stats: crate::terrain::scatter::TerrainScatterFrameStats,
}

mod core;
mod overlays;
mod pbr_compute;
mod pipeline_init;
#[cfg(feature = "enable-gpu-instancing")]
pub(in crate::viewer::terrain) mod scatter;
mod terrain_load;
