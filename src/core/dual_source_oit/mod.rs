//! B16: Dual-source blending Order Independent Transparency
//! High-quality OIT using dual-source color blending with WBOIT fallback

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

mod constructor;
mod controls;
mod pass;
mod pipeline;

/// Dual-source OIT rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualSourceOITMode {
    Disabled,
    DualSource,
    WBOITFallback,
    Automatic,
}

/// Quality settings for dual-source OIT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualSourceOITQuality {
    Low,
    Medium,
    High,
    Ultra,
}

/// Dual-source OIT uniforms matching WGSL layout
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DualSourceOITUniforms {
    pub alpha_correction: f32,
    pub depth_weight_scale: f32,
    pub max_fragments: f32,
    pub premultiply_factor: f32,
}

impl Default for DualSourceOITUniforms {
    fn default() -> Self {
        Self {
            alpha_correction: 1.0,
            depth_weight_scale: 1.0,
            max_fragments: 8.0,
            premultiply_factor: 1.0,
        }
    }
}

/// Composition uniforms for final blending
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DualSourceComposeUniforms {
    pub use_dual_source: u32,
    pub tone_mapping_mode: u32,
    pub exposure: f32,
    pub gamma: f32,
}

impl Default for DualSourceComposeUniforms {
    fn default() -> Self {
        Self {
            use_dual_source: 0,
            tone_mapping_mode: 1,
            exposure: 0.0,
            gamma: 2.2,
        }
    }
}

/// Dual-source OIT renderer state
pub struct DualSourceOITRenderer {
    mode: DualSourceOITMode,
    quality: DualSourceOITQuality,
    enabled: bool,
    width: u32,
    height: u32,
    dual_source_supported: bool,
    _max_dual_source_targets: u32,
    uniforms_buffer: wgpu::Buffer,
    compose_uniforms_buffer: wgpu::Buffer,
    dual_source_color_texture: Option<wgpu::Texture>,
    dual_source_color_view: Option<wgpu::TextureView>,
    wboit_color_accum: Option<wgpu::Texture>,
    wboit_reveal_accum: Option<wgpu::Texture>,
    wboit_color_view: Option<wgpu::TextureView>,
    wboit_reveal_view: Option<wgpu::TextureView>,
    dual_source_shader: wgpu::ShaderModule,
    _compose_shader: wgpu::ShaderModule,
    dual_source_bind_group_layout: wgpu::BindGroupLayout,
    _compose_bind_group_layout: wgpu::BindGroupLayout,
    dual_source_pipeline: Option<wgpu::RenderPipeline>,
    compose_pipeline: wgpu::RenderPipeline,
    _dual_source_bind_group: Option<wgpu::BindGroup>,
    compose_bind_group: Option<wgpu::BindGroup>,
    _sampler: wgpu::Sampler,
    frame_stats: DualSourceOITStats,
    uniforms: DualSourceOITUniforms,
    compose_uniforms: DualSourceComposeUniforms,
}

/// Performance and quality statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct DualSourceOITStats {
    pub frames_rendered: u64,
    pub dual_source_frames: u64,
    pub wboit_fallback_frames: u64,
    pub average_fragment_count: f32,
    pub peak_fragment_count: f32,
    pub quality_score: f32,
}
