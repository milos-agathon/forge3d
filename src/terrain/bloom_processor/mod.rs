//! Standalone bloom processor for terrain offline rendering.

mod config;
mod constructor;
mod execute;
mod uniforms;

pub use config::TerrainBloomConfig;

use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};

pub struct TerrainBloomProcessor {
    brightpass_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
    brightpass_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    brightpass_uniform_buffer: TrackedBuffer,
    blur_uniform_buffer: TrackedBuffer,
    composite_uniform_buffer: TrackedBuffer,
    bright_texture: Option<TrackedTexture>,
    bright_view: Option<wgpu::TextureView>,
    blur_temp_texture: Option<TrackedTexture>,
    blur_temp_view: Option<wgpu::TextureView>,
    blur_result_texture: Option<TrackedTexture>,
    blur_result_view: Option<wgpu::TextureView>,
    current_size: (u32, u32),
}
