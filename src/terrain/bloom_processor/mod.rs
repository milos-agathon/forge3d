//! Standalone bloom processor for terrain offline rendering.

mod config;
mod constructor;
mod execute;
mod uniforms;

pub use config::TerrainBloomConfig;

pub struct TerrainBloomProcessor {
    brightpass_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
    brightpass_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    brightpass_uniform_buffer: wgpu::Buffer,
    blur_uniform_buffer: wgpu::Buffer,
    composite_uniform_buffer: wgpu::Buffer,
    bright_texture: Option<wgpu::Texture>,
    bright_view: Option<wgpu::TextureView>,
    blur_temp_texture: Option<wgpu::Texture>,
    blur_temp_view: Option<wgpu::TextureView>,
    blur_result_texture: Option<wgpu::Texture>,
    blur_result_view: Option<wgpu::TextureView>,
    current_size: (u32, u32),
}
