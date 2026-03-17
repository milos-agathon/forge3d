mod config;
mod effect;
mod execute;
mod init;
mod resources;

pub use config::BloomConfig;

pub struct BloomEffect {
    pub(super) config: crate::core::postfx::PostFxConfig,
    pub(super) bloom_config: BloomConfig,
    pub(super) brightpass_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) blur_h_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) blur_v_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) composite_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) brightpass_layout: Option<wgpu::BindGroupLayout>,
    pub(super) blur_layout: Option<wgpu::BindGroupLayout>,
    pub(super) composite_layout: Option<wgpu::BindGroupLayout>,
    pub(super) brightpass_uniform_buffer: Option<wgpu::Buffer>,
    pub(super) blur_uniform_buffer: Option<wgpu::Buffer>,
    pub(super) composite_uniform_buffer: Option<wgpu::Buffer>,
    pub(super) brightpass_texture_index: Option<usize>,
    pub(super) blur_temp_texture_index: Option<usize>,
}
