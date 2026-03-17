mod execute;
mod setup;

use std::sync::Arc;

/// Depth of Field pass manager with two-pass separable blur
pub struct DofPass {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sampler: wgpu::Sampler,
    pub(super) uniform_buffer_h: wgpu::Buffer,
    pub(super) uniform_buffer_v: wgpu::Buffer,
    pub(super) input_texture: Option<wgpu::Texture>,
    pub input_view: Option<wgpu::TextureView>,
    pub(super) intermediate_texture: Option<wgpu::Texture>,
    pub intermediate_view: Option<wgpu::TextureView>,
    pub(super) current_size: (u32, u32),
}
