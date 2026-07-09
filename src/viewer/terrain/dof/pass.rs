mod execute;
mod setup;

use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};
use std::sync::Arc;

/// Depth of Field pass manager with two-pass separable blur
pub struct DofPass {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sampler: wgpu::Sampler,
    pub(super) uniform_buffer_h: TrackedBuffer,
    pub(super) uniform_buffer_v: TrackedBuffer,
    pub(super) input_texture: Option<TrackedTexture>,
    pub input_view: Option<wgpu::TextureView>,
    pub(super) intermediate_texture: Option<TrackedTexture>,
    pub intermediate_view: Option<wgpu::TextureView>,
    pub(super) current_size: (u32, u32),
}
