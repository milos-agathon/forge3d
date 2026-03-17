use std::sync::Arc;

use super::OverlayLayerGpu;

pub struct OverlayStack {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) layers: Vec<OverlayLayerGpu>,
    pub(super) next_id: u32,
    pub(super) composite_texture: Option<wgpu::Texture>,
    pub(super) composite_view: Option<wgpu::TextureView>,
    pub(super) composite_dimensions: (u32, u32),
    pub(super) dirty: bool,
    pub(super) sampler: wgpu::Sampler,
}

mod composite;
mod core;
