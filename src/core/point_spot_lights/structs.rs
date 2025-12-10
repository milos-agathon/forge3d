use super::types::{Light, PointSpotLightUniforms};
use wgpu;
use std::collections::HashMap;

/// Core renderer for point and spot lights
pub struct PointSpotLightRenderer {
    // Rendering pipelines
    pub(crate) deferred_pipeline: wgpu::RenderPipeline,
    pub(crate) _forward_pipeline: wgpu::RenderPipeline,
    pub(crate) _shadow_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    pub(crate) main_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) shadow_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    pub(crate) uniforms_buffer: wgpu::Buffer,
    pub(crate) lights_buffer: wgpu::Buffer,

    // Shadow mapping resources
    pub(crate) _shadow_map_array: Option<wgpu::Texture>,
    pub(crate) shadow_map_view: Option<wgpu::TextureView>,
    pub(crate) shadow_sampler: wgpu::Sampler,

    // Bind groups
    pub(crate) main_bind_group: Option<wgpu::BindGroup>,
    pub(crate) shadow_bind_group: Option<wgpu::BindGroup>,

    // Light management
    pub(crate) lights: Vec<Light>,
    pub(crate) light_id_counter: u32,
    pub(crate) light_id_map: HashMap<u32, usize>,
    pub(crate) max_lights: usize,

    // Configuration
    pub(crate) uniforms: PointSpotLightUniforms,
    pub(crate) _shadow_map_size: u32,
}
