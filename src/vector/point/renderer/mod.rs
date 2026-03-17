use super::types::*;
use crate::core::error::RenderError;
use crate::vector::api::PointDef;
use crate::vector::data::{validate_point_instances, PointInstance};
use crate::vector::layer::Layer;
use glam::Vec2;

mod config;
mod draw;
mod init;
mod pipelines;
mod upload;

#[cfg(test)]
mod tests;

pub use upload::cluster_points;

/// Instanced point renderer with H20,H21,H22 enhancements
pub struct PointRenderer {
    render_pipeline: wgpu::RenderPipeline,
    instance_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pick_pipeline: wgpu::RenderPipeline,
    pick_uniform_buffer: wgpu::Buffer,
    pick_bind_group: wgpu::BindGroup,
    oit_pipeline: wgpu::RenderPipeline,
    instance_capacity: usize,
    debug_flags: DebugFlags,
    texture_atlas: Option<TextureAtlas>,
    enable_clip_w_scaling: bool,
    depth_range: (f32, f32),
    shape_mode: u32,
    lod_threshold: f32,
}
