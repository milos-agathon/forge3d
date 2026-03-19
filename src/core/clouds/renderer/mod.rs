use glam::{Mat4, Vec2, Vec3};
use std::borrow::Cow;
use wgpu::{
    vertex_attr_array, BindGroup, BindGroupLayout, Buffer, BufferAddress, ComputePipeline, Device,
    Queue, RenderPipeline, Sampler, Texture, TextureFormat, TextureView,
};

use super::types::*;

mod controls;
mod data;
mod init;
mod render;
mod resources;
mod textures;

/// Main cloud rendering system
pub struct CloudRenderer {
    pub uniforms: CloudUniforms,
    pub params: CloudParams,
    pub uniform_buffer: Buffer,
    pub cloud_pipeline: RenderPipeline,
    pub compute_pipeline: Option<ComputePipeline>,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub bind_group_layout_uniforms: BindGroupLayout,
    pub bind_group_layout_textures: BindGroupLayout,
    pub bind_group_layout_ibl: BindGroupLayout,
    pub bind_group_uniforms: BindGroup,
    pub bind_group_textures: Option<BindGroup>,
    pub bind_group_ibl: Option<BindGroup>,
    pub noise_texture: Option<Texture>,
    pub noise_view: Option<TextureView>,
    pub shape_texture: Option<Texture>,
    pub shape_view: Option<TextureView>,
    pub ibl_irradiance_texture: Option<Texture>,
    pub ibl_irradiance_view: Option<TextureView>,
    pub ibl_prefilter_texture: Option<Texture>,
    pub ibl_prefilter_view: Option<TextureView>,
    pub cloud_sampler: Sampler,
    pub shape_sampler: Sampler,
    pub ibl_sampler: Sampler,
    pub noise_resolution: u32,
    pub time: f32,
    pub enabled: bool,
}
