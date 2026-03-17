// B11: Water Surface Color Toggle - Configurable water surface rendering system
// Provides pipeline uniform controlling water albedo/hue with Python setter
// Supports water tint toggling, transparency, and basic wave animation

use glam::{Mat4, Vec2, Vec3};
use std::borrow::Cow;
use wgpu::{
    vertex_attr_array, AddressMode, BindGroup, BindGroupLayout, BlendComponent, BlendFactor,
    BlendOperation, BlendState, Buffer, BufferAddress, Device, Queue, RenderPipeline, Sampler,
    Texture, TextureFormat, TextureView,
};

mod constructor;
mod controls;
mod render;
mod uniforms;

/// Water surface rendering modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaterSurfaceMode {
    Disabled,
    Transparent,
    Reflective,
    Animated,
}

/// Water surface configuration parameters
#[derive(Debug, Clone)]
pub struct WaterSurfaceParams {
    pub mode: WaterSurfaceMode,
    pub size: f32,
    pub height: f32,
    pub alpha: f32,
    pub base_color: Vec3,
    pub hue_shift: f32,
    pub tint_color: Vec3,
    pub tint_strength: f32,
    pub wave_amplitude: f32,
    pub wave_frequency: f32,
    pub wave_speed: f32,
    pub ripple_scale: f32,
    pub ripple_speed: f32,
    pub flow_direction: Vec2,
    pub reflection_strength: f32,
    pub refraction_strength: f32,
    pub fresnel_power: f32,
    pub roughness: f32,
    pub foam_enabled: bool,
    pub foam_width_px: f32,
    pub foam_intensity: f32,
    pub foam_noise_scale: f32,
    pub debug_mode: u32,
}

impl Default for WaterSurfaceParams {
    fn default() -> Self {
        Self {
            mode: WaterSurfaceMode::Transparent,
            size: 1000.0,
            height: 0.0,
            alpha: 0.7,
            base_color: Vec3::new(0.1, 0.3, 0.6),
            hue_shift: 0.0,
            tint_color: Vec3::new(0.0, 0.8, 1.0),
            tint_strength: 0.2,
            wave_amplitude: 0.1,
            wave_frequency: 2.0,
            wave_speed: 1.0,
            ripple_scale: 1.0,
            ripple_speed: 0.5,
            flow_direction: Vec2::new(1.0, 0.0),
            reflection_strength: 0.8,
            refraction_strength: 0.3,
            fresnel_power: 5.0,
            roughness: 0.1,
            foam_enabled: false,
            foam_width_px: 2.0,
            foam_intensity: 0.85,
            foam_noise_scale: 20.0,
            debug_mode: 0,
        }
    }
}

/// Water surface uniforms structure (must match WGSL exactly)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterSurfaceUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub world_transform: [[f32; 4]; 4],
    pub surface_params: [f32; 4],
    pub color_params: [f32; 4],
    pub wave_params: [f32; 4],
    pub tint_params: [f32; 4],
    pub lighting_params: [f32; 4],
    pub animation_params: [f32; 4],
    pub foam_params: [f32; 4],
    pub debug_params: [f32; 4],
}

impl Default for WaterSurfaceUniforms {
    fn default() -> Self {
        let params = WaterSurfaceParams::default();
        let mut uniforms = Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            world_transform: Mat4::IDENTITY.to_cols_array_2d(),
            surface_params: [params.size, params.height, 1.0, params.alpha],
            color_params: [
                params.base_color.x,
                params.base_color.y,
                params.base_color.z,
                params.hue_shift,
            ],
            wave_params: [
                params.wave_amplitude,
                params.wave_frequency,
                params.wave_speed,
                0.0,
            ],
            tint_params: [
                params.tint_color.x,
                params.tint_color.y,
                params.tint_color.z,
                params.tint_strength,
            ],
            lighting_params: [
                params.reflection_strength,
                params.refraction_strength,
                params.fresnel_power,
                params.roughness,
            ],
            animation_params: [
                params.ripple_scale,
                params.ripple_speed,
                params.flow_direction.x,
                params.flow_direction.y,
            ],
            foam_params: [
                params.foam_width_px,
                params.foam_intensity,
                params.foam_noise_scale,
                0.0,
            ],
            debug_params: [0.0; 4],
        };
        uniforms.world_transform =
            Mat4::from_translation(Vec3::new(0.0, params.height, 0.0)).to_cols_array_2d();
        uniforms
    }
}

/// Main water surface rendering system
pub struct WaterSurfaceRenderer {
    pub uniforms: WaterSurfaceUniforms,
    pub params: WaterSurfaceParams,
    pub uniform_buffer: Buffer,
    pub water_pipeline: RenderPipeline,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub mask_bind_group_layout: BindGroupLayout,
    pub mask_bind_group: BindGroup,
    pub mask_texture: Texture,
    pub mask_view: TextureView,
    pub mask_sampler: Sampler,
    pub mask_size: (u32, u32),
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub animation_time: f32,
    pub enabled: bool,
}
