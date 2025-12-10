// src/core/ltc_types.rs
// Type definitions for LTC rectangular area lights (B14)
// RELEVANT FILES: shaders/ltc_area_lights.wgsl

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// LTC matrix and scale lookup table dimensions
pub const LTC_LUT_SIZE: u32 = 64;
pub const LTC_LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Rectangular area light configuration
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RectAreaLight {
    /// Light position in world space
    pub position: [f32; 3],
    /// Light intensity
    pub intensity: f32,

    /// Light right vector (half-width direction)
    pub right: [f32; 3],
    /// Light width (full width)
    pub width: f32,

    /// Light up vector (half-height direction)
    pub up: [f32; 3],
    /// Light height (full height)
    pub height: f32,

    /// Light color (RGB)
    pub color: [f32; 3],
    /// Light emission power
    pub power: f32,

    /// Light normal (computed from right × up)
    pub normal: [f32; 3],
    /// Two-sided lighting flag
    pub two_sided: f32,
}

impl Default for RectAreaLight {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            intensity: 1.0,
            right: [1.0, 0.0, 0.0],
            width: 2.0,
            up: [0.0, 0.0, 1.0],
            height: 2.0,
            color: [1.0, 1.0, 1.0],
            power: 10.0,
            normal: [0.0, -1.0, 0.0],
            two_sided: 0.0,
        }
    }
}

impl RectAreaLight {
    /// Create a new rectangular area light
    pub fn new(
        position: Vec3,
        right: Vec3,
        up: Vec3,
        width: f32,
        height: f32,
        color: Vec3,
        intensity: f32,
        two_sided: bool,
    ) -> Self {
        let normal = right.cross(up).normalize();
        let power = intensity * width * height * std::f32::consts::PI;

        Self {
            position: position.to_array(),
            intensity,
            right: right.normalize().to_array(),
            width: width.max(0.01),
            up: up.normalize().to_array(),
            height: height.max(0.01),
            color: color.to_array(),
            power,
            normal: normal.to_array(),
            two_sided: if two_sided { 1.0 } else { 0.0 },
        }
    }

    /// Create a simple rectangular area light facing down
    pub fn quad(position: Vec3, width: f32, height: f32, color: Vec3, intensity: f32) -> Self {
        Self::new(position, Vec3::X, Vec3::Z, width, height, color, intensity, false)
    }

    /// Update the normal vector from right and up vectors
    pub fn update_normal(&mut self) {
        let right = Vec3::from(self.right);
        let up = Vec3::from(self.up);
        let normal = right.cross(up).normalize();
        self.normal = normal.to_array();
    }

    /// Update power based on current intensity and dimensions
    pub fn update_power(&mut self) {
        self.power = self.intensity * self.width * self.height * std::f32::consts::PI;
    }

    /// Validate light parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.width <= 0.0 {
            return Err("Light width must be positive".to_string());
        }
        if self.height <= 0.0 {
            return Err("Light height must be positive".to_string());
        }
        if self.intensity <= 0.0 {
            return Err("Light intensity must be positive".to_string());
        }

        let right = Vec3::from(self.right);
        let up = Vec3::from(self.up);
        let cross = right.cross(up);
        if cross.length() < 0.001 {
            return Err("Right and up vectors cannot be parallel".to_string());
        }

        Ok(())
    }
}

/// LTC uniform data for GPU shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LTCUniforms {
    /// Number of active rect area lights
    pub light_count: u32,
    /// LTC lookup texture size
    pub lut_size: u32,
    /// Global LTC intensity multiplier
    pub global_intensity: f32,
    /// Enable LTC approximation (vs. exact computation)
    pub enable_ltc: f32,

    /// Quality settings
    pub sample_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for LTCUniforms {
    fn default() -> Self {
        Self {
            light_count: 0,
            lut_size: LTC_LUT_SIZE,
            global_intensity: 1.0,
            enable_ltc: 1.0,
            sample_count: 8,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}
