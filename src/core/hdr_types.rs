// src/core/hdr_types.rs
// Type definitions for HDR rendering and tone mapping
// RELEVANT FILES: shaders/tonemap.wgsl

use wgpu::TextureFormat;

/// HDR tone mapping operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMappingOperator {
    /// Simple Reinhard tone mapping: color / (color + 1)
    Reinhard,
    /// Extended Reinhard with white point: color * (1 + color/white²) / (1 + color)
    ReinhardExtended,
    /// Filmic ACES tone mapping
    Aces,
    /// Uncharted 2 filmic tone mapping
    Uncharted2,
    /// Linear exposure-based mapping
    Exposure,
}

impl Default for ToneMappingOperator {
    fn default() -> Self {
        ToneMappingOperator::Reinhard
    }
}

impl ToneMappingOperator {
    /// Convert to shader index
    pub fn as_index(self) -> u32 {
        match self {
            ToneMappingOperator::Reinhard => 0,
            ToneMappingOperator::ReinhardExtended => 1,
            ToneMappingOperator::Aces => 2,
            ToneMappingOperator::Uncharted2 => 3,
            ToneMappingOperator::Exposure => 4,
        }
    }
}

/// HDR rendering configuration
#[derive(Debug, Clone)]
pub struct HdrConfig {
    pub width: u32,
    pub height: u32,
    pub hdr_format: TextureFormat,
    pub tone_mapping: ToneMappingOperator,
    pub exposure: f32,
    pub white_point: f32,
    pub gamma: f32,
}

impl Default for HdrConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            hdr_format: TextureFormat::Rgba16Float,
            tone_mapping: ToneMappingOperator::Reinhard,
            exposure: 1.0,
            white_point: 4.0,
            gamma: 2.2,
        }
    }
}

/// Tone mapping uniforms for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMappingUniforms {
    pub exposure: f32,
    pub white_point: f32,
    pub gamma: f32,
    pub operator_index: u32,
}

impl ToneMappingUniforms {
    /// Create uniforms from config
    pub fn from_config(config: &HdrConfig) -> Self {
        Self {
            exposure: config.exposure,
            white_point: config.white_point,
            gamma: config.gamma,
            operator_index: config.tone_mapping.as_index(),
        }
    }
}
