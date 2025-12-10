use wgpu::TextureFormat;

/// HDR tone mapping operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMappingOperator {
    /// Simple Reinhard tone mapping: color / (color + 1)
    Reinhard = 0,
    /// Extended Reinhard with white point: color * (1 + color/white²) / (1 + color)
    ReinhardExtended = 1,
    /// Filmic ACES tone mapping
    Aces = 2,
    /// Uncharted 2 filmic tone mapping
    Uncharted2 = 3,
    /// Linear exposure-based mapping
    Exposure = 4,
}

impl Default for ToneMappingOperator {
    fn default() -> Self {
        ToneMappingOperator::Reinhard
    }
}

/// HDR off-screen pipeline configuration
#[derive(Debug, Clone)]
pub struct HdrOffscreenConfig {
    pub width: u32,
    pub height: u32,
    pub hdr_format: TextureFormat,
    pub ldr_format: TextureFormat,
    pub tone_mapping: ToneMappingOperator,
    pub exposure: f32,
    pub white_point: f32,
    pub gamma: f32,
    pub sample_count: u32,
}

impl Default for HdrOffscreenConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            hdr_format: TextureFormat::Rgba16Float,
            ldr_format: TextureFormat::Rgba8UnormSrgb,
            tone_mapping: ToneMappingOperator::Reinhard,
            exposure: 1.0,
            white_point: 4.0,
            gamma: 2.2,
            sample_count: 1,
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
    pub operator_index: u32, // 0=Reinhard, 1=ReinhardExtended, 2=ACES, etc.
}
