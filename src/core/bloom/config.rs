/// Bloom effect configuration
#[derive(Debug, Clone, Copy)]
pub struct BloomConfig {
    /// Enabled flag (false = passthrough, no bloom)
    pub enabled: bool,
    /// Brightness threshold for bloom extraction (default 1.5 = HDR only)
    pub threshold: f32,
    /// Softness of threshold transition (0.0 = hard, 1.0 = very soft)
    pub softness: f32,
    /// Bloom intensity/strength when compositing (0.0-1.0+)
    pub strength: f32,
    /// Blur radius multiplier (affects spread)
    pub radius: f32,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 1.5,
            softness: 0.5,
            strength: 0.3,
            radius: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomBrightPassUniforms {
    pub(super) threshold: f32,
    pub(super) softness: f32,
    pub(super) _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomBlurUniforms {
    pub(super) radius: f32,
    pub(super) strength: f32,
    pub(super) _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomCompositeUniforms {
    pub(super) intensity: f32,
    pub(super) _pad: [f32; 3],
}
