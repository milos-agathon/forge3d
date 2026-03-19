/// Uniform data for bloom bright-pass
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomBrightPassUniforms {
    pub(super) threshold: f32,
    pub(super) softness: f32,
    pub(super) _pad: [f32; 2],
}

/// Uniform data for bloom blur passes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomBlurUniforms {
    pub(super) radius: f32,
    pub(super) strength: f32,
    pub(super) _pad: [f32; 2],
}

/// Uniform data for bloom composite pass
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BloomCompositeUniforms {
    pub(super) intensity: f32,
    pub(super) _pad: [f32; 3],
}
