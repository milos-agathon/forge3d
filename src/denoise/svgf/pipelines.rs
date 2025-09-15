// src/denoise/svgf/pipelines.rs
// Pipeline descriptors and resource layout notes for SVGF stages (stub)
// Purpose: document intended wgpu pipeline wiring; no functional code to avoid build impact
// RELEVANT FILES:src/shaders/svgf_reproject.wgsl,src/shaders/svgf_variance.wgsl,src/shaders/svgf_atrous.wgsl

#[allow(dead_code)]
pub struct SvgfParams {
    pub iterations: u32,
    pub alpha: f32,
}

#[allow(dead_code)]
impl Default for SvgfParams {
    fn default() -> Self {
        Self { iterations: 5, alpha: 0.2 }
    }
}

