// src/shadows/mod.rs
// Shadow mapping implementations for Workstream B
// Exists to centralize GPU/CPU shadow utilities shared across bindings and pipelines
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py, tests/test_b4_csm.py

mod cascade_math;
mod csm_depth_control;
mod csm_renderer;
mod csm_types;

pub mod blur_pass;
pub mod manager;
pub mod moment_pass;
pub mod state;

// Re-export CSM types from split modules
pub use cascade_math::detect_peter_panning;
pub use csm_renderer::CsmRenderer;
pub use csm_types::{CascadeStatistics, CsmConfig, CsmUniforms, ShadowCascade};

pub use blur_pass::ShadowBlurPass;
pub use manager::{ShadowManager, ShadowManagerConfig};
pub use moment_pass::{create_moment_storage_view, MomentGenerationPass};

// Re-export common shadow types and utilities
pub use csm_renderer::CsmRenderer as CascadedShadowMaps;

/// Largest EVSM exponent an `Rgba16Float` moment atlas can carry.
///
/// The moment atlas stores `exp(c * d)` AND its square for `d` in `[0, 1]`, so the
/// binding constraint is `exp(2 * c) <= 65504` => `c <= 5.545`. Above that the second
/// moment saturates to `+Inf`, `E[x^2] - E[x]^2` becomes `NaN`, and every Chebyshev
/// bound downstream collapses - which renders the whole scene as if fully shadowed.
pub const EVSM_MAX_EXPONENT_RGBA16F: f32 = 5.54;

/// Clamp an EVSM exponent to the range the moment atlas can actually represent.
///
/// Must be applied identically to the moment-generation pass and to the shader
/// uniforms that sample it: producer and consumer have to warp by the same constant.
pub fn clamp_evsm_exponent(exponent: f32) -> f32 {
    exponent.clamp(0.0, EVSM_MAX_EXPONENT_RGBA16F)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evsm_exponent_clamp_keeps_squared_moment_finite_in_rgba16f() {
        assert_eq!(clamp_evsm_exponent(-1.0), 0.0);
        assert_eq!(clamp_evsm_exponent(40.0), EVSM_MAX_EXPONENT_RGBA16F);

        let largest_squared_moment = (2.0 * EVSM_MAX_EXPONENT_RGBA16F).exp();
        assert!(largest_squared_moment.is_finite());
        assert!(largest_squared_moment <= 65_504.0);
    }
}
