// src/viewer/state/gi_state.rs
// GI (Global Illumination) parameters for the Viewer

/// GI rendering parameters (runtime adjustable)
#[derive(Clone, Debug)]
pub struct GiParams {
    pub ao_weight: f32,
    pub ssgi_weight: f32,
    pub ssr_weight: f32,
    pub use_ssao_composite: bool,
    pub ssao_composite_mul: f32,
    pub ssao_blur_enabled: bool,
}

impl Default for GiParams {
    fn default() -> Self {
        Self {
            ao_weight: 1.0,
            ssgi_weight: 1.0,
            ssr_weight: 1.0,
            use_ssao_composite: true,
            ssao_composite_mul: 1.0,
            ssao_blur_enabled: true,
        }
    }
}

impl GiParams {
    /// Set AO weight (clamped 0.0-2.0)
    pub fn set_ao_weight(&mut self, w: f32) {
        self.ao_weight = w.clamp(0.0, 2.0);
    }

    /// Set SSGI weight (clamped 0.0-2.0)
    pub fn set_ssgi_weight(&mut self, w: f32) {
        self.ssgi_weight = w.clamp(0.0, 2.0);
    }

    /// Set SSR weight (clamped 0.0-2.0)
    pub fn set_ssr_weight(&mut self, w: f32) {
        self.ssr_weight = w.clamp(0.0, 2.0);
    }

    /// Toggle SSAO composite
    pub fn set_use_ssao_composite(&mut self, on: bool) {
        self.use_ssao_composite = on;
    }

    /// Set SSAO composite multiplier
    pub fn set_ssao_composite_mul(&mut self, m: f32) {
        self.ssao_composite_mul = m.clamp(0.0, 2.0);
    }

    /// Toggle SSAO blur
    pub fn set_ssao_blur(&mut self, on: bool) {
        self.ssao_blur_enabled = on;
    }
}
