// src/viewer/state/fog_state.rs
// Fog rendering parameters for the Viewer

use super::super::viewer_enums::FogMode;

/// Fog rendering parameters (runtime adjustable)
#[derive(Clone, Debug)]
pub struct FogParams {
    pub enabled: bool,
    pub density: f32,
    pub g: f32, // phase function asymmetry (-0.999 to 0.999)
    pub steps: u32,
    pub temporal_alpha: f32, // 0.0-0.9
    pub use_shadows: bool,
    pub mode: FogMode,
    pub half_res_enabled: bool,
    pub bilateral: bool,
    pub upsigma: f32,
    pub frame_index: u32,
}

impl Default for FogParams {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 0.02,
            g: 0.5,
            steps: 64,
            temporal_alpha: 0.8,
            use_shadows: false,
            mode: FogMode::Raymarch,
            half_res_enabled: false,
            bilateral: true,
            upsigma: 0.5,
            frame_index: 0,
        }
    }
}

impl FogParams {
    /// Toggle fog on/off
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// Set density (>= 0.0)
    pub fn set_density(&mut self, v: f32) {
        self.density = v.max(0.0);
    }

    /// Set phase function g (-0.999 to 0.999)
    pub fn set_g(&mut self, v: f32) {
        self.g = v.clamp(-0.999, 0.999);
    }

    /// Set ray march steps (>= 1)
    pub fn set_steps(&mut self, v: u32) {
        self.steps = v.max(1);
    }

    /// Set temporal alpha (0.0-0.9)
    pub fn set_temporal_alpha(&mut self, v: f32) {
        self.temporal_alpha = v.clamp(0.0, 0.9);
    }

    /// Toggle shadows
    pub fn set_use_shadows(&mut self, on: bool) {
        self.use_shadows = on;
    }

    /// Set fog mode
    pub fn set_mode(&mut self, m: FogMode) {
        self.mode = m;
    }

    /// Toggle half-resolution rendering
    pub fn set_half_res(&mut self, on: bool) {
        self.half_res_enabled = on;
    }

    /// Increment frame index (for temporal jitter)
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}
