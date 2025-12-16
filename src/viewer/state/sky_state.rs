// src/viewer/state/sky_state.rs
// Sky rendering parameters for the Viewer

/// Sky rendering parameters (runtime adjustable)
#[derive(Clone, Debug)]
pub struct SkyParams {
    pub enabled: bool,
    pub model_id: u32,      // 0=Preetham, 1=Hosek-Wilkie
    pub turbidity: f32,     // 1.0-10.0
    pub ground_albedo: f32, // 0.0-1.0
    pub exposure: f32,      // >= 0.0
    pub sun_intensity: f32, // >= 0.0
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            enabled: true,
            model_id: 1, // Hosek-Wilkie
            turbidity: 2.0,
            ground_albedo: 0.3,
            exposure: 1.0,
            sun_intensity: 1.0,
        }
    }
}

impl SkyParams {
    /// Apply toggle
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// Set model (0=Preetham, 1=Hosek-Wilkie)
    pub fn set_model(&mut self, id: u32) {
        self.model_id = id;
        self.enabled = true;
    }

    /// Set turbidity (clamped 1.0-10.0)
    pub fn set_turbidity(&mut self, t: f32) {
        self.turbidity = t.clamp(1.0, 10.0);
    }

    /// Set ground albedo (clamped 0.0-1.0)
    pub fn set_ground_albedo(&mut self, a: f32) {
        self.ground_albedo = a.clamp(0.0, 1.0);
    }

    /// Set exposure (>= 0.0)
    pub fn set_exposure(&mut self, e: f32) {
        self.exposure = e.max(0.0);
    }

    /// Set sun intensity (>= 0.0)
    pub fn set_sun_intensity(&mut self, i: f32) {
        self.sun_intensity = i.max(0.0);
    }
}
