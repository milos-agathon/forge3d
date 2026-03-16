#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DofSettingsNative {
    pub enabled: bool,
    pub f_stop: f32,
    pub focus_distance: f32,
    pub focal_length: f32,
    pub tilt_pitch: f32,
    pub tilt_yaw: f32,
    pub method: DofMethodNative,
    pub quality: DofQualityNative,
    pub show_coc: bool,
    pub debug_mode: u32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DofMethodNative {
    Gather,
    Separable,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DofQualityNative {
    Low,
    Medium,
    High,
    Ultra,
}

#[cfg(feature = "extension-module")]
impl Default for DofSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            f_stop: 5.6,
            focus_distance: 100.0,
            focal_length: 50.0,
            tilt_pitch: 0.0,
            tilt_yaw: 0.0,
            method: DofMethodNative::Gather,
            quality: DofQualityNative::Medium,
            show_coc: false,
            debug_mode: 0,
        }
    }
}

#[cfg(feature = "extension-module")]
impl DofSettingsNative {
    pub fn aperture(&self) -> f32 {
        1.0 / self.f_stop.max(0.1)
    }

    pub fn has_tilt(&self) -> bool {
        self.tilt_pitch.abs() > 0.001 || self.tilt_yaw.abs() > 0.001
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct MotionBlurSettingsNative {
    pub enabled: bool,
    pub samples: u32,
    pub shutter_open: f32,
    pub shutter_close: f32,
    pub cam_phi_delta: f32,
    pub cam_theta_delta: f32,
    pub cam_radius_delta: f32,
    pub seed: Option<u64>,
}

#[cfg(feature = "extension-module")]
impl Default for MotionBlurSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: 8,
            shutter_open: 0.0,
            shutter_close: 0.5,
            cam_phi_delta: 0.0,
            cam_theta_delta: 0.0,
            cam_radius_delta: 0.0,
            seed: None,
        }
    }
}

#[cfg(feature = "extension-module")]
impl MotionBlurSettingsNative {
    pub fn has_camera_motion(&self) -> bool {
        self.cam_phi_delta.abs() > 0.001
            || self.cam_theta_delta.abs() > 0.001
            || self.cam_radius_delta.abs() > 0.001
    }

    pub fn shutter_angle(&self) -> f32 {
        (self.shutter_close - self.shutter_open) * 360.0
    }

    pub fn interpolate_camera(&self, t: f32) -> (f32, f32, f32) {
        let shutter_t = self.shutter_open + t * (self.shutter_close - self.shutter_open);
        let phi_offset = self.cam_phi_delta * shutter_t;
        let theta_offset = self.cam_theta_delta * shutter_t;
        let radius_offset = self.cam_radius_delta * shutter_t;
        (phi_offset, theta_offset, radius_offset)
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct LensEffectsSettingsNative {
    pub enabled: bool,
    pub distortion: f32,
    pub chromatic_aberration: f32,
    pub vignette_strength: f32,
    pub vignette_radius: f32,
    pub vignette_softness: f32,
}

#[cfg(feature = "extension-module")]
impl Default for LensEffectsSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            distortion: 0.0,
            chromatic_aberration: 0.0,
            vignette_strength: 0.0,
            vignette_radius: 0.7,
            vignette_softness: 0.3,
        }
    }
}

#[cfg(feature = "extension-module")]
impl LensEffectsSettingsNative {
    pub fn has_any_effect(&self) -> bool {
        self.distortion.abs() > 0.001
            || self.chromatic_aberration.abs() > 0.001
            || self.vignette_strength > 0.001
    }
}
