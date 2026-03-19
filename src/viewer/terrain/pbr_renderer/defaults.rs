use super::types::{
    DenoiseConfig, DofConfig, HeightAoConfig, LensEffectsConfig, MaterialLayerConfig,
    MotionBlurConfig, SunVisConfig, TonemapConfig, VectorOverlayConfig, ViewerTerrainPbrConfig,
    VolumetricsConfig,
};
use crate::viewer::terrain::overlay::OverlayConfig;

impl Default for HeightAoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            directions: 6,
            steps: 16,
            max_distance: 200.0,
            strength: 1.0,
            resolution_scale: 0.5,
        }
    }
}

impl Default for SunVisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "soft".to_string(),
            samples: 4,
            steps: 24,
            max_distance: 400.0,
            softness: 1.0,
            bias: 0.01,
            resolution_scale: 0.5,
        }
    }
}

impl Default for MaterialLayerConfig {
    fn default() -> Self {
        Self {
            snow_enabled: false,
            snow_altitude_min: 2500.0,
            snow_altitude_blend: 200.0,
            snow_slope_max: 45.0,
            rock_enabled: false,
            rock_slope_min: 45.0,
            wetness_enabled: false,
            wetness_strength: 0.3,
        }
    }
}

impl Default for VectorOverlayConfig {
    fn default() -> Self {
        Self {
            depth_test: false,
            depth_bias: 0.001,
            halo_enabled: false,
            halo_width: 2.0,
            halo_color: [0.0, 0.0, 0.0, 0.5],
        }
    }
}

impl Default for LensEffectsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            vignette_strength: 0.0,
            vignette_radius: 0.7,
            vignette_softness: 0.3,
            distortion: 0.0,
            chromatic_aberration: 0.0,
        }
    }
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 500.0,
            f_stop: 5.6,
            focal_length: 50.0,
            quality: 8,
            max_blur_radius: 32.0,
            blur_strength: 25.0,
            tilt_pitch: 0.0,
            tilt_yaw: 0.0,
        }
    }
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: 16,
            shutter_open: 0.0,
            shutter_close: 0.5,
            cam_phi_delta: 0.0,
            cam_theta_delta: 0.0,
            cam_radius_delta: 0.0,
        }
    }
}

impl Default for VolumetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "height".to_string(),
            density: 0.01,
            scattering: 0.5,
            absorption: 0.1,
            height_falloff: 0.01,
            light_shafts: false,
            shaft_intensity: 1.0,
            steps: 32,
            half_res: false,
        }
    }
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: "atrous".to_string(),
            iterations: 3,
            sigma_color: 10.0,
        }
    }
}

impl Default for TonemapConfig {
    fn default() -> Self {
        Self {
            operator: "aces".to_string(),
            white_point: 4.0,
            white_balance_enabled: false,
            temperature: 6500.0,
            tint: 0.0,
        }
    }
}

impl Default for ViewerTerrainPbrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hdr_path: None,
            ibl_intensity: 1.0,
            shadow_technique: "pcss".to_string(),
            shadow_map_res: 2048,
            exposure: 1.0,
            msaa: 1,
            normal_strength: 1.0,
            height_ao: HeightAoConfig::default(),
            sun_visibility: SunVisConfig::default(),
            materials: MaterialLayerConfig::default(),
            vector_overlay: VectorOverlayConfig::default(),
            tonemap: TonemapConfig::default(),
            lens_effects: LensEffectsConfig::default(),
            dof: DofConfig::default(),
            motion_blur: MotionBlurConfig::default(),
            volumetrics: VolumetricsConfig::default(),
            denoise: DenoiseConfig::default(),
            overlay: OverlayConfig::new(),
            vector_overlays_enabled: false,
            debug_mode: 0,
        }
    }
}
