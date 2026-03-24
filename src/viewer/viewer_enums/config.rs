use crate::geometry::MeshBuffers;
#[cfg(feature = "enable-gpu-instancing")]
use crate::terrain::scatter::ScatterWindSettingsNative;

/// Heightfield ray-traced AO configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerHeightAoConfig {
    pub enabled: bool,
    pub directions: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub strength: f32,
    pub resolution_scale: f32,
}

/// Heightfield ray-traced sun visibility configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerSunVisConfig {
    pub enabled: bool,
    pub mode: String,
    pub samples: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub softness: f32,
    pub bias: f32,
    pub resolution_scale: f32,
}

/// M4: Material layer configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerMaterialLayerConfig {
    pub snow_enabled: bool,
    pub snow_altitude_min: f32,
    pub snow_altitude_blend: f32,
    pub snow_slope_max: f32,
    pub rock_enabled: bool,
    pub rock_slope_min: f32,
    pub wetness_enabled: bool,
    pub wetness_strength: f32,
}

/// M5: Vector overlay configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerVectorOverlayConfig {
    pub depth_test: bool,
    pub depth_bias: f32,
    pub halo_enabled: bool,
    pub halo_width: f32,
    pub halo_color: [f32; 4],
}

/// M6: Tonemap configuration for viewer
#[derive(Debug, Clone)]
pub struct ViewerTonemapConfig {
    pub operator: String,
    pub white_point: f32,
    pub white_balance_enabled: bool,
    pub temperature: f32,
    pub tint: f32,
}

impl Default for ViewerTonemapConfig {
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

/// M3: Depth of Field configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerDofConfig {
    pub enabled: bool,
    pub f_stop: f32,
    pub focus_distance: f32,
    pub focal_length: f32,
    pub tilt_pitch: f32,
    pub tilt_yaw: f32,
    pub quality: String,
}

/// M4: Motion blur configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerMotionBlurConfig {
    pub enabled: bool,
    pub samples: u32,
    pub shutter_open: f32,
    pub shutter_close: f32,
    pub cam_phi_delta: f32,
    pub cam_theta_delta: f32,
    pub cam_radius_delta: f32,
}

/// M5: Lens effects configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerLensEffectsConfig {
    pub enabled: bool,
    pub distortion: f32,
    pub chromatic_aberration: f32,
    pub vignette_strength: f32,
    pub vignette_radius: f32,
    pub vignette_softness: f32,
}

/// M5: Denoise configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerDenoiseConfig {
    pub enabled: bool,
    pub method: String,
    pub iterations: u32,
    pub sigma_color: f32,
}

/// M6: Volumetrics configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerVolumetricsConfig {
    pub enabled: bool,
    pub mode: String,
    pub density: f32,
    pub scattering: f32,
    pub absorption: f32,
    pub light_shafts: bool,
    pub shaft_intensity: f32,
    pub half_res: bool,
}

/// M6: Sky configuration for viewer
#[derive(Debug, Clone)]
pub struct ViewerSkyConfig {
    pub enabled: bool,
    pub turbidity: f32,
    pub ground_albedo: f32,
    pub sun_intensity: f32,
    pub aerial_perspective: bool,
    pub sky_exposure: f32,
}

impl Default for ViewerSkyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            turbidity: 2.0,
            ground_albedo: 0.3,
            sun_intensity: 1.0,
            aerial_perspective: true,
            sky_exposure: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ViewerTerrainScatterLevelConfig {
    pub mesh: MeshBuffers,
    pub max_distance: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct ViewerTerrainScatterBatchConfig {
    pub name: Option<String>,
    pub color: [f32; 4],
    pub max_draw_distance: Option<f32>,
    pub transforms: Vec<[f32; 16]>,
    pub levels: Vec<ViewerTerrainScatterLevelConfig>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub wind: ScatterWindSettingsNative,
}
