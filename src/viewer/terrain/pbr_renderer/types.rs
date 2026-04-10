use crate::viewer::terrain::overlay::OverlayConfig;
use std::path::PathBuf;

/// Configuration for PBR terrain rendering mode
#[derive(Debug, Clone)]
pub struct ViewerTerrainPbrConfig {
    pub enabled: bool,
    pub hdr_path: Option<PathBuf>,
    pub ibl_intensity: f32,
    pub hdr_rotate_deg: f32,
    pub shadow_technique: String,
    pub shadow_map_res: u32,
    pub exposure: f32,
    pub msaa: u32,
    pub normal_strength: f32,
    pub height_ao: HeightAoConfig,
    pub sun_visibility: SunVisConfig,
    pub materials: MaterialLayerConfig,
    pub vector_overlay: VectorOverlayConfig,
    pub tonemap: TonemapConfig,
    pub lens_effects: LensEffectsConfig,
    pub dof: DofConfig,
    pub motion_blur: MotionBlurConfig,
    pub volumetrics: VolumetricsConfig,
    pub denoise: DenoiseConfig,
    pub overlay: OverlayConfig,
    pub debug_mode: u32,
}

#[derive(Debug, Clone)]
pub struct HeightAoConfig {
    pub enabled: bool,
    pub directions: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub strength: f32,
    pub resolution_scale: f32,
}

#[derive(Debug, Clone)]
pub struct SunVisConfig {
    pub enabled: bool,
    pub mode: String,
    pub samples: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub softness: f32,
    pub bias: f32,
    pub resolution_scale: f32,
}

#[derive(Debug, Clone)]
pub struct MaterialLayerConfig {
    pub snow_enabled: bool,
    pub snow_altitude_min: f32,
    pub snow_altitude_blend: f32,
    pub snow_slope_max: f32,
    pub rock_enabled: bool,
    pub rock_slope_min: f32,
    pub wetness_enabled: bool,
    pub wetness_strength: f32,
}

#[derive(Debug, Clone)]
pub struct VectorOverlayConfig {
    pub depth_test: bool,
    pub depth_bias: f32,
    pub halo_enabled: bool,
    pub halo_width: f32,
    pub halo_color: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct TonemapConfig {
    pub operator: String,
    pub white_point: f32,
    pub white_balance_enabled: bool,
    pub temperature: f32,
    pub tint: f32,
}

#[derive(Debug, Clone)]
pub struct LensEffectsConfig {
    pub enabled: bool,
    pub vignette_strength: f32,
    pub vignette_radius: f32,
    pub vignette_softness: f32,
    pub distortion: f32,
    pub chromatic_aberration: f32,
}

#[derive(Debug, Clone)]
pub struct DofConfig {
    pub enabled: bool,
    pub focus_distance: f32,
    pub f_stop: f32,
    pub focal_length: f32,
    pub quality: u32,
    pub max_blur_radius: f32,
    pub blur_strength: f32,
    pub tilt_pitch: f32,
    pub tilt_yaw: f32,
}

#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    pub enabled: bool,
    pub samples: u32,
    pub shutter_open: f32,
    pub shutter_close: f32,
    pub cam_phi_delta: f32,
    pub cam_theta_delta: f32,
    pub cam_radius_delta: f32,
}

#[derive(Debug, Clone)]
pub struct DensityVolumeConfig {
    pub preset: String,
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub resolution: [u32; 3],
    pub density_scale: f32,
    pub edge_softness: f32,
    pub noise_strength: f32,
    pub floor_offset: f32,
    pub ceiling: f32,
    pub plume_spread: f32,
    pub wind: [f32; 3],
    pub seed: u32,
}

#[derive(Debug, Clone)]
pub struct VolumetricsConfig {
    pub enabled: bool,
    pub mode: String,
    pub density: f32,
    pub scattering: f32,
    pub absorption: f32,
    pub height_falloff: f32,
    pub light_shafts: bool,
    pub shaft_intensity: f32,
    pub steps: u32,
    pub half_res: bool,
    pub density_volumes: Vec<DensityVolumeConfig>,
}

impl VolumetricsConfig {
    pub fn is_effectively_enabled(&self) -> bool {
        self.enabled && (self.density > 0.0001 || !self.density_volumes.is_empty())
    }
}

#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    pub enabled: bool,
    pub method: String,
    pub iterations: u32,
    pub sigma_color: f32,
}
