use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcHeightAoConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub directions: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub resolution_scale: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSunVisConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub samples: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
    #[serde(default)]
    pub softness: Option<f32>,
    #[serde(default)]
    pub bias: Option<f32>,
    #[serde(default)]
    pub resolution_scale: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcMaterialLayerConfig {
    #[serde(default)]
    pub snow_enabled: Option<bool>,
    #[serde(default)]
    pub snow_altitude_min: Option<f32>,
    #[serde(default)]
    pub snow_altitude_blend: Option<f32>,
    #[serde(default)]
    pub snow_slope_max: Option<f32>,
    #[serde(default)]
    pub rock_enabled: Option<bool>,
    #[serde(default)]
    pub rock_slope_min: Option<f32>,
    #[serde(default)]
    pub wetness_enabled: Option<bool>,
    #[serde(default)]
    pub wetness_strength: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcVectorOverlayConfig {
    #[serde(default)]
    pub depth_test: Option<bool>,
    #[serde(default)]
    pub depth_bias: Option<f32>,
    #[serde(default)]
    pub halo_enabled: Option<bool>,
    #[serde(default)]
    pub halo_width: Option<f32>,
    #[serde(default)]
    pub halo_color: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTonemapConfig {
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub white_point: Option<f32>,
    #[serde(default)]
    pub white_balance_enabled: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub tint: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcDofConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub f_stop: Option<f32>,
    #[serde(default)]
    pub focus_distance: Option<f32>,
    #[serde(default)]
    pub focal_length: Option<f32>,
    #[serde(default)]
    pub tilt_pitch: Option<f32>,
    #[serde(default)]
    pub tilt_yaw: Option<f32>,
    #[serde(default)]
    pub quality: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcMotionBlurConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub samples: Option<u32>,
    #[serde(default)]
    pub shutter_open: Option<f32>,
    #[serde(default)]
    pub shutter_close: Option<f32>,
    #[serde(default)]
    pub cam_phi_delta: Option<f32>,
    #[serde(default)]
    pub cam_theta_delta: Option<f32>,
    #[serde(default)]
    pub cam_radius_delta: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcLensEffectsConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub distortion: Option<f32>,
    #[serde(default)]
    pub chromatic_aberration: Option<f32>,
    #[serde(default)]
    pub vignette_strength: Option<f32>,
    #[serde(default)]
    pub vignette_radius: Option<f32>,
    #[serde(default)]
    pub vignette_softness: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcDenoiseConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub iterations: Option<u32>,
    #[serde(default)]
    pub sigma_color: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcDensityVolumeConfig {
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub center: Option<[f32; 3]>,
    #[serde(default)]
    pub size: Option<[f32; 3]>,
    #[serde(default)]
    pub resolution: Option<[u32; 3]>,
    #[serde(default)]
    pub density_scale: Option<f32>,
    #[serde(default)]
    pub edge_softness: Option<f32>,
    #[serde(default)]
    pub noise_strength: Option<f32>,
    #[serde(default)]
    pub floor_offset: Option<f32>,
    #[serde(default)]
    pub ceiling: Option<f32>,
    #[serde(default)]
    pub plume_spread: Option<f32>,
    #[serde(default)]
    pub wind: Option<[f32; 3]>,
    #[serde(default)]
    pub seed: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcVolumetricsConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub density: Option<f32>,
    #[serde(default)]
    pub height_falloff: Option<f32>,
    #[serde(default)]
    pub scattering: Option<f32>,
    #[serde(default)]
    pub absorption: Option<f32>,
    #[serde(default)]
    pub light_shafts: Option<bool>,
    #[serde(default)]
    pub shaft_intensity: Option<f32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub half_res: Option<bool>,
    #[serde(default)]
    pub density_volumes: Vec<IpcDensityVolumeConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSkyConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub turbidity: Option<f32>,
    #[serde(default)]
    pub ground_albedo: Option<f32>,
    #[serde(default)]
    pub sun_intensity: Option<f32>,
    #[serde(default)]
    pub aerial_perspective: Option<bool>,
    #[serde(default)]
    pub sky_exposure: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterLevel {
    #[serde(default)]
    pub positions: Vec<[f32; 3]>,
    #[serde(default)]
    pub normals: Vec<[f32; 3]>,
    #[serde(default)]
    pub indices: Vec<u32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IpcHlodConfig {
    pub hlod_distance: f32,
    pub cluster_radius: f32,
    pub simplify_ratio: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterBatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    #[serde(default)]
    pub max_draw_distance: Option<f32>,
    #[serde(default)]
    pub transforms: Vec<[f32; 16]>,
    #[serde(default)]
    pub levels: Vec<IpcTerrainScatterLevel>,
    #[serde(default)]
    pub hlod: Option<IpcHlodConfig>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ViewerStats {
    pub vb_ready: bool,
    pub vertex_count: u32,
    pub index_count: u32,
    pub scene_has_mesh: bool,
    pub transform_version: u64,
    pub transform_is_identity: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TerrainVolumetricsVolumeReport {
    pub preset: String,
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub resolution: [u32; 3],
    pub atlas_offset: [u32; 3],
    pub voxel_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TerrainVolumetricsReport {
    pub active_volume_count: u32,
    pub atlas_dimensions: [u32; 3],
    pub total_voxels: u64,
    pub texture_bytes: u64,
    pub memory_budget_bytes: u64,
    pub raymarch_steps: u32,
    pub half_res: bool,
    pub volumes: Vec<TerrainVolumetricsVolumeReport>,
}

impl Default for TerrainVolumetricsReport {
    fn default() -> Self {
        Self {
            active_volume_count: 0,
            atlas_dimensions: [0, 0, 0],
            total_voxels: 0,
            texture_bytes: 0,
            memory_budget_bytes: 16 * 1024 * 1024,
            raymarch_steps: 0,
            half_res: false,
            volumes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BundleRequest {
    pub pending: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl BundleRequest {
    pub fn none() -> Self {
        Self {
            pending: false,
            path: None,
            name: None,
        }
    }

    pub fn save(path: String, name: Option<String>) -> Self {
        Self {
            pending: true,
            path: Some(path),
            name,
        }
    }

    pub fn load(path: String) -> Self {
        Self {
            pending: true,
            path: Some(path),
            name: None,
        }
    }
}
