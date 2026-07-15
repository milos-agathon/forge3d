use crate::geometry::MeshBuffers;
#[cfg(feature = "enable-gpu-instancing")]
use crate::terrain::scatter::ScatterWindSettingsNative;

/// Legacy eight-lane vector IPC row with typed storage:
/// XYZ=f64, RGBA=f32, feature ID=u32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewerVectorVertex {
    pub position: [f64; 3],
    pub color: [f32; 4],
    pub feature_id: u32,
}

impl<'de> serde::Deserialize<'de> for ViewerVectorVertex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let lanes = <Vec<f64> as serde::Deserialize>::deserialize(deserializer)?;
        if lanes.len() != 8 {
            return Err(serde::de::Error::custom(format!(
                "vector vertex must contain exactly 8 lanes, got {}",
                lanes.len()
            )));
        }
        if lanes[..7].iter().any(|value| !value.is_finite()) {
            return Err(serde::de::Error::custom(
                "vector XYZ/RGBA lanes must be finite",
            ));
        }
        let id = lanes[7];
        if !id.is_finite() || id < 0.0 || id.fract() != 0.0 || id > f64::from(u32::MAX) {
            return Err(serde::de::Error::custom(
                "vector feature ID must be an integer in [0, u32::MAX]",
            ));
        }
        let local = crate::camera::Anchor::new();
        let rgb = local.to_render_direction(glam::DVec3::new(lanes[3], lanes[4], lanes[5]));
        let alpha = local.to_render_direction(glam::DVec3::new(lanes[6], 0.0, 0.0));
        Ok(Self {
            position: [lanes[0], lanes[1], lanes[2]],
            color: [rgb.x, rgb.y, rgb.z, alpha.x],
            feature_id: id as u32,
        })
    }
}

#[cfg(test)]
mod vector_vertex_tests {
    use super::ViewerVectorVertex;

    #[test]
    fn legacy_eight_lane_shape_preserves_earth_scale_xyz_and_typed_id() {
        let vertex: ViewerVectorVertex = serde_json::from_str(
            "[6378137.001,500000.002,-5500000.003,1.0,0.5,0.25,1.0,4294967295]",
        )
        .unwrap();
        assert_eq!(
            vertex.position,
            [6_378_137.001, 500_000.002, -5_500_000.003]
        );
        assert_eq!(vertex.color, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(vertex.feature_id, u32::MAX);
    }

    #[test]
    fn malformed_lengths_and_ids_are_rejected_by_the_custom_visitor() {
        for json in [
            "[0,0,0,1,1,1,1]",
            "[0,0,0,1,1,1,1,1,9]",
            "[0,0,0,1,1,1,1,1.5]",
            "[0,0,0,1,1,1,1,-1]",
            "[0,0,0,1,1,1,1,4294967296]",
        ] {
            assert!(
                serde_json::from_str::<ViewerVectorVertex>(json).is_err(),
                "{json}"
            );
        }
    }
}

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
pub struct ViewerDensityVolumeConfig {
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

/// M6: Volumetrics configuration for viewer
#[derive(Debug, Clone, Default)]
pub struct ViewerVolumetricsConfig {
    pub enabled: bool,
    pub mode: String,
    pub density: f32,
    pub height_falloff: f32,
    pub scattering: f32,
    pub absorption: f32,
    pub light_shafts: bool,
    pub shaft_intensity: f32,
    pub steps: u32,
    pub half_res: bool,
    pub density_volumes: Vec<ViewerDensityVolumeConfig>,
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

#[derive(Debug, Clone)]
pub struct ViewerTerrainScatterBlendConfig {
    pub enabled: bool,
    pub bury_depth: f32,
    pub fade_distance: f32,
}

impl Default for ViewerTerrainScatterBlendConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bury_depth: 0.75,
            fade_distance: 2.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViewerTerrainScatterContactConfig {
    pub enabled: bool,
    pub distance: f32,
    pub strength: f32,
    pub vertical_weight: f32,
}

impl Default for ViewerTerrainScatterContactConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            distance: 3.0,
            strength: 0.35,
            vertical_weight: 0.65,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ViewerTerrainScatterBatchConfig {
    pub name: Option<String>,
    pub color: [f32; 4],
    pub max_draw_distance: Option<f32>,
    pub terrain_blend: ViewerTerrainScatterBlendConfig,
    pub terrain_contact: ViewerTerrainScatterContactConfig,
    pub transforms: Vec<[f32; 16]>,
    pub levels: Vec<ViewerTerrainScatterLevelConfig>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub wind: ScatterWindSettingsNative,
    pub hlod_config: Option<crate::terrain::scatter::HlodConfig>,
}
