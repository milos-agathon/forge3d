// src/viewer/ipc_split/protocol.rs
// IPC protocol definitions for non-blocking viewer control via TCP + NDJSON

use serde::{Deserialize, Serialize};

use crate::viewer::viewer_enums::ViewerCmd;

/// IPC request envelope (NDJSON format)
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum IpcRequest {
    /// Get viewer stats (geometry readiness, vertex/index counts)
    GetStats,
    /// Load an OBJ file
    LoadObj { path: String },
    /// Load a glTF/GLB file
    LoadGltf { path: String },
    /// Set object transform
    SetTransform {
        #[serde(default)]
        translation: Option<[f32; 3]>,
        #[serde(default)]
        rotation_quat: Option<[f32; 4]>,
        #[serde(default)]
        scale: Option<[f32; 3]>,
    },
    /// Set camera look-at
    CamLookat {
        eye: [f32; 3],
        target: [f32; 3],
        #[serde(default = "default_up")]
        up: [f32; 3],
    },
    /// Set field of view
    SetFov { deg: f32 },
    /// Set sun lighting (azimuth/elevation)
    LitSun {
        azimuth_deg: f32,
        elevation_deg: f32,
    },
    /// Set IBL (environment map)
    LitIbl {
        path: String,
        #[serde(default = "default_intensity")]
        intensity: f32,
    },
    /// Set terrain z-scale (height exaggeration)
    SetZScale { value: f32 },
    /// Take a snapshot
    Snapshot {
        path: String,
        #[serde(default)]
        width: Option<u32>,
        #[serde(default)]
        height: Option<u32>,
    },
    /// Close the viewer
    Close,
    /// Load terrain DEM file for interactive viewing
    LoadTerrain { path: String },
    /// Set terrain camera parameters
    SetTerrainCamera {
        #[serde(default = "default_phi")]
        phi_deg: f32,
        #[serde(default = "default_theta")]
        theta_deg: f32,
        #[serde(default = "default_radius")]
        radius: f32,
        #[serde(default = "default_fov")]
        fov_deg: f32,
    },
    /// Set terrain sun parameters
    SetTerrainSun {
        #[serde(default = "default_sun_azimuth")]
        azimuth_deg: f32,
        #[serde(default = "default_sun_elevation")]
        elevation_deg: f32,
        #[serde(default = "default_sun_intensity")]
        intensity: f32,
    },
    /// Set multiple terrain parameters at once (like rayshader::render_camera)
    SetTerrain {
        #[serde(default)]
        phi: Option<f32>,
        #[serde(default)]
        theta: Option<f32>,
        #[serde(default)]
        radius: Option<f32>,
        #[serde(default)]
        fov: Option<f32>,
        #[serde(default)]
        sun_azimuth: Option<f32>,
        #[serde(default)]
        sun_elevation: Option<f32>,
        #[serde(default)]
        sun_intensity: Option<f32>,
        #[serde(default)]
        ambient: Option<f32>,
        #[serde(default)]
        zscale: Option<f32>,
        #[serde(default)]
        shadow: Option<f32>,
        #[serde(default)]
        background: Option<[f32; 3]>,
        #[serde(default)]
        water_level: Option<f32>,
        #[serde(default)]
        water_color: Option<[f32; 3]>,
    },
    /// Get current terrain parameters
    GetTerrainParams,
    /// Configure terrain PBR+POM rendering mode
    SetTerrainPbr {
        #[serde(default)]
        enabled: Option<bool>,
        #[serde(default)]
        hdr_path: Option<String>,
        #[serde(default)]
        ibl_intensity: Option<f32>,
        #[serde(default)]
        shadow_technique: Option<String>,
        #[serde(default)]
        shadow_map_res: Option<u32>,
        #[serde(default)]
        exposure: Option<f32>,
        #[serde(default)]
        msaa: Option<u32>,
        #[serde(default)]
        normal_strength: Option<f32>,
        #[serde(default)]
        height_ao: Option<IpcHeightAoConfig>,
        #[serde(default)]
        sun_visibility: Option<IpcSunVisConfig>,
        /// M4: Material layer settings
        #[serde(default)]
        materials: Option<IpcMaterialLayerConfig>,
        /// M5: Vector overlay settings
        #[serde(default)]
        vector_overlay: Option<IpcVectorOverlayConfig>,
        /// M6: Tonemap settings
        #[serde(default)]
        tonemap: Option<IpcTonemapConfig>,
        /// M3: Depth of Field settings
        #[serde(default)]
        dof: Option<IpcDofConfig>,
        /// M4: Motion blur settings
        #[serde(default)]
        motion_blur: Option<IpcMotionBlurConfig>,
        /// M5: Lens effects settings
        #[serde(default)]
        lens_effects: Option<IpcLensEffectsConfig>,
        /// M5: Denoise settings
        #[serde(default)]
        denoise: Option<IpcDenoiseConfig>,
        /// M6: Volumetrics settings
        #[serde(default)]
        volumetrics: Option<IpcVolumetricsConfig>,
        /// M6: Sky settings
        #[serde(default)]
        sky: Option<IpcSkyConfig>,
    },
    /// Load an overlay texture from an image file
    LoadOverlay {
        /// Unique name for this overlay layer
        name: String,
        /// Path to image file (PNG, JPEG, etc.)
        path: String,
        /// Extent in terrain UV space: [u_min, v_min, u_max, v_max]
        /// None means full terrain coverage [0, 0, 1, 1]
        #[serde(default)]
        extent: Option<[f32; 4]>,
        /// Opacity (0.0 - 1.0)
        #[serde(default)]
        opacity: Option<f32>,
        /// Z-order for stacking (lower = behind)
        #[serde(default)]
        z_order: Option<i32>,
    },
    /// Remove an overlay by ID
    RemoveOverlay {
        id: u32,
    },
    /// Set overlay visibility
    SetOverlayVisible {
        id: u32,
        visible: bool,
    },
    /// Set overlay opacity (0.0 - 1.0)
    SetOverlayOpacity {
        id: u32,
        opacity: f32,
    },
    /// Set global overlay opacity multiplier (0.0 - 1.0)
    SetGlobalOverlayOpacity {
        opacity: f32,
    },
    /// Enable or disable overlays
    SetOverlaysEnabled {
        enabled: bool,
    },
    /// Set overlay solid surface mode (true=show base surface, false=hide where alpha=0)
    SetOverlaySolid {
        solid: bool,
    },
    /// List all overlay IDs
    ListOverlays,
}

/// Heightfield ray-traced AO configuration (IPC)
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

/// Heightfield ray-traced sun visibility configuration (IPC)
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

/// M4: Material layer configuration (IPC)
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

/// M5: Vector overlay configuration (IPC)
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

/// M6: Tonemap configuration (IPC)
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

/// M3: Depth of Field configuration (IPC)
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

/// M4: Motion blur configuration (IPC)
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

/// M5: Lens effects configuration (IPC)
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

/// M5: Denoise configuration (IPC)
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

/// M6: Volumetrics configuration (IPC)
#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcVolumetricsConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub density: Option<f32>,
    #[serde(default)]
    pub scattering: Option<f32>,
    #[serde(default)]
    pub absorption: Option<f32>,
    #[serde(default)]
    pub light_shafts: Option<bool>,
    #[serde(default)]
    pub shaft_intensity: Option<f32>,
    #[serde(default)]
    pub half_res: Option<bool>,
}

/// M6: Sky configuration (IPC)
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

/// Stats about the currently loaded scene
#[derive(Debug, Clone, Serialize, Default)]
pub struct ViewerStats {
    /// Whether vertex buffer is ready for drawing
    pub vb_ready: bool,
    /// Number of vertices in the current mesh
    pub vertex_count: u32,
    /// Number of indices in the current mesh
    pub index_count: u32,
    /// Whether the scene has any mesh loaded
    pub scene_has_mesh: bool,
    /// Monotonically increasing transform version (incremented on each set_transform)
    pub transform_version: u64,
    /// Whether the current transform is identity (no translation, rotation, or scale applied)
    pub transform_is_identity: bool,
}

fn default_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn default_intensity() -> f32 {
    1.0
}

fn default_phi() -> f32 {
    135.0
}

fn default_theta() -> f32 {
    45.0
}

fn default_radius() -> f32 {
    1000.0
}

fn default_fov() -> f32 {
    55.0
}

fn default_sun_azimuth() -> f32 {
    135.0
}

fn default_sun_elevation() -> f32 {
    35.0
}

fn default_sun_intensity() -> f32 {
    1.0
}

/// IPC response envelope
#[derive(Debug, Clone, Serialize)]
pub struct IpcResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ViewerStats>,
}

impl IpcResponse {
    pub fn success() -> Self {
        Self {
            ok: true,
            error: None,
            stats: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: Some(msg.into()),
            stats: None,
        }
    }

    pub fn with_stats(stats: ViewerStats) -> Self {
        Self {
            ok: true,
            error: None,
            stats: Some(stats),
        }
    }
}

/// Parse an IPC request from a JSON line
pub fn parse_ipc_request(line: &str) -> Result<IpcRequest, String> {
    serde_json::from_str(line).map_err(|e| format!("JSON parse error: {}", e))
}

/// Convert an IpcRequest to a ViewerCmd (if possible)
/// Returns None for requests that are handled specially (like GetStats)
pub fn ipc_request_to_viewer_cmd(req: &IpcRequest) -> Result<Option<ViewerCmd>, String> {
    match req {
        IpcRequest::GetStats => Ok(None),
        IpcRequest::LoadObj { path } => Ok(Some(ViewerCmd::LoadObj(path.clone()))),
        IpcRequest::LoadGltf { path } => Ok(Some(ViewerCmd::LoadGltf(path.clone()))),
        IpcRequest::SetTransform {
            translation,
            rotation_quat,
            scale,
        } => Ok(Some(ViewerCmd::SetTransform {
            translation: *translation,
            rotation_quat: *rotation_quat,
            scale: *scale,
        })),
        IpcRequest::CamLookat { eye, target, up } => Ok(Some(ViewerCmd::SetCamLookAt {
            eye: *eye,
            target: *target,
            up: *up,
        })),
        IpcRequest::SetFov { deg } => Ok(Some(ViewerCmd::SetFov(*deg))),
        IpcRequest::LitSun {
            azimuth_deg,
            elevation_deg,
        } => Ok(Some(ViewerCmd::SetSunDirection {
            azimuth_deg: *azimuth_deg,
            elevation_deg: *elevation_deg,
        })),
        IpcRequest::LitIbl { path, intensity } => Ok(Some(ViewerCmd::SetIbl {
            path: path.clone(),
            intensity: *intensity,
        })),
        IpcRequest::SetZScale { value } => Ok(Some(ViewerCmd::SetZScale(*value))),
        IpcRequest::Snapshot {
            path,
            width,
            height,
        } => Ok(Some(ViewerCmd::SnapshotWithSize {
            path: path.clone(),
            width: *width,
            height: *height,
        })),
        IpcRequest::Close => Ok(Some(ViewerCmd::Quit)),
        IpcRequest::LoadTerrain { path } => Ok(Some(ViewerCmd::LoadTerrain(path.clone()))),
        IpcRequest::SetTerrainCamera {
            phi_deg,
            theta_deg,
            radius,
            fov_deg,
        } => Ok(Some(ViewerCmd::SetTerrainCamera {
            phi_deg: *phi_deg,
            theta_deg: *theta_deg,
            radius: *radius,
            fov_deg: *fov_deg,
        })),
        IpcRequest::SetTerrainSun {
            azimuth_deg,
            elevation_deg,
            intensity,
        } => Ok(Some(ViewerCmd::SetTerrainSun {
            azimuth_deg: *azimuth_deg,
            elevation_deg: *elevation_deg,
            intensity: *intensity,
        })),
        IpcRequest::SetTerrain {
            phi,
            theta,
            radius,
            fov,
            sun_azimuth,
            sun_elevation,
            sun_intensity,
            ambient,
            zscale,
            shadow,
            background,
            water_level,
            water_color,
        } => Ok(Some(ViewerCmd::SetTerrain {
            phi: *phi,
            theta: *theta,
            radius: *radius,
            fov: *fov,
            sun_azimuth: *sun_azimuth,
            sun_elevation: *sun_elevation,
            sun_intensity: *sun_intensity,
            ambient: *ambient,
            zscale: *zscale,
            shadow: *shadow,
            background: *background,
            water_level: *water_level,
            water_color: *water_color,
        })),
        IpcRequest::GetTerrainParams => Ok(Some(ViewerCmd::GetTerrainParams)),
        IpcRequest::SetTerrainPbr {
            enabled,
            hdr_path,
            ibl_intensity,
            shadow_technique,
            shadow_map_res,
            exposure,
            msaa,
            normal_strength,
            height_ao,
            sun_visibility,
            materials,
            vector_overlay,
            tonemap,
            dof,
            motion_blur,
            lens_effects,
            denoise,
            volumetrics,
            sky,
        } => {
            use crate::viewer::viewer_enums::{
                ViewerHeightAoConfig, ViewerSunVisConfig,
                ViewerMaterialLayerConfig, ViewerVectorOverlayConfig, ViewerTonemapConfig,
                ViewerDofConfig, ViewerMotionBlurConfig, ViewerLensEffectsConfig,
                ViewerDenoiseConfig, ViewerVolumetricsConfig, ViewerSkyConfig,
            };
            
            let height_ao_config = height_ao.as_ref().map(|c| ViewerHeightAoConfig {
                enabled: c.enabled.unwrap_or(false),
                directions: c.directions.unwrap_or(6),
                steps: c.steps.unwrap_or(16),
                max_distance: c.max_distance.unwrap_or(200.0),
                strength: c.strength.unwrap_or(1.0),
                resolution_scale: c.resolution_scale.unwrap_or(0.5),
            });
            
            let sun_vis_config = sun_visibility.as_ref().map(|c| ViewerSunVisConfig {
                enabled: c.enabled.unwrap_or(false),
                mode: c.mode.clone().unwrap_or_else(|| "soft".to_string()),
                samples: c.samples.unwrap_or(4),
                steps: c.steps.unwrap_or(24),
                max_distance: c.max_distance.unwrap_or(400.0),
                softness: c.softness.unwrap_or(1.0),
                bias: c.bias.unwrap_or(0.01),
                resolution_scale: c.resolution_scale.unwrap_or(0.5),
            });
            
            // M4: Material layer config
            let materials_config = materials.as_ref().map(|c| ViewerMaterialLayerConfig {
                snow_enabled: c.snow_enabled.unwrap_or(false),
                snow_altitude_min: c.snow_altitude_min.unwrap_or(2500.0),
                snow_altitude_blend: c.snow_altitude_blend.unwrap_or(200.0),
                snow_slope_max: c.snow_slope_max.unwrap_or(45.0),
                rock_enabled: c.rock_enabled.unwrap_or(false),
                rock_slope_min: c.rock_slope_min.unwrap_or(45.0),
                wetness_enabled: c.wetness_enabled.unwrap_or(false),
                wetness_strength: c.wetness_strength.unwrap_or(0.3),
            });
            
            // M5: Vector overlay config
            let vector_overlay_config = vector_overlay.as_ref().map(|c| ViewerVectorOverlayConfig {
                depth_test: c.depth_test.unwrap_or(false),
                depth_bias: c.depth_bias.unwrap_or(0.001),
                halo_enabled: c.halo_enabled.unwrap_or(false),
                halo_width: c.halo_width.unwrap_or(2.0),
                halo_color: c.halo_color.unwrap_or([0.0, 0.0, 0.0, 0.5]),
            });
            
            // M6: Tonemap config
            let tonemap_config = tonemap.as_ref().map(|c| ViewerTonemapConfig {
                operator: c.operator.clone().unwrap_or_else(|| "aces".to_string()),
                white_point: c.white_point.unwrap_or(4.0),
                white_balance_enabled: c.white_balance_enabled.unwrap_or(false),
                temperature: c.temperature.unwrap_or(6500.0),
                tint: c.tint.unwrap_or(0.0),
            });
            
            // M3: DoF config
            let dof_config = dof.as_ref().map(|c| ViewerDofConfig {
                enabled: c.enabled.unwrap_or(false),
                f_stop: c.f_stop.unwrap_or(5.6),
                focus_distance: c.focus_distance.unwrap_or(100.0),
                focal_length: c.focal_length.unwrap_or(50.0),
                tilt_pitch: c.tilt_pitch.unwrap_or(0.0),
                tilt_yaw: c.tilt_yaw.unwrap_or(0.0),
                quality: c.quality.clone().unwrap_or_else(|| "medium".to_string()),
            });
            
            // M4: Motion blur config
            let motion_blur_config = motion_blur.as_ref().map(|c| ViewerMotionBlurConfig {
                enabled: c.enabled.unwrap_or(false),
                samples: c.samples.unwrap_or(8),
                shutter_open: c.shutter_open.unwrap_or(0.0),
                shutter_close: c.shutter_close.unwrap_or(0.5),
                cam_phi_delta: c.cam_phi_delta.unwrap_or(0.0),
                cam_theta_delta: c.cam_theta_delta.unwrap_or(0.0),
                cam_radius_delta: c.cam_radius_delta.unwrap_or(0.0),
            });
            
            // M5: Lens effects config
            let lens_effects_config = lens_effects.as_ref().map(|c| ViewerLensEffectsConfig {
                enabled: c.enabled.unwrap_or(false),
                distortion: c.distortion.unwrap_or(0.0),
                chromatic_aberration: c.chromatic_aberration.unwrap_or(0.0),
                vignette_strength: c.vignette_strength.unwrap_or(0.0),
                vignette_radius: c.vignette_radius.unwrap_or(0.7),
                vignette_softness: c.vignette_softness.unwrap_or(0.3),
            });
            
            // M5: Denoise config
            let denoise_config = denoise.as_ref().map(|c| ViewerDenoiseConfig {
                enabled: c.enabled.unwrap_or(false),
                method: c.method.clone().unwrap_or_else(|| "atrous".to_string()),
                iterations: c.iterations.unwrap_or(3),
                sigma_color: c.sigma_color.unwrap_or(0.1),
            });
            
            // M6: Volumetrics config
            let volumetrics_config = volumetrics.as_ref().map(|c| ViewerVolumetricsConfig {
                enabled: c.enabled.unwrap_or(false),
                mode: c.mode.clone().unwrap_or_else(|| "uniform".to_string()),
                density: c.density.unwrap_or(0.01),
                scattering: c.scattering.unwrap_or(0.5),
                absorption: c.absorption.unwrap_or(0.1),
                light_shafts: c.light_shafts.unwrap_or(false),
                shaft_intensity: c.shaft_intensity.unwrap_or(1.0),
                half_res: c.half_res.unwrap_or(false),
            });
            
            // M6: Sky config
            let sky_config = sky.as_ref().map(|c| ViewerSkyConfig {
                enabled: c.enabled.unwrap_or(false),
                turbidity: c.turbidity.unwrap_or(2.0),
                ground_albedo: c.ground_albedo.unwrap_or(0.3),
                sun_intensity: c.sun_intensity.unwrap_or(1.0),
                aerial_perspective: c.aerial_perspective.unwrap_or(true),
                sky_exposure: c.sky_exposure.unwrap_or(1.0),
            });
            
            Ok(Some(ViewerCmd::SetTerrainPbr {
                enabled: *enabled,
                hdr_path: hdr_path.clone(),
                ibl_intensity: *ibl_intensity,
                shadow_technique: shadow_technique.clone(),
                shadow_map_res: *shadow_map_res,
                exposure: *exposure,
                msaa: *msaa,
                normal_strength: *normal_strength,
                height_ao: height_ao_config,
                sun_visibility: sun_vis_config,
                materials: materials_config,
                vector_overlay: vector_overlay_config,
                tonemap: tonemap_config,
                dof: dof_config,
                motion_blur: motion_blur_config,
                lens_effects: lens_effects_config,
                denoise: denoise_config,
                volumetrics: volumetrics_config,
                sky: sky_config,
            }))
        },
        IpcRequest::LoadOverlay {
            name,
            path,
            extent,
            opacity,
            z_order,
        } => Ok(Some(ViewerCmd::LoadOverlay {
            name: name.clone(),
            path: path.clone(),
            extent: *extent,
            opacity: *opacity,
            z_order: *z_order,
        })),
        IpcRequest::RemoveOverlay { id } => Ok(Some(ViewerCmd::RemoveOverlay { id: *id })),
        IpcRequest::SetOverlayVisible { id, visible } => {
            Ok(Some(ViewerCmd::SetOverlayVisible {
                id: *id,
                visible: *visible,
            }))
        }
        IpcRequest::SetOverlayOpacity { id, opacity } => {
            Ok(Some(ViewerCmd::SetOverlayOpacity {
                id: *id,
                opacity: *opacity,
            }))
        }
        IpcRequest::SetGlobalOverlayOpacity { opacity } => {
            Ok(Some(ViewerCmd::SetGlobalOverlayOpacity { opacity: *opacity }))
        }
        IpcRequest::SetOverlaysEnabled { enabled } => {
            Ok(Some(ViewerCmd::SetOverlaysEnabled { enabled: *enabled }))
        }
        IpcRequest::SetOverlaySolid { solid } => {
            Ok(Some(ViewerCmd::SetOverlaySolid { solid: *solid }))
        }
        IpcRequest::ListOverlays => Ok(Some(ViewerCmd::ListOverlays)),
    }
}
