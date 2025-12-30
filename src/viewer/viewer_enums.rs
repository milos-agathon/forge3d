// src/viewer/viewer_enums.rs
// Enum types for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

use crate::cli::args::GiVizMode;

/// Viewer command enum for input parsing
#[derive(Debug, Clone)]
pub enum ViewerCmd {
    GiToggle(&'static str, bool),
    SetGiSeed(u32),
    Snapshot(Option<String>),
    DumpGbuffer,
    Quit,
    LoadObj(String),
    LoadGltf(String),
    SetViz(String),
    SetGiViz(GiVizMode),
    QueryGiViz,
    LoadSsrPreset,
    LoadIbl(String),
    IblToggle(bool),
    IblIntensity(f32),
    IblRotate(f32),
    IblCache(Option<String>),
    IblRes(u32),
    #[allow(dead_code)]
    SetSsaoRadius(f32),
    #[allow(dead_code)]
    SetSsaoIntensity(f32),
    #[allow(dead_code)]
    SetSsaoBias(f32),
    #[allow(dead_code)]
    SetSsgiSteps(u32),
    #[allow(dead_code)]
    SetSsgiRadius(f32),
    SetSsrMaxSteps(u32),
    SetSsrThickness(f32),
    #[allow(dead_code)]
    SetSsgiHalf(bool),
    #[allow(dead_code)]
    SetSsgiTemporalAlpha(f32),
    #[allow(dead_code)]
    SetSsgiTemporalEnabled(bool),
    #[allow(dead_code)]
    SetAoTemporalAlpha(f32),
    #[allow(dead_code)]
    SetSsaoSamples(u32),
    #[allow(dead_code)]
    SetSsaoDirections(u32),
    #[allow(dead_code)]
    SetSsaoTemporalAlpha(f32),
    #[allow(dead_code)]
    SetSsaoTemporalEnabled(bool),
    #[allow(dead_code)]
    SetSsaoTechnique(u32),
    #[allow(dead_code)]
    SetVizDepthMax(f32),
    #[allow(dead_code)]
    SetFov(f32),
    #[allow(dead_code)]
    SetCamLookAt {
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
    },
    #[allow(dead_code)]
    SetSize(u32, u32),
    #[allow(dead_code)]
    SetSsaoComposite(bool),
    #[allow(dead_code)]
    SetSsaoCompositeMul(f32),
    #[allow(dead_code)]
    SetAoBlur(bool),
    QuerySsaoRadius,
    QuerySsaoIntensity,
    QuerySsaoBias,
    QuerySsaoSamples,
    QuerySsaoDirections,
    QuerySsaoTemporalAlpha,
    QuerySsaoTemporalEnabled,
    QuerySsaoBlur,
    QuerySsaoComposite,
    QuerySsaoMul,
    QuerySsaoTechnique,
    QuerySsgiSteps,
    QuerySsgiRadius,
    QuerySsgiHalf,
    QuerySsgiTemporalAlpha,
    QuerySsgiTemporalEnabled,
    QuerySsgiEdges,
    QuerySsgiUpsampleSigmaDepth,
    QuerySsgiUpsampleSigmaNormal,
    QuerySsrEnable,
    QuerySsrMaxSteps,
    QuerySsrThickness,
    QueryGiAoWeight,
    QueryGiSsgiWeight,
    QueryGiSsrWeight,
    #[allow(dead_code)]
    SetSsgiEdges(bool),
    #[allow(dead_code)]
    SetSsgiUpsampleSigmaDepth(f32),
    #[allow(dead_code)]
    SetSsgiUpsampleSigmaNormal(f32),
    SetGiAoWeight(f32),
    SetGiSsgiWeight(f32),
    SetGiSsrWeight(f32),
    SetLitSun(f32),
    SetLitIbl(f32),
    SetLitBrdf(u32),
    SetLitRough(f32),
    SetLitDebug(u32),
    SkyToggle(bool),
    SkySetModel(u32),
    SkySetTurbidity(f32),
    SkySetGround(f32),
    SkySetExposure(f32),
    SkySetSunIntensity(f32),
    FogToggle(bool),
    FogSetDensity(f32),
    FogSetG(f32),
    FogSetSteps(u32),
    FogSetShadow(bool),
    FogSetTemporal(f32),
    SetFogMode(u32),
    FogPreset(u32),
    FogHalf(bool),
    FogEdges(bool),
    FogUpsigma(f32),
    HudToggle(bool),
    CaptureP51Cornell,
    CaptureP51Grid,
    CaptureP51Sweep,
    CaptureP52SsgiCornell,
    CaptureP52SsgiTemporal,
    CaptureP53SsrGlossy,
    CaptureP53SsrThickness,
    CaptureP54GiStack,
    QueryGiSeed,
    GiStatus,
    // IPC-specific commands for non-blocking viewer workflow
    /// Set sun direction (azimuth/elevation in degrees)
    SetSunDirection {
        azimuth_deg: f32,
        elevation_deg: f32,
    },
    /// Set IBL with path and intensity
    SetIbl {
        path: String,
        intensity: f32,
    },
    /// Set terrain z-scale (height exaggeration)
    SetZScale(f32),
    /// Snapshot with explicit width/height override
    SnapshotWithSize {
        path: String,
        width: Option<u32>,
        height: Option<u32>,
    },
    /// Set object transform (translation, rotation quaternion, scale)
    SetTransform {
        translation: Option<[f32; 3]>,
        rotation_quat: Option<[f32; 4]>,
        scale: Option<[f32; 3]>,
    },
    /// Load terrain DEM file for interactive viewing
    LoadTerrain(String),
    /// Set terrain camera (phi, theta, radius, fov in degrees)
    SetTerrainCamera {
        phi_deg: f32,
        theta_deg: f32,
        radius: f32,
        fov_deg: f32,
    },
    /// Set terrain sun (azimuth, elevation in degrees, intensity)
    SetTerrainSun {
        azimuth_deg: f32,
        elevation_deg: f32,
        intensity: f32,
    },
    /// Set multiple terrain parameters at once
    SetTerrain {
        phi: Option<f32>,
        theta: Option<f32>,
        radius: Option<f32>,
        fov: Option<f32>,
        sun_azimuth: Option<f32>,
        sun_elevation: Option<f32>,
        sun_intensity: Option<f32>,
        ambient: Option<f32>,
        zscale: Option<f32>,
        shadow: Option<f32>,
        background: Option<[f32; 3]>,
        water_level: Option<f32>,
        water_color: Option<[f32; 3]>,
    },
    /// Get current terrain parameters
    GetTerrainParams,
    /// Configure terrain PBR+POM rendering mode (opt-in, default off)
    SetTerrainPbr {
        /// Enable PBR mode (false = legacy simple shader)
        enabled: Option<bool>,
        /// Path to HDR environment map for IBL
        hdr_path: Option<String>,
        /// IBL intensity multiplier
        ibl_intensity: Option<f32>,
        /// Shadow technique: "none", "hard", "pcf", "pcss"
        shadow_technique: Option<String>,
        /// Shadow map resolution (default 2048)
        shadow_map_res: Option<u32>,
        /// ACES exposure multiplier
        exposure: Option<f32>,
        /// MSAA samples (1, 4, or 8)
        msaa: Option<u32>,
        /// Terrain normal strength multiplier
        normal_strength: Option<f32>,
        /// Heightfield ray-traced AO settings
        height_ao: Option<ViewerHeightAoConfig>,
        /// Heightfield ray-traced sun visibility settings
        sun_visibility: Option<ViewerSunVisConfig>,
        /// M4: Material layer settings (snow/rock/wetness)
        materials: Option<ViewerMaterialLayerConfig>,
        /// M5: Vector overlay settings (depth test, halos)
        vector_overlay: Option<ViewerVectorOverlayConfig>,
        /// M6: Tonemap settings (operator, white balance)
        tonemap: Option<ViewerTonemapConfig>,
        /// M3: Depth of Field settings
        dof: Option<ViewerDofConfig>,
        /// M4: Motion blur settings
        motion_blur: Option<ViewerMotionBlurConfig>,
        /// M5: Lens effects settings
        lens_effects: Option<ViewerLensEffectsConfig>,
        /// M5: Denoise settings
        denoise: Option<ViewerDenoiseConfig>,
        /// M6: Volumetrics settings
        volumetrics: Option<ViewerVolumetricsConfig>,
        /// M6: Sky settings
        sky: Option<ViewerSkyConfig>,
    },
    /// Load an overlay texture from an image file
    LoadOverlay {
        /// Unique name for this overlay layer
        name: String,
        /// Path to image file (PNG, JPEG, etc.)
        path: String,
        /// Extent in terrain UV space: [u_min, v_min, u_max, v_max]
        /// None means full terrain coverage [0, 0, 1, 1]
        extent: Option<[f32; 4]>,
        /// Opacity (0.0 - 1.0)
        opacity: Option<f32>,
        /// Z-order for stacking (lower = behind)
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
    
    // === OPTION B: VECTOR OVERLAY GEOMETRY COMMANDS ===
    
    /// Add a vector overlay layer with geometry
    AddVectorOverlay {
        /// Unique name for this overlay layer
        name: String,
        /// Vertices: list of [x, y, z, r, g, b, a]
        vertices: Vec<[f32; 7]>,
        /// Indices for indexed drawing
        indices: Vec<u32>,
        /// Primitive type: "points", "lines", "line_strip", "triangles", "triangle_strip"
        primitive: String,
        /// Drape onto terrain surface
        drape: bool,
        /// Height offset above terrain when draped
        drape_offset: f32,
        /// Opacity (0.0 - 1.0)
        opacity: f32,
        /// Depth bias for z-fighting prevention
        depth_bias: f32,
        /// Line width for line primitives
        line_width: f32,
        /// Point size for point primitives
        point_size: f32,
        /// Z-order for stacking (lower = behind)
        z_order: i32,
    },
    /// Remove a vector overlay by ID
    RemoveVectorOverlay {
        id: u32,
    },
    /// Set vector overlay visibility
    SetVectorOverlayVisible {
        id: u32,
        visible: bool,
    },
    /// Set vector overlay opacity (0.0 - 1.0)
    SetVectorOverlayOpacity {
        id: u32,
        opacity: f32,
    },
    /// List all vector overlay IDs
    ListVectorOverlays,
    /// Enable or disable vector overlays
    SetVectorOverlaysEnabled {
        enabled: bool,
    },
    /// Set global vector overlay opacity multiplier (0.0 - 1.0)
    SetGlobalVectorOverlayOpacity {
        opacity: f32,
    },
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

/// Visualization mode for viewer output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VizMode {
    #[default]
    Material,
    Normal,
    Depth,
    Gi,
    Lit,
}

/// Fog rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FogMode {
    #[default]
    Raymarch,
    Froxels,
}

/// P5 capture output types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    P51CornellSplit,
    P51AoGrid,
    P51ParamSweep,
    P52SsgiCornell,
    P52SsgiTemporal,
    P53SsrGlossy,
    P53SsrThickness,
    P54GiStack,
}

/// Parse GI visualization mode from string token
pub fn parse_gi_viz_mode_token(tok: &str) -> Option<GiVizMode> {
    match tok {
        "none" => Some(GiVizMode::None),
        "composite" => Some(GiVizMode::Composite),
        "ao" => Some(GiVizMode::Ao),
        "ssgi" => Some(GiVizMode::Ssgi),
        "ssr" => Some(GiVizMode::Ssr),
        _ => None,
    }
}
