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
    },
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
