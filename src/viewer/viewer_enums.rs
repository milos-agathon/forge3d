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
