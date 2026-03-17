use crate::cli::args::GiVizMode;

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
