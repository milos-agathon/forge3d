// src/viewer/cmd/scene_handlers.rs
// Scene/visualization command handlers for the interactive viewer

use super::super::viewer_enums::{FogMode, VizMode};
use crate::cli::args::GiVizMode;

/// Parse and return the VizMode for a given mode string
pub fn parse_viz_mode(mode: &str, current: VizMode) -> VizMode {
    match mode {
        "material" | "mat" => VizMode::Material,
        "normal" | "normals" => VizMode::Normal,
        "depth" => VizMode::Depth,
        "gi" => VizMode::Gi,
        "lit" => VizMode::Lit,
        _ => {
            eprintln!("Unknown viz mode: {}", mode);
            current
        }
    }
}

/// Get the name string for a GiVizMode
pub fn gi_viz_mode_name(mode: GiVizMode) -> &'static str {
    match mode {
        GiVizMode::None => "none",
        GiVizMode::Composite => "composite",
        GiVizMode::Ao => "ao",
        GiVizMode::Ssgi => "ssgi",
        GiVizMode::Ssr => "ssr",
    }
}

/// Apply GI viz mode and return the appropriate coarse viz mode
pub fn apply_gi_viz_mode(mode: GiVizMode) -> VizMode {
    match mode {
        GiVizMode::None => VizMode::Lit,
        _ => VizMode::Gi,
    }
}

/// Query and print the current GI viz mode
pub fn query_gi_viz(mode: GiVizMode) {
    println!("viz-gi = {}", gi_viz_mode_name(mode));
}

/// Parse fog preset and return (steps, temporal_alpha, density)
pub fn parse_fog_preset(preset: u32) -> (u32, f32, f32) {
    match preset {
        0 => (32, 0.7, 0.02), // low
        1 => (64, 0.6, 0.04), // medium
        _ => (96, 0.5, 0.06), // high
    }
}

/// Parse fog mode from integer
pub fn parse_fog_mode(mode: u32) -> FogMode {
    if mode != 0 {
        FogMode::Froxels
    } else {
        FogMode::Raymarch
    }
}
