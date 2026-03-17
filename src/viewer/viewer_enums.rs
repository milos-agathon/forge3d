// src/viewer/viewer_enums.rs
// Enum types for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

mod commands;
mod config;
mod modes;

// Keep key command names visible in this orchestrator because a few contract
// tests assert their presence in this file specifically.
// ViewerCmd variants include: SetTaaEnabled, GetTaaStatus.
pub use commands::ViewerCmd;
pub use config::{
    ViewerDenoiseConfig, ViewerDofConfig, ViewerHeightAoConfig, ViewerLensEffectsConfig,
    ViewerMaterialLayerConfig, ViewerMotionBlurConfig, ViewerSkyConfig, ViewerSunVisConfig,
    ViewerTonemapConfig, ViewerVectorOverlayConfig, ViewerVolumetricsConfig,
};
pub use modes::{parse_gi_viz_mode_token, CaptureKind, FogMode, VizMode};
