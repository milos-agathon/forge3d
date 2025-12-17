// src/viewer/viewer_constants.rs
// Constants and thresholds for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

/// Quick sanity-check version for viewer lit WGSL
pub const LIT_WGSL_VERSION: u32 = 2;

/// Limit for P5.1 capture outputs (in megapixels). Images larger than this will be downscaled.
pub const P51_MAX_MEGAPIXELS: f32 = 2.0;

/// Limit for P5.2 capture outputs (in megapixels)
pub const P52_MAX_MEGAPIXELS: f32 = 2.0;

/// Soft limit for interactive viewer snapshots (in megapixels).
/// User-provided snapshot overrides will be clamped to this size to keep
/// memory usage within budget while still allowing high-resolution captures.
/// 16k = 16384x16384 = 268 MP, so we allow up to 270 MP.
pub const VIEWER_SNAPSHOT_MAX_MEGAPIXELS: f32 = 270.0;

/// Diffuse scale factor for P5 SSGI captures
pub const P5_SSGI_DIFFUSE_SCALE: f32 = 0.5;

/// Number of warmup frames for P5 SSGI Cornell box captures
pub const P5_SSGI_CORNELL_WARMUP_FRAMES: u32 = 64;
