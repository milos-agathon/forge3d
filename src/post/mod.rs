//! Post-processing effects for rendered imagery
//!
//! Includes denoising, ambient occlusion, and other image-space effects.

pub mod ambient_occlusion;
pub mod denoise;

pub use denoise::{denoise_rgba, DenoiseConfig, DenoiserType, compute_patch_variance};
