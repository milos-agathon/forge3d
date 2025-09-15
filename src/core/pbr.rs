//! Physically-Based Rendering (PBR) materials system
//!
//! This module re-exports PBR components from their specialized modules:
//! - Core material definitions from `material.rs`
//! - GPU pipeline components from `pipeline::pbr` (when available)

// Re-export core material types
pub use crate::core::material::{brdf, presets, texture_flags, PbrLighting, PbrMaterial};

// Re-export GPU pipeline types (feature-gated)
#[cfg(feature = "enable-pbr")]
pub use crate::pipeline::pbr::{create_pbr_sampler, PbrMaterialGpu, PbrTextures};
