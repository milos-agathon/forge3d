//! Physically-Based Rendering (PBR) materials system
//! 
//! This module re-exports PBR components from their specialized modules:
//! - Core material definitions from `material.rs`
//! - GPU pipeline components from `pipeline::pbr` (when available)

// Re-export core material types
pub use crate::core::material::{
    PbrMaterial, PbrLighting, texture_flags, brdf, presets
};

// Re-export GPU pipeline types (feature-gated)
#[cfg(feature = "enable-pbr")]
pub use crate::pipeline::pbr::{
    PbrTextures, PbrMaterialGpu, create_pbr_sampler
};