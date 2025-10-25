// src/lighting/mod.rs
// P0 Milestone: Production-ready lighting stack
// Implements light types, BRDFs, shadows, and IBL for terrain rendering
// RELEVANT FILES: src/shaders/lighting.wgsl, examples/terrain_demo.py

pub mod types;
pub mod ibl_cache;

#[cfg(feature = "extension-module")]
pub mod py_bindings;

// Re-export main types
pub use types::{
    LightType, BrdfModel, ShadowTechnique, GiTechnique,
    Light, MaterialShading, ShadowSettings, GiSettings, Atmosphere,
};
pub use ibl_cache::IblResourceCache;

#[cfg(feature = "extension-module")]
pub use py_bindings::{
    PyLight, PyMaterialShading, PyShadowSettings, PyGiSettings, PyAtmosphere,
};

#[cfg(test)]
mod tests;
