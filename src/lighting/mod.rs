// src/lighting/mod.rs
// P0 Milestone: Production-ready lighting stack
// Implements light types, BRDFs, shadows, and IBL for terrain rendering
// RELEVANT FILES: src/shaders/lighting.wgsl, examples/terrain_demo.py

pub mod types;
pub mod ibl_cache;
pub mod shadow_map;
pub mod light_buffer;

#[cfg(feature = "extension-module")]
pub mod py_bindings;

// Re-export main types
pub use types::{
    LightType, BrdfModel, ShadowTechnique, GiTechnique, ScreenSpaceEffect,
    SkyModel, VolumetricPhase,
    Light, MaterialShading, ShadowSettings, GiSettings, Atmosphere,
    SSAOSettings, SSGISettings, SSRSettings, ScreenSpaceSettings,
    SkySettings, VolumetricSettings, AtmosphericsSettings,
};
pub use ibl_cache::IblResourceCache;
pub use shadow_map::{ShadowMap, ShadowMatrixCalculator, SceneBounds};
pub use light_buffer::LightBuffer;

#[cfg(feature = "extension-module")]
pub use py_bindings::{
    PyLight, PyMaterialShading, PyShadowSettings, PyGiSettings, PyAtmosphere,
    PySSAOSettings, PySSGISettings, PySSRSettings,
    PySkySettings, PyVolumetricSettings,
};

#[cfg(test)]
mod tests;
