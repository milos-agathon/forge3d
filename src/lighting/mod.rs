// src/lighting/mod.rs
// P0 Milestone: Production-ready lighting stack
// Implements light types, BRDFs, shadows, and IBL for terrain rendering
// RELEVANT FILES: src/shaders/lighting.wgsl, examples/terrain_demo.py

// Focused submodules (refactored from types.rs)
pub mod atmospherics;
pub mod ephemeris;
pub mod light;
pub mod material;
pub mod screen_space;
pub mod shadow;

// Core modules
pub mod ibl_cache;
#[cfg(feature = "extension-module")]
pub mod ibl_wrapper;
pub mod light_buffer;
pub mod shadow_map;
pub mod types;

#[cfg(feature = "extension-module")]
pub mod py_bindings;

// Re-export main types
pub use ibl_cache::IblResourceCache;
pub use light_buffer::LightBuffer;
pub use shadow_map::{SceneBounds, ShadowMap, ShadowMatrixCalculator};
pub use types::{
    Atmosphere, AtmosphericsSettings, BrdfModel, GiSettings, GiTechnique, Light, LightType,
    MaterialShading, SSAOSettings, SSGISettings, SSRSettings, ScreenSpaceEffect,
    ScreenSpaceSettings, ShadowSettings, ShadowTechnique, SkyModel, SkySettings, VolumetricPhase,
    VolumetricSettings,
};

#[cfg(feature = "extension-module")]
pub use py_bindings::{
    PyAtmosphere, PyGiSettings, PyLight, PyMaterialShading, PySSAOSettings, PySSGISettings,
    PySSRSettings, PyShadowSettings, PySkySettings, PyVolumetricSettings,
};
