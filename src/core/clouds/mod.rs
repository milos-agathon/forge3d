// src/core/clouds/mod.rs
// Implements realtime cloud renderer with procedural billboards and volumetric shading.
// Exists so the terrain Scene can composite realtime clouds that react to lighting inputs.
// RELEVANT FILES: src/shaders/clouds.wgsl, src/scene/mod.rs, python/forge3d/__init__.py, tests/test_b8_clouds.py

pub mod types;
pub mod renderer;

pub use types::*;
pub use renderer::CloudRenderer;
