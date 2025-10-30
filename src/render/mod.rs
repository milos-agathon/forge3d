// src/render/mod.rs
// Shared entry point for renderer utilities and feature gates
// Exists to group CPU helpers with emerging GPU configuration surfaces
// RELEVANT FILES: src/render/params.rs, src/render/instancing.rs, src/lib.rs, python/forge3d/__init__.py
//! Rendering utilities and CPU fallbacks.

pub mod colormap;
pub mod instancing;
pub mod params;
pub mod memory_budget;
#[cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]
pub mod pbr_pass;
#[cfg(feature = "enable-gpu-instancing")]
pub mod mesh_instanced;
