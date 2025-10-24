// src/render/mod.rs
//! Rendering utilities and CPU fallbacks.

pub mod colormap;
pub mod instancing;
#[cfg(feature = "enable-gpu-instancing")]
pub mod mesh_instanced;
