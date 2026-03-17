// src/material_set.rs
//! Material set for terrain rendering with triplanar mapping support

mod core;
#[cfg(feature = "extension-module")]
mod gpu;
#[cfg(feature = "extension-module")]
mod gpu_helpers;
mod py_api;

pub use core::MaterialSet;
#[cfg(feature = "extension-module")]
pub(crate) use gpu::GpuMaterialSet;
