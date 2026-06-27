//! Physical smoke volume representation, CPU reference simulation, and ray marcher.
//!
//! The GPU viewer can already sample bounded density volumes. This module adds the
//! first-class smoke state needed to drive those volumes with deterministic,
//! testable transport and physically motivated rendering contracts.

mod render;
mod sampling;
mod sim;
mod types;

#[cfg(feature = "extension-module")]
pub mod py;

pub use types::{
    SmokeDomainConfig, SmokeEmitter, SmokeMemoryReport, SmokeRenderSettings, SmokeStepSettings,
    SmokeVolume,
};
