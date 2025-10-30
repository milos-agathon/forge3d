//! Rendering pipeline implementations
//!
//! This module contains GPU rendering pipelines for various advanced rendering techniques
//! including normal mapping, PBR materials, shadow mapping, and environment mapping.

#[cfg(feature = "enable-normal-mapping")]
pub mod normal_mapping;

#[cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]
pub mod pbr;

#[cfg(feature = "enable-hdr-offscreen")]
pub mod hdr_offscreen;

#[cfg(feature = "enable-normal-mapping")]
pub use normal_mapping::{
    compute_normal_matrix, create_checkerboard_normal_texture, NormalMappingPipeline,
    NormalMappingUniforms,
};

#[cfg(all(feature = "enable-pbr", feature = "enable-tbn"))]
pub use pbr::{create_pbr_sampler, PbrMaterialGpu, PbrTextures};

#[cfg(feature = "enable-hdr-offscreen")]
pub use hdr_offscreen::{
    HdrOffscreenConfig, HdrOffscreenPipeline, ToneMappingOperator, ToneMappingUniforms,
};
