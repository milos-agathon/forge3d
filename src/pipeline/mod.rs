//! Rendering pipeline implementations
//!
//! This module contains GPU rendering pipelines for various advanced rendering techniques
//! including normal mapping, PBR materials, shadow mapping, and environment mapping.

#[cfg(feature = "enable-normal-mapping")]
pub mod normal_mapping;

#[cfg(feature = "enable-normal-mapping")]
pub use normal_mapping::{NormalMappingPipeline, NormalMappingUniforms, compute_normal_matrix, create_checkerboard_normal_texture};