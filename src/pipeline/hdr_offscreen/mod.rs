//! HDR off-screen rendering pipeline
//!
//! Provides high dynamic range off-screen rendering to RGBA16Float textures with
//! tone mapping post-process to sRGB8 output suitable for PNG export and readback.

mod pipeline;
mod types;

pub use pipeline::HdrOffscreenPipeline;
pub use types::{HdrOffscreenConfig, ToneMappingOperator, ToneMappingUniforms};
