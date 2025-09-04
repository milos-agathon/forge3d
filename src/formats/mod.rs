//! Image format loaders and utilities
//!
//! Supports loading various image formats for texture and HDR usage.

// L6: HDR (Radiance) format loader
pub mod hdr;
pub use hdr::{HdrImage, load_hdr};

// L6: Optional EXR format loader (feature-gated)
#[cfg(feature = "exr")]
pub mod exr;
#[cfg(feature = "exr")]
pub use exr::{ExrImage, load_exr};