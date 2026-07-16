//! Image format loaders and utilities
//!
//! Supports loading various image formats for texture and HDR usage.

// L6: HDR (Radiance) format loader
pub mod hdr;
pub use hdr::{load_hdr, HdrImage};

pub mod exr;
pub use exr::load_exr;
