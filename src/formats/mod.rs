//! Image format loaders and utilities
//!
//! Supports loading various image formats for texture and HDR usage.

// L6: HDR (Radiance) format loader
pub mod hdr;
pub use hdr::{load_hdr, HdrImage};

// L6: Optional EXR format loader (feature-gated)
// EXR support is planned but not yet implemented
