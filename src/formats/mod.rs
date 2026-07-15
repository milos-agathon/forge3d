//! Image format loaders and utilities
//!
//! Supports loading various image formats for texture and HDR usage.

// L6: Radiance HDR (.hdr/.rgbe) format loader.
pub mod hdr;
pub use hdr::{load_hdr, HdrImage};

// OpenEXR (.exr) environment decoder. `load_exr` is compiled in every build:
// with the default `images` feature it decodes EXR, and without it returns an
// explicit feature-required error.
pub mod exr;
pub use exr::load_exr;
