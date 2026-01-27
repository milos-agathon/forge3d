// src/geo/mod.rs
// Geographic utilities including CRS reprojection
// RELEVANT FILES: src/geo/reproject.rs, python/forge3d/crs.py

pub mod reproject;

// Re-export main types and functions
pub use reproject::GeoError;

#[cfg(feature = "proj")]
pub use reproject::reproject_coords;

/// Check if the proj feature is available
pub fn proj_available() -> bool {
    cfg!(feature = "proj")
}
