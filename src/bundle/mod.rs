//! Scene bundle module for `.forge3d` portable scene packages.
//!
//! A bundle is a directory containing:
//! - `manifest.json` - version, name, checksums
//! - `terrain/` - DEM and colormap data
//! - `overlays/` - vectors and labels
//! - `camera/` - camera bookmarks
//! - `render/` - preset configuration
//! - `assets/` - fonts, HDRI, etc.

mod manifest;

pub use manifest::{BundleManifest, BundleError, BundleResult};

use std::path::Path;

/// Bundle format version
pub const BUNDLE_VERSION: u32 = 1;

/// Bundle file extension
pub const BUNDLE_EXTENSION: &str = "forge3d";

/// Check if a path is a valid bundle directory
pub fn is_bundle(path: &Path) -> bool {
    path.is_dir() && path.join("manifest.json").exists()
}

/// Get the manifest path for a bundle
pub fn manifest_path(bundle_path: &Path) -> std::path::PathBuf {
    bundle_path.join("manifest.json")
}

/// Standard subdirectories in a bundle
pub mod dirs {
    pub const TERRAIN: &str = "terrain";
    pub const OVERLAYS: &str = "overlays";
    pub const CAMERA: &str = "camera";
    pub const RENDER: &str = "render";
    pub const ASSETS: &str = "assets";
}
