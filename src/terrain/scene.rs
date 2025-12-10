// src/terrain/scene.rs
// TerrainScene helper re-exported from terrain_renderer.
// This module exists to provide a reusable terrain scene type for the viewer
// and other Rust code without pulling in PyO3.

#[cfg(feature = "extension-module")]
pub use super::renderer::TerrainScene;
