// src/viewer/terrain/mod.rs
// Terrain viewer module - standalone terrain rendering without PyO3 dependencies
// Split from viewer_terrain.rs as part of the viewer refactoring

mod render;
mod scene;
mod shader;

pub use scene::ViewerTerrainScene;
