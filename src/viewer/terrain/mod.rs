// src/viewer/terrain/mod.rs
// Terrain viewer module - standalone terrain rendering without PyO3 dependencies
// Split from viewer_terrain.rs as part of the viewer refactoring

mod pbr_renderer;
mod render;
mod scene;
mod shader;
mod shader_pbr;

#[allow(unused_imports)]
pub use pbr_renderer::ViewerTerrainPbrConfig;
pub use scene::ViewerTerrainScene;
