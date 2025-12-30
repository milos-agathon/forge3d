// src/viewer/terrain/mod.rs
// Terrain viewer module - standalone terrain rendering without PyO3 dependencies
// Split from viewer_terrain.rs as part of the viewer refactoring

mod dof;
mod motion_blur;
pub mod overlay;
mod pbr_renderer;
mod post_process;
mod render;
mod scene;
mod shader;
mod shader_pbr;
pub mod vector_overlay;
mod volumetrics;

#[allow(unused_imports)]
pub use overlay::{OverlayStack, OverlayLayer, OverlayData, BlendMode, OverlayConfig};
#[allow(unused_imports)]
pub use pbr_renderer::ViewerTerrainPbrConfig;
pub use scene::ViewerTerrainScene;

// Option B: Vector overlay geometry exports
#[allow(unused_imports)]
pub use vector_overlay::{
    VectorOverlayStack, VectorOverlayLayer, VectorVertex, 
    OverlayPrimitive, VectorOverlayUniforms, VectorOverlayGpu,
    drape_vertices, VECTOR_OVERLAY_SHADER,
};

