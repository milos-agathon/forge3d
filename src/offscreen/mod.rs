// src/offscreen/mod.rs
// Offscreen rendering harness for BRDF galleries and CI goldens
// Provides headless, reproducible PBR tile rendering without viewer dependencies
// RELEVANT FILES: src/offscreen/brdf_tile.rs, src/pipeline/pbr.rs, src/shaders/pbr.wgsl

pub mod brdf_tile;
pub mod pipeline;
pub mod sphere;

pub use brdf_tile::render_brdf_tile_offscreen;
pub use sphere::generate_uv_sphere;
