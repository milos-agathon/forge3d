// src/viewer/terrain/shader_pbr.rs
// Enhanced PBR-like shader for terrain rendering with better lighting

pub const TERRAIN_PBR_SHADER: &str = concat!(
    include_str!("../../shaders/includes/shadow_moments.wgsl"),
    "\n",
    include_str!("shader_pbr/terrain_pbr.wgsl")
);
