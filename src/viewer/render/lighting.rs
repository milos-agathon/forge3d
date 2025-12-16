// src/viewer/render/lighting.rs
// Lighting uniform helpers for the interactive viewer

use wgpu::{Buffer, Queue};

/// Lighting parameters for the uniform buffer
pub struct LitParams {
    pub sun_intensity: f32,
    pub ibl_intensity: f32,
    pub use_ibl: bool,
    pub brdf_index: u32,
    pub roughness: f32,
    pub debug_mode: u32,
}

/// Update the lighting uniform buffer
pub fn update_lit_uniform(queue: &Queue, lit_uniform: &Buffer, params: &LitParams) {
    // Keep sun_dir consistent with compute shader default
    let sun_dir = [0.3f32, 0.6, -1.0];
    let uniform_data: [f32; 12] = [
        // sun_dir.xyz, sun_intensity
        sun_dir[0],
        sun_dir[1],
        sun_dir[2],
        params.sun_intensity,
        // ibl_intensity, use_ibl, brdf_index, pad
        params.ibl_intensity,
        if params.use_ibl { 1.0 } else { 0.0 },
        params.brdf_index as f32,
        0.0,
        // roughness, debug_mode, pad, pad
        params.roughness.clamp(0.0, 1.0),
        params.debug_mode as f32,
        0.0,
        0.0,
    ];
    queue.write_buffer(lit_uniform, 0, bytemuck::cast_slice(&uniform_data));
}
