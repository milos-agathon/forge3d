// src/path_tracing/hybrid_compute/terrain_albedo.rs
// Encodings of the optional per-texel terrain albedo map uploaded by
// `TerrainPtScene::new_with_options`.
// RELEVANT FILES: src/path_tracing/hybrid_compute/terrain_heightfield.rs

/// Optional per-texel albedo map on the DEM grid.
pub enum TerrainAlbedoMap<'a> {
    None,
    /// Linear RGBA, `w * h * 4` floats (RGB >= 0, alpha in [0, 1]; alpha < 1
    /// falls back to the constant albedo). Stored as Rgba32Float.
    Rgba32F(&'a [f32]),
    /// 8-bit sRGB RGBA, `w * h * 4` bytes (4 B per cell). Alpha 255 marks a
    /// mapped texel; any other alpha falls back to the constant albedo.
    /// Stored as Rgba8UnormSrgb, so the kernel reads linear values.
    Rgba8Srgb(&'a [u8]),
}
