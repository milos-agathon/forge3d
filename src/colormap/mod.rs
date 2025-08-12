//! Central colormap registry.
//! - Single source for supported names
//! - Embedded 256×1 PNG bytes via `include_bytes!`
//! - Small helpers (enum mapping + PyO3 error)

/// Built-in colormap names (case-sensitive).
pub static SUPPORTED: [&str; 3] = ["viridis", "magma", "terrain"];

/// Resolve embedded 256×1 PNG bytes for the given name.
pub fn resolve_bytes(name: &str) -> Result<&'static [u8], String> {
    match name {
        "viridis" => Ok(include_bytes!("assets/viridis_256x1.png")),
        "magma"   => Ok(include_bytes!("assets/magma_256x1.png")),
        "terrain" => Ok(include_bytes!("assets/terrain_256x1.png")),
        _ => Err(format!("Unknown colormap '{}'. Supported: {}", name, SUPPORTED.join(", "))),
    }
}

/// Optional typed mapping if you keep a ColormapType in your pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColormapType { 
    Viridis, 
    Magma, 
    Terrain 
}

pub fn map_name_to_type(name: &str) -> Result<ColormapType, String> {
    match name {
        "viridis" => Ok(ColormapType::Viridis),
        "magma"   => Ok(ColormapType::Magma),
        "terrain" => Ok(ColormapType::Terrain),
        _ => Err(format!("Unknown colormap '{}'. Supported: {}", name, SUPPORTED.join(", "))),
    }
}

/// PyO3-friendly error helper (always compiled; crate already depends on pyo3).
pub fn py_err_unknown(name: &str) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        format!("Unknown colormap '{}'. Supported: {}", name, SUPPORTED.join(", "))
    )
}

/// Export supported colormap names for Python (unconditionally available)
#[pyo3::prelude::pyfunction]
pub fn colormap_supported() -> Vec<&'static str> {
    SUPPORTED.to_vec()
}

/// Decode embedded PNG to raw RGBA8 bytes (sRGB encoded)
pub fn decode_png_rgba8(name: &str) -> Result<Vec<u8>, String> {
    let png_bytes = resolve_bytes(name)?;
    let img = image::load_from_memory(png_bytes)
        .map_err(|e| format!("Failed to decode PNG for '{}': {}", name, e))?;
    let rgba = img.to_rgba8();
    Ok(rgba.as_raw().clone())
}

/// Convert sRGB RGBA8 bytes to linear RGBA8 (apply sRGB→linear curve to RGB channels only)
pub fn to_linear_u8_rgba(src_srgb_rgba8: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(src_srgb_rgba8.len());
    
    for chunk in src_srgb_rgba8.chunks_exact(4) {
        let r_srgb = chunk[0] as f32 / 255.0;
        let g_srgb = chunk[1] as f32 / 255.0;
        let b_srgb = chunk[2] as f32 / 255.0;
        let a = chunk[3]; // Alpha unchanged
        
        let r_linear = if r_srgb <= 0.04045 { r_srgb / 12.92 } else { ((r_srgb + 0.055) / 1.055).powf(2.4) };
        let g_linear = if g_srgb <= 0.04045 { g_srgb / 12.92 } else { ((g_srgb + 0.055) / 1.055).powf(2.4) };
        let b_linear = if b_srgb <= 0.04045 { b_srgb / 12.92 } else { ((b_srgb + 0.055) / 1.055).powf(2.4) };
        
        result.push((r_linear.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        result.push((g_linear.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        result.push((b_linear.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        result.push(a);
    }
    
    result
}