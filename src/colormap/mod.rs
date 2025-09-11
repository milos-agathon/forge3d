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

/// O3: Compressed texture integration for colormaps
use crate::core::compressed_textures::{CompressedImage, CompressionOptions};
use crate::core::texture_format::{TextureUseCase, CompressionQuality};

/// Create compressed colormap texture from name
pub fn create_compressed_colormap(
    name: &str,
    device: &wgpu::Device,
    quality: CompressionQuality,
) -> Result<CompressedImage, String> {
    // Get raw RGBA data
    let rgba_data = decode_png_rgba8(name)?;
    
    // Colormaps are 256x1, so let's create a larger texture for better compression
    let expanded_data = expand_colormap_for_compression(&rgba_data, 256, 256)?;
    
    // Configure compression options for colormap use case
    let options = CompressionOptions {
        target_format: None, // Auto-select
        quality,
        generate_mipmaps: true,
        use_case: TextureUseCase::Albedo, // Colormaps are like albedo textures
        max_size: 256,
        force_power_of_2: true,
    };
    
    // Create compressed image
    CompressedImage::from_rgba_data(&expanded_data, 256, 256, device, &options)
}

/// Expand 256x1 colormap to 256x256 for better compression efficiency
fn expand_colormap_for_compression(
    colormap_data: &[u8],
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, String> {
    if colormap_data.len() != 256 * 4 {
        return Err("Invalid colormap data size".to_string());
    }
    
    let mut expanded = vec![0u8; (target_width * target_height * 4) as usize];
    
    // Replicate the 1D colormap vertically to create a 2D texture
    for y in 0..target_height {
        for x in 0..target_width {
            let src_offset = (x * 4) as usize;
            let dst_offset = ((y * target_width + x) * 4) as usize;
            
            if src_offset + 3 < colormap_data.len() && dst_offset + 3 < expanded.len() {
                expanded[dst_offset..dst_offset + 4]
                    .copy_from_slice(&colormap_data[src_offset..src_offset + 4]);
            }
        }
    }
    
    Ok(expanded)
}

/// Get compression statistics for a colormap
pub fn get_colormap_compression_stats(name: &str) -> Result<String, String> {
    let original_data = decode_png_rgba8(name)?;
    let original_size = original_data.len();
    
    // Estimate compression ratio for different formats
    let bc1_ratio = 4.0; // BC1 has ~4:1 compression
    let bc7_ratio = 2.0; // BC7 has ~2:1 compression
    let etc2_ratio = 3.0; // ETC2 has ~3:1 compression
    
    Ok(format!(
        "Colormap '{}' compression estimates:\n\
         Original size: {} bytes\n\
         BC1 compressed: ~{} bytes ({:.1}:1 ratio)\n\
         BC7 compressed: ~{} bytes ({:.1}:1 ratio)\n\
         ETC2 compressed: ~{} bytes ({:.1}:1 ratio)",
        name, original_size,
        original_size as f32 / bc1_ratio, bc1_ratio,
        original_size as f32 / bc7_ratio, bc7_ratio,
        original_size as f32 / etc2_ratio, etc2_ratio,
    ))
}

/// Check if compressed texture formats are available for colormaps
pub fn check_compressed_colormap_support(device: &wgpu::Device) -> Vec<String> {
    use crate::core::texture_format::global_format_registry;
    
    let registry = global_format_registry();
    let features = device.features();
    
    let mut supported_formats = Vec::new();
    
    // Check BC formats
    if registry.is_format_supported(wgpu::TextureFormat::Bc1RgbaUnorm, &features) {
        supported_formats.push("BC1 (4:1 compression)".to_string());
    }
    if registry.is_format_supported(wgpu::TextureFormat::Bc7RgbaUnorm, &features) {
        supported_formats.push("BC7 (high quality)".to_string());
    }
    
    // Check ETC2 formats
    if registry.is_format_supported(wgpu::TextureFormat::Etc2Rgba8Unorm, &features) {
        supported_formats.push("ETC2 (mobile optimized)".to_string());
    }
    
    supported_formats
}