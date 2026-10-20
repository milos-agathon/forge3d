//! Texture format definitions and presets
//!
//! Contains the static format definitions used by TextureFormatRegistry.

use super::texture_format::TextureFormatInfo;
use wgpu::TextureFormat;

fn format_info(
    format: TextureFormat,
    channels: u32,
    bit_depth: u32,
    supports_linear: bool,
) -> TextureFormatInfo {
    let layout = super::texture_layout::FormatLayout::new(format);
    TextureFormatInfo {
        format,
        is_compressed: format.is_compressed(),
        bytes_per_pixel: layout.resident_bytes_per_block,
        block_size: layout.block_width,
        channels,
        bit_depth,
        supports_linear,
        is_srgb: format.is_srgb(),
    }
}

/// Create all uncompressed format definitions
pub fn uncompressed_formats() -> Vec<TextureFormatInfo> {
    vec![
        format_info(TextureFormat::R8Unorm, 1, 8, true),
        format_info(TextureFormat::Rg8Unorm, 2, 8, true),
        format_info(TextureFormat::Rgba8Unorm, 4, 8, true),
        format_info(TextureFormat::Rgba8UnormSrgb, 4, 8, false),
        format_info(TextureFormat::Bgra8Unorm, 4, 8, true),
        format_info(TextureFormat::R16Float, 1, 16, true),
        format_info(TextureFormat::Rgba16Float, 4, 16, true),
        format_info(TextureFormat::R32Float, 1, 32, false),
        format_info(TextureFormat::Rgba32Float, 4, 32, true),
    ]
}

/// Create all BC (DirectX) compressed format definitions
pub fn bc_compressed_formats() -> Vec<TextureFormatInfo> {
    vec![
        format_info(TextureFormat::Bc1RgbaUnorm, 4, 8, true),
        format_info(TextureFormat::Bc1RgbaUnormSrgb, 4, 8, false),
        format_info(TextureFormat::Bc3RgbaUnorm, 4, 8, true),
        format_info(TextureFormat::Bc3RgbaUnormSrgb, 4, 8, false),
        format_info(TextureFormat::Bc4RUnorm, 1, 8, true),
        format_info(TextureFormat::Bc5RgUnorm, 2, 8, true),
        format_info(TextureFormat::Bc6hRgbUfloat, 3, 16, true),
        format_info(TextureFormat::Bc7RgbaUnorm, 4, 8, true),
    ]
}

/// Create all ETC2 (mobile) compressed format definitions
pub fn etc2_compressed_formats() -> Vec<TextureFormatInfo> {
    vec![
        format_info(TextureFormat::Etc2Rgb8Unorm, 3, 8, true),
        format_info(TextureFormat::Etc2Rgba8Unorm, 4, 8, true),
    ]
}

/// Get all format definitions
pub fn all_format_definitions() -> Vec<TextureFormatInfo> {
    let mut formats = Vec::with_capacity(20);
    formats.extend(uncompressed_formats());
    formats.extend(bc_compressed_formats());
    formats.extend(etc2_compressed_formats());
    formats
}
