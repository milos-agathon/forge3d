//! Texture format definitions and presets
//!
//! Contains the static format definitions used by TextureFormatRegistry.

use super::texture_format::TextureFormatInfo;
use wgpu::TextureFormat;

/// Create all uncompressed format definitions
pub fn uncompressed_formats() -> Vec<TextureFormatInfo> {
    vec![
        TextureFormatInfo {
            format: TextureFormat::R8Unorm,
            is_compressed: false,
            bytes_per_pixel: 1,
            block_size: 1,
            channels: 1,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Rg8Unorm,
            is_compressed: false,
            bytes_per_pixel: 2,
            block_size: 1,
            channels: 2,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Rgba8Unorm,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Rgba8UnormSrgb,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: false,
            is_srgb: true,
        },
        TextureFormatInfo {
            format: TextureFormat::Bgra8Unorm,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::R16Float,
            is_compressed: false,
            bytes_per_pixel: 2,
            block_size: 1,
            channels: 1,
            bit_depth: 16,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Rgba16Float,
            is_compressed: false,
            bytes_per_pixel: 8,
            block_size: 1,
            channels: 4,
            bit_depth: 16,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::R32Float,
            is_compressed: false,
            bytes_per_pixel: 4,
            block_size: 1,
            channels: 1,
            bit_depth: 32,
            supports_linear: false,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Rgba32Float,
            is_compressed: false,
            bytes_per_pixel: 16,
            block_size: 1,
            channels: 4,
            bit_depth: 32,
            supports_linear: true,
            is_srgb: false,
        },
    ]
}

/// Create all BC (DirectX) compressed format definitions
pub fn bc_compressed_formats() -> Vec<TextureFormatInfo> {
    vec![
        TextureFormatInfo {
            format: TextureFormat::Bc1RgbaUnorm,
            is_compressed: true,
            bytes_per_pixel: 8,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc1RgbaUnormSrgb,
            is_compressed: true,
            bytes_per_pixel: 8,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: false,
            is_srgb: true,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc3RgbaUnorm,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc3RgbaUnormSrgb,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: false,
            is_srgb: true,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc4RUnorm,
            is_compressed: true,
            bytes_per_pixel: 8,
            block_size: 4,
            channels: 1,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc5RgUnorm,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 2,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc6hRgbUfloat,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 3,
            bit_depth: 16,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Bc7RgbaUnorm,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
    ]
}

/// Create all ETC2 (mobile) compressed format definitions
pub fn etc2_compressed_formats() -> Vec<TextureFormatInfo> {
    vec![
        TextureFormatInfo {
            format: TextureFormat::Etc2Rgb8Unorm,
            is_compressed: true,
            bytes_per_pixel: 8,
            block_size: 4,
            channels: 3,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
        TextureFormatInfo {
            format: TextureFormat::Etc2Rgba8Unorm,
            is_compressed: true,
            bytes_per_pixel: 16,
            block_size: 4,
            channels: 4,
            bit_depth: 8,
            supports_linear: true,
            is_srgb: false,
        },
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
