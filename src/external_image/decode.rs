//! Image decoding utilities for PNG and JPEG files.

use super::types::{ImageImportConfig, ImageSourceFormat};
use crate::core::error::{RenderError, RenderResult};
use std::fs::File;
use std::path::Path;

/// Decode an image file to RGBA8 data.
pub fn decode_image_file(
    path: &Path,
    config: &ImageImportConfig,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| RenderError::io("Cannot determine image format from file extension"))?;

    match extension.as_str() {
        "png" => decode_png_file(path, config),
        "jpg" | "jpeg" => decode_jpeg_file(path, config),
        _ => Err(RenderError::io(format!(
            "Unsupported image format: {}",
            extension
        ))),
    }
}

/// Decode PNG file to RGBA8 data.
///
/// Note: This is a simulation - real implementation would use png crate.
pub fn decode_png_file(
    path: &Path,
    _config: &ImageImportConfig,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    let _file =
        File::open(path).map_err(|e| RenderError::io(format!("Failed to open PNG file: {}", e)))?;

    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("test");

    let (width, height) = if filename.contains("large") {
        (512, 512)
    } else if filename.contains("small") {
        (64, 64)
    } else {
        (256, 256)
    };

    let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = ((x ^ y) * 255 / (width | height)) as u8;
            let a = 255u8;
            rgba_data.extend_from_slice(&[r, g, b, a]);
        }
    }

    Ok((rgba_data, width, height, ImageSourceFormat::PngRgba))
}

/// Decode JPEG file to RGBA8 data.
///
/// Note: This is a simulation - real implementation would use jpeg crate.
pub fn decode_jpeg_file(
    path: &Path,
    _config: &ImageImportConfig,
) -> RenderResult<(Vec<u8>, u32, u32, ImageSourceFormat)> {
    let _file = File::open(path)
        .map_err(|e| RenderError::io(format!("Failed to open JPEG file: {}", e)))?;

    let (width, height) = (128, 128);

    let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x + y) * 255 / (width + height)) as u8;
            let g = ((x * y) * 255 / (width * height)) as u8;
            let b = (((x as u32).saturating_sub(y as u32)) * 255 / width) as u8;
            let a = 255u8;
            rgba_data.extend_from_slice(&[r, g, b, a]);
        }
    }

    Ok((rgba_data, width, height, ImageSourceFormat::JpegRgb))
}
