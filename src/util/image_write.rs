//! PNG encoding utilities for writing tightly packed RGBA buffers.
//!
//! Centralizes output validation for GPU readback pipelines.

use anyhow::{ensure, Context, Result};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ImageEncoder};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Number of channels in RGBA8 format.
const RGBA8_CHANNELS: usize = 4;

/// Write PNG with fast compression (5–10× faster, larger files).
///
/// Uses compression level 1 with no filtering for maximum encoding speed.
pub fn write_png_rgba8(path: &Path, data: &[u8], width: u32, height: u32) -> Result<()> {
    write_png_rgba8_with_settings(
        path,
        data,
        width,
        height,
        CompressionType::Fast,
        FilterType::NoFilter,
    )
}

/// Write PNG with default compression (slower, smaller files).
///
/// Uses default zlib compression with adaptive filtering for better file size.
pub fn write_png_rgba8_small(path: &Path, data: &[u8], width: u32, height: u32) -> Result<()> {
    write_png_rgba8_with_settings(
        path,
        data,
        width,
        height,
        CompressionType::Default,
        FilterType::Adaptive,
    )
}

/// Core PNG writer with configurable compression and filter settings.
fn write_png_rgba8_with_settings(
    path: &Path,
    data: &[u8],
    width: u32,
    height: u32,
    compression: CompressionType,
    filter: FilterType,
) -> Result<()> {
    let expected = compute_expected_buffer_size(width, height)?;

    ensure!(
        data.len() == expected,
        "PNG writer requires tight RGBA8 buffer: expected {} bytes, got {}",
        expected,
        data.len()
    );

    let file = File::create(path)
        .with_context(|| format!("failed to create output PNG at {}", path.display()))?;

    let encoder = PngEncoder::new_with_quality(BufWriter::new(file), compression, filter);
    encoder
        .write_image(data, width, height, ColorType::Rgba8.into())
        .context("failed to encode RGBA8 PNG")?;

    Ok(())
}

/// Compute expected buffer size with overflow checking.
fn compute_expected_buffer_size(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(RGBA8_CHANNELS))
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow when computing buffer size"))
}
