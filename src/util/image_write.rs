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
const RGBA16_BYTES_PER_PIXEL: usize = 8;

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

/// Write a tightly packed big-endian RGBA16 PNG with fast compression.
pub fn write_png_rgba16(path: &Path, data: &[u8], width: u32, height: u32) -> Result<()> {
    write_png_rgba16_with_settings(
        path,
        data,
        width,
        height,
        CompressionType::Fast,
        FilterType::NoFilter,
    )
}

/// Write a tightly packed big-endian RGBA16 PNG with default compression.
pub fn write_png_rgba16_small(path: &Path, data: &[u8], width: u32, height: u32) -> Result<()> {
    write_png_rgba16_with_settings(
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
    let expected = compute_expected_buffer_size(width, height, RGBA8_CHANNELS)?;

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

fn write_png_rgba16_with_settings(
    path: &Path,
    data: &[u8],
    width: u32,
    height: u32,
    compression: CompressionType,
    filter: FilterType,
) -> Result<()> {
    let expected = compute_expected_buffer_size(width, height, RGBA16_BYTES_PER_PIXEL)?;

    ensure!(
        data.len() == expected,
        "PNG writer requires tight RGBA16 buffer: expected {} bytes, got {}",
        expected,
        data.len()
    );

    let file = File::create(path)
        .with_context(|| format!("failed to create output PNG at {}", path.display()))?;

    let encoder = PngEncoder::new_with_quality(BufWriter::new(file), compression, filter);
    encoder
        .write_image(data, width, height, ColorType::Rgba16.into())
        .context("failed to encode RGBA16 PNG")?;

    Ok(())
}

/// Compute expected buffer size with overflow checking.
fn compute_expected_buffer_size(width: u32, height: u32, bytes_per_pixel: usize) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(bytes_per_pixel))
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow when computing buffer size"))
}
