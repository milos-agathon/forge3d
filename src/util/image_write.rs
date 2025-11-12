// src/util/image_write.rs
// PNG encoding utilities for writing tightly packed RGBA buffers
// Exists to centralize output validation for GPU readback pipelines
// RELEVANT FILES: src/renderer/readback.rs, src/lib.rs, src/terrain_renderer.rs, tools/validate_rows.py

use anyhow::{ensure, Context, Result};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const RGBA8_CHANNELS: usize = 4;

pub fn write_png_rgba8(path: &Path, data: &[u8], width: u32, height: u32) -> Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(RGBA8_CHANNELS))
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow when computing buffer size"))?;

    ensure!(
        data.len() == expected,
        "PNG writer requires tight RGBA8 buffer: expected {} bytes, got {}",
        expected,
        data.len()
    );

    let file = File::create(path)
        .with_context(|| format!("failed to create output PNG at {}", path.display()))?;
    let writer = BufWriter::new(file);
    // Use fast compression (level 1) instead of default (level 6) for 5-10x faster encoding
    let encoder = PngEncoder::new_with_quality(writer, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::NoFilter);
    encoder
        .write_image(data, width, height, ColorType::Rgba8.into())
        .context("failed to encode RGBA8 PNG")?;
    Ok(())
}

/// Write PNG favoring smaller file size (slower than `write_png_rgba8`).
/// Uses default zlib compression and adaptive filtering.
pub fn write_png_rgba8_small(path: &Path, data: &[u8], width: u32, height: u32) -> Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(RGBA8_CHANNELS))
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow when computing buffer size"))?;

    ensure!(
        data.len() == expected,
        "PNG writer requires tight RGBA8 buffer: expected {} bytes, got {}",
        expected,
        data.len()
    );

    let file = File::create(path)
        .with_context(|| format!("failed to create output PNG at {}", path.display()))?;
    let writer = BufWriter::new(file);
    // Default compression + adaptive filtering for better file size
    let encoder = PngEncoder::new_with_quality(
        writer,
        image::codecs::png::CompressionType::Default,
        image::codecs::png::FilterType::Adaptive,
    );
    encoder
        .write_image(data, width, height, ColorType::Rgba8.into())
        .context("failed to encode RGBA8 PNG (small)")?;
    Ok(())
}
