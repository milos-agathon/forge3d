//! EXR encoding utilities for HDR output.
//!
//! Centralizes channel naming and validation for EXR write paths.

use anyhow::{ensure, Context, Result};
use exr::prelude::*;
use std::path::Path;

#[derive(Debug)]
pub struct ExrChannelData {
    pub name: String,
    pub data: Vec<f32>,
    pub quantize_linearly: bool,
}

#[derive(Debug)]
struct ChannelSpec {
    name: String,
    quantize_linearly: bool,
}

fn channel_specs(prefix: &str, channel_count: usize) -> Result<Vec<ChannelSpec>> {
    let prefix = prefix.trim();
    ensure!(!prefix.is_empty(), "EXR channel prefix must be non-empty");

    match channel_count {
        1 => {
            let name = if prefix.eq_ignore_ascii_case("depth") {
                format!("{prefix}.Z")
            } else {
                prefix.to_string()
            };
            Ok(vec![ChannelSpec {
                name,
                quantize_linearly: true,
            }])
        }
        3 => {
            let (suffixes, linear) = if prefix.eq_ignore_ascii_case("normal") {
                (["X", "Y", "Z"], true)
            } else {
                (["R", "G", "B"], false)
            };
            Ok(suffixes
                .iter()
                .map(|suffix| ChannelSpec {
                    name: format!("{prefix}.{suffix}"),
                    quantize_linearly: linear,
                })
                .collect())
        }
        4 => {
            let suffixes = ["R", "G", "B", "A"];
            Ok(suffixes
                .iter()
                .map(|suffix| ChannelSpec {
                    name: format!("{prefix}.{suffix}"),
                    quantize_linearly: *suffix == "A",
                })
                .collect())
        }
        _ => anyhow::bail!("unsupported EXR channel count: {}", channel_count),
    }
}

fn pixel_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow"))
}

fn split_rgba(
    data: &[f32],
    expected_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    ensure!(
        data.len() == expected_pixels * 4,
        "expected RGBA buffer with {} floats, got {}",
        expected_pixels * 4,
        data.len()
    );
    let mut r = Vec::with_capacity(expected_pixels);
    let mut g = Vec::with_capacity(expected_pixels);
    let mut b = Vec::with_capacity(expected_pixels);
    let mut a = Vec::with_capacity(expected_pixels);
    for px in data.chunks_exact(4) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
        a.push(px[3]);
    }
    Ok((r, g, b, a))
}

fn split_rgb_from_rgba(
    data: &[f32],
    expected_pixels: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    ensure!(
        data.len() == expected_pixels * 4,
        "expected RGBA buffer with {} floats, got {}",
        expected_pixels * 4,
        data.len()
    );
    let mut r = Vec::with_capacity(expected_pixels);
    let mut g = Vec::with_capacity(expected_pixels);
    let mut b = Vec::with_capacity(expected_pixels);
    for px in data.chunks_exact(4) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }
    Ok((r, g, b))
}

fn split_rgb(data: &[f32], expected_pixels: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    ensure!(
        data.len() == expected_pixels * 3,
        "expected RGB buffer with {} floats, got {}",
        expected_pixels * 3,
        data.len()
    );
    let mut r = Vec::with_capacity(expected_pixels);
    let mut g = Vec::with_capacity(expected_pixels);
    let mut b = Vec::with_capacity(expected_pixels);
    for px in data.chunks_exact(3) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }
    Ok((r, g, b))
}

fn to_any_channel(channel: ExrChannelData, expected_pixels: usize) -> Result<AnyChannel<FlatSamples>> {
    ensure!(
        channel.data.len() == expected_pixels,
        "channel {} expected {} samples, got {}",
        channel.name,
        expected_pixels,
        channel.data.len()
    );
    let name = Text::new_or_none(&channel.name)
        .ok_or_else(|| anyhow::anyhow!("invalid EXR channel name: {}", channel.name))?;
    Ok(AnyChannel {
        name,
        sample_data: FlatSamples::F32(channel.data),
        quantize_linearly: channel.quantize_linearly,
        sampling: Vec2(1, 1),
    })
}

pub fn write_exr_f32_channels(
    path: &Path,
    width: u32,
    height: u32,
    channels: Vec<ExrChannelData>,
) -> Result<()> {
    ensure!(width > 0 && height > 0, "EXR dimensions must be positive");
    ensure!(!channels.is_empty(), "EXR requires at least one channel");

    let expected_pixels = pixel_count(width, height)?;
    let mut list = SmallVec::<[AnyChannel<FlatSamples>; 4]>::new();
    for channel in channels {
        list.push(to_any_channel(channel, expected_pixels)?);
    }

    let channels = AnyChannels::sort(list);
    let image = Image::from_channels((width as usize, height as usize), channels);
    image
        .write()
        .to_file(path)
        .with_context(|| format!("failed to write EXR to {}", path.display()))?;
    Ok(())
}

pub fn write_exr_rgba_f32(
    path: &Path,
    width: u32,
    height: u32,
    data: &[f32],
    prefix: &str,
) -> Result<()> {
    let expected_pixels = pixel_count(width, height)?;
    let (r, g, b, a) = split_rgba(data, expected_pixels)?;
    let specs = channel_specs(prefix, 4)?;
    let channels = vec![
        ExrChannelData {
            name: specs[0].name.clone(),
            data: r,
            quantize_linearly: specs[0].quantize_linearly,
        },
        ExrChannelData {
            name: specs[1].name.clone(),
            data: g,
            quantize_linearly: specs[1].quantize_linearly,
        },
        ExrChannelData {
            name: specs[2].name.clone(),
            data: b,
            quantize_linearly: specs[2].quantize_linearly,
        },
        ExrChannelData {
            name: specs[3].name.clone(),
            data: a,
            quantize_linearly: specs[3].quantize_linearly,
        },
    ];
    write_exr_f32_channels(path, width, height, channels)
}

pub fn write_exr_rgb_f32_from_rgba(
    path: &Path,
    width: u32,
    height: u32,
    data: &[f32],
    prefix: &str,
) -> Result<()> {
    let expected_pixels = pixel_count(width, height)?;
    let (r, g, b) = split_rgb_from_rgba(data, expected_pixels)?;
    let specs = channel_specs(prefix, 3)?;
    let channels = vec![
        ExrChannelData {
            name: specs[0].name.clone(),
            data: r,
            quantize_linearly: specs[0].quantize_linearly,
        },
        ExrChannelData {
            name: specs[1].name.clone(),
            data: g,
            quantize_linearly: specs[1].quantize_linearly,
        },
        ExrChannelData {
            name: specs[2].name.clone(),
            data: b,
            quantize_linearly: specs[2].quantize_linearly,
        },
    ];
    write_exr_f32_channels(path, width, height, channels)
}

pub fn write_exr_rgb_f32(
    path: &Path,
    width: u32,
    height: u32,
    data: &[f32],
    prefix: &str,
) -> Result<()> {
    let expected_pixels = pixel_count(width, height)?;
    let (r, g, b) = split_rgb(data, expected_pixels)?;
    let specs = channel_specs(prefix, 3)?;
    let channels = vec![
        ExrChannelData {
            name: specs[0].name.clone(),
            data: r,
            quantize_linearly: specs[0].quantize_linearly,
        },
        ExrChannelData {
            name: specs[1].name.clone(),
            data: g,
            quantize_linearly: specs[1].quantize_linearly,
        },
        ExrChannelData {
            name: specs[2].name.clone(),
            data: b,
            quantize_linearly: specs[2].quantize_linearly,
        },
    ];
    write_exr_f32_channels(path, width, height, channels)
}

pub fn write_exr_scalar_f32(
    path: &Path,
    width: u32,
    height: u32,
    data: &[f32],
    prefix: &str,
) -> Result<()> {
    let expected_pixels = pixel_count(width, height)?;
    ensure!(
        data.len() == expected_pixels,
        "expected scalar buffer with {} floats, got {}",
        expected_pixels,
        data.len()
    );
    let specs = channel_specs(prefix, 1)?;
    let channels = vec![ExrChannelData {
        name: specs[0].name.clone(),
        data: data.to_vec(),
        quantize_linearly: specs[0].quantize_linearly,
    }];
    write_exr_f32_channels(path, width, height, channels)
}
