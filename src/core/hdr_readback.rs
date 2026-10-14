// GPU texture readback format adapters. Transfer mechanics live in texture_readback.
use crate::core::texture_readback::read_texture_sync;
use wgpu::{Device, Queue, Texture, TextureFormat};

/// Read HDR data from a floating-point texture.
pub fn read_hdr_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<Vec<f32>, String> {
    if !matches!(
        format,
        TextureFormat::Rgba16Float | TextureFormat::Rgba32Float
    ) {
        return Err("Unsupported HDR format for readback".to_string());
    }
    let bytes = read_texture_sync(
        device,
        queue,
        texture,
        width,
        height,
        format,
        "hdr_readback.hdr_staging",
    )
    .map_err(|e| e.to_string())?;
    Ok(match format {
        TextureFormat::Rgba16Float => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        TextureFormat::Rgba32Float => decode_f32(&bytes),
        _ => unreachable!("HDR format was validated before transfer"),
    })
}

/// Read LDR bytes, preserving the source channel order and color space.
pub fn read_ldr_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    read_texture_sync(
        device,
        queue,
        texture,
        width,
        height,
        TextureFormat::Rgba8Unorm,
        "hdr_readback.ldr_staging",
    )
    .map_err(|e| e.to_string())
}

/// Read scalar R32Float data from a texture.
pub fn read_r32_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, String> {
    let bytes = read_texture_sync(
        device,
        queue,
        texture,
        width,
        height,
        TextureFormat::R32Float,
        "hdr_readback.r32_staging",
    )
    .map_err(|e| e.to_string())?;
    Ok(decode_f32(&bytes))
}

/// Read scalar R32Uint source IDs. The tracked staging allocation is released
/// before this function returns, including on a mapping failure.
pub fn read_r32uint_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u32>, String> {
    let bytes = read_texture_sync(
        device,
        queue,
        texture,
        width,
        height,
        TextureFormat::R32Uint,
        "hdr_readback.r32uint_staging",
    )
    .map_err(|e| e.to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
