// src/renderer/readback.rs
// Canonical RGBA8 texture readback helper yielding tight CPU buffers
// Exists to guarantee consistent depadded downloads for image export paths
// RELEVANT FILES: src/renderer.rs, src/terrain_renderer.rs, src/util/image_write.rs, src/gpu.rs

use anyhow::{bail, ensure, Result};

pub fn read_texture_tight(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Texture,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> Result<Vec<u8>> {
    let (width, height) = size;

    ensure!(width > 0 && height > 0, "readback size must be positive");
    ensure!(
        src.sample_count() == 1,
        "readback requires single-sample texture (sample_count=1), got {}",
        src.sample_count()
    );
    ensure!(
        src.format() == format,
        "texture format mismatch: texture={:?}, requested={:?}",
        src.format(),
        format
    );

    match format {
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb
        | wgpu::TextureFormat::Bgra8Unorm
        | wgpu::TextureFormat::Bgra8UnormSrgb
        | wgpu::TextureFormat::Rgba16Float => {}
        _ => bail!(
            "read_texture_tight only supports RGBA8/BGRA8/RGBA16F formats, got {:?}",
            format
        ),
    }

    // Frame beauty downloads must use the same completion-owned transfer as
    // HDR/AOV downloads. A separate Maintain::Wait path can prematurely retire
    // unfinished Metal work under wgpu 0.19's internal blocking-wait timeout.
    Ok(crate::core::texture_readback::read_texture_sync(
        device,
        queue,
        src,
        width,
        height,
        format,
        "forge3d-readback-staging",
    )?)
}
