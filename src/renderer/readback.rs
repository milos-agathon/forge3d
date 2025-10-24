// src/renderer/readback.rs
// Canonical RGBA8 texture readback helper yielding tight CPU buffers
// Exists to guarantee consistent depadded downloads for image export paths
// RELEVANT FILES: src/renderer.rs, src/terrain_renderer.rs, src/util/image_write.rs, src/gpu.rs

use anyhow::{anyhow, bail, ensure, Result};
use futures_intrusive::channel::shared::oneshot_channel;
use std::num::NonZeroU32;

/// Align number to WebGPU's copy row alignment (256 bytes).
fn align_bpr(value: usize) -> usize {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    ((value + align - 1) / align) * align
}

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

    let bytes_per_pixel = match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => 4,
        wgpu::TextureFormat::Rgba16Float => 8,
        _ => bail!(
            "read_texture_tight only supports RGBA8/RGBA16F formats, got {:?}",
            format
        ),
    };

    let tight_bpr = bytes_per_pixel * width as usize;
    let padded_bpr = align_bpr(tight_bpr);
    ensure!(
        padded_bpr <= u32::MAX as usize,
        "padded bytes per row exceeds u32::MAX"
    );
    let rows_per_image =
        NonZeroU32::new(height).ok_or_else(|| anyhow!("rows_per_image must be non-zero"))?;
    let bytes_per_row = NonZeroU32::new(padded_bpr as u32)
        .ok_or_else(|| anyhow!("bytes_per_row must be non-zero"))?;
    let buffer_size = (padded_bpr * height as usize) as wgpu::BufferAddress;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forge3d-readback-staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("forge3d-readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row.get()),
                rows_per_image: Some(rows_per_image.get()),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = staging.slice(..);
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver.receive())
        .ok_or_else(|| anyhow!("map_async callback channel dropped"))??;

    let data = slice.get_mapped_range();
    let expected_tight_size = tight_bpr * height as usize;
    let mut tight = vec![0u8; expected_tight_size];

    // Depad rows: copy tight_bpr bytes from each padded row
    for row in 0..height as usize {
        let src_offset = row * padded_bpr;
        let dst_offset = row * tight_bpr;
        tight[dst_offset..dst_offset + tight_bpr]
            .copy_from_slice(&data[src_offset..src_offset + tight_bpr]);
    }
    drop(data);
    staging.unmap();

    Ok(tight)
}
