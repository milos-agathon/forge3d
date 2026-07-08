// src/core/hdr_readback.rs
// GPU texture readback utilities for HDR rendering
// RELEVANT FILES: shaders/postprocess_tonemap.wgsl

use wgpu::{
    BufferDescriptor, BufferUsages, Device, Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d,
    Queue, Texture, TextureAspect, TextureFormat,
};

/// Read HDR data from a floating-point texture
pub fn read_hdr_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<Vec<f32>, String> {
    let bpp = match format {
        TextureFormat::Rgba16Float => 8,  // 4 channels * 2 bytes
        TextureFormat::Rgba32Float => 16, // 4 channels * 4 bytes
        _ => return Err("Unsupported HDR format for readback".to_string()),
    };

    let unpadded_bytes_per_row = width * bpp;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
    };

    let buffer_size = padded_bytes_per_row * height;

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("hdr_staging_buffer"),
        size: buffer_size as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hdr_copy_encoder"),
    });

    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Map and read the buffer
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();

    // Convert to float data
    let hdr_data = convert_to_floats(&data, width, height, padded_bytes_per_row, format)?;

    drop(data);
    staging_buffer.unmap();

    Ok(hdr_data)
}

/// Read LDR data from RGBA8 texture
pub fn read_ldr_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let bpp = 4; // RGBA8
    let unpadded_bytes_per_row = width * bpp;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
    };

    let buffer_size = padded_bytes_per_row * height;

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("ldr_staging_buffer"),
        size: buffer_size as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ldr_copy_encoder"),
    });

    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Map and read the buffer
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();

    // Copy LDR data (remove padding)
    let mut ldr_data = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        let row_offset = (y * padded_bytes_per_row) as usize;
        let row_data = &data[row_offset..row_offset + unpadded_bytes_per_row as usize];
        ldr_data.extend_from_slice(row_data);
    }

    drop(data);
    staging_buffer.unmap();

    Ok(ldr_data)
}

/// Read scalar R32Float data from a texture.
pub fn read_r32_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, String> {
    let bpp = 4u32;
    let unpadded_bytes_per_row = width * bpp;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
    };

    let buffer_size = padded_bytes_per_row * height;
    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("r32_staging_buffer"),
        size: buffer_size as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("r32_copy_encoder"),
    });

    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let mut values = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        let row_offset = (y * padded_bytes_per_row) as usize;
        for x in 0..width {
            let pixel_offset = row_offset + (x * bpp) as usize;
            let bytes = [
                data[pixel_offset],
                data[pixel_offset + 1],
                data[pixel_offset + 2],
                data[pixel_offset + 3],
            ];
            values.push(f32::from_le_bytes(bytes));
        }
    }

    drop(data);
    staging_buffer.unmap();
    Ok(values)
}

/// VERITAS: read scalar R32Uint data (per-pixel source-id map) from a
/// texture. The transient host-visible staging buffer is routed through the
/// memory tracker and released before returning.
pub fn read_r32uint_texture(
    device: &Device,
    queue: &Queue,
    texture: &Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u32>, String> {
    let bpp = 4u32;
    let unpadded_bytes_per_row = width * bpp;
    let padded_bytes_per_row = {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((unpadded_bytes_per_row + alignment - 1) / alignment) * alignment
    };

    let buffer_size = (padded_bytes_per_row * height) as u64;
    let memory_tracker = crate::core::memory_tracker::global_tracker();
    memory_tracker.track_buffer_allocation(buffer_size, true);
    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("r32uint_staging_buffer"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let release_staging = |staging_buffer: &wgpu::Buffer| {
        staging_buffer.destroy();
        memory_tracker.free_buffer_allocation(buffer_size, true);
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("r32uint_copy_encoder"),
    });
    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    let map_result = receiver
        .recv()
        .map_err(|e| format!("Buffer mapping callback lost: {e}"));
    let map_result = match map_result {
        Ok(inner) => inner.map_err(|e| format!("Buffer mapping failed: {e:?}")),
        Err(e) => Err(e),
    };
    if let Err(e) = map_result {
        release_staging(&staging_buffer);
        return Err(e);
    }

    let data = buffer_slice.get_mapped_range();
    let mut values = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        let row_offset = (y * padded_bytes_per_row) as usize;
        for x in 0..width {
            let pixel_offset = row_offset + (x * bpp) as usize;
            let bytes = [
                data[pixel_offset],
                data[pixel_offset + 1],
                data[pixel_offset + 2],
                data[pixel_offset + 3],
            ];
            values.push(u32::from_le_bytes(bytes));
        }
    }

    drop(data);
    staging_buffer.unmap();
    release_staging(&staging_buffer);
    Ok(values)
}

/// Convert raw buffer data to f32 vector based on texture format
fn convert_to_floats(
    data: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    format: TextureFormat,
) -> Result<Vec<f32>, String> {
    let mut hdr_data = Vec::new();

    match format {
        TextureFormat::Rgba16Float => {
            for y in 0..height {
                let row_offset = (y * padded_bytes_per_row) as usize;
                for x in 0..width {
                    let pixel_offset = row_offset + (x * 8) as usize;

                    for c in 0..4 {
                        let half_bytes =
                            [data[pixel_offset + c * 2], data[pixel_offset + c * 2 + 1]];
                        let half_val = half::f16::from_le_bytes(half_bytes);
                        hdr_data.push(half_val.to_f32());
                    }
                }
            }
        }
        TextureFormat::Rgba32Float => {
            for y in 0..height {
                let row_offset = (y * padded_bytes_per_row) as usize;
                for x in 0..width {
                    let pixel_offset = row_offset + (x * 16) as usize;

                    for c in 0..4 {
                        let float_bytes = [
                            data[pixel_offset + c * 4],
                            data[pixel_offset + c * 4 + 1],
                            data[pixel_offset + c * 4 + 2],
                            data[pixel_offset + c * 4 + 3],
                        ];
                        let float_val = f32::from_le_bytes(float_bytes);
                        hdr_data.push(float_val);
                    }
                }
            }
        }
        _ => return Err("Unsupported HDR format".to_string()),
    }

    Ok(hdr_data)
}
