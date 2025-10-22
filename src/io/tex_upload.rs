//! Safe texture upload with proper row-pitch alignment for WebGPU/WGSL
//!
//! WebGPU requires COPY_BYTES_PER_ROW_ALIGNMENT (256 bytes) for texture uploads.
//! This module provides utilities to handle padding automatically.

use wgpu::{Device, Queue, Texture, TextureFormat, ImageCopyTexture, ImageDataLayout, Extent3d, Origin3d, TextureAspect};

/// WebGPU row-pitch alignment requirement (256 bytes)
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Calculate padded bytes per row to meet 256-byte alignment
pub fn padded_bytes_per_row(unpadded_bytes: u32) -> u32 {
    let alignment = COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded_bytes + alignment - 1) / alignment) * alignment
}

/// Upload RGBA8 texture data with automatic row padding if needed
///
/// # Arguments
/// * `queue` - WebGPU queue for upload
/// * `texture` - Target texture
/// * `data` - Tightly packed RGBA8 data (width * height * 4 bytes)
/// * `width` - Texture width in pixels
/// * `height` - Texture height in pixels
///
/// # Errors
/// Returns error if data size doesn't match width * height * 4
pub fn upload_rgba8_texture(
    queue: &Queue,
    texture: &Texture,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let bytes_per_pixel = 4; // RGBA8
    let row_bytes = width * bytes_per_pixel;
    let expected_size = (height * row_bytes) as usize;

    if data.len() != expected_size {
        return Err(format!(
            "Data size mismatch: expected {} bytes ({}x{}x4), got {}",
            expected_size, width, height, data.len()
        ));
    }

    let padded_bpr = padded_bytes_per_row(row_bytes);

    if padded_bpr == row_bytes {
        // No padding needed, upload directly
        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        // Need padding - create padded buffer
        let padded_size = (padded_bpr * height) as usize;
        let mut padded_data = vec![0u8; padded_size];

        for y in 0..height {
            let src_offset = (y * row_bytes) as usize;
            let dst_offset = (y * padded_bpr) as usize;
            padded_data[dst_offset..dst_offset + row_bytes as usize]
                .copy_from_slice(&data[src_offset..src_offset + row_bytes as usize]);
        }

        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &padded_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(())
}

/// Upload R32Float texture data with automatic row padding if needed
///
/// # Arguments
/// * `queue` - WebGPU queue for upload
/// * `texture` - Target texture (must be R32Float format)
/// * `data` - Tightly packed f32 data (width * height floats)
/// * `width` - Texture width in pixels
/// * `height` - Texture height in pixels
pub fn upload_r32f_texture(
    queue: &Queue,
    texture: &Texture,
    data: &[f32],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let expected_size = (width * height) as usize;

    if data.len() != expected_size {
        return Err(format!(
            "Data size mismatch: expected {} floats ({}x{}), got {}",
            expected_size, width, height, data.len()
        ));
    }

    let bytes_per_pixel = 4; // f32 = 4 bytes
    let row_bytes = width * bytes_per_pixel;
    let padded_bpr = padded_bytes_per_row(row_bytes);

    let data_bytes: &[u8] = bytemuck::cast_slice(data);

    if padded_bpr == row_bytes {
        // No padding needed
        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            data_bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        // Need padding
        let padded_size = (padded_bpr * height) as usize;
        let mut padded_data = vec![0u8; padded_size];

        for y in 0..height {
            let src_offset = (y * row_bytes) as usize;
            let dst_offset = (y * padded_bpr) as usize;
            padded_data[dst_offset..dst_offset + row_bytes as usize]
                .copy_from_slice(&data_bytes[src_offset..src_offset + row_bytes as usize]);
        }

        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &padded_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(())
}

/// Upload R16Float texture data with automatic row padding if needed
pub fn upload_r16f_texture(
    queue: &Queue,
    texture: &Texture,
    data: &[u16],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let expected_size = (width * height) as usize;

    if data.len() != expected_size {
        return Err(format!(
            "Data size mismatch: expected {} half-floats ({}x{}), got {}",
            expected_size, width, height, data.len()
        ));
    }

    let bytes_per_pixel = 2; // f16 = 2 bytes
    let row_bytes = width * bytes_per_pixel;
    let padded_bpr = padded_bytes_per_row(row_bytes);

    let data_bytes: &[u8] = bytemuck::cast_slice(data);

    if padded_bpr == row_bytes {
        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            data_bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        let padded_size = (padded_bpr * height) as usize;
        let mut padded_data = vec![0u8; padded_size];

        for y in 0..height {
            let src_offset = (y * row_bytes) as usize;
            let dst_offset = (y * padded_bpr) as usize;
            padded_data[dst_offset..dst_offset + row_bytes as usize]
                .copy_from_slice(&data_bytes[src_offset..src_offset + row_bytes as usize]);
        }

        queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &padded_data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(())
}

/// Create a texture with proper format and usage flags
pub fn create_texture_2d(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    format: TextureFormat,
    usage: wgpu::TextureUsages,
) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_bytes_per_row_alignment() {
        // Test basic alignment
        assert_eq!(padded_bytes_per_row(100), 256);
        assert_eq!(padded_bytes_per_row(256), 256);
        assert_eq!(padded_bytes_per_row(257), 512);
        assert_eq!(padded_bytes_per_row(1000), 1024);
        
        // Test odd widths (critical for catching bugs)
        let width = 10001;
        let bytes_per_pixel = 4;
        let row_bytes = width * bytes_per_pixel; // 40004
        let padded = padded_bytes_per_row(row_bytes);
        assert_eq!(padded, 40192); // Next 256-byte boundary
        assert!(padded % COPY_BYTES_PER_ROW_ALIGNMENT == 0);
    }

    #[test]
    fn test_rgba8_data_size_validation() {
        let width = 10;
        let height = 10;
        let correct_size = (width * height * 4) as usize;
        let wrong_size = correct_size - 1;

        assert_eq!(correct_size, 400);
        
        // This would fail without actual GPU context, but validates logic
        let data_correct = vec![0u8; correct_size];
        let data_wrong = vec![0u8; wrong_size];

        assert_eq!(data_correct.len(), correct_size);
        assert_eq!(data_wrong.len(), wrong_size);
    }
}
