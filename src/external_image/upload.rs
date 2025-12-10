//! GPU texture creation and upload utilities.

use super::types::ImageImportConfig;
use crate::core::error::RenderResult;

/// Create a texture suitable for image import.
pub fn create_texture_for_import(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    config: &ImageImportConfig,
) -> RenderResult<wgpu::Texture> {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let mip_level_count = if config.generate_mipmaps {
        size.max_mips(wgpu::TextureDimension::D2)
    } else {
        1
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: config.label.as_deref(),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.target_format,
        usage: config.usage,
        view_formats: &[],
    });

    Ok(texture)
}

/// Upload RGBA data to texture with proper row padding.
pub fn upload_rgba_data_to_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rgba_data: &[u8],
    width: u32,
    height: u32,
) -> RenderResult<()> {
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;

    if padded_bytes_per_row == unpadded_bytes_per_row {
        // No padding needed - direct upload
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(unpadded_bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        // Need padding - create padded buffer
        let padded_size = (padded_bytes_per_row * height) as usize;
        let mut padded_data = vec![0u8; padded_size];

        for y in 0..height as usize {
            let src_start = y * unpadded_bytes_per_row as usize;
            let src_end = src_start + unpadded_bytes_per_row as usize;
            let dst_start = y * padded_bytes_per_row as usize;
            let dst_end = dst_start + unpadded_bytes_per_row as usize;

            padded_data[dst_start..dst_end].copy_from_slice(&rgba_data[src_start..src_end]);
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(())
}
