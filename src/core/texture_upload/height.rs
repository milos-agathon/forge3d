use super::hdr::align_copy_bytes_per_row;
use crate::core::error::{RenderError, RenderResult};

pub fn create_r32f_height_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<(wgpu::Texture, wgpu::TextureView)> {
    let size = (width * height) as usize;
    if data.len() != size {
        return Err(RenderError::upload(format!(
            "Data length {} != {} (width*height)",
            data.len(),
            size
        )));
    }

    let texture_bytes = (width as u64) * (height as u64) * 4;
    let limit_bytes = 512 * 1024 * 1024;
    if texture_bytes > limit_bytes {
        return Err(RenderError::upload(format!(
            "Height texture {}x{} ({} bytes) exceeds 512 MiB limit",
            width, height, texture_bytes
        )));
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("height_r32f"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    upload_r32f_data(queue, &texture, data, width, height)?;

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}

fn upload_r32f_data(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<()> {
    let row_bytes = width * 4;
    let padded_bpr = align_copy_bytes_per_row(row_bytes);

    if padded_bpr == row_bytes {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(row_bytes).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        let mut padded_data = vec![0u8; (padded_bpr * height) as usize];
        let input_data = bytemuck::cast_slice::<f32, u8>(data);

        for y in 0..height {
            let src_offset = (y * row_bytes) as usize;
            let dst_offset = (y * padded_bpr) as usize;
            let src_end = src_offset + row_bytes as usize;
            let dst_end = dst_offset + row_bytes as usize;
            padded_data[dst_offset..dst_end].copy_from_slice(&input_data[src_offset..src_end]);
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
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
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

pub fn create_r32f_height_texture_padded(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<(wgpu::Texture, wgpu::TextureView)> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(RenderError::upload(format!(
            "data length {} != {} (width*height)",
            data.len(),
            expected
        )));
    }

    let bytes_per_row = (width * 4) as usize;
    let padded_bpr = ((bytes_per_row + 255) / 256) * 256;
    let mut staged = vec![0u8; padded_bpr * (height as usize)];
    let src = bytemuck::cast_slice::<f32, u8>(data);

    for row in 0..height as usize {
        let src_off = row * bytes_per_row;
        let dst_off = row * padded_bpr;
        staged[dst_off..dst_off + bytes_per_row]
            .copy_from_slice(&src[src_off..src_off + bytes_per_row]);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("height_r32f"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &staged,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr as u32).unwrap().into()),
            rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}
