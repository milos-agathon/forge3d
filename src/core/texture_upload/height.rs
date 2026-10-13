use crate::core::error::{RenderError, RenderResult};
use crate::core::memory_tracker::MEMORY_BUDGET_LIMIT;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::core::texture_layout::{resident_2d_size, TransferLayout};

pub fn create_r32f_height_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    width: u32,
    height: u32,
) -> RenderResult<(TrackedTexture, wgpu::TextureView)> {
    let size = (width * height) as usize;
    if data.len() != size {
        return Err(RenderError::upload(format!(
            "Data length {} != {} (width*height)",
            data.len(),
            size
        )));
    }

    let texture_bytes = resident_2d_size(width, height, wgpu::TextureFormat::R32Float);
    let limit_bytes = MEMORY_BUDGET_LIMIT;
    if texture_bytes > limit_bytes {
        return Err(RenderError::upload(format!(
            "Height texture {}x{} ({} bytes) exceeds 512 MiB limit",
            width, height, texture_bytes
        )));
    }

    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;

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
    let layout = TransferLayout::buffer_copy(
        wgpu::TextureFormat::R32Float,
        wgpu::TextureAspect::All,
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    )
    .map_err(RenderError::upload)?;
    let row_bytes = layout.row_bytes;
    let padded_bpr = layout.bytes_per_row;

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
            layout.image_data_layout(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        let mut padded_data = vec![0u8; layout.transfer_bytes as usize];
        let input_data = bytemuck::cast_slice::<f32, u8>(data);

        for y in 0..height {
            let src_offset = y as usize * row_bytes as usize;
            let dst_offset = y as usize * padded_bpr as usize;
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
            layout.image_data_layout(),
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
) -> RenderResult<(TrackedTexture, wgpu::TextureView)> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(RenderError::upload(format!(
            "data length {} != {} (width*height)",
            data.len(),
            expected
        )));
    }

    let layout = TransferLayout::buffer_copy(
        wgpu::TextureFormat::R32Float,
        wgpu::TextureAspect::All,
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    )
    .map_err(RenderError::upload)?;
    let bytes_per_row = layout.row_bytes as usize;
    let padded_bpr = layout.bytes_per_row as usize;
    let mut staged = vec![0u8; layout.transfer_bytes as usize];
    let src = bytemuck::cast_slice::<f32, u8>(data);

    for row in 0..height as usize {
        let src_off = row * bytes_per_row;
        let dst_off = row * padded_bpr;
        staged[dst_off..dst_off + bytes_per_row]
            .copy_from_slice(&src[src_off..src_off + bytes_per_row]);
    }

    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )?;

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &staged,
        layout.image_data_layout(),
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}
