use super::*;

pub(super) fn read_output_rgba8(
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, RenderError> {
    let data = read_texture_bytes(texture, width, height, 8)?;
    let padded_bpr = align_copy_bpr(width * 8) as usize;
    let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
    let dst_stride = (width as usize) * 4;

    for y in 0..(height as usize) {
        let row = &data[y * padded_bpr..y * padded_bpr + (width as usize) * 8];
        for x in 0..(width as usize) {
            let o = x * 8;
            let r = f16::from_bits(u16::from_le_bytes([row[o], row[o + 1]])).to_f32();
            let g = f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]])).to_f32();
            let b = f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]])).to_f32();
            let ix = y * dst_stride + x * 4;
            out[ix] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out[ix + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out[ix + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out[ix + 3] = 255;
        }
    }

    Ok(out)
}

pub(super) fn read_aov(
    kind: AovKind,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, RenderError> {
    let format = kind.texture_format();
    let (bytes_per_pixel, convert_rgba16f_to_rgb_f32): (u32, bool) = match format {
        wgpu::TextureFormat::Rgba16Float => (8, true),
        wgpu::TextureFormat::R32Float => (4, false),
        wgpu::TextureFormat::R8Unorm => (1, false),
        _ => return Err(RenderError::Readback("Unsupported AOV format".into())),
    };
    let data = read_texture_bytes(texture, width, height, bytes_per_pixel)?;
    let padded_bpr = align_copy_bpr(width * bytes_per_pixel) as usize;

    if convert_rgba16f_to_rgb_f32 {
        let mut out = vec![0u8; (width as usize) * (height as usize) * 12];
        let dst_stride = (width as usize) * 12;
        for y in 0..(height as usize) {
            let row = &data[y * padded_bpr..y * padded_bpr + (width as usize) * 8];
            for x in 0..(width as usize) {
                let o = x * 8;
                let r = f16::from_bits(u16::from_le_bytes([row[o], row[o + 1]])).to_f32();
                let g = f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]])).to_f32();
                let b = f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]])).to_f32();
                let ix = y * dst_stride + x * 12;
                out[ix..ix + 4].copy_from_slice(&r.to_le_bytes());
                out[ix + 4..ix + 8].copy_from_slice(&g.to_le_bytes());
                out[ix + 8..ix + 12].copy_from_slice(&b.to_le_bytes());
            }
        }
        Ok(out)
    } else if bytes_per_pixel == 4 {
        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
        let dst_stride = (width as usize) * 4;
        for y in 0..(height as usize) {
            let src = &data[y * padded_bpr..y * padded_bpr + (width as usize) * 4];
            let dst = &mut out[y * dst_stride..y * dst_stride + (width as usize) * 4];
            dst.copy_from_slice(src);
        }
        Ok(out)
    } else {
        let mut out = vec![0u8; (width as usize) * (height as usize)];
        let dst_stride = width as usize;
        for y in 0..(height as usize) {
            let src = &data[y * padded_bpr..y * padded_bpr + width as usize];
            let dst = &mut out[y * dst_stride..y * dst_stride + width as usize];
            dst.copy_from_slice(src);
        }
        Ok(out)
    }
}

fn read_texture_bytes(
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, RenderError> {
    let g = ctx();
    let row_bytes = width * bytes_per_pixel;
    let padded_bpr = align_copy_bpr(row_bytes);
    let read_buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pt-readback"),
        size: (padded_bpr as u64) * (height as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = g
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pt-readback-encoder"),
        });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &read_buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    g.queue.submit([enc.finish()]);
    g.device.poll(wgpu::Maintain::Wait);

    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    g.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| RenderError::Readback("map_async channel closed".into()))?
        .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;

    let data = slice.get_mapped_range().to_vec();
    read_buf.unmap();
    Ok(data)
}
