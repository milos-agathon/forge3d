use super::*;
use numpy::PyArray1;

fn wait_for_buffer_map(
    device: &wgpu::Device,
    slice: &wgpu::BufferSlice<'_>,
    cancel_message: &'static str,
) -> PyResult<()> {
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    let recv = pollster::block_on(receiver.receive())
        .ok_or_else(|| PyRuntimeError::new_err(cancel_message))?;
    if let Err(error) = recv {
        return Err(PyRuntimeError::new_err(format!(
            "map_async error: {:?}",
            error
        )));
    }
    Ok(())
}

#[cfg(not(feature = "weighted-oit"))]
pub(super) fn weighted_oit_not_enabled_err() -> PyErr {
    PyRuntimeError::new_err("Weighted OIT feature not enabled. Build with --features weighted-oit")
}

pub(super) fn read_rgba_texture_to_vec(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    copy_label: &'static str,
    read_label: &'static str,
    cancel_message: &'static str,
) -> PyResult<Vec<u8>> {
    let bytes_per_row = (width * 4 + 255) / 256 * 256;
    let size = (bytes_per_row * height) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(read_label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(copy_label),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buffer.slice(..);
    wait_for_buffer_map(device, &slice, cancel_message)?;
    let data = slice.get_mapped_range();
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for row in 0..height as usize {
        let src = &data[(row as u32 * bytes_per_row) as usize..][..(width * 4) as usize];
        let dst = &mut rgba[row * width as usize * 4..][..width as usize * 4];
        dst.copy_from_slice(src);
    }
    drop(data);
    buffer.unmap();
    Ok(rgba)
}

pub(super) fn read_rgba_texture_to_py(
    py: Python<'_>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    copy_label: &'static str,
    read_label: &'static str,
    cancel_message: &'static str,
) -> PyResult<Py<PyAny>> {
    let rgba = read_rgba_texture_to_vec(
        device,
        queue,
        texture,
        width,
        height,
        copy_label,
        read_label,
        cancel_message,
    )?;
    let array =
        PyArray1::<u8>::from_vec_bound(py, rgba).reshape([height as usize, width as usize, 4])?;
    Ok(array.into_py(py))
}

pub(super) fn read_u32_texture_to_py(
    py: Python<'_>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    copy_label: &'static str,
    read_label: &'static str,
    cancel_message: &'static str,
) -> PyResult<Py<PyAny>> {
    let bytes_per_row = (width * 4 + 255) / 256 * 256;
    let size = (bytes_per_row * height) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(read_label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(copy_label),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buffer.slice(..);
    wait_for_buffer_map(device, &slice, cancel_message)?;
    let data = slice.get_mapped_range();
    let mut ids = vec![0u32; (width * height) as usize];
    for row in 0..height as usize {
        let src = &data[(row as u32 * bytes_per_row) as usize..][..(width * 4) as usize];
        let row_ids = bytemuck::cast_slice::<u8, u32>(src);
        let dst = &mut ids[row * width as usize..][..width as usize];
        dst.copy_from_slice(row_ids);
    }
    drop(data);
    buffer.unmap();

    let array =
        PyArray1::<u32>::from_vec_bound(py, ids).reshape([height as usize, width as usize])?;
    Ok(array.into_py(py))
}

pub(super) fn read_single_u32_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    x: u32,
    y: u32,
    copy_label: &'static str,
    read_label: &'static str,
    cancel_message: &'static str,
) -> PyResult<u32> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(read_label),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(copy_label),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buffer.slice(..);
    wait_for_buffer_map(device, &slice, cancel_message)?;
    let data = slice.get_mapped_range();
    let pick_id = bytemuck::from_bytes::<u32>(&data[..4]).to_owned();
    drop(data);
    buffer.unmap();
    Ok(pick_id)
}
