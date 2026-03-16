use super::*;

impl IBLRenderer {
    pub(super) fn upload_cubemap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base_size: u32,
        mip_levels: u32,
        bytes: &[u8],
    ) -> Result<(wgpu::Texture, wgpu::TextureView), String> {
        let expected_len = cubemap_data_len(base_size, mip_levels, 8);
        if bytes.len() != expected_len {
            return Err("IBL cache cubemap payload size mismatch".into());
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.cache.cubemap"),
            size: wgpu::Extent3d {
                width: base_size,
                height: base_size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut offset = 0usize;
        for mip in 0..mip_levels {
            let mip_size = (base_size >> mip).max(1);
            let stride = (mip_size * mip_size * 8) as usize;
            for face in 0..CUBE_FACE_COUNT {
                let slice = &bytes[offset..offset + stride];
                offset += stride;
                let (padded, bpr) = pad_image_rows(slice, mip_size, mip_size, 8);
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &padded,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bpr),
                        rows_per_image: Some(mip_size),
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.cache.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        Ok((texture, view))
    }

    pub(super) fn upload_2d(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<(wgpu::Texture, wgpu::TextureView), String> {
        let expected = (width * height * 8) as usize;
        if bytes.len() != expected {
            return Err("IBL cache BRDF payload size mismatch".into());
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.cache.brdf"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (padded, bpr) = pad_image_rows(bytes, width, height, 8);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.cache.brdf.view"),
            ..Default::default()
        });
        Ok((texture, view))
    }

    pub(super) fn download_cubemap(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        base_size: u32,
        mip_levels: u32,
    ) -> Result<Vec<u8>, String> {
        let bytes_per_pixel = 8usize;
        let total_len = cubemap_data_len(base_size, mip_levels, bytes_per_pixel);
        let mut result = Vec::with_capacity(total_len);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.download.cubemap.encoder"),
        });

        let mut buffer_slices = Vec::new();

        for mip in 0..mip_levels {
            let mip_size = (base_size >> mip).max(1);
            let padded_row = align_to(bytes_per_pixel * mip_size as usize, COPY_ALIGNMENT);
            let padded_face = padded_row * mip_size as usize;
            let padded_mip = padded_face * CUBE_FACE_COUNT as usize;

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ibl.download.cubemap.buffer.mip{mip}")),
                size: padded_mip as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            for face in 0..CUBE_FACE_COUNT {
                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: (face as usize * padded_face) as u64,
                            bytes_per_row: Some(padded_row as u32),
                            rows_per_image: Some(mip_size),
                        },
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );
            }

            buffer_slices.push((buffer, padded_row, mip_size));
        }

        queue.submit(Some(encoder.finish()));

        for (buffer, _, _) in buffer_slices.iter() {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        }
        device.poll(wgpu::Maintain::Wait);

        for (buffer, padded_row, mip_size) in buffer_slices.iter() {
            let data = buffer.slice(..).get_mapped_range();
            for face in 0..CUBE_FACE_COUNT as usize {
                let face_offset = face * (*padded_row as usize * *mip_size as usize);
                let face_slice =
                    &data[face_offset..face_offset + (*padded_row as usize * *mip_size as usize)];
                let tight = strip_image_padding(face_slice, *mip_size, *mip_size, 8);
                result.extend_from_slice(&tight);
            }
            drop(data);
            buffer.unmap();
        }

        Ok(result)
    }

    pub(super) fn download_2d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let bytes_per_pixel = 8usize;
        let padded_row = align_to(bytes_per_pixel * width as usize, COPY_ALIGNMENT);
        let padded_total = padded_row * height as usize;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl.download.brdf.buffer"),
            size: padded_total as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.download.brdf.encoder"),
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
                    bytes_per_row: Some(padded_row as u32),
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
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::Wait);

        let data = buffer.slice(..).get_mapped_range();
        let tight = strip_image_padding(&data, width, height, bytes_per_pixel);
        buffer.unmap();
        Ok(tight)
    }
}
