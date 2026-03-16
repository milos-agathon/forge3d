use super::*;

impl IBLRenderer {
    pub fn set_base_resolution(&mut self, base_resolution: u32) {
        let safe = base_resolution.max(16);
        self.base_resolution = safe;
        self.uniforms.env_size = safe;
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn load_environment_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hdr_data: &[f32],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("HDR image dimensions must be positive".into());
        }
        let pixel_count = (width as usize) * (height as usize);
        if hdr_data.len() != pixel_count * 3 && hdr_data.len() != pixel_count * 4 {
            return Err(format!(
                "HDR data length {} does not match width*height*{{3|4}}",
                hdr_data.len()
            ));
        }
        let channel_count = if hdr_data.len() == pixel_count * 4 {
            4
        } else {
            3
        };

        // Clamp dimensions to GPU limits
        let max_dim = device.limits().max_texture_dimension_2d;
        let (target_width, target_height) = if width > max_dim || height > max_dim {
            let scale = (max_dim as f32 / width.max(height) as f32).min(1.0);
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;
            if new_width != width || new_height != height {
                warn!(
                    "HDR image {}x{} exceeds GPU limit {}, resizing to {}x{}",
                    width, height, max_dim, new_width, new_height
                );
            }
            // Ensure both dimensions are within limits
            (
                new_width.max(1).min(max_dim),
                new_height.max(1).min(max_dim),
            )
        } else {
            // Even if original dimensions are within limits, ensure they don't exceed
            (width.min(max_dim), height.min(max_dim))
        };

        // Resize HDR data if needed
        let (resized_data, resized_width, resized_height) =
            if target_width != width || target_height != height {
                let resized = resize_hdr_data(
                    hdr_data,
                    width,
                    height,
                    target_width,
                    target_height,
                    channel_count,
                );
                (resized, target_width, target_height)
            } else {
                (hdr_data.to_vec(), width, height)
            };

        let resized_pixel_count = (resized_width as usize) * (resized_height as usize);
        let mut texels = Vec::with_capacity(resized_pixel_count * 4);
        for idx in 0..resized_pixel_count {
            let src = idx * channel_count;
            texels.push(f16::from_f32(resized_data[src]).to_bits());
            texels.push(f16::from_f32(resized_data[src + 1]).to_bits());
            texels.push(f16::from_f32(resized_data[src + 2]).to_bits());
            let alpha = if channel_count == 4 {
                resized_data[src + 3]
            } else {
                1.0
            };
            texels.push(f16::from_f32(alpha).to_bits());
        }

        // Final safety check: ensure dimensions never exceed device limits
        // This should not be necessary if resize logic works correctly, but serves as a safeguard
        let max_dim_final = device.limits().max_texture_dimension_2d;
        let final_width = resized_width.min(max_dim_final).max(1);
        let final_height = resized_height.min(max_dim_final).max(1);

        // If we need to clamp further, we need to resize the data again
        let (padded, bpr) = if final_width != resized_width || final_height != resized_height {
            warn!(
                "CRITICAL: Resized dimensions {}x{} still exceed device limit {}! Clamping to {}x{}.",
                resized_width, resized_height, max_dim_final, final_width, final_height
            );
            // Resize the data to the final clamped dimensions
            let clamped_data = resize_hdr_data(
                &resized_data,
                resized_width,
                resized_height,
                final_width,
                final_height,
                4,
            );
            // Convert to f16 and pad
            let clamped_pixel_count = (final_width as usize) * (final_height as usize);
            let mut clamped_texels = Vec::with_capacity(clamped_pixel_count * 4);
            for idx in 0..clamped_pixel_count {
                let src = idx * 4;
                clamped_texels.push(f16::from_f32(clamped_data[src]).to_bits());
                clamped_texels.push(f16::from_f32(clamped_data[src + 1]).to_bits());
                clamped_texels.push(f16::from_f32(clamped_data[src + 2]).to_bits());
                clamped_texels.push(f16::from_f32(clamped_data[src + 3]).to_bits());
            }
            let clamped_raw_bytes = bytemuck::cast_slice(&clamped_texels);
            pad_image_rows(clamped_raw_bytes, final_width, final_height, 8)
        } else {
            let raw_bytes = bytemuck::cast_slice(&texels);
            pad_image_rows(raw_bytes, resized_width, resized_height, 8)
        };

        let equirect = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.environment.equirect"),
            size: wgpu::Extent3d {
                width: final_width,
                height: final_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &equirect,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(final_height),
            },
            wgpu::Extent3d {
                width: final_width,
                height: final_height,
                depth_or_array_layers: 1,
            },
        );

        let env_size = self.base_resolution;
        let cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.environment.cubemap"),
            size: wgpu::Extent3d {
                width: env_size,
                height: env_size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let cubemap_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.environment.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        self.uniforms.src_width = final_width;
        self.uniforms.src_height = final_height;
        self.uniforms.env_size = env_size;
        self.uniforms.face_count = CUBE_FACE_COUNT;
        self.uniforms.mip_level = 0;
        self.uniforms.roughness = 0.0;
        self.write_uniforms(queue);

        let storage_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.environment.cubemap.storage"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        let equirect_view = equirect.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.environment.equirect.view"),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl.precompute.equirect.bind_group"),
            layout: &self.equirect_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&equirect_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.equirect_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.precompute.encoder.equirect"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl.precompute.pass.equirect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.equirect_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = 8u32;
            let groups_x = (env_size + work - 1) / work;
            let groups_y = (env_size + work - 1) / work;
            pass.dispatch_workgroups(groups_x, groups_y, CUBE_FACE_COUNT);
        }

        queue.submit(Some(encoder.finish()));

        self.environment_equirect = Some(equirect);
        self.environment_cubemap = Some(cubemap);
        self.environment_view = Some(cubemap_view);
        self.invalidate_cache_key();
        if let Some(ref mut cfg) = self.cache {
            cfg.hdr_width = width;
            cfg.hdr_height = height;
        }
        self.is_initialized = false;
        Ok(())
    }

    pub fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        if self.environment_cubemap.is_none() {
            self.create_default_environment(device, queue)?;
        }

        if self.try_load_cache(device, queue)? {
            self.create_pbr_bind_group(device);
            self.is_initialized = true;
            return Ok(());
        }

        // Cache miss - building IBL resources
        info!(
            "IBL cache miss: building irradiance_{}.cube, prefilter_mips.cube, brdf_{}.png",
            self.quality.irradiance_size(),
            self.quality.brdf_size()
        );
        self.generate_irradiance_map(device, queue)?;
        self.generate_specular_map(device, queue)?;
        self.generate_brdf_lut(device, queue)?;

        self.create_pbr_bind_group(device);
        self.is_initialized = true;

        if let Err(err) = self.write_cache(device, queue) {
            warn!("Failed to write IBL cache: {err}");
        }

        Ok(())
    }
}
