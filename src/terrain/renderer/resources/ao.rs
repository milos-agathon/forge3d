use super::*;

impl TerrainScene {
    pub fn light_debug_info(&self) -> PyResult<String> {
        let light_buffer = self
            .light_buffer
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock light buffer: {}", e)))?;
        Ok(light_buffer.debug_info())
    }

    pub fn set_ao_debug_view(&mut self, view: Option<wgpu::TextureView>) {
        self.ao_debug_view = view;
    }

    pub fn compute_coarse_ao_from_heightmap(
        &mut self,
        width: u32,
        height: u32,
        heightmap_data: &[f32],
    ) -> Result<()> {
        let mut ao_data = vec![1.0f32; (width * height) as usize];
        let sample_radius = 8i32;
        let height_scale = 10.0f32;

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let idx = (y as u32 * width + x as u32) as usize;
                let center_h = heightmap_data[idx];

                let mut occlusion = 0.0f32;
                let mut sample_count = 0;

                for dy in -sample_radius..=sample_radius {
                    for dx in -sample_radius..=sample_radius {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let nidx = (ny as u32 * width + nx as u32) as usize;
                            let neighbor_h = heightmap_data[nidx];
                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let h_diff = (neighbor_h - center_h) * height_scale;
                            if h_diff > 0.0 {
                                let angle = (h_diff / dist).atan();
                                occlusion += (angle / std::f32::consts::FRAC_PI_2).min(1.0);
                            }
                            sample_count += 1;
                        }
                    }
                }

                if sample_count > 0 {
                    let avg_occlusion = occlusion / sample_count as f32;
                    ao_data[idx] = (1.0 - avg_occlusion.min(0.9)).max(0.01);
                }
            }
        }

        let coarse_ao_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.coarse_ao"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &coarse_ao_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&ao_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let coarse_ao_view = coarse_ao_texture.create_view(&wgpu::TextureViewDescriptor::default());
        log::info!(
            target: "terrain.ao",
            "P5: Computed coarse horizon AO from heightmap ({}x{})",
            width, height
        );

        self.coarse_ao_texture = Some(coarse_ao_texture);
        self.coarse_ao_view = Some(coarse_ao_view);
        Ok(())
    }

    pub fn coarse_ao_view(&self) -> Option<&wgpu::TextureView> {
        self.coarse_ao_view.as_ref()
    }
}
