use super::*;

impl VirtualTexture {
    /// Update virtual texture for current camera
    pub fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        camera: &CameraInfo,
    ) -> Result<(), String> {
        self.requested_tiles.clear();
        self.stats.cache_hits = 0;
        self.stats.cache_misses = 0;
        self.stats.tiles_streamed = 0;

        let visible_tiles = self.calculate_visible_tiles(camera);

        for tile_id in visible_tiles {
            self.requested_tiles.insert(tile_id);

            if self.tile_cache.is_resident(&tile_id) {
                self.stats.cache_hits += 1;
                self.tile_cache.access_tile(&tile_id);
            } else {
                self.stats.cache_misses += 1;
                self.request_tile_load(device, queue, tile_id)?;
            }
        }

        if let Some(ref mut feedback_buffer) = self.feedback_buffer {
            let feedback_tiles = feedback_buffer.read_feedback(device, queue)?;
            for tile_id in feedback_tiles {
                if !self.tile_cache.is_resident(&tile_id) {
                    self.request_tile_load(device, queue, tile_id)?;
                }
            }
        }

        self.update_page_table(queue)?;
        self.stats.resident_pages = self.tile_cache.resident_count() as u32;
        self.publish_resident_metrics();
        self.stats.memory_usage = self.calculate_memory_usage();

        Ok(())
    }

    fn calculate_visible_tiles(&self, camera: &CameraInfo) -> Vec<TileId> {
        let mut visible_tiles = Vec::new();
        let pages_x = (self.config.width + self.config.tile_size - 1) / self.config.tile_size;
        let pages_y = (self.config.height + self.config.tile_size - 1) / self.config.tile_size;

        let view_distance = 1000.0;
        let visible_size = (camera.fov_degrees.to_radians().tan() * view_distance) as u32;
        let center_x = ((camera.position[0] / self.config.width as f32) * pages_x as f32) as u32;
        let center_y = ((camera.position[2] / self.config.height as f32) * pages_y as f32) as u32;
        let visible_radius = (visible_size / self.config.tile_size / 2).max(1);

        for y in center_y.saturating_sub(visible_radius)
            ..=center_y.saturating_add(visible_radius).min(pages_y - 1)
        {
            for x in center_x.saturating_sub(visible_radius)
                ..=center_x.saturating_add(visible_radius).min(pages_x - 1)
            {
                let distance = ((x as f32 - center_x as f32).powi(2)
                    + (y as f32 - center_y as f32).powi(2))
                .sqrt();
                let mip_level = (((distance / visible_radius as f32)
                    * self.config.max_mip_levels as f32) as u32)
                    .min(self.config.max_mip_levels - 1);

                visible_tiles.push(TileId { x, y, mip_level });
            }
        }

        visible_tiles
    }

    fn request_tile_load(
        &mut self,
        device: &Device,
        queue: &Queue,
        tile_id: TileId,
    ) -> Result<(), String> {
        if let Some(atlas_slot) = self.tile_cache.allocate_tile(tile_id) {
            let tile_data = self.load_tile_data(tile_id)?;
            self.upload_tile_to_atlas(device, queue, &tile_data, atlas_slot)?;

            let page_index = self.tile_id_to_page_index(tile_id);
            if let Some(entry) = self.page_table_data.get_mut(page_index) {
                entry.atlas_u = atlas_slot.atlas_u;
                entry.atlas_v = atlas_slot.atlas_v;
                entry.is_resident = 1;
                entry.mip_bias = atlas_slot.mip_bias;
            }

            self.stats.tiles_streamed += 1;
        }

        Ok(())
    }

    fn update_page_table(&self, queue: &Queue) -> Result<(), String> {
        let gpu_data = bytemuck::cast_slice(&self.page_table_data);
        let pages_x = (self.config.width + self.config.tile_size - 1) / self.config.tile_size;
        let pages_y = (self.config.height + self.config.tile_size - 1) / self.config.tile_size;

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.page_table,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            gpu_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(pages_x * 16),
                rows_per_image: Some(pages_y),
            },
            wgpu::Extent3d {
                width: pages_x,
                height: pages_y,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Get current virtual texture statistics
    pub fn stats(&self) -> &VirtualTextureStats {
        &self.stats
    }

    /// Get atlas texture for rendering
    pub fn atlas_texture(&self) -> &Texture {
        &self.atlas_texture
    }

    /// Get page table texture for rendering
    pub fn page_table_texture(&self) -> &Texture {
        &self.page_table
    }

    /// Create bind group for virtual texture rendering
    pub fn create_bind_group(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VirtualTexture_BindGroup"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &self.atlas_texture.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self.page_table.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}
