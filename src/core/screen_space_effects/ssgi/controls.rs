use super::*;

impl SsgiRenderer {
    pub fn set_seed(&mut self, queue: &Queue, seed: u32) {
        self.frame_index = seed;
        self.settings.frame_index = self.frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_settings(&mut self, queue: &Queue, settings: SsgiSettings) {
        self.settings = settings;
        self.settings.use_half_res = if self.half_res { 1 } else { 0 };
        self.settings.frame_index = self.frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn update_camera(&mut self, queue: &Queue, camera: &CameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    pub fn get_settings(&self) -> SsgiSettings {
        self.settings
    }

    pub fn set_environment(&mut self, _env_view: &TextureView, _env_sampler: &Sampler) {
        // Deprecated in favor of set_environment_texture; kept for API-compat.
    }

    pub fn set_environment_texture(&mut self, device: &Device, env_texture: &Texture) {
        // Create a cube view from the provided texture
        let view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("gi.env.cube.view"),
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        self.env_view = view;
        // Create a linear sampler suitable for sampling the environment
        self.env_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("gi.env.cube.sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
    }

    pub fn advance_frame(&mut self, queue: &Queue) {
        self.frame_index = self.frame_index.wrapping_add(1);
        self.settings.frame_index = self.frame_index;
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
    }

    pub fn set_half_res(&mut self, device: &Device, queue: &Queue, on: bool) -> RenderResult<()> {
        self.half_res = on;
        let (w, h) = if on {
            (self.width.max(2) / 2, self.height.max(2) / 2)
        } else {
            (self.width, self.height)
        };

        // Recreate output and temporal textures at new resolution
        self.ssgi_hit = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_hit"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.ssgi_hit_view = self.ssgi_hit.create_view(&TextureViewDescriptor::default());
        self.ssgi_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_texture"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.ssgi_view = self
            .ssgi_texture
            .create_view(&TextureViewDescriptor::default());

        self.ssgi_history = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_history"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.ssgi_history_view = self
            .ssgi_history
            .create_view(&TextureViewDescriptor::default());

        self.ssgi_filtered = device.create_texture(&TextureDescriptor {
            label: Some("ssgi_filtered"),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.ssgi_filtered_view = self
            .ssgi_filtered
            .create_view(&TextureViewDescriptor::default());

        // Update inv_resolution in settings
        self.settings.inv_resolution = [1.0 / w as f32, 1.0 / h as f32];
        self.settings.use_half_res = if on { 1 } else { 0 };
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&self.settings));
        Ok(())
    }
}
