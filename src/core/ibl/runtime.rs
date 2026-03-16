use super::*;

impl IBLRenderer {
    pub fn override_specular_mip_levels(&mut self, levels: u32) {
        let lv = levels.max(1);
        self.uniforms.max_mip_levels = lv;
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn override_specular_face_size(&mut self, size: u32) {
        self.specular_size_override = Some(size.max(32));
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn override_irradiance_size(&mut self, size: u32) {
        self.irradiance_size_override = Some(size.max(32));
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn override_brdf_size(&mut self, size: u32) {
        let s = size.max(16);
        self.brdf_size_override = Some(s);
        self.uniforms.brdf_size = s;
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn set_quality(&mut self, quality: IBLQuality) {
        if self.quality == quality {
            return;
        }
        self.quality = quality;
        self.base_resolution = quality.base_environment_size();
        self.uniforms.env_size = self.base_resolution;
        self.uniforms.max_mip_levels = quality.specular_mip_levels();
        self.uniforms.brdf_size = quality.brdf_size();
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn quality(&self) -> IBLQuality {
        self.quality
    }

    pub fn textures(
        &self,
    ) -> (
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
    ) {
        (
            self.irradiance_map.as_ref(),
            self.specular_map.as_ref(),
            self.brdf_lut.as_ref(),
        )
    }

    pub fn pbr_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.pbr_layout
    }

    pub fn pbr_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.pbr_bind_group.as_ref()
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.env_sampler
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub(super) fn write_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub(super) fn create_pbr_bind_group(&mut self, device: &wgpu::Device) {
        if let (Some(spec), Some(irr), Some(brdf)) =
            (&self.specular_view, &self.irradiance_view, &self.brdf_view)
        {
            self.pbr_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl.runtime.pbr.bind_group"),
                layout: &self.pbr_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(spec),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(irr),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(brdf),
                    },
                ],
            }));
        }
    }

    pub(super) fn create_default_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let width = 16;
        let height = 8;
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            let v = y as f32 / (height - 1) as f32;
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let color = [0.1 + 0.9 * u, 0.1 + 0.5 * (1.0 - v), 0.3 + 0.7 * v];
                data.extend_from_slice(&color);
            }
        }
        self.load_environment_map(device, queue, &data, width, height)
    }
}
