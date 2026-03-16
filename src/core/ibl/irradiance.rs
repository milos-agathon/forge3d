use super::*;

impl IBLRenderer {
    pub fn generate_irradiance_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let env_view = self
            .environment_view
            .as_ref()
            .ok_or("Environment cube not available")?;

        let size = self
            .irradiance_size_override
            .unwrap_or(self.quality.irradiance_size());
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.irradiance.cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.irradiance.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        let storage_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.irradiance.storage.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        self.uniforms.env_size = size;
        // Irradiance uses fixed 128 samples (handled in shader, but set for consistency)
        self.uniforms.sample_count = 128;
        self.uniforms.mip_level = 0;
        self.uniforms.roughness = 0.0;
        self.write_uniforms(queue);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl.irradiance.bind_group"),
            layout: &self.convolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.irradiance.encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl.irradiance.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.irradiance_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = 8;
            let groups = (size + work - 1) / work;
            pass.dispatch_workgroups(groups, groups, CUBE_FACE_COUNT);
        }

        queue.submit(Some(encoder.finish()));

        self.irradiance_map = Some(texture);
        self.irradiance_view = Some(cube_view);
        Ok(())
    }
}
