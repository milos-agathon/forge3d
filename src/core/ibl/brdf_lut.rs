use super::*;

impl IBLRenderer {
    pub fn generate_brdf_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let size = self
            .brdf_size_override
            .unwrap_or(self.quality.brdf_size())
            .max(16);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.brdf.lut"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.brdf.lut.view"),
            ..Default::default()
        });

        self.uniforms.brdf_size = size;
        // Use fixed sample count for deterministic BRDF LUT (spec requirement: no random seeds)
        self.uniforms.sample_count = 1024;
        self.write_uniforms(queue);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl.brdf.bind_group"),
            layout: &self.brdf_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.brdf.encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl.brdf.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brdf_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = 8;
            let groups = (size + work - 1) / work;
            pass.dispatch_workgroups(groups, groups, 1);
        }

        queue.submit(Some(encoder.finish()));

        self.brdf_lut = Some(texture);
        self.brdf_view = Some(view);
        Ok(())
    }
}
