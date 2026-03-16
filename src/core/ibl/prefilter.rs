use super::*;

impl IBLRenderer {
    pub fn generate_specular_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let env_view = self
            .environment_view
            .as_ref()
            .ok_or("Environment cube not available")?;

        let size = self
            .specular_size_override
            .unwrap_or(self.quality.specular_size());
        let mip_levels = self
            .uniforms
            .max_mip_levels
            .min(self.quality.specular_mip_levels())
            .max(1);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.specular.cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.specular.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        self.uniforms.max_mip_levels = mip_levels;
        self.write_uniforms(queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.specular.encoder"),
        });

        for mip in 0..mip_levels {
            let mip_size = (size >> mip).max(1);
            self.uniforms.env_size = mip_size;
            self.uniforms.mip_level = mip;
            // Spec: mip0=1024, mip1=512, mip2=256, ... min 64
            let sample_count = (1024u32 >> mip).max(64);
            self.uniforms.sample_count = sample_count;
            // Roughness mapping: mip = roughness^2 * (mipCount-1)
            // For prefilter, we use: roughness = sqrt(mip / (mipCount-1))
            self.uniforms.roughness = if mip_levels > 1 {
                (mip as f32 / ((mip_levels - 1) as f32)).sqrt()
            } else {
                0.0
            };
            self.write_uniforms(queue);

            let storage_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("ibl.specular.storage.mip{mip}")),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(CUBE_FACE_COUNT),
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ibl.specular.bind_group.mip{mip}")),
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

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("ibl.specular.pass.mip{mip}")),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.specular_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let work = 8;
                let groups = (mip_size + work - 1) / work;
                pass.dispatch_workgroups(groups, groups, CUBE_FACE_COUNT);
            }
        }

        queue.submit(Some(encoder.finish()));

        self.specular_map = Some(texture);
        self.specular_view = Some(cube_view);
        Ok(())
    }
}
