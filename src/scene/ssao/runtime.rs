impl SsaoResources {
    pub(super) fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        _color: &wgpu::Texture,
        _normal: &wgpu::Texture,
    ) -> Result<(), String> {
        self.width = width;
        self.height = height;
        let (ao_texture, ao_view) = create_ssao_texture(device, width, height, "scene-ssao");
        let (blur_texture, blur_view) =
            create_ssao_texture(device, width, height, "scene-ssao-blur");
        self.ao_texture = ao_texture;
        self.ao_view = ao_view;
        self.blur_texture = blur_texture;
        self.blur_view = blur_view;
        self.update_inv_resolution(queue);
        Ok(())
    }

    pub(super) fn set_params(
        &mut self,
        radius: f32,
        intensity: f32,
        bias: f32,
        queue: &wgpu::Queue,
    ) {
        self.radius = radius.max(0.05);
        self.intensity = intensity.max(0.0);
        self.bias = bias.max(0.0);
        self.update_inv_resolution(queue);
    }

    fn update_inv_resolution(&self, queue: &wgpu::Queue) {
        let default_proj_scale = 0.5 * self.height.max(1) as f32;
        self.write_settings_uniforms(queue, default_proj_scale, SSAO_AO_MIN_DEFAULT);
    }

    fn write_settings_uniforms(&self, queue: &wgpu::Queue, proj_scale: f32, ao_min: f32) {
        let inv_res = [
            1.0 / self.width.max(1) as f32,
            1.0 / self.height.max(1) as f32,
        ];
        let uniform = SsaoSettingsUniform {
            radius: self.radius,
            intensity: self.intensity,
            bias: self.bias,
            num_samples: 16,
            technique: 0,
            frame_index: 0,
            inv_resolution: inv_res,
            proj_scale,
            ao_min,
        };
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&self.blur_settings_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub(super) fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        normal_view: &wgpu::TextureView,
        color_texture: &wgpu::Texture,
        projection: &glam::Mat4,
    ) -> Result<(), String> {
        // ssao.wgsl expects proj_scale = 0.5 * height * P[1][1].
        let proj_scale = compute_ssao_proj_scale(self.height, projection);
        self.write_settings_uniforms(queue, proj_scale, SSAO_AO_MIN_DEFAULT);

        // Group 0: matches cs_ssao bindings in ssao.wgsl.
        let ssao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-bind-group"),
            layout: &self.ssao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    // HZB is optional in this path; reuse depth texture as conservative fallback.
                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.view_buffer.as_entire_binding(),
                },
            ],
        });

        let _blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-blur-bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blur_settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
            ],
        });

        let color_input_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let color_storage_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao-composite-bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    // Use AO output directly as blurred AO input (blur pass disabled)
                    resource: wgpu::BindingResource::TextureView(&self.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = (self.width + 7) / 8;
        let workgroups_y = (self.height + 7) / 8;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &ssao_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssao-composite-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            // Composite pipeline layout expects the bind group at index 0
            pass.set_bind_group(0, &composite_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        Ok(())
    }

    pub(super) fn params(&self) -> (f32, f32, f32) {
        (self.radius, self.intensity, self.bias)
    }

    fn blur_texture(&self) -> &wgpu::Texture {
        &self.blur_texture
    }
}

