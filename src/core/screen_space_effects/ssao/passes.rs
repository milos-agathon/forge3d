use super::*;

impl SsaoRenderer {
    pub fn encode_ssao(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        hzb_view: &TextureView,
    ) -> RenderResult<()> {
        let t0 = Instant::now();
        let mut settings_shadow = self.settings;
        settings_shadow.frame_index = self.settings.frame_index.wrapping_add(1);
        settings_shadow.inv_resolution = [1.0 / self.width as f32, 1.0 / self.height as f32];
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao.settings.staging"),
            contents: bytemuck::bytes_of(&settings_shadow),
            usage: BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(
            &staging,
            0,
            &self.settings_buffer,
            0,
            std::mem::size_of::<SsaoSettings>() as u64,
        );
        self.settings = settings_shadow;

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_bind_group"),
            layout: &self.ssao_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(hzb_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.noise_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.noise_sampler),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&self.ssao_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_pass"),
            timestamp_writes: None,
        });
        let kernel = if self.settings.technique == 1 {
            &self.gtao_pipeline
        } else {
            &self.ssao_pipeline
        };
        pass.set_pipeline(kernel);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        drop(pass);

        self.last_ao_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    pub fn encode_blur(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
    ) -> RenderResult<()> {
        let t0 = Instant::now();
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;

        let blur_bg_h = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_blur_bg_h"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssao_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssao_tmp_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.blur_settings.as_entire_binding(),
                },
            ],
        });
        let mut blur_h = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_blur_h"),
            timestamp_writes: None,
        });
        blur_h.set_pipeline(&self.blur_h_pipeline);
        blur_h.set_bind_group(0, &blur_bg_h, &[]);
        blur_h.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(blur_h);

        let blur_bg_v = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_blur_bg_v"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssao_tmp_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssao_blurred_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.blur_settings.as_entire_binding(),
                },
            ],
        });
        let mut blur_v = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_blur_v"),
            timestamp_writes: None,
        });
        blur_v.set_pipeline(&self.blur_v_pipeline);
        blur_v.set_bind_group(0, &blur_bg_v, &[]);
        blur_v.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        drop(blur_v);

        self.last_blur_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    pub fn encode_composite(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
    ) -> RenderResult<()> {
        let comp_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssao_composite_bind_group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssao_composited_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssao_resolved_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.comp_uniform.as_entire_binding(),
                },
            ],
        });

        let mut comp_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssao_composite_pass"),
            timestamp_writes: None,
        });
        comp_pass.set_pipeline(&self.composite_pipeline);
        comp_pass.set_bind_group(0, &comp_bg, &[]);
        comp_pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        Ok(())
    }

    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        hzb_view: &TextureView,
    ) -> RenderResult<()> {
        self.encode_ssao(device, encoder, gbuffer, hzb_view)?;
        self.encode_blur(device, encoder, gbuffer)?;
        self.encode_temporal(device, encoder)?;
        self.encode_composite(device, encoder, gbuffer)?;
        Ok(())
    }
}
