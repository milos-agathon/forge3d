use super::*;

impl WavefrontScheduler {
    fn create_restir_spatial_bind_group(&self) -> Result<BindGroup, Box<dyn std::error::Error>> {
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-spatial-bind-group"),
            layout: &self.pipelines.restir_spatial_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.restir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.restir_out.as_entire_binding(),
                },
            ],
        }))
    }

    pub fn init_restir_scene_spatial_bind_group(
        &mut self,
        area_lights: &Buffer,
        directional_lights: &Buffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-scene-spatial-bind-group"),
            layout: &self.pipelines.restir_scene_spatial_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: area_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: directional_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.restir_gbuffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.restir_gbuffer_pos.as_entire_binding(),
                },
            ],
        });
        self.restir_scene_spatial_bind_group = Some(bg);
        Ok(())
    }

    pub(super) fn dispatch_restir_spatial(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        _scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let spatial_bind_group = self.create_restir_spatial_bind_group()?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir-spatial-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.restir_spatial);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        if let Some(ref scene_spatial_bg) = self.restir_scene_spatial_bind_group {
            pass.set_bind_group(1, scene_spatial_bg, &[]);
        } else {
            return Ok(());
        }
        pass.set_bind_group(2, &spatial_bind_group, &[]);
        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    pub(super) fn dispatch_restir_temporal(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let temporal_bind_group = self.create_restir_temporal_bind_group()?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir-temporal-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.restir_temporal);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &temporal_bind_group, &[]);
        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    fn create_restir_bind_group(&self) -> Result<BindGroup, Box<dyn std::error::Error>> {
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-bind-group"),
            layout: &self.pipelines.restir_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.restir_reservoirs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.restir_light_samples.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.restir_alias_entries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.restir_light_probs.as_entire_binding(),
                },
            ],
        }))
    }

    fn create_restir_temporal_bind_group(&self) -> Result<BindGroup, Box<dyn std::error::Error>> {
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-temporal-bind-group"),
            layout: &self.pipelines.restir_temporal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.restir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.restir_reservoirs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.restir_out.as_entire_binding(),
                },
            ],
        }))
    }

    pub(super) fn dispatch_restir_init(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let restir_bind_group = self.create_restir_bind_group()?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir-init-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.restir_init);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &restir_bind_group, &[]);
        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}
