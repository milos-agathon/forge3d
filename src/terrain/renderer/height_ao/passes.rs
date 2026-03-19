use super::*;

impl TerrainScene {
    pub(in crate::terrain::renderer) fn compute_height_ao_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        heightmap_view: &wgpu::TextureView,
        internal_width: u32,
        internal_height: u32,
        width: u32,
        height: u32,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
    ) -> Result<bool> {
        if !decoded.height_ao.enabled {
            return Ok(false);
        }

        let ao_resolution_scale = decoded.height_ao.resolution_scale.clamp(0.1, 1.0);
        self.ensure_height_ao_texture_size(internal_width, internal_height, ao_resolution_scale)?;

        let ao_size = self
            .height_ao_size
            .lock()
            .map_err(|_| anyhow!("height_ao_size mutex poisoned"))?;
        let (ao_width, ao_height) = *ao_size;
        drop(ao_size);

        let ao_uniforms = HeightAoUniforms {
            params0: [
                decoded.height_ao.directions as f32,
                decoded.height_ao.steps as f32,
                decoded.height_ao.max_distance,
                decoded.height_ao.strength,
            ],
            params1: [
                params.terrain_span / width as f32,
                params.terrain_span / height as f32,
                params.z_scale,
                decoded.clamp.height_range.0,
            ],
            params2: [
                ao_width as f32,
                ao_height as f32,
                width as f32,
                height as f32,
            ],
        };
        self.queue.write_buffer(
            &self.height_ao_uniform_buffer,
            0,
            bytemuck::bytes_of(&ao_uniforms),
        );

        let storage_view_guard = self
            .height_ao_storage_view
            .lock()
            .map_err(|_| anyhow!("height_ao_storage_view mutex poisoned"))?;
        let storage_view = storage_view_guard
            .as_ref()
            .ok_or_else(|| anyhow!("height_ao_storage_view not initialized"))?;

        let ao_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height_ao.bind_group"),
            layout: &self.height_ao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.height_ao_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(storage_view),
                },
            ],
        });
        drop(storage_view_guard);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("height_ao.compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.height_ao_compute_pipeline);
            compute_pass.set_bind_group(0, &ao_bind_group, &[]);
            compute_pass.dispatch_workgroups((ao_width + 7) / 8, (ao_height + 7) / 8, 1);
        }

        Ok(true)
    }

    pub(in crate::terrain::renderer) fn compute_sun_visibility_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        heightmap_view: &wgpu::TextureView,
        internal_width: u32,
        internal_height: u32,
        width: u32,
        height: u32,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
    ) -> Result<bool> {
        if !decoded.sun_visibility.enabled {
            return Ok(false);
        }

        let sv_resolution_scale = decoded.sun_visibility.resolution_scale.clamp(0.1, 1.0);
        self.ensure_sun_vis_texture_size(internal_width, internal_height, sv_resolution_scale)?;

        let sv_size = self
            .sun_vis_size
            .lock()
            .map_err(|_| anyhow!("sun_vis_size mutex poisoned"))?;
        let (sv_width, sv_height) = *sv_size;
        drop(sv_size);

        let sun_dir = glam::Vec3::new(
            -decoded.light.direction[0],
            -decoded.light.direction[1],
            -decoded.light.direction[2],
        )
        .normalize();

        let sv_uniforms = SunVisUniforms {
            params0: [
                decoded.sun_visibility.samples as f32,
                decoded.sun_visibility.steps as f32,
                decoded.sun_visibility.max_distance,
                decoded.sun_visibility.softness,
            ],
            params1: [
                params.terrain_span / width as f32,
                params.terrain_span / height as f32,
                params.z_scale,
                decoded.clamp.height_range.0,
            ],
            params2: [
                sv_width as f32,
                sv_height as f32,
                width as f32,
                height as f32,
            ],
            params3: [sun_dir.x, sun_dir.y, sun_dir.z, decoded.sun_visibility.bias],
        };
        self.queue.write_buffer(
            &self.sun_vis_uniform_buffer,
            0,
            bytemuck::bytes_of(&sv_uniforms),
        );

        let storage_view_guard = self
            .sun_vis_storage_view
            .lock()
            .map_err(|_| anyhow!("sun_vis_storage_view mutex poisoned"))?;
        let storage_view = storage_view_guard
            .as_ref()
            .ok_or_else(|| anyhow!("sun_vis_storage_view not initialized"))?;

        let sv_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sun_vis.bind_group"),
            layout: &self.sun_vis_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sun_vis_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(storage_view),
                },
            ],
        });
        drop(storage_view_guard);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sun_vis.compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.sun_vis_compute_pipeline);
            compute_pass.set_bind_group(0, &sv_bind_group, &[]);
            compute_pass.dispatch_workgroups((sv_width + 7) / 8, (sv_height + 7) / 8, 1);
        }

        Ok(true)
    }
}
