use super::*;

impl TerrainScene {
    pub(super) fn render_shadow_depth_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        heightmap_view: &wgpu::TextureView,
        terrain_spacing: f32,
        height_exag: f32,
        height_min: f32,
        height_max: f32,
        _view_matrix: glam::Mat4,
        _proj_matrix: glam::Mat4,
        sun_direction: glam::Vec3,
        near_plane: f32,
        far_plane: f32,
        height_curve: [f32; 4],
    ) -> wgpu::BindGroup {
        let light_dir = sun_direction.normalize();
        let light_up = if light_dir.z.abs() > 0.99 {
            glam::Vec3::Y
        } else {
            glam::Vec3::Z
        };

        let half_spacing = terrain_spacing * 0.5;
        let shadow_z_min = 0.0;
        let shadow_z_max = height_exag;

        let terrain_min = glam::Vec3::new(-half_spacing, -half_spacing, shadow_z_min);
        let terrain_max = glam::Vec3::new(half_spacing, half_spacing, shadow_z_max);
        let terrain_center = (terrain_min + terrain_max) * 0.5;

        let terrain_diagonal = (terrain_max - terrain_min).length();
        let light_camera_distance = terrain_diagonal * 2.0;
        let light_camera_pos = terrain_center - light_dir * light_camera_distance;
        let light_view = glam::Mat4::look_to_rh(light_camera_pos, light_dir, light_up);

        let corners = [
            glam::Vec3::new(terrain_min.x, terrain_min.y, terrain_min.z),
            glam::Vec3::new(terrain_max.x, terrain_min.y, terrain_min.z),
            glam::Vec3::new(terrain_min.x, terrain_max.y, terrain_min.z),
            glam::Vec3::new(terrain_max.x, terrain_max.y, terrain_min.z),
            glam::Vec3::new(terrain_min.x, terrain_min.y, terrain_max.z),
            glam::Vec3::new(terrain_max.x, terrain_min.y, terrain_max.z),
            glam::Vec3::new(terrain_min.x, terrain_max.y, terrain_max.z),
            glam::Vec3::new(terrain_max.x, terrain_max.y, terrain_max.z),
        ];

        let mut light_min = glam::Vec3::splat(f32::MAX);
        let mut light_max = glam::Vec3::splat(f32::MIN);
        for corner in &corners {
            let light_pos = (light_view * corner.extend(1.0)).truncate();
            light_min = light_min.min(light_pos);
            light_max = light_max.max(light_pos);
        }

        let padding = terrain_spacing * 0.3;
        light_min -= glam::Vec3::splat(padding);
        light_max += glam::Vec3::splat(padding);

        let z_padding = terrain_spacing * 0.1;
        let proj_near = -light_max.z - z_padding;
        let proj_far = -light_min.z + z_padding;
        let light_proj = glam::Mat4::orthographic_rh(
            light_min.x,
            light_max.x,
            light_min.y,
            light_max.y,
            proj_near,
            proj_far,
        );

        let mut cascade_count = self
            .csm_renderer
            .config
            .cascade_count
            .max(1)
            .min(self.csm_renderer.shadow_map_views.len() as u32);
        let splits = if self.csm_renderer.config.cascade_splits.len() >= cascade_count as usize + 1
        {
            self.csm_renderer.config.cascade_splits.clone()
        } else {
            let mut fallback = Vec::with_capacity(cascade_count as usize + 1);
            fallback.push(near_plane);
            let step = (far_plane - near_plane) / cascade_count as f32;
            for i in 1..=cascade_count {
                fallback.push((near_plane + step * i as f32).min(far_plane));
            }
            fallback
        };
        if splits.len() < 2 {
            cascade_count = 1;
        }

        let light_view_proj = light_proj * light_view;
        let texel_size =
            (light_max.x - light_min.x) / self.csm_renderer.config.shadow_map_size as f32;

        self.csm_renderer.uniforms.light_direction = [light_dir.x, light_dir.y, light_dir.z, 0.0];
        self.csm_renderer.uniforms.light_view = light_view.to_cols_array();
        self.csm_renderer.uniforms.cascade_count = cascade_count;
        self.csm_renderer.uniforms.pcf_kernel_size = self.csm_renderer.config.pcf_kernel_size;
        self.csm_renderer.uniforms.depth_bias = self.csm_renderer.config.depth_bias;
        self.csm_renderer.uniforms.slope_bias = self.csm_renderer.config.slope_bias;
        self.csm_renderer.uniforms.shadow_map_size =
            self.csm_renderer.config.shadow_map_size as f32;
        self.csm_renderer.uniforms.peter_panning_offset =
            self.csm_renderer.config.peter_panning_offset;
        self.csm_renderer.uniforms.debug_mode = self.csm_renderer.config.debug_mode;

        for cascade_idx in 0..cascade_count as usize {
            let near_d = splits.get(cascade_idx).copied().unwrap_or(near_plane);
            let far_d = splits.get(cascade_idx + 1).copied().unwrap_or(far_plane);
            self.csm_renderer.uniforms.cascades[cascade_idx].light_projection =
                light_proj.to_cols_array();
            self.csm_renderer.uniforms.cascades[cascade_idx].light_view_proj =
                light_view_proj.to_cols_array_2d();
            self.csm_renderer.uniforms.cascades[cascade_idx].near_distance = near_d;
            self.csm_renderer.uniforms.cascades[cascade_idx].far_distance = far_d;
            self.csm_renderer.uniforms.cascades[cascade_idx].texel_size = texel_size;
            self.csm_renderer.uniforms.cascades[cascade_idx]._padding = 0.0;
        }
        self.csm_renderer.upload_uniforms(&self.queue);

        const SHADOW_GRID_RES: u32 = 1024;
        let vertices_per_cascade = (SHADOW_GRID_RES - 1) * (SHADOW_GRID_RES - 1) * 6;

        for cascade_idx in 0..cascade_count {
            let cascade = &self.csm_renderer.uniforms.cascades[cascade_idx as usize];
            let stored_light_view_proj = cascade.light_view_proj;
            let shadow_uniforms = ShadowPassUniforms {
                light_view_proj: stored_light_view_proj,
                terrain_params: [terrain_spacing, height_exag, height_min, height_max],
                grid_params: [SHADOW_GRID_RES as f32, 0.0, 0.0, 0.0],
                height_curve,
            };

            let shadow_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("terrain.shadow.cascade_{}.uniforms", cascade_idx)),
                        contents: bytemuck::bytes_of(&shadow_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let shadow_depth_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "terrain.shadow.cascade_{}.bind_group",
                        cascade_idx
                    )),
                    layout: &self.shadow_depth_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: shadow_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(heightmap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                        },
                    ],
                });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("terrain.shadow.cascade_{}.pass", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.csm_renderer.shadow_map_views[cascade_idx as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.shadow_depth_pipeline);
            pass.set_bind_group(0, &shadow_depth_bind_group, &[]);
            pass.draw(0..vertices_per_cascade, 0..1);
        }

        self.create_shadow_bind_group()
    }
}
