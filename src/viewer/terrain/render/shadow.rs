use super::*;

impl ViewerTerrainScene {
    pub fn render_shadow_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_view: glam::Mat4,
        camera_proj: glam::Mat4,
        sun_direction: glam::Vec3,
    ) {
        // Early exit if shadow infrastructure not ready
        let csm = match self.csm_renderer.as_mut() {
            Some(c) => c,
            None => return,
        };

        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };

        if self.shadow_pipeline.is_none() || self.shadow_bind_groups.is_empty() {
            return;
        }

        let cascade_count = csm.config.cascade_count;
        let shadow_map_size = csm.config.shadow_map_size;

        // Mark that shadow passes are running (will be copied to csm_uniforms later)
        csm.uniforms.technique_reserved[0] = 1.0; // Flag: shadow passes executed

        // Update CSM cascade matrices based on camera and light
        let near_plane = 1.0;
        let far_plane = csm.config.max_shadow_distance;
        csm.update_cascades(
            camera_view,
            camera_proj,
            sun_direction,
            near_plane,
            far_plane,
        );

        // Get terrain parameters for shadow pass uniforms
        let (min_h, max_h) = terrain.domain;
        let terrain_span = terrain.dimensions.0.max(terrain.dimensions.1) as f32;

        let z_scale = terrain.z_scale;
        let grid_res = 512u32; // Shadow pass grid resolution

        // Height curve params: use linear (mode=0) with no curve transformation
        // This matches the default terrain rendering behavior
        let height_curve_mode = 0.0_f32; // 0=linear
        let height_curve_strength = 0.0_f32; // No curve applied
        let height_curve_power = 1.0_f32; // Default power

        // Compute a terrain-covering light view-projection matrix
        // The cascade's light_view_proj is based on camera frustum which doesn't cover full terrain
        // We need a projection that covers the entire terrain from light's perspective
        let terrain_height = (max_h - min_h) * z_scale;
        let terrain_center =
            glam::Vec3::new(terrain_span * 0.5, terrain_height * 0.5, terrain_span * 0.5);
        let light_distance = terrain_span * 2.0; // Place light far enough to see entire terrain
        let light_pos = terrain_center - sun_direction * light_distance;
        let light_view = glam::Mat4::look_at_rh(light_pos, terrain_center, glam::Vec3::Y);

        // Orthographic projection covering terrain bounds with margin
        let half_extent = terrain_span * 0.75; // Cover terrain with some margin
        let terrain_light_proj = glam::Mat4::orthographic_rh(
            -half_extent,
            half_extent,
            -half_extent,
            half_extent,
            0.1,
            light_distance * 2.0 + terrain_span,
        );
        let terrain_light_view_proj = terrain_light_proj * light_view;
        let terrain_light_view_proj_arr = terrain_light_view_proj.to_cols_array_2d();

        // Render each cascade
        for cascade_idx in 0..cascade_count as usize {
            if cascade_idx >= self.shadow_bind_groups.len()
                || cascade_idx >= self.shadow_uniform_buffers.len()
            {
                break;
            }

            // Use terrain-covering projection for shadow depth pass
            // This ensures the entire terrain is rendered to shadow map, not just camera frustum portion
            let light_view_proj = terrain_light_view_proj_arr;

            // Build shadow pass uniforms
            // Match main shader terrain_params layout: [min_h, h_range, terrain_width, z_scale]
            let shadow_uniforms = ShadowPassUniforms {
                light_view_proj,
                terrain_params: [min_h, max_h - min_h, terrain_span, z_scale],
                grid_params: [grid_res as f32, 0.0, 0.0, 0.0],
                height_curve: [
                    height_curve_mode,
                    height_curve_strength,
                    height_curve_power,
                    0.0,
                ],
            };

            // Upload uniforms
            self.queue.write_buffer(
                &self.shadow_uniform_buffers[cascade_idx],
                0,
                bytemuck::cast_slice(&[shadow_uniforms]),
            );

            // Get shadow map view for this cascade
            let shadow_map_view = &csm.shadow_map_views[cascade_idx];

            // Begin depth-only render pass for this cascade
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_depth_pass_cascade_{}", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &self.shadow_bind_groups[cascade_idx], &[]);

            // Draw terrain grid (6 vertices per quad, (grid_res-1)^2 quads)
            let vertex_count = 6 * (grid_res - 1) * (grid_res - 1);
            render_pass.draw(0..vertex_count, 0..1);
        }

        // Sync CSM cascade matrices to match the terrain-covering projection
        // used above.  The shader samples shadow_maps using these matrices, so
        // they MUST match what was used to render the depth pass.
        for i in 0..cascade_count as usize {
            csm.uniforms.cascades[i].light_view_proj = terrain_light_view_proj_arr;
        }

        // Execute moment generation pass for VSM/EVSM/MSM techniques
        // This converts the depth maps into moment statistics
        let technique = match self.pbr_config.shadow_technique.to_lowercase().as_str() {
            "vsm" => crate::lighting::shadow::ShadowTechnique::VSM,
            "evsm" => crate::lighting::shadow::ShadowTechnique::EVSM,
            "msm" => crate::lighting::shadow::ShadowTechnique::MSM,
            _ => return, // No moment generation needed for HARD/PCF/PCSS
        };

        // Prepare and execute moment pass if we have the resources
        if let (Some(ref mut moment_pass), Some(ref csm)) =
            (&mut self.moment_pass, &self.csm_renderer)
        {
            if let Some(ref moment_texture) = csm.evsm_maps {
                let depth_view = csm.shadow_texture_view();
                let moment_view =
                    crate::shadows::create_moment_storage_view(moment_texture, cascade_count);

                moment_pass.prepare_bind_group(&self.device, &depth_view, &moment_view);
                moment_pass.execute(
                    &self.queue,
                    encoder,
                    technique,
                    cascade_count,
                    shadow_map_size,
                    csm.config.evsm_positive_exp,
                    csm.config.evsm_negative_exp,
                );
            }
        }
    }
}
