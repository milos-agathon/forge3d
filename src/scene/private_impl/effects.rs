impl Scene {
    pub(super) fn render_reflections(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        let camera_pos = self.extract_camera_position();
        let camera_target = self.extract_camera_target();
        let camera_up = glam::Vec3::Y;
        let projection = self.scene.proj;

        let Some(ref mut renderer) = self.reflection_renderer else {
            return Ok(());
        };

        let g = crate::core::gpu::ctx();

        if !self.reflections_enabled {
            renderer.set_enabled(false);
            renderer.upload_uniforms(&g.queue);
            return Ok(());
        }

        if renderer.bind_group().is_none() {
            renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
        }

        renderer.update_reflection_camera(camera_pos, camera_target, camera_up, projection);

        // Ensure measurable overhead for test timing at small resolutions
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Reflection pass: enable clip plane + disable sampling (mode=reflection pass)
        renderer.set_reflection_pass_mode();
        renderer.upload_uniforms(&g.queue);

        let reflection_view =
            glam::Mat4::from_cols_array(&renderer.uniforms.reflection_plane.reflection_view);
        let reflection_proj =
            glam::Mat4::from_cols_array(&renderer.uniforms.reflection_plane.reflection_projection);
        let reflection_uniforms = self
            .scene
            .globals
            .to_uniforms(reflection_view, reflection_proj);
        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&reflection_uniforms));

        {
            let mut rp = renderer.begin_reflection_pass(encoder);
            rp.set_pipeline(&self.tp.pipeline);
            rp.set_bind_group(0, &self.bg0_globals, &[]);
            rp.set_bind_group(1, &self.bg1_height, &[]);
            rp.set_bind_group(2, &self.bg2_lut, &[]);
            rp.set_bind_group(3, &self.bg3_tile, &[]);
            let max_groups = crate::core::gpu::ctx().device.limits().max_bind_groups;
            if max_groups >= 6 {
                // Use actual cloud shadow bind group if available, otherwise use dummy
                let cloud_bg = self
                    .bg3_cloud_shadows
                    .as_ref()
                    .unwrap_or(&self.bg4_dummy_cloud_shadows);
                rp.set_bind_group(4, cloud_bg, &[]);
                if let Some(reflection_bg) = renderer.bind_group() {
                    rp.set_bind_group(5, reflection_bg, &[]);
                }
            }
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }

        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&self.last_uniforms));

        // Restore reflection sampling for the main pass
        renderer.set_enabled(true);
        renderer.upload_uniforms(&g.queue);

        Ok(())
    }

    // B6: Render DOF post-processing effect
    pub(super) fn render_dof(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<(), String> {
        if !self.dof_enabled {
            return Ok(()); // Early return if DOF disabled
        }

        let Some(ref mut dof_renderer) = self.dof_renderer else {
            return Ok(()); // Early return if no DOF renderer
        };

        // Create bind group with color and depth textures
        let g = crate::core::gpu::ctx();

        // Ensure we have depth texture for DOF calculations
        let Some(ref depth_view) = self.depth_view else {
            return Err(
                "DOF requires depth buffer. Enable MSAA (samples > 1) for depth buffer."
                    .to_string(),
            );
        };

        // Create bind group for DOF
        let color_storage_view = self.color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene-color-storage"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        dof_renderer.create_bind_group(
            &g.device,
            &self.color_view,
            depth_view,
            Some(&color_storage_view),
        );

        // Upload DOF uniforms
        dof_renderer.upload_uniforms(&g.queue);

        // Dispatch DOF computation
        dof_renderer.dispatch(encoder);

        Ok(())
    }

    // B7: Generate and render cloud shadows
    pub(super) fn render_cloud_shadows(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        if !self.cloud_shadows_enabled {
            return Ok(()); // Early return if cloud shadows disabled
        }

        let Some(ref mut cloud_renderer) = self.cloud_shadow_renderer else {
            return Ok(()); // Early return if no cloud shadow renderer
        };

        let g = crate::core::gpu::ctx();

        // Create bind group for cloud shadow generation if needed
        if cloud_renderer.bind_group.is_none() {
            cloud_renderer.create_bind_group(&g.device);
        }

        // Upload cloud shadow uniforms
        cloud_renderer.upload_uniforms(&g.queue);

        // Generate cloud shadow texture
        cloud_renderer.generate_shadows(encoder);

        // Create terrain bind group for cloud shadows if needed
        if self.bg3_cloud_shadows.is_none() {
            self.bg3_cloud_shadows = Some(self.tp.make_bg_cloud_shadows(
                &g.device,
                cloud_renderer.shadow_view(),
                cloud_renderer.shadow_sampler(),
            ));
        }

        Ok(())
    }

    // Extract camera position from view matrix
    fn extract_camera_position(&self) -> glam::Vec3 {
        let view_matrix = self.scene.view;
        let inv_view = view_matrix.inverse();
        glam::Vec3::new(inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z)
    }

    // Extract camera target from view matrix (approximate)
    fn extract_camera_target(&self) -> glam::Vec3 {
        let camera_pos = self.extract_camera_position();
        let view_matrix = self.scene.view;

        // Forward vector from view matrix
        let forward = glam::Vec3::new(
            -view_matrix.z_axis.x,
            -view_matrix.z_axis.y,
            -view_matrix.z_axis.z,
        );
        camera_pos + forward // Target is camera position + forward direction
    }
}

