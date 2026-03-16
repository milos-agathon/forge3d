impl Scene {
    pub(super) fn render_clouds(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        if !self.clouds_enabled {
            return Ok(());
        }

        let camera_pos = self.extract_camera_position();
        let view_proj = self.scene.proj * self.scene.view;
        let sun_dir = self.scene.globals.sun_dir.normalize_or_zero();
        let sun_intensity = self.scene.globals.exposure.max(0.1);
        let sky_color = glam::Vec3::new(0.58, 0.72, 0.92);

        let Some(ref mut renderer) = self.cloud_renderer else {
            return Ok(());
        };

        let g = crate::core::gpu::ctx();
        renderer.prepare_frame(&g.device, &g.queue)?;
        renderer.set_camera(view_proj, camera_pos);
        renderer.set_sky_params(sky_color, sun_dir, sun_intensity);
        renderer.upload_uniforms(&g.queue);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene-clouds-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        renderer.draw(&mut pass);

        Ok(())
    }
}

