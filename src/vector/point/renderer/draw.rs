use super::*;

impl PointRenderer {
    pub fn render_oit<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        instance_count: u32,
    ) -> Result<(), RenderError> {
        if let Some(instance_buffer) = &self.instance_buffer {
            let uniform = self.build_uniform(transform, viewport_size, pixel_scale);
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
            render_pass.set_pipeline(&self.oit_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }
        Ok(())
    }

    pub fn render_pick<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        instance_count: u32,
        base_pick_id: u32,
    ) -> Result<(), RenderError> {
        if let Some(instance_buffer) = &self.instance_buffer {
            let uniform = self.build_uniform(transform, viewport_size, pixel_scale);
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
            let pick_data: [u32; 4] = [base_pick_id, 0, 0, 0];
            queue.write_buffer(&self.pick_uniform_buffer, 0, bytemuck::bytes_of(&pick_data));

            render_pass.set_pipeline(&self.pick_pipeline);
            render_pass.set_bind_group(0, &self.pick_bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }
        Ok(())
    }

    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
        instance_count: u32,
    ) -> Result<(), RenderError> {
        if let Some(instance_buffer) = &self.instance_buffer {
            let uniform = self.build_uniform(transform, viewport_size, pixel_scale);
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }

        Ok(())
    }

    fn build_uniform(
        &self,
        transform: &[[f32; 4]; 4],
        viewport_size: [f32; 2],
        pixel_scale: f32,
    ) -> PointUniform {
        let atlas_size = if let Some(atlas) = &self.texture_atlas {
            [atlas.width as f32, atlas.height as f32]
        } else {
            [1.0, 1.0]
        };

        PointUniform {
            transform: *transform,
            viewport_size,
            pixel_scale,
            debug_mode: self.debug_flags.to_bitfield(),
            atlas_size,
            enable_clip_w_scaling: self.enable_clip_w_scaling as u32,
            _pad0: 0.0,
            depth_range: [self.depth_range.0, self.depth_range.1],
            shape_mode: self.shape_mode,
            lod_threshold: self.lod_threshold,
        }
    }
}
