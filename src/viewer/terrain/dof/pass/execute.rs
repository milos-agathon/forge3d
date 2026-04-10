use super::DofPass;
use crate::viewer::terrain::dof::{DofConfig, DofUniforms};

impl DofPass {
    pub fn apply_from_input(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        config: &DofConfig,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.ensure_textures(width, height, format);

        let input_view = self.input_view.as_ref().unwrap();
        let intermediate_view = self.intermediate_view.as_ref().unwrap();

        let uniforms_h = DofUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            dof_params: [
                config.focus_distance,
                config.f_stop,
                config.focal_length,
                config.max_blur_radius,
            ],
            dof_params2: [near_plane, far_plane, 0.0, config.quality as f32],
            camera_params: [
                24.0,
                config.blur_strength,
                config.tilt_pitch,
                config.tilt_yaw,
            ],
        };
        queue.write_buffer(
            &self.uniform_buffer_h,
            0,
            bytemuck::cast_slice(&[uniforms_h]),
        );

        let uniforms_v = DofUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            dof_params: [
                config.focus_distance,
                config.f_stop,
                config.focal_length,
                config.max_blur_radius,
            ],
            dof_params2: [near_plane, far_plane, 1.0, config.quality as f32],
            camera_params: [
                24.0,
                config.blur_strength,
                config.tilt_pitch,
                config.tilt_yaw,
            ],
        };
        queue.write_buffer(
            &self.uniform_buffer_v,
            0,
            bytemuck::cast_slice(&[uniforms_v]),
        );

        self.render_pass_with_buffer(
            encoder,
            input_view,
            depth_view,
            intermediate_view,
            &self.uniform_buffer_h,
        );
        self.render_pass_with_buffer(
            encoder,
            intermediate_view,
            depth_view,
            output_view,
            &self.uniform_buffer_v,
        );
    }

    pub fn apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        config: &DofConfig,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.ensure_textures(width, height, format);

        let uniforms_h = DofUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            dof_params: [
                config.focus_distance,
                config.f_stop,
                config.focal_length,
                config.max_blur_radius,
            ],
            dof_params2: [near_plane, far_plane, 0.0, config.quality as f32],
            camera_params: [
                24.0,
                config.blur_strength,
                config.tilt_pitch,
                config.tilt_yaw,
            ],
        };
        queue.write_buffer(
            &self.uniform_buffer_h,
            0,
            bytemuck::cast_slice(&[uniforms_h]),
        );

        let uniforms_v = DofUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            dof_params: [
                config.focus_distance,
                config.f_stop,
                config.focal_length,
                config.max_blur_radius,
            ],
            dof_params2: [near_plane, far_plane, 1.0, config.quality as f32],
            camera_params: [
                24.0,
                config.blur_strength,
                config.tilt_pitch,
                config.tilt_yaw,
            ],
        };
        queue.write_buffer(
            &self.uniform_buffer_v,
            0,
            bytemuck::cast_slice(&[uniforms_v]),
        );

        self.render_pass_with_buffer(
            encoder,
            color_view,
            depth_view,
            self.intermediate_view.as_ref().unwrap(),
            &self.uniform_buffer_h,
        );
        self.render_pass_with_buffer(
            encoder,
            self.intermediate_view.as_ref().unwrap(),
            depth_view,
            output_view,
            &self.uniform_buffer_v,
        );
    }

    fn render_pass_with_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dof.render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
