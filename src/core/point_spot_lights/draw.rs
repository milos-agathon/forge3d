use super::structs::PointSpotLightRenderer;
use wgpu;
use bytemuck;

impl PointSpotLightRenderer {
    /// Update uniforms and lights buffers
    pub fn update_buffers(&self, queue: &wgpu::Queue) {
        // Update uniforms
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        // Update lights buffer
        if !self.lights.is_empty() {
            queue.write_buffer(&self.lights_buffer, 0, bytemuck::cast_slice(&self.lights));
        }
    }

    /// Create bind groups with G-buffer textures
    pub fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        g_buffer_sampler: &wgpu::Sampler,
    ) {
        self.main_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point_spot_lights_main_bind_group"),
            layout: &self.main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(g_buffer_sampler),
                },
            ],
        }));

        // Create shadow bind group if shadow map exists
        if let Some(shadow_view) = &self.shadow_map_view {
            self.shadow_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point_spot_lights_shadow_bind_group"),
                layout: &self.shadow_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                ],
            }));
        }
    }

    /// Render lights using deferred shading
    pub fn render_deferred<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.lights.is_empty() {
            return;
        }

        if let (Some(main_bind_group), Some(shadow_bind_group)) =
            (&self.main_bind_group, &self.shadow_bind_group)
        {
            render_pass.set_pipeline(&self.deferred_pipeline);
            render_pass.set_bind_group(0, main_bind_group, &[]);
            render_pass.set_bind_group(1, shadow_bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Full-screen triangle
        }
    }
}
