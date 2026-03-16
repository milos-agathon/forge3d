use super::*;

impl GpuBvhBuilder {
    pub(super) fn generate_morton_codes(
        &self,
        buffers: &GpuBuffers,
        world_aabb: &Aabb,
        prim_count: u32,
    ) -> Result<()> {
        if prim_count == 0 {
            return Ok(());
        }
        // Uniforms
        let uniforms = MortonUniforms {
            prim_count,
            frame_index: 0,
            _pad0: 0,
            _pad1: 0,
            world_min: world_aabb.min,
            _pad2: 0.0,
            world_extent: [
                (world_aabb.max[0] - world_aabb.min[0]).max(1e-6),
                (world_aabb.max[1] - world_aabb.min[1]).max(1e-6),
                (world_aabb.max[2] - world_aabb.min[2]).max(1e-6),
            ],
            _pad3: 0.0,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lbvh-morton-uniforms"),
                contents: cast_slice(&[uniforms]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // Bind groups from pipeline layout
        let bgl0 = self.morton_pipeline.get_bind_group_layout(0);
        let bgl1 = self.morton_pipeline.get_bind_group_layout(1);
        let bgl2 = self.morton_pipeline.get_bind_group_layout(2);

        let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-morton-bg0"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-morton-bg1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.centroids_buffer.as_entire_binding(),
                },
                // Read-only primitive index stream
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.prim_indices_buffer.as_entire_binding(),
                },
            ],
        });
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-morton-bg2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.morton_codes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.sorted_indices_buffer.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbvh-morton-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbvh-morton-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.morton_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            let wg = ((prim_count + 255) / 256) as u32;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
}
