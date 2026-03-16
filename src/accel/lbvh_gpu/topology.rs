use super::*;

impl GpuBvhBuilder {
    pub(super) fn build_bvh_topology(
        &self,
        buffers: &GpuBuffers,
        prim_count: u32,
        node_count: u32,
    ) -> Result<()> {
        if prim_count == 0 {
            return Ok(());
        }
        // Uniforms
        let uniforms = LinkUniforms {
            prim_count,
            node_count,
            _pad0: 0,
            _pad1: 0,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lbvh-link-uniforms"),
                contents: cast_slice(&[uniforms]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bgl0_leaves = self.init_leaves_pipeline.get_bind_group_layout(0);
        let bgl0_link = self.link_nodes_pipeline.get_bind_group_layout(0);
        // For init_leaves entry: group(1) uses codes, indices, and primitive AABBs
        let bgl1_leaves = self.init_leaves_pipeline.get_bind_group_layout(1);
        let bgl2_leaves = self.init_leaves_pipeline.get_bind_group_layout(2);
        // For link_nodes entry: group(1) uses only codes and indices
        let bgl1_link = self.link_nodes_pipeline.get_bind_group_layout(1);
        let bgl2_link = self.link_nodes_pipeline.get_bind_group_layout(2);

        let bg0_leaves = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-link-bg0-leaves"),
            layout: &bgl0_leaves,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let bg0_link = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-link-bg0-link"),
            layout: &bgl0_link,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let bg1_leaves = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-link-bg1-leaves"),
            layout: &bgl1_leaves,
            entries: &[
                // init_leaves uses bindings 1 and 2; binding 0 (sorted_codes) is not referenced in this entry point
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.sorted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.primitive_aabbs_buffer.as_entire_binding(),
                },
            ],
        });
        let bg2_leaves = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-link-bg2-leaves"),
            layout: &bgl2_leaves,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.nodes_buffer.as_entire_binding(),
            }],
        });
        let bg1_link = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-link-bg1-link"),
            layout: &bgl1_link,
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
        let bg2_link = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh-link-bg2-link"),
            layout: &bgl2_link,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.nodes_buffer.as_entire_binding(),
            }],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbvh-link-enc"),
            });
        // Initialize leaves
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbvh-init-leaves"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.init_leaves_pipeline);
            pass.set_bind_group(0, &bg0_leaves, &[]);
            pass.set_bind_group(1, &bg1_leaves, &[]);
            pass.set_bind_group(2, &bg2_leaves, &[]);
            let wg = ((prim_count + 63) / 64) as u32;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        // Link internal nodes (best-effort minimal)
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbvh-link-nodes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.link_nodes_pipeline);
            pass.set_bind_group(0, &bg0_link, &[]);
            pass.set_bind_group(1, &bg1_link, &[]);
            pass.set_bind_group(2, &bg2_link, &[]);
            let wg = (((prim_count.saturating_sub(1)) + 63) / 64) as u32;
            if wg > 0 {
                pass.dispatch_workgroups(wg, 1, 1);
            }
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
}
