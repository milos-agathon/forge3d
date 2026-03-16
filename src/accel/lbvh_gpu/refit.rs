use super::*;

impl GpuBvhBuilder {
    pub fn refit(&mut self, handle: &mut BvhHandle, triangles: &[Triangle]) -> Result<()> {
        let gpu_data = match &handle.backend {
            BvhBackend::Gpu(data) => data,
            BvhBackend::Cpu(_) => anyhow::bail!("Cannot refit CPU BVH with GPU builder"),
        };

        if triangles.len() as u32 != gpu_data.primitive_count {
            anyhow::bail!(
                "Triangle count mismatch: expected {}, got {}",
                gpu_data.primitive_count,
                triangles.len()
            );
        }

        // Update primitive AABBs
        let triangle_aabbs = crate::accel::types::compute_triangle_aabbs(triangles);
        let aabb_data = cast_slice(&triangle_aabbs);

        // Create temporary buffer for updated AABBs
        let aabb_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Updated Primitive AABBs"),
                contents: aabb_data,
                usage: BufferUsages::STORAGE,
            });

        // Execute refit passes; GPU kernels are pending.
        self.execute_refit(
            &aabb_buffer,
            &gpu_data.nodes_buffer,
            &gpu_data.indices_buffer,
            gpu_data.primitive_count,
        )?;

        // Keep handle AABB in sync on the CPU so callers see refit effects.
        let new_world = crate::accel::types::compute_scene_aabb(triangles);
        handle.world_aabb = new_world;

        Ok(())
    }

    pub(super) fn execute_refit(
        &self,
        aabb_buffer: &Buffer,
        nodes_buffer: &Buffer,
        indices_buffer: &Buffer,
        prim_count: u32,
    ) -> Result<()> {
        if prim_count == 0 {
            return Ok(());
        }
        let node_count = 2 * prim_count - 1;
        let uniforms = LinkUniforms {
            prim_count,
            node_count,
            _pad0: 0,
            _pad1: 0,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bvh-refit-uniforms"),
                contents: cast_slice(&[uniforms]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bgl0 = self.refit_iterative_pipeline.get_bind_group_layout(0);
        let bgl1 = self.refit_iterative_pipeline.get_bind_group_layout(1);
        let bgl2 = self.refit_iterative_pipeline.get_bind_group_layout(2);

        let node_flags = &self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh-refit-node-flags"),
            size: (node_count * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bvh-refit-bg0"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        // refit_iterative ignores sorted_indices; bind a small buffer to satisfy the layout.
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bvh-refit-bg1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aabb_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indices_buffer.as_entire_binding(),
                },
            ],
        });
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bvh-refit-bg2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_flags.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bvh-refit-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bvh-refit-iterative"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.refit_iterative_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
}
