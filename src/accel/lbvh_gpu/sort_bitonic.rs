use super::*;

impl GpuBvhBuilder {
    pub(super) fn sort_morton_codes_bitonic(
        &self,
        buffers: &GpuBuffers,
        prim_count: u32,
        size_bytes: u64,
    ) -> Result<()> {
        let uniforms = SortUniforms {
            prim_count,
            pass_shift: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let uniforms_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bitonic-sort-uniforms"),
                contents: cast_slice(&[uniforms]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let bgl0 = self.sort_bitonic_pipeline.get_bind_group_layout(0);
        let bgl1 = self.sort_bitonic_pipeline.get_bind_group_layout(1);
        let bgl2 = self.sort_bitonic_pipeline.get_bind_group_layout(2);

        let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitonic-bg0"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitonic-bg1"),
            layout: &bgl1,
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
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitonic-bg2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.sort_temp_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.sort_temp_values.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bitonic-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bitonic-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sort_bitonic_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &buffers.sort_temp_keys,
            0,
            &buffers.morton_codes_buffer,
            0,
            size_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.sort_temp_values,
            0,
            &buffers.sorted_indices_buffer,
            0,
            size_bytes,
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}
