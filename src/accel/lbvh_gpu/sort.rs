use super::*;

impl GpuBvhBuilder {
    pub(super) fn sort_morton_codes(&self, buffers: &GpuBuffers, prim_count: u32) -> Result<()> {
        if prim_count == 0 {
            return Ok(());
        }
        let size_bytes = (prim_count * 4) as u64;

        // For very small arrays, use GPU bitonic sorter (single workgroup)
        if prim_count <= 256 {
            return self.sort_morton_codes_bitonic(buffers, prim_count, size_bytes);
        }

        // GPU radix sort multi-pass (4-bit digits)
        let mut keys_in = &buffers.morton_codes_buffer;
        let mut vals_in = &buffers.sorted_indices_buffer;
        let mut keys_out = &buffers.sort_temp_keys;
        let mut vals_out = &buffers.sort_temp_values;

        // Bind group layouts per entry
        let bgl0_count = self.sort_count_pipeline.get_bind_group_layout(0);
        let bgl1_count = self.sort_count_pipeline.get_bind_group_layout(1);
        let bgl3_count = self.sort_count_pipeline.get_bind_group_layout(3);
        let bgl3_scan = self.sort_scan_pipeline.get_bind_group_layout(3);
        let bgl0_scat = self.sort_scatter_pipeline.get_bind_group_layout(0);
        let bgl1_scat = self.sort_scatter_pipeline.get_bind_group_layout(1);
        let bgl2_scat = self.sort_scatter_pipeline.get_bind_group_layout(2);
        let bgl3_scat = self.sort_scatter_pipeline.get_bind_group_layout(3);
        let bgl3_clear = self.sort_clear_pipeline.get_bind_group_layout(3);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbvh-radix-enc"),
            });

        for shift in (0..32).step_by(4) {
            let u = SortUniforms {
                prim_count,
                pass_shift: shift as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let ubuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("radix-uniforms"),
                    contents: cast_slice(&[u]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

            // Clear histogram
            let bg3c = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg3-clear"),
                layout: &bgl3_clear,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.sort_histogram.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.sort_prefix_sums.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-clear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sort_clear_pipeline);
                pass.set_bind_group(3, &bg3c, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Count
            let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg0-count"),
                layout: &bgl0_count,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            });
            let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg1-count"),
                layout: &bgl1_count,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: keys_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vals_in.as_entire_binding(),
                    },
                ],
            });
            let bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg3-count"),
                layout: &bgl3_count,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.sort_histogram.as_entire_binding(),
                }],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-count"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sort_count_pipeline);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.set_bind_group(3, &bg3, &[]);
                let wg = ((prim_count + 255) / 256) as u32;
                pass.dispatch_workgroups(wg, 1, 1);
            }

            // Scan
            let bg3s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg3-scan"),
                layout: &bgl3_scan,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.sort_histogram.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.sort_prefix_sums.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-scan"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sort_scan_pipeline);
                pass.set_bind_group(3, &bg3s, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Clear histogram again to reuse as per-bucket offsets
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-clear2"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sort_clear_pipeline);
                pass.set_bind_group(3, &bg3c, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Scatter
            let bg0_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg0-scatter"),
                layout: &bgl0_scat,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            });
            let bg1_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg1-scatter"),
                layout: &bgl1_scat,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: keys_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vals_in.as_entire_binding(),
                    },
                ],
            });
            let bg2_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg2-scatter"),
                layout: &bgl2_scat,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: keys_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vals_out.as_entire_binding(),
                    },
                ],
            });
            let bg3_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bg3-scatter"),
                layout: &bgl3_scat,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.sort_histogram.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.sort_prefix_sums.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sort_scatter_pipeline);
                pass.set_bind_group(0, &bg0_s, &[]);
                pass.set_bind_group(1, &bg1_s, &[]);
                pass.set_bind_group(2, &bg2_s, &[]);
                pass.set_bind_group(3, &bg3_s, &[]);
                let wg = ((prim_count + 255) / 256) as u32;
                pass.dispatch_workgroups(wg, 1, 1);
            }

            // Ping-pong
            std::mem::swap(&mut keys_in, &mut keys_out);
            std::mem::swap(&mut vals_in, &mut vals_out);
        }

        // After 8 passes, data are back in primary buffers; no copy needed.
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
}
