use super::*;

impl SsrRenderer {
    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        gbuffer: &GBuffer,
        mut stats: Option<&mut SsrStats>,
    ) -> RenderResult<()> {
        let stats_requested = stats.is_some();
        if stats_requested {
            encoder.clear_buffer(&self.counters_buffer, 0, None);
        }

        let (w, h) = (self.width, self.height);
        let gx = (w + 7) / 8;
        let gy = (h + 7) / 8;

        let trace_start = Instant::now();

        // Trace rays against the hierarchical depth
        let trace_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.trace.bg"),
            layout: &self.trace_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssr_hit_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Shade pass converts hit data into specular contributions
        let shade_start = Instant::now();
        let scene_color_view = self
            .scene_color_override
            .as_ref()
            .unwrap_or(&gbuffer.material_view);
        let shade_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.shade.bg"),
            layout: &self.shade_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(scene_color_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.linear_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssr_hit_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&self.ssr_spec_view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.shade"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shade_pipeline);
            pass.set_bind_group(0, &shade_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Environment fallback for misses
        let fallback_start = Instant::now();
        let fallback_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.fallback.bg"),
            layout: &self.fallback_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssr_spec_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssr_hit_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&gbuffer.depth_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&gbuffer.normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&self.env_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&self.env_sampler),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&self.ssr_final_view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.fallback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fallback_pipeline);
            pass.set_bind_group(0, &fallback_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Temporal accumulation smooths the reflection
        let fallback_end = Instant::now();
        let temporal_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.temporal.bg"),
            layout: &self.temporal_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.ssr_final_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssr_history_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.ssr_filtered_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.temporal_params.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &temporal_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Composite SSR into the lit buffer with tone mapping/boost parameters
        let composite_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.ssr.composite.bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gbuffer.material_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.ssr_filtered_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.composite_params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.ssr_composited_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.ssr.composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &composite_bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        // Preserve filtered result for next frame's temporal accumulation
        encoder.copy_texture_to_texture(
            self.ssr_filtered_texture.as_image_copy(),
            self.ssr_history_texture.as_image_copy(),
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.clear_scene_color_override();

        self.last_trace_ms = (shade_start - trace_start).as_secs_f32() * 1000.0;
        self.last_shade_ms = (fallback_start - shade_start).as_secs_f32() * 1000.0;
        self.last_fallback_ms = (fallback_end - fallback_start).as_secs_f32() * 1000.0;

        self.stats_readback_pending = stats_requested;
        if stats_requested {
            let counter_bytes = size_of::<[u32; 5]>() as BufferAddress;
            encoder.copy_buffer_to_buffer(
                &self.counters_buffer,
                0,
                &self.counters_readback,
                0,
                counter_bytes,
            );
        }

        if let Some(stats) = stats.as_deref_mut() {
            stats.trace_ms = self.last_trace_ms;
            stats.shade_ms = self.last_shade_ms;
            stats.fallback_ms = self.last_fallback_ms;
        }

        Ok(())
    }
}
