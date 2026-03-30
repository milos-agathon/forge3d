use super::*;

impl ViewerTerrainScene {
    pub fn render_with_motion_blur(
        &mut self,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<wgpu::Texture> {
        if self.terrain.is_none() {
            return None;
        }

        let config = self.pbr_config.motion_blur.clone();
        if !config.enabled || config.samples <= 1 {
            // No motion blur needed, fall back to regular render
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain_viewer.motion_blur_fallback"),
                });
            let result = self.render_to_texture(&mut encoder, target_format, width, height, 0);
            self.queue.submit(std::iter::once(encoder.finish()));
            return result;
        }

        // Initialize motion blur pass if needed
        if self.motion_blur_pass.is_none() {
            self.init_motion_blur_pass();
        }

        // Store original camera params
        let terrain = self.terrain.as_ref().unwrap();
        let base_phi = terrain.cam_phi_deg;
        let base_theta = terrain.cam_theta_deg;
        let base_radius = terrain.cam_radius;
        let base_target = terrain.cam_target;
        let _ = terrain;

        // Create accumulation texture (Rgba32Float for HDR)
        let accum_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.motion_blur_accum"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Clear accumulation buffer
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain_viewer.motion_blur_clear"),
                });
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.motion_blur_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &accum_view,
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
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Render N sub-frames with interpolated camera
        let orig_distortion = self.pbr_config.lens_effects.distortion;
        let orig_chromatic_aberration = self.pbr_config.lens_effects.chromatic_aberration;
        self.pbr_config.lens_effects.distortion = 0.0;
        self.pbr_config.lens_effects.chromatic_aberration = 0.0;

        let samples = config.samples.max(1);

        for i in 0..samples {
            // Calculate interpolation factor using the shutter timing
            // sample_t spans [shutter_open, shutter_close] across the sample set
            // This allows the cam_*_delta values to represent motion per full frame,
            // with the shutter timing determining how much of that motion is captured
            let shutter_range = config.shutter_close - config.shutter_open;
            let relative_t = (i as f32 + 0.5) / samples as f32; // 0..1
            let sample_t = config.shutter_open + shutter_range * relative_t;

            // Interpolate camera position across the shutter interval
            // cam_*_delta represents motion per full frame (frame time = 1.0)
            // The shutter timing naturally scales the effective motion captured
            let phi = base_phi + config.cam_phi_delta * sample_t;
            let theta = base_theta + config.cam_theta_delta * sample_t;
            let radius = base_radius + config.cam_radius_delta * sample_t;

            // Temporarily set camera params
            if let Some(ref mut terrain) = self.terrain {
                terrain.cam_phi_deg = phi;
                terrain.cam_theta_deg = theta;
                terrain.cam_radius = radius;
            }

            // Render frame to temporary texture
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain_viewer.motion_blur_sample"),
                });

            let frame_tex = self.render_to_texture(&mut encoder, target_format, width, height, 0);
            self.queue.submit(std::iter::once(encoder.finish()));

            // Add to accumulation (additive blend)
            if let Some(ref frame) = frame_tex {
                let frame_view = frame.create_view(&wgpu::TextureViewDescriptor::default());
                self.accumulate_frame(&frame_view, &accum_view, width, height);
            }
        }

        self.pbr_config.lens_effects.distortion = orig_distortion;
        self.pbr_config.lens_effects.chromatic_aberration = orig_chromatic_aberration;

        // Restore original camera params
        if let Some(ref mut terrain) = self.terrain {
            terrain.cam_phi_deg = base_phi;
            terrain.cam_theta_deg = base_theta;
            terrain.cam_radius = base_radius;
            terrain.cam_target = base_target;
        }

        // Resolve: create final output and divide by sample count
        let output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.motion_blur_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Use motion blur pass to resolve
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);

        let mut final_tex = output_tex;
        if let Some(ref motion_blur) = self.motion_blur_pass {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain_viewer.motion_blur_resolve"),
                });
            motion_blur.resolve(
                &mut encoder,
                &self.queue,
                &accum_view,
                &output_view,
                width,
                height,
                samples,
            );

            if needs_post_process {
                if self.post_process.is_none() {
                    self.init_post_process();
                }
                if let Some(ref mut pp) = self.post_process {
                    let lens = &self.pbr_config.lens_effects;
                    let lens_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("terrain_viewer.motion_blur_lens_output"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: target_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let lens_output_view =
                        lens_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

                    pp.apply_from_input(
                        &mut encoder,
                        &self.queue,
                        &output_view,
                        &lens_output_view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                    final_tex = lens_output_tex;
                }
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        println!("[terrain] Motion blur: {} samples rendered", samples);
        Some(final_tex)
    }

    /// Accumulate a frame into the accumulation buffer (additive blend)
    fn accumulate_frame(
        &self,
        frame_view: &wgpu::TextureView,
        accum_view: &wgpu::TextureView,
        _width: u32,
        _height: u32,
    ) {
        // Create a simple additive blit pipeline if needed
        // Use a simple additive pass until a dedicated accumulation pipeline is wired.
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("motion_blur.accumulate_shader"),
                source: wgpu::ShaderSource::Wgsl(ACCUMULATE_SHADER.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("motion_blur.accumulate_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("motion_blur.accumulate_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("motion_blur.accumulate_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("motion_blur.accumulate_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_blur.accumulate_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("motion_blur.accumulate"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion_blur.accumulate_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: accum_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
