use crate::cli::gi_types::GiVizMode;
use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::viewer_enums::VizMode;
use crate::viewer::Viewer;
use wgpu::util::DeviceExt;

impl Viewer {
    pub(super) fn render_postfx_stage(
        &mut self,
        gi: &mut ScreenSpaceEffectsManager,
        encoder: &mut wgpu::CommandEncoder,
        zv: &wgpu::TextureView,
    ) {
        // If SSR is enabled, compute the pre-tonemap lighting now so SSR can sample it
        if gi.is_enabled(crate::core::screen_space_effects::ScreenSpaceEffect::SSR) {
            // Build lighting into lit_output_view
            let env_view = if let Some(ref v) = self.ibl_env_view {
                v
            } else {
                &self.ibl_env_view.as_ref().unwrap()
            };
            let env_samp = if let Some(ref s) = self.ibl_sampler {
                s
            } else {
                &self.ibl_sampler.as_ref().unwrap()
            };
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.lit.bg.pre_ssr"),
                layout: &self.lit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(env_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.lit_uniform.as_entire_binding(),
                    },
                ],
            });
            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.lit.compute.pre_ssr"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.lit_pipeline);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            // Provide SSR with the lit buffer as scene color
            let lit_view_for_ssr = self
                .lit_output
                .create_view(&wgpu::TextureViewDescriptor::default());
            gi.set_ssr_scene_color_view(lit_view_for_ssr);
        }

        // Build Hierarchical Z (HZB) pyramid from the real depth buffer (Depth32Float)
        // Use regular-Z convention (reversed_z=false) for viewer
        gi.build_hzb(&self.device, encoder, zv, false);
        // Execute effects
        let _ = gi.execute(&self.device, encoder, None, None);

        // Composite the material GBuffer to the swapchain
        if let (Some(comp_pl), Some(comp_bgl)) = (
            self.comp_pipeline.as_ref(),
            self.comp_bind_group_layout.as_ref(),
        ) {
            // Select source texture based on viz_mode
            // If Lit, compute into lit_output first
            if matches!(self.viz_mode, VizMode::Lit) {
                let env_view = if let Some(ref v) = self.ibl_env_view {
                    v
                } else {
                    &self.ibl_env_view.as_ref().unwrap()
                };
                let env_samp = if let Some(ref s) = self.ibl_sampler {
                    s
                } else {
                    &self.ibl_sampler.as_ref().unwrap()
                };
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.lit.bg"),
                    layout: &self.lit_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &gi.gbuffer().material_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(env_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(env_samp),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.lit_uniform.as_entire_binding(),
                        },
                    ],
                });
                let gx = (self.config.width + 7) / 8;
                let gy = (self.config.height + 7) / 8;
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("viewer.lit.compute"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.lit_pipeline);
                    cpass.set_bind_group(0, &bg, &[]);
                    cpass.dispatch_workgroups(gx, gy, 1);
                }
            }

            // P1.3: Execute TAA resolve if enabled
            let taa_applied = if let Some(ref mut taa) = self.taa_renderer {
                if taa.is_enabled() {
                    // Update TAA settings with current jitter
                    taa.update_settings(
                        &self.queue,
                        self.taa_jitter.offset_array(),
                        self.frame_count as u32,
                    );
                    // Execute TAA resolve
                    taa.execute(
                        &self.device,
                        encoder,
                        &self.queue,
                        &self.lit_output_view,
                        &gi.gbuffer().velocity_view,
                        &gi.gbuffer().depth_view,
                    )
                } else {
                    false
                }
            } else {
                false
            };

            // When taking snapshot, use raw GBuffer to avoid SSR/SSAO temporal caching issues
            let use_raw_gbuffer = self.snapshot_request.is_some();
            let (mode_u32, src_view) = match self.viz_mode {
                VizMode::Material => {
                    if use_raw_gbuffer {
                        (0u32, &gi.gbuffer().material_view)
                    } else if let Some(v) = gi.material_with_ssr_view() {
                        (0u32, v)
                    } else if self.use_ssao_composite {
                        if let Some(v) = gi.material_with_ao_view() {
                            (0u32, v)
                        } else {
                            (0u32, &gi.gbuffer().material_view)
                        }
                    } else {
                        (0u32, &gi.gbuffer().material_view)
                    }
                }
                VizMode::Normal => (1u32, &gi.gbuffer().normal_view),
                VizMode::Depth => (2u32, &gi.gbuffer().depth_view),
                VizMode::Gi => match self.gi_viz_mode {
                    GiVizMode::None => {
                        if let Some(v) = gi.gi_debug_view() {
                            (3u32, v)
                        } else {
                            (0u32, &gi.gbuffer().material_view)
                        }
                    }
                    GiVizMode::Composite => (0u32, &self.gi_debug_view),
                    GiVizMode::Ao => {
                        if let Some(v) = gi.ao_resolved_view() {
                            (3u32, v)
                        } else {
                            (3u32, &gi.gbuffer().material_view)
                        }
                    }
                    GiVizMode::Ssgi => {
                        if let Some(v) = gi.ssgi_output_for_display_view() {
                            (0u32, v)
                        } else {
                            (0u32, &gi.gbuffer().material_view)
                        }
                    }
                    GiVizMode::Ssr => {
                        if let Some(v) = gi.ssr_final_view() {
                            (0u32, v)
                        } else {
                            (0u32, &self.lit_output_view)
                        }
                    }
                },
                VizMode::Lit => {
                    // P1.3: Use TAA output if TAA was applied
                    if taa_applied {
                        if let Some(ref taa) = self.taa_renderer {
                            (0u32, taa.output_view())
                        } else {
                            (0u32, &self.lit_output_view)
                        }
                    } else {
                        (0u32, &self.lit_output_view)
                    }
                }
            };
            // Prepare comp uniform (mode, far)
            let params: [f32; 4] = [
                mode_u32 as f32,
                self.viz_depth_max_override.unwrap_or(self.view_config.zfar),
                0.0,
                0.0,
            ];
            let buf_ref: &wgpu::Buffer = if let Some(ref ub) = self.comp_uniform {
                self.queue
                    .write_buffer(ub, 0, bytemuck::cast_slice(&params));
                ub
            } else {
                let ub = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("viewer.comp.uniform"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                self.comp_uniform = Some(ub);
                self.comp_uniform.as_ref().unwrap()
            };
            let comp_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.comp.bg"),
                layout: comp_bgl,
                entries: &[
                    // binding 0: sky_tex
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                    },
                    // binding 1: depth_tex
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                    },
                    // binding 2: fog_tex
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(if self.fog_enabled {
                            &self.fog_output_view
                        } else {
                            &self.fog_zero_view
                        }),
                    },
                    // binding 3: params (uniform)
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_ref.as_entire_binding(),
                    },
                    // binding 4: color_tex
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                ],
            });

            // If a snapshot is requested, render the composite to an offscreen texture too
            if self.snapshot_request.is_some() {
                let snap_w = self.config.width;
                let snap_h = self.config.height;

                let snap_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.snapshot.offscreen"),
                    size: wgpu::Extent3d {
                        width: snap_w,
                        height: snap_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
                {
                    let mut pass_snap = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.comp.pass.snapshot"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &snap_view,
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
                    pass_snap.set_pipeline(comp_pl);
                    pass_snap.set_bind_group(0, &comp_bg, &[]);
                    pass_snap.draw(0..3, 0..1);
                }
                self.pending_snapshot_tex = Some(snap_tex);
            }
        }
    }
}
