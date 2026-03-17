use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::passes::gi::{GiCompositeParams, GiPass};
use crate::passes::ssr::SsrStats;
use crate::viewer::Viewer;
use anyhow::{anyhow, Context};

impl Viewer {
    pub(crate) fn reexecute_gi(&mut self, ssr_stats: Option<&mut SsrStats>) -> anyhow::Result<()> {
        let depth_view = self.z_view.as_ref().context("Depth view unavailable")?;
        if let Some(ref mut gi) = self.gi {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p5.gi.reexec"),
                });
            gi.advance_frame(&self.queue);

            let mut timing_opt = self.gi_timing.as_mut().filter(|t| t.is_supported());

            if let Some(timer) = timing_opt.as_deref_mut() {
                let scope_id = timer.begin_scope(&mut enc, "p5.hzb");
                gi.build_hzb(&self.device, &mut enc, depth_view, false);
                timer.end_scope(&mut enc, scope_id);
            } else {
                gi.build_hzb(&self.device, &mut enc, depth_view, false);
            }

            gi.execute(&self.device, &mut enc, ssr_stats, timing_opt.as_deref_mut())?;

            let env_view = self
                .ibl_env_view
                .as_ref()
                .context("IBL env view unavailable")?;
            let env_samp = self
                .ibl_sampler
                .as_ref()
                .context("IBL sampler unavailable")?;

            let lit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.lit.bg.gi_baseline"),
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
                        resource: wgpu::BindingResource::TextureView(env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(env_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.lit_uniform.as_entire_binding(),
                    },
                ],
            });

            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;

            let baseline_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gi.baseline.bg"),
                layout: &self.gi_baseline_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.gi_baseline_hdr_view),
                    },
                ],
            });

            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.lit.baseline"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.lit_pipeline);
                cpass.set_bind_group(0, &lit_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.gi.baseline.copy"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.gi_baseline_pipeline);
                cpass.set_bind_group(0, &baseline_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            let split_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gi.baseline.split.bg"),
                layout: &self.gi_split_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &self.gi_baseline_diffuse_hdr_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &self.gi_baseline_spec_hdr_view,
                        ),
                    },
                ],
            });

            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.gi.baseline.split"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.gi_split_pipeline);
                cpass.set_bind_group(0, &split_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }

            let (w, h) = (self.config.width, self.config.height);
            if self.gi_pass.is_none() {
                match GiPass::new(&self.device, w, h) {
                    Ok(pass) => {
                        self.gi_pass = Some(pass);
                    }
                    Err(e) => {
                        return Err(anyhow!("Failed to create GiPass: {}", e));
                    }
                }
            }

            if let Some(ref mut gi_pass) = self.gi_pass {
                let ao_view = gi.ao_resolved_view().unwrap_or(&gi.gbuffer().material_view);
                let ssgi_view = gi
                    .ssgi_output_for_display_view()
                    .unwrap_or(&gi.gbuffer().material_view);
                let ssr_view = gi.ssr_final_view().unwrap_or(&self.lit_output_view);

                let params = GiCompositeParams {
                    ao_enable: gi.is_enabled(SSE::SSAO),
                    ssgi_enable: gi.is_enabled(SSE::SSGI),
                    ssr_enable: gi.is_enabled(SSE::SSR) && self.ssr_params.ssr_enable,
                    ao_weight: self.gi_ao_weight,
                    ssgi_weight: self.gi_ssgi_weight,
                    ssr_weight: self.gi_ssr_weight,
                    energy_cap: 1.05,
                };

                gi_pass.update_params(&self.queue, |p| {
                    *p = params;
                });

                gi_pass.execute(
                    &self.device,
                    &mut enc,
                    &self.gi_baseline_hdr_view,
                    &self.gi_baseline_diffuse_hdr_view,
                    &self.gi_baseline_spec_hdr_view,
                    ao_view,
                    ssgi_view,
                    ssr_view,
                    &gi.gbuffer().normal_view,
                    &gi.gbuffer().material_view,
                    &self.gi_output_hdr_view,
                    timing_opt.as_deref_mut(),
                )?;

                gi_pass.execute_debug(
                    &self.device,
                    &mut enc,
                    ao_view,
                    ssgi_view,
                    ssr_view,
                    &self.gi_debug_view,
                )?;
            }

            if let Some(timer) = timing_opt.as_deref_mut() {
                timer.resolve_queries(&mut enc);
            }

            self.queue.submit(std::iter::once(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            if let Some(timer) = self.gi_timing.as_mut() {
                if timer.is_supported() {
                    match pollster::block_on(timer.get_results()) {
                        Ok(results) => {
                            self.gi_gpu_hzb_ms = 0.0;
                            self.gi_gpu_ssao_ms = 0.0;
                            self.gi_gpu_ssgi_ms = 0.0;
                            self.gi_gpu_ssr_ms = 0.0;
                            self.gi_gpu_composite_ms = 0.0;
                            for r in results {
                                if !r.timestamp_valid {
                                    continue;
                                }
                                match r.name.as_str() {
                                    "p5.hzb" => self.gi_gpu_hzb_ms = r.gpu_time_ms,
                                    "p5.ssao" => self.gi_gpu_ssao_ms = r.gpu_time_ms,
                                    "p5.ssgi" => self.gi_gpu_ssgi_ms = r.gpu_time_ms,
                                    "p5.ssr" => self.gi_gpu_ssr_ms = r.gpu_time_ms,
                                    "p5.composite" => self.gi_gpu_composite_ms = r.gpu_time_ms,
                                    _ => {}
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[P5.6] GPU timing readback failed: {e}");
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
