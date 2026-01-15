// src/viewer/state/viewer_helpers.rs
// Helper methods for the Viewer struct
// Restored from mod.rs during refactoring

use std::path::Path;

use crate::core::ibl::{IBLQuality, IBLRenderer};
use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::p5::meta as p5_meta;
use crate::passes::gi::{GiCompositeParams, GiPass};
use crate::passes::ssr::SsrStats;
use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;
use crate::viewer::viewer_render_helpers::render_view_to_rgba8_ex;
use crate::viewer::Viewer;
use anyhow::{anyhow, bail, Context};
use glam::Mat4;

impl Viewer {
    pub(crate) fn write_p5_meta<F: FnOnce(&mut std::collections::BTreeMap<String, serde_json::Value>)>(
        &self,
        patch: F,
    ) -> anyhow::Result<()> {
        p5_meta::write_p5_meta(Path::new("reports/p5"), patch)
    }

    pub(crate) fn sync_ssr_params_to_gi(&mut self) {
        if let Some(ref mut gi) = self.gi {
            gi.set_ssr_params(&self.queue, &self.ssr_params);
        }
    }

    pub(crate) fn with_comp_pipeline<T>(
        &self,
        f: impl FnOnce(&wgpu::RenderPipeline, &wgpu::BindGroupLayout) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let comp_pl = self
            .comp_pipeline
            .as_ref()
            .context("comp pipeline")?;
        let comp_bgl = self
            .comp_bind_group_layout
            .as_ref()
            .context("comp bgl")?;
        f(comp_pl, comp_bgl)
    }

    pub(crate) fn snapshot_swapchain_to_png(
        &mut self,
        tex: &wgpu::Texture,
        path: &str,
    ) -> anyhow::Result<()> {
        let size = tex.size();
        let w = size.width;
        let h = size.height;
        let fmt = tex.format();

        match fmt {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                for px in data.chunks_exact_mut(4) {
                    px[3] = 255;
                }
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                let mut data = read_texture_tight(&self.device, &self.queue, tex, (w, h), fmt)
                    .context("readback failed")?;
                for px in data.chunks_exact_mut(4) {
                    px.swap(0, 2);
                    px[3] = 255;
                }
                image_write::write_png_rgba8(Path::new(path), &data, w, h)
                    .context("failed to write PNG")?;
                Ok(())
            }
            other => {
                bail!(
                    "snapshot only supports RGBA8/BGRA8 surfaces (got {:?})",
                    other
                )
            }
        }
    }

    pub(crate) fn load_ibl(&mut self, path: &str) -> anyhow::Result<()> {
        let hdr_img = crate::formats::hdr::load_hdr(path)
            .map_err(|e| anyhow!("failed to load HDR '{}': {}", path, e))?;

        let mut ibl = IBLRenderer::new(&self.device, IBLQuality::Low);

        if let Some(res) = self.ibl_base_resolution {
            ibl.set_base_resolution(res);
        } else {
            ibl.set_base_resolution(IBLQuality::Low.base_environment_size());
        }

        if let Some(ref cache_dir) = self.ibl_cache_dir {
            ibl.configure_cache(cache_dir, std::path::Path::new(path))
                .map_err(|e| anyhow!("failed to configure IBL cache: {}", e))?;
        }

        ibl.load_environment_map(
            &self.device,
            &self.queue,
            &hdr_img.data,
            hdr_img.width,
            hdr_img.height,
        )
        .map_err(|e| anyhow!("failed to upload environment: {}", e))?;

        ibl.initialize(&self.device, &self.queue)
            .map_err(|e| anyhow!("failed to initialize IBL: {}", e))?;

        let (irr_tex_opt, spec_tex_opt, _) = ibl.textures();
        if let Some(ref mut gi) = self.gi {
            if let Some(irr_tex) = irr_tex_opt {
                gi.set_ssgi_env(&self.device, irr_tex);
            }
            if let Some(spec_tex) = spec_tex_opt {
                gi.set_ssr_env(&self.device, spec_tex);
                let cube_view = spec_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("viewer.ibl.specular.cube.view"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                });
                let env_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.ibl.env.sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
                self.ibl_env_view = Some(cube_view);
                self.ibl_sampler = Some(env_sampler);
            }
        }

        self.ibl_renderer = Some(ibl);
        self.ibl_hdr_path = Some(path.to_string());
        Ok(())
    }

    pub(crate) fn update_lit_uniform(&mut self) {
        let sun_dir = [0.3f32, 0.6, -1.0];
        let params: [f32; 12] = [
            sun_dir[0],
            sun_dir[1],
            sun_dir[2],
            self.lit_sun_intensity,
            self.lit_ibl_intensity,
            if self.lit_use_ibl { 1.0 } else { 0.0 },
            self.lit_brdf as f32,
            0.0,
            self.lit_roughness.clamp(0.0, 1.0),
            self.lit_debug_mode as f32,
            0.0,
            0.0,
        ];
        self.queue
            .write_buffer(&self.lit_uniform, 0, bytemuck::cast_slice(&params));
    }

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
                        resource: wgpu::BindingResource::TextureView(&self.gi_baseline_diffuse_hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.gi_baseline_spec_hdr_view),
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

    pub(crate) fn capture_material_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        self.with_comp_pipeline(|comp_pl, comp_bgl| {
            let fog_view = if self.fog_enabled {
                &self.fog_output_view
            } else {
                &self.fog_zero_view
            };
            render_view_to_rgba8_ex(crate::viewer::viewer_render_helpers::RenderViewArgs {
                device: &self.device,
                queue: &self.queue,
                comp_pl,
                comp_bgl,
                sky_view: &self.sky_output_view,
                depth_view: &gi.gbuffer().depth_view,
                fog_view,
                surface_format: self.config.format,
                width: self.config.width,
                height: self.config.height,
                far,
                src_view: gi.material_with_ssr_view()
                    .or_else(|| gi.material_with_ssgi_view())
                    .or_else(|| gi.material_with_ao_view())
                    .unwrap_or(&gi.gbuffer().material_view),
                mode: 0,
            })
        })
    }

    pub(crate) fn read_ssgi_filtered_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi.ssgi_dimensions().context("SSGI dimensions unavailable")?;
        let tex = gi
            .ssgi_filtered_texture()
            .context("SSGI filtered texture unavailable")?;
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSGI filtered texture")?;
        Ok((bytes, dims))
    }

    pub(crate) fn read_ssgi_hit_bytes(&self) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let dims = gi.ssgi_dimensions().context("SSGI dimensions unavailable")?;
        let tex = gi
            .ssgi_hit_texture()
            .context("SSGI hit texture unavailable")?;
        let bytes = read_texture_tight(
            &self.device,
            &self.queue,
            tex,
            dims,
            wgpu::TextureFormat::Rgba16Float,
        )
        .context("read SSGI hit texture")?;
        Ok((bytes, dims))
    }

    pub(crate) fn render_geometry_to_gbuffer_once(&mut self) -> anyhow::Result<()> {
        if self.geom_vb.is_none()
            || self.geom_pipeline.is_none()
            || self.gi.is_none()
            || self.z_view.is_none()
        {
            return Ok(());
        }

        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov = self.view_config.fov_deg.to_radians();
        let proj = Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
        let view_mat = self.camera.view_matrix();
        let model_view = view_mat * self.object_transform;

        fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
            let c = m.to_cols_array();
            [
                [c[0], c[1], c[2], c[3]],
                [c[4], c[5], c[6], c[7]],
                [c[8], c[9], c[10], c[11]],
                [c[12], c[13], c[14], c[15]],
            ]
        }
        let cam_pack = [to_arr4(model_view), to_arr4(proj)];

        {
            let cam_buf = self.geom_camera_buffer.as_ref().unwrap();
            self.queue
                .write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_pack));
        }

        {
            if let Some(ref mut gi_mgr) = self.gi {
                let inv_proj = proj.inverse();
                let eye = self.camera.eye();
                let inv_model_view = model_view.inverse();
                let view_proj = proj * model_view;
                let cam = crate::core::screen_space_effects::CameraParams {
                    view_matrix: to_arr4(model_view),
                    inv_view_matrix: to_arr4(inv_model_view),
                    proj_matrix: to_arr4(proj),
                    inv_proj_matrix: to_arr4(inv_proj),
                    prev_view_proj_matrix: to_arr4(self.prev_view_proj),
                    camera_pos: [eye.x, eye.y, eye.z],
                    frame_index: self.frame_count as u32,
                    // P1.2: Pass jitter offset to shaders for TAA unjitter
                    jitter_offset: self.taa_jitter.offset_array(),
                    _pad_jitter: [0.0, 0.0],
                };
                gi_mgr.update_camera(&self.queue, &cam);
                // P1.1: Store current view_proj for next frame's motion vectors
                self.prev_view_proj = view_proj;
            }
        }

        if self.geom_bind_group.is_none() {
            if let Err(err) = self.ensure_geom_bind_group() {
                eprintln!("[viewer] failed to build geometry bind group for P5.1: {err}");
            }
        }

        if self.geom_bind_group.is_none() {
            let sampler = self.albedo_sampler.get_or_insert_with(|| {
                self.device
                    .create_sampler(&wgpu::SamplerDescriptor::default())
            });
            let white_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer.geom.albedo.fallback.p51"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &white_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255, 255, 255, 255],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.albedo_texture = Some(white_tex);
            let cam_buf = self.geom_camera_buffer.as_ref().unwrap();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.gbuf.geom.bg.p51"),
                layout: self.geom_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cam_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.albedo_view = Some(view);
            self.geom_bind_group = Some(bg);
        }

        let pipe = self.geom_pipeline.as_ref().unwrap();
        let vb = self.geom_vb.as_ref().unwrap();
        let zv = self.z_view.as_ref().unwrap();
        let gi = self.gi.as_ref().unwrap();
        let bg_ref = self.geom_bind_group.as_ref().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("p51.cornell.geom.encoder"),
            });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewer.geom.p51"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &gi.gbuffer().normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gi.gbuffer().material_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gi.gbuffer().depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: zv,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg_ref, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        if let Some(ib) = self.geom_ib.as_ref() {
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
        } else {
            pass.draw(0..self.geom_index_count, 0..1);
        }
        drop(pass);

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}
