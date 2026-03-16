use super::RenderAvailability;
use glam::Mat4;

use crate::viewer::viewer_enums::FogMode;
use crate::viewer::{
    CameraFrustum, FogCameraUniforms, FogUpsampleParamsStd140, Viewer, VolumetricUniformsStd140,
};

impl Viewer {
    pub(super) fn render_geometry_stage(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> RenderAvailability {
        // Execute screen-space effects if any are enabled
        let have_gi = self.gi.is_some();
        let have_pipe = self.geom_pipeline.is_some();
        let have_cam = self.geom_camera_buffer.is_some();
        let have_vb = self.geom_vb.is_some();
        let have_z = self.z_view.is_some();
        let have_bgl = self.geom_bind_group_layout.is_some();

        // D1: Log render gate status on every snapshot frame (to file for diagnosis)
        if self.snapshot_request.is_some() {
            let msg = format!(
                "[D1-GATE] frame={} gi={} pipe={} cam={} vb={} z={} bgl={} idx_cnt={} transform_identity={}\n",
                self.frame_count, have_gi, have_pipe, have_cam, have_vb, have_z, have_bgl, self.geom_index_count,
                self.object_transform == glam::Mat4::IDENTITY
            );
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("examples/out/d1_debug.log")
                .and_then(|mut f| {
                    use std::io::Write;
                    f.write_all(msg.as_bytes())
                });
        }

        if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            if !self.debug_logged_render_gate {
                eprintln!(
                    "[viewer-debug] render gate: gi={} pipe={} cam={} vb={} z={} bgl={}",
                    have_gi, have_pipe, have_cam, have_vb, have_z, have_bgl
                );
                self.debug_logged_render_gate = true;
            }
        }

        if self.geom_bind_group.is_none() {
            if let Err(err) = self.ensure_geom_bind_group() {
                eprintln!("[viewer] failed to build geometry bind group: {err}");
            }
        }
        if let (Some(mut gi), Some(pipe), Some(cam_buf), Some(vb), Some(zv), Some(_bgl)) = (
            self.gi.take(),
            self.geom_pipeline.as_ref(),
            self.geom_camera_buffer.as_ref(),
            self.geom_vb.as_ref(),
            self.z_view.take(),
            self.geom_bind_group_layout.as_ref(),
        ) {
            // Update geometry camera uniform (view, proj)
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj_base =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);

            // P1.2: Apply TAA jitter to projection matrix
            let proj = if self.taa_jitter.enabled {
                crate::core::jitter::apply_jitter(
                    proj_base,
                    self.taa_jitter.offset.0,
                    self.taa_jitter.offset.1,
                    self.config.width,
                    self.config.height,
                )
            } else {
                proj_base
            };
            // Apply object transform to create model-view matrix
            let view_mat = self.camera.view_matrix();
            let model_view = view_mat * self.object_transform;

            // D1: Snapshot-time truth instrumentation - log to file
            if self.snapshot_request.is_some() {
                let is_identity = self.object_transform == glam::Mat4::IDENTITY;
                let t = self.object_translation;
                let s = self.object_scale;
                let msg = format!(
                    "[D1-GEOM] frame={} transform_identity={} index_count={} trans=[{:.3},{:.3},{:.3}] scale=[{:.3},{:.3},{:.3}]\n",
                    self.frame_count,
                    is_identity,
                    self.geom_index_count,
                    t.x, t.y, t.z, s.x, s.y, s.z
                );
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("examples/out/d1_debug.log")
                    .and_then(|mut f| {
                        use std::io::Write;
                        f.write_all(msg.as_bytes())
                    });
            }
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
            // Always write to the camera buffer
            self.queue
                .write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_pack));

            // Keep GI camera uniforms in sync with this geometry pass so that
            // SSAO/GTAO reconstruction uses the correct view/projection.
            // IMPORTANT: Use model_view (which includes object_transform) to ensure
            // GI lighting matches the transformed geometry.
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
            gi.update_camera(&self.queue, &cam);
            // P1.1: Store current view_proj for next frame's motion vectors
            self.prev_view_proj = view_proj;
            // P1.2: Advance jitter sequence for next frame
            self.taa_jitter.advance();

            // Geometry bind group (camera + albedo)
            let bg_ref = match self.geom_bind_group.as_ref() {
                Some(bg) => bg,
                None => {
                    // Create a minimal bind group if missing (shouldn't happen)
                    let sampler = self.albedo_sampler.get_or_insert_with(|| {
                        self.device
                            .create_sampler(&wgpu::SamplerDescriptor::default())
                    });
                    let white_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("viewer.geom.albedo.fallback2"),
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
                    let bgl = self.geom_bind_group_layout.as_ref().unwrap();
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.gbuf.geom.bg.autogen"),
                        layout: bgl,
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
                    self.geom_bind_group.as_ref().unwrap()
                }
            };
            let _layout = self.geom_bind_group_layout.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.geom"),
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
                    view: &zv,
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

            // P6: Volumetric fog compute (after depth is available)
            if self.fog_enabled {
                // Prepare camera uniforms
                let aspect = self.config.width as f32 / self.config.height as f32;
                let fov = self.view_config.fov_deg.to_radians();
                let proj = Mat4::perspective_rh(
                    fov,
                    aspect,
                    self.view_config.znear,
                    self.view_config.zfar,
                );
                let view_mat = self.camera.view_matrix();

                // Update CSM cascade transforms for current camera. The actual
                // shadow depth rendering into the CSM atlas is a separate
                // milestone; here we only keep the light-space matrices in sync
                // so fog can reuse them.
                if let Some(ref mut csm) = self.csm {
                    let frustum = CameraFrustum::from_matrices(&view_mat, &proj);
                    csm.update_cascades(&self.queue, &frustum);
                }

                if self.fog_use_shadows {
                    if let (Some(ref csm), Some(ref csm_pipe), Some(ref csm_cam_buf)) = (
                        self.csm.as_ref(),
                        self.csm_depth_pipeline.as_ref(),
                        self.csm_depth_camera.as_ref(),
                    ) {
                        let cascade_count = csm.cascade_count() as usize;
                        let bgl = csm_pipe.get_bind_group_layout(0);
                        for cascade_idx in 0..cascade_count {
                            if let (Some(depth_view), Some(light_vp)) = (
                                csm.cascade_depth_view(cascade_idx),
                                csm.cascade_projection(cascade_idx),
                            ) {
                                let light_vp_arr = light_vp.to_cols_array();
                                self.queue.write_buffer(
                                    csm_cam_buf,
                                    0,
                                    bytemuck::cast_slice(&light_vp_arr),
                                );
                                let csm_bg =
                                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("viewer.csm.depth.bg"),
                                        layout: &bgl,
                                        entries: &[wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: csm_cam_buf.as_entire_binding(),
                                        }],
                                    });
                                let mut shadow_pass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("viewer.csm.depth"),
                                        color_attachments: &[],
                                        depth_stencil_attachment: Some(
                                            wgpu::RenderPassDepthStencilAttachment {
                                                view: depth_view,
                                                depth_ops: Some(wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(1.0),
                                                    store: wgpu::StoreOp::Store,
                                                }),
                                                stencil_ops: None,
                                            },
                                        ),
                                        occlusion_query_set: None,
                                        timestamp_writes: None,
                                    });
                                shadow_pass.set_pipeline(csm_pipe);
                                shadow_pass.set_bind_group(0, &csm_bg, &[]);
                                shadow_pass.set_vertex_buffer(0, vb.slice(..));
                                if let Some(ib) = self.geom_ib.as_ref() {
                                    shadow_pass
                                        .set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                    shadow_pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
                                } else {
                                    shadow_pass.draw(0..self.geom_index_count, 0..1);
                                }
                            }
                        }
                    }
                }

                let inv_view = view_mat.inverse();
                let inv_proj = proj.inverse();
                let eye = self.camera.eye();
                fn to_arr(m: Mat4) -> [[f32; 4]; 4] {
                    let c = m.to_cols_array();
                    [
                        [c[0], c[1], c[2], c[3]],
                        [c[4], c[5], c[6], c[7]],
                        [c[8], c[9], c[10], c[11]],
                        [c[12], c[13], c[14], c[15]],
                    ]
                }
                let fog_cam = FogCameraUniforms {
                    view: to_arr(view_mat),
                    proj: to_arr(proj),
                    inv_view: to_arr(inv_view),
                    inv_proj: to_arr(inv_proj),
                    view_proj: to_arr(proj * view_mat),
                    eye_position: [eye.x, eye.y, eye.z],
                    near: self.view_config.znear,
                    far: self.view_config.zfar,
                    _pad: [0.0; 3],
                };
                self.queue
                    .write_buffer(&self.fog_camera, 0, bytemuck::bytes_of(&fog_cam));
                // Params
                let sun_dir_ws = (inv_view * glam::Vec4::new(0.3, 0.6, -1.0, 0.0))
                    .truncate()
                    .normalize();
                let steps = if self.fog_half_res_enabled {
                    (self.fog_steps.max(1) / 2).max(16)
                } else {
                    self.fog_steps.max(1)
                };
                let fog_params_packed = VolumetricUniformsStd140 {
                    density: self.fog_density.max(0.0),
                    height_falloff: 0.1,
                    phase_g: self.fog_g.clamp(-0.999, 0.999),
                    max_steps: steps,
                    start_distance: 0.1,
                    max_distance: self.view_config.zfar,
                    _pad_a0: 0.0,
                    _pad_a1: 0.0,
                    scattering_color: [1.0, 1.0, 1.0],
                    absorption: 1.0,
                    sun_direction: [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
                    sun_intensity: self.sky_sun_intensity.max(0.0),
                    ambient_color: [0.2, 0.25, 0.3],
                    temporal_alpha: self.fog_temporal_alpha.clamp(0.0, 0.9),
                    use_shadows: if self.fog_use_shadows { 1 } else { 0 },
                    jitter_strength: 0.8,
                    frame_index: self.fog_frame_index,
                    _pad0: 0,
                };
                self.queue.write_buffer(
                    &self.fog_params,
                    0,
                    bytemuck::bytes_of(&fog_params_packed),
                );

                // Select a light-space matrix for fog shadows. Use the
                // first CSM cascade's light_projection if available; otherwise
                // fall back to identity. Depth still comes from fog_shadow_map,
                // which is initialized to depth=1.0 until a real shadow pass
                // is wired in.
                let mut fog_shadow_mat = Mat4::IDENTITY;
                if self.fog_use_shadows {
                    if let Some(ref csm) = self.csm {
                        let cascades = csm.cascades();
                        if let Some(c0) = cascades.get(0) {
                            fog_shadow_mat = Mat4::from_cols_array_2d(&c0.light_projection);
                        }
                    }
                }
                self.queue.write_buffer(
                    &self.fog_shadow_matrix,
                    0,
                    bytemuck::bytes_of(&to_arr(fog_shadow_mat)),
                );

                // Bind groups (shared among both modes)
                let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg0"),
                    layout: &self.fog_bgl0,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.fog_params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.fog_camera.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.fog_depth_sampler),
                        },
                    ],
                });
                let (shadow_tex_view, shadow_uniform_buf) = if let Some(ref csm) = self.csm {
                    (csm.shadow_array_view(), csm.uniform_buffer())
                } else {
                    (&self.fog_shadow_view, &self.fog_shadow_matrix)
                };
                let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg1"),
                    layout: &self.fog_bgl1,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(shadow_tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.fog_shadow_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: shadow_uniform_buf.as_entire_binding(),
                        },
                    ],
                });
                let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.fog.bg2"),
                    layout: &self.fog_bgl2,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.fog_output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.fog_history_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.fog_history_sampler),
                        },
                    ],
                });
                if matches!(self.fog_mode, FogMode::Raymarch) {
                    if self.fog_half_res_enabled {
                        // Half-resolution path: bind half-res output/history
                        let bg2_half = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("viewer.fog.bg2.half"),
                            layout: &self.fog_bgl2,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_output_half_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_history_half_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.fog_history_sampler,
                                    ),
                                },
                            ],
                        });
                        let gx = ((self.config.width / 2) + 7) / 8;
                        let gy = ((self.config.height / 2) + 7) / 8;
                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("viewer.fog.raymarch.half"),
                                    timestamp_writes: None,
                                });
                            cpass.set_pipeline(&self.fog_pipeline);
                            cpass.set_bind_group(0, &bg0, &[]);
                            cpass.set_bind_group(1, &bg1, &[]);
                            cpass.set_bind_group(2, &bg2_half, &[]);
                            cpass.dispatch_workgroups(gx, gy, 1);
                        }
                        // Copy half output to half history
                        encoder.copy_texture_to_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_output_half,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_history_half,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: self.config.width.max(1) / 2,
                                height: self.config.height.max(1) / 2,
                                depth_or_array_layers: 1,
                            },
                        );
                        // Upsample to full-res for composition
                        let upsampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                            label: Some("viewer.fog.upsampler"),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            mipmap_filter: wgpu::FilterMode::Nearest,
                            ..Default::default()
                        });
                        let params = FogUpsampleParamsStd140 {
                            sigma: self.fog_upsigma.max(0.0),
                            use_bilateral: if self.fog_bilateral { 1 } else { 0 },
                            _pad: [0.0; 2],
                        };
                        self.queue.write_buffer(
                            &self.fog_upsample_params,
                            0,
                            bytemuck::bytes_of(&params),
                        );
                        let up_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("viewer.fog.upsample.bg"),
                            layout: &self.fog_upsample_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_output_half_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&upsampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &self.fog_output_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &gi.gbuffer().depth_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.fog_depth_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: self.fog_upsample_params.as_entire_binding(),
                                },
                            ],
                        });
                        let ugx = (self.config.width + 7) / 8;
                        let ugy = (self.config.height + 7) / 8;
                        let mut up_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("viewer.fog.upsample"),
                                timestamp_writes: None,
                            });
                        up_pass.set_pipeline(&self.fog_upsample_pipeline);
                        up_pass.set_bind_group(0, &up_bg, &[]);
                        up_pass.dispatch_workgroups(ugx, ugy, 1);
                    } else {
                        // Full-resolution path (original)
                        let gx = (self.config.width + 7) / 8;
                        let gy = (self.config.height + 7) / 8;
                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("viewer.fog.raymarch"),
                                    timestamp_writes: None,
                                });
                            cpass.set_pipeline(&self.fog_pipeline);
                            cpass.set_bind_group(0, &bg0, &[]);
                            cpass.set_bind_group(1, &bg1, &[]);
                            cpass.set_bind_group(2, &bg2, &[]);
                            cpass.dispatch_workgroups(gx, gy, 1);
                        }
                        // Copy output to full-res history
                        encoder.copy_texture_to_texture(
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_output,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::ImageCopyTexture {
                                texture: &self.fog_history,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: self.config.width,
                                height: self.config.height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                } else {
                    // Froxel build then apply
                    let bg3 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg3"),
                        layout: &self.fog_bgl3,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                            }, // storage view
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.froxel_sampler),
                            },
                        ],
                    });
                    // Build froxels: workgroup_size(4,4,4) over 16x8x64
                    let gx3d = (16u32 + 3) / 4;
                    let gy3d = (8u32 + 3) / 4;
                    let gz3d = (64u32 + 3) / 4;
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("viewer.fog.froxel.build"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.froxel_build_pipeline);
                        pass.set_bind_group(0, &bg0, &[]);
                        pass.set_bind_group(1, &bg1, &[]);
                        pass.set_bind_group(3, &bg3, &[]);
                        pass.dispatch_workgroups(gx3d, gy3d, gz3d);
                    }
                    // Apply froxels: workgroup_size(8,8,1) across viewport
                    let gx2d = (self.config.width + 7) / 8;
                    let gy2d = (self.config.height + 7) / 8;
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("viewer.fog.froxel.apply"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.froxel_apply_pipeline);
                        pass.set_bind_group(0, &bg0, &[]);
                        pass.set_bind_group(2, &bg2, &[]);
                        pass.set_bind_group(3, &bg3, &[]);
                        pass.dispatch_workgroups(gx2d, gy2d, 1);
                    }
                    // For froxels, history is full-res; copy as before
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: &self.fog_output,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: &self.fog_history,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: self.config.width,
                            height: self.config.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                self.fog_frame_index = self.fog_frame_index.wrapping_add(1);
            }

            self.render_postfx_stage(&mut gi, encoder, &zv);
            self.gi = Some(gi);
            self.z_view = Some(zv);
        }

        RenderAvailability {
            have_gi,
            have_pipe,
            have_cam,
            have_vb,
            have_z,
            have_bgl,
        }
    }
}
