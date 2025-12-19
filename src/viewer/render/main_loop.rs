// src/viewer/render/main_loop.rs
// Main render loop for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring
//
// NOTE: This file exceeds the 300 LOC target due to the complexity of the render loop.
// Further refactoring would require significant architectural changes.

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::cli::gi_types::GiVizMode;
use crate::viewer::viewer_enums::{CaptureKind, FogMode, VizMode};
use crate::viewer::{
    hud_push_number, hud_push_text_3x5, CameraFrustum, FogCameraUniforms, FogUpsampleParamsStd140,
    SkyUniforms, Viewer, VolumetricUniformsStd140, VIEWER_SNAPSHOT_MAX_MEGAPIXELS,
};

impl Viewer {
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if self.frame_count == 0 {
            eprintln!("[viewer-debug] entering render loop (first frame)");
        }

        // Ensure auto-snapshot request is registered before encoding so we render to an offscreen texture
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
                eprintln!("[viewer-debug] auto snapshot requested: {}", p);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render sky background (compute) before opaques
        if self.sky_enabled {
            // Build camera matrices (view, proj, inv_view, inv_proj) and eye
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
            let view_mat = self.camera.view_matrix();
            let inv_view = view_mat.inverse();
            let inv_proj = proj.inverse();
            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let eye = self.camera.eye();
            let cam_buf: [[[f32; 4]; 4]; 4] = [
                to_arr4(view_mat),
                to_arr4(proj),
                to_arr4(inv_view),
                to_arr4(inv_proj),
            ];
            // Write matrices
            self.queue
                .write_buffer(&self.sky_camera, 0, bytemuck::cast_slice(&cam_buf));
            // Write eye position (vec4 packed)
            let eye4: [f32; 4] = [eye.x, eye.y, eye.z, 0.0];
            let base = (std::mem::size_of::<[[f32; 4]; 4]>() * 4) as u64;
            self.queue
                .write_buffer(&self.sky_camera, base, bytemuck::cast_slice(&eye4));

            // Update sky params each frame based on viewer-set fields
            let sun_dir_vs = glam::Vec3::new(0.3, 0.6, -1.0).normalize();
            let sun_dir_ws = (inv_view
                * glam::Vec4::new(sun_dir_vs.x, sun_dir_vs.y, sun_dir_vs.z, 0.0))
            .truncate()
            .normalize();
            let model_id: u32 = self.sky_model_id;
            let turb: f32 = self.sky_turbidity.clamp(1.0, 10.0);
            let ground: f32 = self.sky_ground_albedo.clamp(0.0, 1.0);
            let expose: f32 = self.sky_exposure.max(0.0);
            let sun_i: f32 = self.sky_sun_intensity.max(0.0);

            let sky_params_frame = SkyUniforms {
                sun_direction: [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
                turbidity: turb,
                ground_albedo: ground,
                model: model_id,
                sun_intensity: sun_i,
                exposure: expose,
                _pad: [0.0; 4],
            };
            self.queue
                .write_buffer(&self.sky_params, 0, bytemuck::bytes_of(&sky_params_frame));

            // Bind and dispatch compute
            let sky_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.sky.bg0"),
                layout: &self.sky_bind_group_layout0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.sky_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                    },
                ],
            });
            let sky_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("viewer.sky.bg1"),
                layout: &self.sky_bind_group_layout1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sky_camera.as_entire_binding(),
                }],
            });
            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.sky.compute"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sky_pipeline);
                cpass.set_bind_group(0, &sky_bg0, &[]);
                cpass.set_bind_group(1, &sky_bg1, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
        }

        // Composite debug: after GI/geometry, show GBuffer material on swapchain

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
        if let (Some(gi), Some(pipe), Some(cam_buf), Some(vb), Some(zv), Some(_bgl)) = (
            self.gi.as_mut(),
            self.geom_pipeline.as_ref(),
            self.geom_camera_buffer.as_ref(),
            self.geom_vb.as_ref(),
            self.z_view.as_ref(),
            self.geom_bind_group_layout.as_ref(),
        ) {
            // Update geometry camera uniform (view, proj)
            let aspect = self.config.width as f32 / self.config.height as f32;
            let fov = self.view_config.fov_deg.to_radians();
            let proj =
                Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
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
            let cam = crate::core::screen_space_effects::CameraParams {
                view_matrix: to_arr4(model_view),
                inv_view_matrix: to_arr4(inv_model_view),
                proj_matrix: to_arr4(proj),
                inv_proj_matrix: to_arr4(inv_proj),
                camera_pos: [eye.x, eye.y, eye.z],
                _pad: 0.0,
            };
            gi.update_camera(&self.queue, &cam);

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
                                let csm_bg = self
                                    .device
                                    .create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("viewer.csm.depth.bg"),
                                        layout: &bgl,
                                        entries: &[wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: csm_cam_buf.as_entire_binding(),
                                        }],
                                    });
                                let mut shadow_pass = encoder.begin_render_pass(
                                    &wgpu::RenderPassDescriptor {
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
                                    },
                                );
                                shadow_pass.set_pipeline(csm_pipe);
                                shadow_pass.set_bind_group(0, &csm_bg, &[]);
                                shadow_pass.set_vertex_buffer(0, vb.slice(..));
                                if let Some(ib) = self.geom_ib.as_ref() {
                                    shadow_pass.set_index_buffer(
                                        ib.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    shadow_pass.draw_indexed(
                                        0..self.geom_index_count,
                                        0,
                                        0..1,
                                    );
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

                // Select a light-space matrix for fog shadows. For now, use the
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
            gi.build_hzb(&self.device, &mut encoder, zv, false);
            // Execute effects
            let _ = gi.execute(&self.device, &mut encoder, None, None);

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
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().normal_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().material_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().depth_view,
                                ),
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
                    VizMode::Lit => (0u32, &self.lit_output_view),
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
                // Sampler: non-filtering so we can bind depth/non-filterable textures safely
                let comp_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("viewer.comp.sampler"),
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                });
                let comp_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.comp.bg"),
                    layout: comp_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&comp_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(if self.fog_enabled {
                                &self.fog_output_view
                            } else {
                                &self.fog_zero_view
                            }),
                        },
                    ],
                });
                // If a snapshot is requested, render the composite to an offscreen texture too
                if self.snapshot_request.is_some() {
                    let (mut snap_w, mut snap_h) =
                        if let (Some(w), Some(h)) =
                            (self.view_config.snapshot_width, self.view_config.snapshot_height)
                        {
                            (w, h)
                        } else {
                            (self.config.width, self.config.height)
                        };

                    // Apply a soft megapixel clamp only when the user has requested
                    // an explicit override resolution via ViewerConfig.
                    if self.view_config.snapshot_width.is_some()
                        && self.view_config.snapshot_height.is_some()
                    {
                        let pixels = snap_w as u64 * snap_h as u64;
                        let max_pixels = (VIEWER_SNAPSHOT_MAX_MEGAPIXELS * 1_000_000.0) as u64;
                        if pixels > max_pixels {
                            let scale = (max_pixels as f32 / pixels as f32).sqrt();
                            snap_w = ((snap_w as f32) * scale).floor().max(1.0) as u32;
                            snap_h = ((snap_h as f32) * scale).floor().max(1.0) as u32;
                        }
                    }

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
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut pass_snap = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.comp.pass.snapshot"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &snap_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
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
                    drop(pass_snap);
                    // Store to be read back after submit
                    self.pending_snapshot_tex = Some(snap_tex);
                }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.comp.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(comp_pl);
                pass.set_bind_group(0, &comp_bg, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);

                if self.hud_enabled {
                    // HUD overlay after composite
                    // Build simple bars for sky/fog settings + numeric readouts
                    let mut hud_instances: Vec<crate::core::text_overlay::TextInstance> =
                        Vec::new();
                    let sx = 8.0f32;
                    let sy = 8.0f32; // start position
                    let bar_w = 120.0f32;
                    let bar_h = 10.0f32;
                    let gap = 4.0f32;
                    let num_scale = 0.6f32; // ~11px tall digits
                    let num_dx = 8.0f32; // spacing from end of bar
                    let mut y = sy;
                    // Sky enabled bar (green if on, gray if off)
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "SKY",
                        [0.8, 0.95, 0.8, 0.9],
                    );
                    let sky_on = if self.sky_enabled { 1.0 } else { 0.25 };
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.2, 0.8, 0.2, sky_on],
                    });
                    // Label model (0=Preetham,1=Hosek)
                    let model_val = if self.sky_model_id == 0 { 0.0 } else { 1.0 };
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0; // slightly above bar baseline
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        model_val,
                        1,
                        0,
                        [0.7, 0.9, 0.7, 0.9],
                    );
                    y += bar_h + gap;
                    // Sky turbidity bar length + number
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "TURB",
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    let tfrac = (self.sky_turbidity.clamp(1.0, 10.0) - 1.0) / 9.0;
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w * tfrac, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.2, 0.5, 1.0, 0.8],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        self.sky_turbidity,
                        4,
                        1,
                        [0.6, 0.8, 1.0, 0.9],
                    );
                    y += bar_h + gap;
                    // Fog enabled bar (blue if on)
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "FOG",
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    let fog_on = if self.fog_enabled { 0.9 } else { 0.2 };
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.2, 0.6, 1.0, fog_on],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        if self.fog_enabled { 1.0 } else { 0.0 },
                        1,
                        0,
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    y += bar_h + gap;
                    // Fog density bar + number
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "DENS",
                        [0.7, 0.85, 1.0, 0.9],
                    );
                    let dfrac = (self.fog_density / 0.1).clamp(0.0, 1.0);
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w * dfrac, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [0.6, 0.8, 1.0, 0.8],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        self.fog_density,
                        5,
                        3,
                        [0.6, 0.8, 1.0, 0.9],
                    );
                    y += bar_h + gap;
                    // Fog temporal alpha bar + number
                    hud_push_text_3x5(
                        &mut hud_instances,
                        sx,
                        y - 9.0,
                        1.0,
                        "TEMP",
                        [1.0, 0.85, 0.6, 0.95],
                    );
                    let afrac = self.fog_temporal_alpha.clamp(0.0, 0.9) / 0.9;
                    hud_instances.push(crate::core::text_overlay::TextInstance {
                        rect_min: [sx, y],
                        rect_max: [sx + bar_w * afrac, y + bar_h],
                        uv_min: [0.0, 0.0],
                        uv_max: [1.0, 1.0],
                        color: [1.0, 0.6, 0.2, 0.8],
                    });
                    let nx = sx + bar_w + num_dx;
                    let ny = y - 1.0;
                    hud_push_number(
                        &mut hud_instances,
                        nx,
                        ny,
                        num_scale,
                        self.fog_temporal_alpha,
                        4,
                        2,
                        [1.0, 0.8, 0.5, 0.95],
                    );

                    self.hud
                        .upload_instances(&self.device, &self.queue, &hud_instances);
                    self.hud.upload_uniforms(&self.queue);
                    // Render overlay
                    let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.hud.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
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
                    self.hud.render(&mut overlay_pass);
                    drop(overlay_pass);
                }
            }
        }

        // If we didn't composite anything (GI path unavailable), either let an attached
        // TerrainScene render, or fall back to the purple debug pipeline.
        // Helper closure to render fallback with optional snapshot texture
        let render_fallback = |encoder: &mut wgpu::CommandEncoder,
                               view: &wgpu::TextureView,
                               pipeline: &wgpu::RenderPipeline,
                               device: &wgpu::Device,
                               config: &wgpu::SurfaceConfiguration,
                               snapshot_request: &Option<String>,
                               view_config: &crate::viewer::viewer_config::ViewerConfig|
         -> Option<wgpu::Texture> {
            // If snapshot requested, create offscreen texture at requested size
            let snap_tex = if snapshot_request.is_some() {
                let (snap_w, snap_h) = if let (Some(w), Some(h)) =
                    (view_config.snapshot_width, view_config.snapshot_height)
                {
                    (w, h)
                } else {
                    (config.width, config.height)
                };
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("viewer.fallback.snapshot"),
                    size: wgpu::Extent3d {
                        width: snap_w,
                        height: snap_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let snap_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.fallback.pass.snapshot"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &snap_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05,
                                g: 0.0,
                                b: 0.15,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(pipeline);
                pass.draw(0..3, 0..1);
                drop(pass);
                Some(tex)
            } else {
                None
            };
            // Always render to swapchain view too
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.fallback.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.0,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.draw(0..3, 0..1);
            drop(pass);
            snap_tex
        };

        // Standalone terrain viewer (works without extension-module)
        let mut terrain_rendered = false;
        // Check if snapshot requested with custom resolution (for terrain viewer)
        let terrain_snap_size = if self.snapshot_request.is_some() {
            if let (Some(w), Some(h)) = (self.view_config.snapshot_width, self.view_config.snapshot_height) {
                Some((w, h))
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref mut tv) = self.terrain_viewer {
            if tv.has_terrain() {
                // Always render to screen first at window resolution
                terrain_rendered = tv.render(&mut encoder, &view, self.config.width, self.config.height);

                // Then render to offscreen texture at snapshot resolution (if requested)
                // This must be LAST so the uniform buffer has the correct aspect ratio for the snapshot
                if let Some((snap_w, snap_h)) = terrain_snap_size {
                    println!("[terrain] Rendering snapshot at {}x{}", snap_w, snap_h);
                    if let Some(tex) = tv.render_to_texture(&mut encoder, self.config.format, snap_w, snap_h) {
                        self.pending_snapshot_tex = Some(tex);
                    }
                }
            }
        }

        #[cfg(feature = "extension-module")]
        if !terrain_rendered {
            if let Some(ref mut scene) = self.terrain_scene {
                if scene.has_viewer_terrain() {
                    terrain_rendered = scene.render_viewer_terrain(
                        &mut encoder,
                        &view,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                    );
                }
            }
        }

        if !terrain_rendered && !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            if let Some(tex) = render_fallback(
                &mut encoder,
                &view,
                &self.fallback_pipeline,
                &self.device,
                &self.config,
                &self.snapshot_request,
                &self.view_config,
            ) {
                self.pending_snapshot_tex = Some(tex);
            }
        }

        #[cfg(any())] // Dead code - kept for reference
        if !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl) {
            if let Some(tex) = render_fallback(
                &mut encoder,
                &view,
                &self.fallback_pipeline,
                &self.device,
                &self.config,
                &self.snapshot_request,
                &self.view_config,
            ) {
                self.pending_snapshot_tex = Some(tex);
            }
        }

        // Submit rendering
        self.queue.submit(std::iter::once(encoder.finish()));

        // Auto-snapshot once if env var is set and we haven't requested yet
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
            }
        }

        // Snapshot if requested (read back the swapchain texture before present)
        if let Some(path) = self.snapshot_request.take() {
            // Prefer offscreen snapshot texture if we rendered one; otherwise fallback to surface texture
            if let Some(tex) = self.pending_snapshot_tex.take() {
                let dims = tex.size();
                println!("[snapshot] Using offscreen tex {}x{}", dims.width, dims.height);
                if let Err(e) = self.snapshot_swapchain_to_png(&tex, &path) {
                    eprintln!("Snapshot failed: {}", e);
                } else {
                    println!("Saved snapshot to {}", path);
                }
            } else {
                println!("[snapshot] Using swapchain texture (no offscreen)");
                if let Err(e) = self.snapshot_swapchain_to_png(&output.texture, &path) {
                    eprintln!("Snapshot failed: {}", e);
                } else {
                    println!("Saved snapshot to {}", path);
                }
            }
        }
        output.present();

        // Optionally dump P5 artifacts after finishing all passes
        if self.dump_p5_requested {
            if let Err(e) = self.dump_gbuffer_artifacts() {
                eprintln!("[P5] dump failed: {}", e);
            }
            self.dump_p5_requested = false;
        }
        self.frame_count += 1;
        if let Some(fps) = self.fps_counter.tick() {
            let viz = match self.viz_mode {
                VizMode::Material => "material",
                VizMode::Normal => "normal",
                VizMode::Depth => "depth",
                VizMode::Gi => "gi",
                VizMode::Lit => "lit",
            };
            self.window.set_title(&format!(
                "{} | FPS: {:.1} | Mode: {:?} | Viz: {}",
                self.view_config.title,
                fps,
                self.camera.mode(),
                viz
            ));
        }

        // Process any pending P5 capture requests using current frame data.
        // Handle all queued captures before the viewer exits so that scripts
        // like p5_golden.sh (which enqueue multiple :p5 commands followed by
        // :quit) still produce every expected artifact.
        while let Some(kind) = self.pending_captures.pop_front() {
            match kind {
                CaptureKind::P51CornellSplit => {
                    if let Err(e) = self.capture_p51_cornell_with_scene() {
                        eprintln!("[P5.1] Cornell split failed: {}", e);
                    }
                }
                CaptureKind::P51AoGrid => {
                    if let Err(e) = self.capture_p51_ao_grid() {
                        eprintln!("[P5.1] AO grid failed: {}", e);
                    }
                }
                CaptureKind::P51ParamSweep => {
                    if let Err(e) = self.capture_p51_param_sweep() {
                        eprintln!("[P5.1] AO sweep failed: {}", e);
                    }
                }
                CaptureKind::P52SsgiCornell => {
                    if let Err(e) = self.capture_p52_ssgi_cornell() {
                        eprintln!("[P5.2] SSGI Cornell capture failed: {}", e);
                    }
                }
                CaptureKind::P52SsgiTemporal => {
                    if let Err(e) = self.capture_p52_ssgi_temporal() {
                        eprintln!("[P5.2] SSGI temporal compare failed: {}", e);
                    }
                }
                CaptureKind::P53SsrGlossy => {
                    if let Err(e) = self.capture_p53_ssr_glossy() {
                        eprintln!("[P5.3] SSR glossy capture failed: {}", e);
                    }
                }
                CaptureKind::P53SsrThickness => {
                    if let Err(e) = self.capture_p53_ssr_thickness_ablation() {
                        eprintln!("[P5.3] SSR thickness capture failed: {}", e);
                    }
                }
                CaptureKind::P54GiStack => {
                    if let Err(e) = self.capture_p54_gi_stack_ablation() {
                        eprintln!("[P5.4] GI stack ablation capture failed: {}", e);
                    }
                }
            }
        }

        Ok(())
    }
}
