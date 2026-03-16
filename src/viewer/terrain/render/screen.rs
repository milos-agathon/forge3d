use super::*;
use crate::viewer::terrain::dof;
use crate::viewer::terrain::post_process::PostProcessPass;
use crate::viewer::terrain::vector_overlay;

impl ViewerTerrainScene {
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        selected_feature_id: u32,
    ) -> bool {
        if self.terrain.is_none() {
            return false;
        }

        self.ensure_depth(width, height);

        // Initialize PBR pipeline if enabled but not yet created
        // This ensures overlays work in interactive mode, not just snapshots
        if self.pbr_config.enabled && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(self.surface_format) {
                eprintln!("[render] Failed to initialize PBR pipeline: {}", e);
            }
        }

        // Pre-compute values needed for PBR bind group before borrowing terrain
        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();

        // Check if DoF is enabled
        let needs_dof = self.pbr_config.dof.enabled;

        // Check if post-process effects need a separate pass (distortion, CA, or vignette)
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);

        // P5: Volumetrics pass
        let needs_volumetrics =
            self.pbr_config.volumetrics.enabled && self.pbr_config.volumetrics.density > 0.0001;

        // M5: Denoise pass
        let needs_denoise = self.pbr_config.denoise.enabled;

        // We need PostProcess pass (for intermediate texture) if PP is active OR Volumetrics is active (since Vol reads from PP intermediate)
        if (needs_post_process || needs_volumetrics) && self.post_process.is_none() {
            self.init_post_process();
        }

        // We need DoF pass if DoF is active OR (Volumetrics AND PP are active) - for scratch buffer
        // Case: Vol+PP (no DoF) needs a scratch buffer for Vol output before PP reads it. We reuse DoF input buffer.
        let needs_dof_scratch = needs_volumetrics && needs_post_process && !needs_dof;

        if (needs_dof || needs_dof_scratch) && self.dof_pass.is_none() {
            self.init_dof_pass();
        }

        // Initialize volumetrics pass if needed
        if needs_volumetrics && self.volumetrics_pass.is_none() {
            self.init_volumetrics_pass();
        }

        if needs_denoise && self.denoise_pass.is_none() {
            self.init_denoise_pass();
        }

        // Ensure textures exist for active passes
        if needs_denoise {
            if let Some(ref mut denoise) = self.denoise_pass {
                // Denoise pass manages its own ping-pong resources
                let _ = denoise.get_input_view(width, height);
            }
        }
        if needs_dof || needs_dof_scratch {
            if let Some(ref mut dof) = self.dof_pass {
                let _ = dof.get_input_view(width, height, self.surface_format);
            }
        }

        // P0.1/M1: Check if OIT is needed and initialize WBOIT resources early
        let has_vector_overlays_early = self
            .vector_overlay_stack
            .as_ref()
            .map(|s| s.visible_layer_count() > 0)
            .unwrap_or(false);
        if has_vector_overlays_early && self.oit_enabled {
            // Ensure WBOIT is fully initialized including textures and bind groups
            if self.wboit_compose_bind_group.is_none() || self.wboit_size != (width, height) {
                self.init_wboit(width, height);
            }
        }

        if needs_post_process || needs_volumetrics {
            if let Some(ref mut pp) = self.post_process {
                let _ = pp.get_intermediate_view(width, height, self.surface_format);
            }
        }

        // Extract all terrain values we need before any mutable operations
        let (
            phi,
            theta,
            r,
            tw,
            th,
            terrain_z_scale,
            domain,
            fov_deg,
            sun_azimuth_deg,
            sun_elevation_deg,
        ) = {
            let terrain = self.terrain.as_ref().unwrap();
            (
                terrain.cam_phi_deg.to_radians(),
                terrain.cam_theta_deg.to_radians(),
                terrain.cam_radius,
                terrain.dimensions.0,
                terrain.dimensions.1,
                terrain.z_scale,
                terrain.domain,
                terrain.cam_fov_deg,
                terrain.sun_azimuth_deg,
                terrain.sun_elevation_deg,
            )
        };

        let terrain_width = tw.max(th) as f32;
        let h_range = domain.1 - domain.0;
        let legacy_z_scale = terrain_z_scale * h_range * 1000.0 / terrain_width.max(1.0);
        let shader_z_scale = if use_pbr {
            terrain_z_scale
        } else {
            legacy_z_scale
        };
        let center_y = if use_pbr {
            h_range * terrain_z_scale * 0.5
        } else {
            terrain_width * legacy_z_scale * 0.001 * 0.5
        };
        let center = glam::Vec3::new(terrain_width * 0.5, center_y, terrain_width * 0.5);

        let eye = glam::Vec3::new(
            center.x + r * theta.sin() * phi.cos(),
            center.y + r * theta.cos(),
            center.z + r * theta.sin() * phi.sin(),
        );

        let view_mat = glam::Mat4::look_at_rh(eye, center, glam::Vec3::Y);
        let proj_base = glam::Mat4::perspective_rh(
            fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
        // P1.4: Apply TAA jitter to projection matrix
        let proj = if self.taa_jitter.enabled {
            crate::core::jitter::apply_jitter(
                proj_base,
                self.taa_jitter.offset.0,
                self.taa_jitter.offset.1,
                width,
                height,
            )
        } else {
            proj_base
        };
        let view_proj = proj * view_mat;

        let sun_az = sun_azimuth_deg.to_radians();
        let sun_el = sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

        // Initialize shadow depth pipeline if PBR mode with shadows enabled
        if use_pbr && self.shadow_pipeline.is_none() {
            self.init_shadow_depth_pipeline();
            self.update_shadow_bind_groups();
        }

        // Render shadow depth passes before main terrain render
        if use_pbr && self.shadow_pipeline.is_some() {
            self.render_shadow_passes(encoder, view_mat, proj, -sun_dir);
        } else if use_pbr {
            eprintln!(
                "[render] Skipping shadow passes: pipeline={}",
                self.shadow_pipeline.is_some()
            );
        }

        // Debug: print uniform values on first render
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            println!(
                "[render] terrain_params: min_h={:.1}, h_range={:.1}, width={:.1}, z_scale={:.2}",
                domain.0, h_range, terrain_width, shader_z_scale
            );
            let max_y = if use_pbr {
                h_range * terrain_z_scale
            } else {
                terrain_width * legacy_z_scale * 0.001
            };
            println!("[render] Expected Y range: 0 to {:.1}", max_y);
            println!(
                "[render] Camera center: ({:.1}, {:.1}, {:.1})",
                center.x, center.y, center.z
            );
        });

        // Re-borrow terrain for the remaining operations
        let terrain = self.terrain.as_ref().unwrap();

        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [
                terrain.domain.0,
                terrain.domain.1 - terrain.domain.0,
                terrain_width,
                shader_z_scale,
            ],
            lighting: [
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
            ],
            background: [
                terrain.background_color[0],
                terrain.background_color[1],
                terrain.background_color[2],
                0.0,
            ],
            water_color: [
                terrain.water_color[0],
                terrain.water_color[1],
                terrain.water_color[2],
                0.0,
            ],
        };

        self.queue.write_buffer(
            &terrain.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Prepare PBR bind group if PBR mode is enabled (must be done before render pass)
        // Extract values needed for PBR uniforms before mutable borrow
        let pbr_uniforms_data = if use_pbr {
            Some((
                terrain.domain,
                terrain.z_scale,
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
                terrain.background_color,
                terrain.water_color,
            ))
        } else {
            None
        };

        // Release the immutable borrow of terrain before mutable operations
        let _ = terrain;

        if let Some((
            domain,
            z_scale,
            sun_intensity,
            ambient,
            shadow_intensity,
            water_level,
            background_color,
            water_color,
        )) = pbr_uniforms_data
        {
            let pbr_uniforms = TerrainPbrUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                terrain_params: [domain.0, domain.1 - domain.0, terrain_width, z_scale],
                lighting: [sun_intensity, ambient, shadow_intensity, water_level],
                background: [
                    background_color[0],
                    background_color[1],
                    background_color[2],
                    0.0,
                ],
                water_color: [water_color[0], water_color[1], water_color[2], 0.0],
                pbr_params: [
                    self.pbr_config.exposure,
                    self.pbr_config.normal_strength,
                    self.pbr_config.ibl_intensity,
                    0.0,
                ],
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                lens_params: [
                    self.pbr_config.lens_effects.vignette_strength,
                    self.pbr_config.lens_effects.vignette_radius,
                    self.pbr_config.lens_effects.vignette_softness,
                    0.0,
                ],
                screen_dims: [width as f32, height as f32, 0.0, 0.0],
                overlay_params: [
                    if self.pbr_config.overlay.enabled {
                        1.0
                    } else {
                        0.0
                    },
                    self.pbr_config.overlay.global_opacity,
                    0.0, // Blend mode: 0 = Normal
                    if self.pbr_config.overlay.solid {
                        1.0
                    } else {
                        0.0
                    },
                ],
            };

            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes for heightfield AO and sun visibility before render
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);

        // Re-borrow terrain after mutable operations
        // Determine primary render target based on active effects
        // Logic:
        // - If Volumetrics active: Scene -> PP Intermediate (so Vol can read it)
        // - Else if DoF active: Scene -> DoF Input
        // - Else if PP active: Scene -> PP Intermediate
        // - Else: Scene -> Final View

        // - If Denoise active: Scene -> Denoise Input
        // - Else if Volumetrics active: Scene -> PP Intermediate (so Vol can read it)
        // - Else if DoF active: Scene -> DoF Input
        // - Else if PP active: Scene -> PP Intermediate
        // - Else: Scene -> Final View

        let render_target: &wgpu::TextureView = if needs_denoise {
            self.denoise_pass
                .as_mut()
                .unwrap()
                .get_input_view(width, height)
        } else if needs_volumetrics {
            self.post_process
                .as_ref()
                .unwrap()
                .intermediate_view
                .as_ref()
                .unwrap()
        } else if needs_dof {
            self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
        } else if needs_post_process {
            self.post_process
                .as_ref()
                .unwrap()
                .intermediate_view
                .as_ref()
                .unwrap()
        } else {
            view
        };

        // Store camera params for DoF
        let cam_radius = self.terrain.as_ref().unwrap().cam_radius;

        let terrain = self.terrain.as_ref().unwrap();
        let depth_view = self.depth_view.as_ref().unwrap();
        let bg = &terrain.background_color;

        // Option B: Prepare vector overlay stack if it has visible layers
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            stack.is_enabled() && stack.visible_layer_count() > 0
        } else {
            false
        };

        if has_vector_overlays {
            // Ensure we have a fallback texture for when sun visibility isn't enabled
            if self.fallback_texture.is_none() {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("vector_overlay_fallback_texture"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                // Initialize to 1.0 (full visibility / no shadow)
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&[1.0f32]),
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
                self.fallback_texture_view =
                    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                self.fallback_texture = Some(texture);
            }

            // Initialize vector overlay pipelines if not yet done
            if let Some(ref mut stack) = self.vector_overlay_stack {
                // P0.1/M1: For OIT mode, ensure OIT pipelines are initialized
                if !stack.pipelines_ready() || (self.oit_enabled && !stack.oit_pipelines_ready()) {
                    stack.init_pipelines(self.surface_format);
                }

                // Prepare bind group with sun visibility texture or fallback
                let texture_view = self
                    .sun_vis_view
                    .as_ref()
                    .or(self.fallback_texture_view.as_ref())
                    .unwrap();
                stack.prepare_bind_group(texture_view);
            }
        }

        // Store values needed for vector overlay rendering
        let vo_view_proj = view_proj.to_cols_array_2d();
        let vo_sun_dir = [sun_dir.x, sun_dir.y, sun_dir.z];
        let vo_lighting = [
            terrain.sun_intensity,
            terrain.ambient,
            terrain.shadow_intensity,
            terrain_width,
        ];
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64,
                            g: bg[1] as f64,
                            b: bg[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Choose pipeline based on PBR mode
            if use_pbr {
                if let Some(ref pbr_bind_group) = self.pbr_bind_group {
                    pass.set_pipeline(self.pbr_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, pbr_bind_group, &[]);
                } else {
                    // Fallback to legacy if bind group not ready
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &terrain.bind_group, &[]);
                }
            } else {
                // Legacy rendering path
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &terrain.bind_group, &[]);
            }

            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);

            // Option B: Render vector overlays after terrain
            // P0.1/M1: Use OIT rendering if enabled, otherwise standard alpha blending
            if has_vector_overlays && !self.oit_enabled {
                // Standard alpha blending path
                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5];
                        for i in 0..layer_count {
                            stack.render_layer_with_highlight(
                                &mut pass,
                                vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: vo_view_proj,
                                    sun_dir: vo_sun_dir,
                                    lighting: vo_lighting,
                                    selected_feature_id,
                                    highlight_color,
                                },
                            );
                        }
                    }
                }
            }
        }

        // P0.1/M1: OIT rendering path - render overlays to WBOIT accumulation buffers
        if has_vector_overlays && self.oit_enabled {
            let depth_view = self.depth_view.as_ref().unwrap();

            // OIT accumulation pass
            if let (Some(color_view), Some(reveal_view)) = (
                self.wboit_color_view.as_ref(),
                self.wboit_reveal_view.as_ref(),
            ) {
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.wboit.accumulation_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve terrain depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.oit_pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5];
                        for i in 0..layer_count {
                            stack.render_layer_oit(
                                &mut oit_pass,
                                vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: vo_view_proj,
                                    sun_dir: vo_sun_dir,
                                    lighting: vo_lighting,
                                    selected_feature_id,
                                    highlight_color,
                                },
                            );
                        }
                    }
                }
            }

            // OIT compose pass - blend accumulated transparency onto scene
            if let (Some(pipeline), Some(bind_group)) = (
                self.wboit_compose_pipeline.as_ref(),
                self.wboit_compose_bind_group.as_ref(),
            ) {
                let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.wboit.compose_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve scene
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                compose_pass.set_pipeline(pipeline);
                compose_pass.set_bind_group(0, bind_group, &[]);
                compose_pass.draw(0..3, 0..1); // Fullscreen triangle
            }

            static OIT_LOG_ONCE: std::sync::Once = std::sync::Once::new();
            OIT_LOG_ONCE.call_once(|| {
                println!("[render] WBOIT active: mode={}", self.oit_mode);
            });
        }

        // M5: Apply Denoise pass if enabled (after scene render, before effects)
        if needs_denoise {
            let (iterations, sigma_color) = {
                let config = &self.pbr_config.denoise;
                (config.iterations, config.sigma_color)
            };

            let ViewerTerrainScene {
                denoise_pass,
                post_process,
                dof_pass,
                depth_view,
                queue,
                device,
                surface_format,
                ..
            } = self;

            if let Some(denoise) = denoise_pass.as_mut() {
                let depth_view = depth_view.as_ref().unwrap();
                denoise.apply(encoder, depth_view, iterations, sigma_color);

                let denoise_result = denoise
                    .get_last_result_view(iterations)
                    .unwrap_or(denoise.view_a.as_ref().unwrap());

                if post_process.is_none() {
                    *post_process = Some(PostProcessPass::new(device.clone(), *surface_format));
                }

                let post_process = post_process.as_mut().unwrap();
                let mut intermediate_view = None;
                let next_target = if needs_volumetrics {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else if needs_dof {
                    dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else if needs_post_process {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else {
                    view
                };

                post_process.apply_from_input(
                    encoder,
                    queue,
                    denoise_result,
                    next_target,
                    width,
                    height,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );

                if let Some(view) = intermediate_view {
                    post_process.intermediate_view = Some(view);
                }
            }
        }

        // P5: Apply volumetrics pass if enabled (after main render, before DoF)
        if needs_volumetrics {
            if let Some(ref vol_pass) = self.volumetrics_pass {
                let depth_view = self.depth_view.as_ref().unwrap();
                let color_input = self
                    .post_process
                    .as_ref()
                    .unwrap()
                    .intermediate_view
                    .as_ref()
                    .unwrap();

                // Determine volumetrics output target with robust chaining:
                // - If DoF enabled: output to DoF input
                // - If DoF disabled but PP enabled: output to DoF input (as scratch buffer)
                // - Otherwise: output to Final View
                let vol_output = if needs_dof || needs_post_process {
                    self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else {
                    view
                };

                // Calculate inverse view-projection matrix
                let inv_view_proj = (proj * view_mat).inverse();

                vol_pass.apply(
                    encoder,
                    &self.queue,
                    color_input,
                    depth_view,
                    &terrain.heightmap_view,
                    vol_output,
                    width,
                    height,
                    inv_view_proj.to_cols_array_2d(),
                    [eye.x, eye.y, eye.z],
                    1.0,               // near
                    cam_radius * 10.0, // far
                    [sun_dir.x, sun_dir.y, sun_dir.z],
                    terrain.sun_intensity,
                    [terrain_width, terrain.domain.0, shader_z_scale, h_range],
                    &self.pbr_config.volumetrics,
                );
            }
        }

        // Apply DoF pass if enabled (before other post-process effects)
        if needs_dof {
            // DoF input source:
            // - If Volumetrics enabled: Read from DoF Input (Volumetrics wrote here)
            // - Else: Read from DoF Input (Scene wrote here)
            // Note: In both cases, data is already in dof.input_view

            // DoF output target:
            // - If PP enabled: Output to PP Intermediate (so PP can read it)
            // - Else: Output to Final View
            let dof_output = if needs_post_process {
                self.post_process
                    .as_ref()
                    .unwrap()
                    .intermediate_view
                    .as_ref()
                    .unwrap()
            } else {
                view
            };

            if let Some(ref mut dof) = self.dof_pass {
                let depth_view = self.depth_view.as_ref().unwrap();

                let dof_cfg = dof::DofConfig {
                    enabled: true,
                    focus_distance: self.pbr_config.dof.focus_distance,
                    f_stop: self.pbr_config.dof.f_stop,
                    focal_length: self.pbr_config.dof.focal_length,
                    quality: self.pbr_config.dof.quality,
                    max_blur_radius: self.pbr_config.dof.max_blur_radius,
                    blur_strength: self.pbr_config.dof.blur_strength,
                    tilt_pitch: self.pbr_config.dof.tilt_pitch,
                    tilt_yaw: self.pbr_config.dof.tilt_yaw,
                };

                // We use apply_from_input to explicit chaining from dof.input_view
                dof.apply_from_input(
                    encoder,
                    &self.queue,
                    depth_view,
                    dof_output,
                    width,
                    height,
                    self.surface_format,
                    &dof_cfg,
                    1.0,               // near plane
                    cam_radius * 10.0, // far plane
                );
            }
        }

        // Apply post-process pass if needed (distortion, CA, vignette)
        if needs_post_process {
            // Determine if we need to read from an external source (scratch buffer)
            // This happens when Volumetrics + PP are active but DoF is not.
            // In that case, Volumetrics wrote to DoF Input (as scratch), so we read from there.
            // In all other cases (including DoF+PP), the input is in our own intermediate_view.
            let external_input = if !needs_dof && needs_volumetrics {
                self.dof_pass
                    .as_ref()
                    .and_then(|dof| dof.input_view.as_ref())
            } else {
                None
            };

            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;

                if let Some(input_view) = external_input {
                    // Read from external texture (DoF scratch)
                    pp.apply_from_input(
                        encoder,
                        &self.queue,
                        input_view,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                } else {
                    // Read from internal intermediate texture (Standard path)
                    pp.apply(
                        encoder,
                        &self.queue,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                }
            }
        }

        true
    }
}
