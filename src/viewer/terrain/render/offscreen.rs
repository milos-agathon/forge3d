use super::*;
use crate::viewer::terrain::dof;
use crate::viewer::terrain::vector_overlay;

impl ViewerTerrainScene {
    pub fn render_to_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        selected_feature_id: u32,
    ) -> Option<wgpu::Texture> {
        eprintln!("[DEBUG render_to_texture ENTRY] {}x{}", width, height);
        if self.terrain.is_none() {
            eprintln!("[DEBUG render_to_texture] No terrain, returning None");
            return None;
        }

        // Ensure critical resources are initialized (needed when called without prior interactive rendering)
        self.ensure_fallback_texture();

        // Ensure shared depth resources exist (used by compute shaders and PBR)
        // Note: render_to_texture creates its own depth texture, but some systems need
        // the shared depth_view to exist even if not directly used
        if self.depth_view.is_none() {
            self.ensure_depth(width, height);
        }

        // Create offscreen color texture at requested resolution
        // Include TEXTURE_BINDING for DoF shader to sample from
        let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.snapshot_color"),
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
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture at requested resolution
        let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.snapshot_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Calculate all the same values as render() but with custom dimensions
        // Ensure PBR pipeline is initialized if PBR mode is enabled
        if self.pbr_config.enabled && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(target_format) {
                eprintln!("[snapshot] Failed to initialize PBR pipeline: {}", e);
            }
        }
        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        let terrain = self.terrain.as_ref().unwrap();

        let phi = terrain.cam_phi_deg.to_radians();
        let theta = terrain.cam_theta_deg.to_radians();
        let r = terrain.cam_radius;
        let (tw, th) = terrain.dimensions;
        let terrain_width = tw.max(th) as f32;
        let h_range = terrain.domain.1 - terrain.domain.0;
        let legacy_z_scale = terrain.z_scale * h_range * 1000.0 / terrain_width.max(1.0);
        let shader_z_scale = if use_pbr {
            terrain.z_scale
        } else {
            legacy_z_scale
        };
        let center_y = if use_pbr {
            h_range * terrain.z_scale * 0.5
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
        // Use requested width/height for aspect ratio
        let proj_base = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
        // P1.4: Apply TAA jitter to projection matrix (snapshot path)
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

        let sun_az = terrain.sun_azimuth_deg.to_radians();
        let sun_el = terrain.sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

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

        // Prepare PBR bind group if enabled
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

        let bg = terrain.background_color;
        let _ = terrain; // Release borrow

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
            // DEBUG: Print overlay_params values
            eprintln!("[DEBUG render_to_texture] overlay_params: enabled={}, opacity={}, blend={}, solid={}",
                pbr_uniforms.overlay_params[0],
                pbr_uniforms.overlay_params[1],
                pbr_uniforms.overlay_params[2],
                pbr_uniforms.overlay_params[3]);
            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);

        // Option B: Prepare vector overlay stack if it has visible layers (for snapshot path)
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            let enabled = stack.is_enabled();
            let count = stack.visible_layer_count();
            // println!("[snapshot] Checking overlays: enabled={}, visible={}", enabled, count);
            enabled && count > 0
        } else {
            false
        };

        if has_vector_overlays {
            // Ensure we have a fallback texture for when sun visibility isn't enabled
            if self.fallback_texture.is_none() {
                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("vector_overlay_fallback_texture_snapshot"),
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
                // P0.1/M1: For OIT snapshots, ensure OIT pipelines are initialized
                if !stack.pipelines_ready() || (self.oit_enabled && !stack.oit_pipelines_ready()) {
                    stack.init_pipelines(self.surface_format);
                }

                // Prepare bind group with sun visibility texture or fallback
                // For snapshots, force fallback texture (fully lit) to avoid potential issues with
                // uninitialized or black sun visibility texture which would hide the overlays.
                let texture_view = self.fallback_texture_view.as_ref().unwrap();
                stack.prepare_bind_group(texture_view);
            }
        }

        // Store values needed for vector overlay rendering
        let vo_view_proj = view_proj.to_cols_array_2d();
        let vo_sun_dir = [sun_dir.x, sun_dir.y, sun_dir.z];
        let terrain = self.terrain.as_ref().unwrap();
        let vo_lighting = [
            terrain.sun_intensity,
            terrain.ambient,
            terrain.shadow_intensity,
            terrain_width,
        ];
        let _ = terrain; // Release borrow for render pass

        // Render to offscreen texture
        {
            let terrain = self.terrain.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.snapshot_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
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
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if use_pbr {
                if let Some(ref pbr_bind_group) = self.pbr_bind_group {
                    pass.set_pipeline(self.pbr_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, pbr_bind_group, &[]);
                } else {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &terrain.bind_group, &[]);
                }
            } else {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &terrain.bind_group, &[]);
            }

            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);

            // Option B: Render vector overlays after terrain (snapshot path)
            // P0.1/M1: Use standard alpha blending if OIT is disabled
            if has_vector_overlays && !self.oit_enabled {
                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5]; // Yellow highlight
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

        // P0.1/M1: OIT rendering path for snapshots - render overlays to temporary WBOIT accumulation buffers
        if has_vector_overlays && self.oit_enabled {
            // Ensure OIT compose pipeline is initialized (size-independent resources)
            // This is safe to call from snapshot path as it doesn't corrupt interactive viewer state
            self.init_wboit_pipeline();

            // Create temporary OIT accumulation textures at snapshot resolution
            let oit_color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_oit_color"),
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
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let oit_color_view = oit_color_tex.create_view(&wgpu::TextureViewDescriptor::default());

            let oit_reveal_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_oit_reveal"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let oit_reveal_view =
                oit_reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

            // OIT accumulation pass
            {
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.snapshot_oit_accumulation"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &oit_color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &oit_reveal_view,
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
                        view: &depth_view,
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
                    } else {
                        println!(
                            "[snapshot] OIT skip: pipelines_ready={} bind_group={}",
                            stack.oit_pipelines_ready(),
                            stack.bind_group.is_some()
                        );
                    }
                }
            }

            // Create temporary OIT compose bind group
            // Note: compose pipeline is initialized by render() method when OIT is first enabled
            if let (Some(ref pipeline), Some(ref sampler)) = (
                self.wboit_compose_pipeline.as_ref(),
                self.wboit_sampler.as_ref(),
            ) {
                println!("[snapshot] Compositing OIT result");
                let layout = &pipeline.get_bind_group_layout(0);
                let compose_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("terrain_viewer.snapshot_oit_compose_bind_group"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&oit_color_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&oit_reveal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });

                // OIT compose pass - blend accumulated transparency onto scene
                let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.snapshot_oit_compose"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
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
                compose_pass.set_bind_group(0, &compose_bind_group, &[]);
                compose_pass.draw(0..3, 0..1); // Fullscreen triangle
            }
        }

        let mut out_tex = color_tex;
        let mut out_view = color_view;

        // P1.4: TAA is disabled for render_to_texture snapshots
        // TAA requires temporal history between consecutive frames with small camera deltas.
        // For animation exports, each frame is rendered independently at potentially large
        // camera deltas, so TAA temporal accumulation doesn't apply correctly.
        // TAA is only used for the interactive render() path where frames are consecutive.

        // P5: Apply volumetrics pass if enabled (after main render, before DoF)
        let needs_volumetrics =
            self.pbr_config.volumetrics.enabled && self.pbr_config.volumetrics.density > 0.0001;
        if needs_volumetrics {
            // Initialize volumetrics pass if needed
            if self.volumetrics_pass.is_none() {
                self.init_volumetrics_pass();
            }

            if let Some(ref vol_pass) = self.volumetrics_pass {
                // Create output texture for volumetrics
                let vol_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.snapshot_vol_output"),
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
                let vol_output_view =
                    vol_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

                // Calculate inverse view-projection matrix
                let inv_view_proj = (proj * view_mat).inverse();
                let terrain = self.terrain.as_ref().unwrap();
                let cam_radius = terrain.cam_radius;
                let terrain_sun_intensity = terrain.sun_intensity;

                vol_pass.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    &depth_view,
                    &terrain.heightmap_view,
                    &vol_output_view,
                    width,
                    height,
                    inv_view_proj.to_cols_array_2d(),
                    [eye.x, eye.y, eye.z],
                    1.0,               // near
                    cam_radius * 10.0, // far
                    [sun_dir.x, sun_dir.y, sun_dir.z],
                    terrain_sun_intensity,
                    [terrain_width, terrain.domain.0, shader_z_scale, h_range],
                    &self.pbr_config.volumetrics,
                );

                out_tex = vol_output_tex;
                out_view = vol_output_view;
            }
        }

        // Apply DoF if enabled
        let needs_dof = self.pbr_config.dof.enabled;
        if needs_dof {
            // Initialize DoF pass if needed
            if self.dof_pass.is_none() {
                self.init_dof_pass();
            }

            if let Some(ref mut dof) = self.dof_pass {
                // Get DoF input view (allocates textures if needed)
                let _ = dof.get_input_view(width, height, target_format);

                // We need to copy the rendered scene to DoF input first
                // Then apply DoF from input to a new output texture
                let dof_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.snapshot_dof_output"),
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
                let dof_output_view =
                    dof_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

                let cam_radius = self
                    .terrain
                    .as_ref()
                    .map(|t| t.cam_radius)
                    .unwrap_or(2000.0);

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

                // Apply DoF: reads from color_view, writes to dof_output_view
                dof.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    &depth_view,
                    &dof_output_view,
                    width,
                    height,
                    target_format,
                    &dof_cfg,
                    1.0,
                    cam_radius * 10.0,
                );

                out_tex = dof_output_tex;
                out_view = dof_output_view;
            }
        }

        // Apply lens effects post-process in snapshot path to match on-screen output
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);
        if needs_post_process {
            if self.post_process.is_none() {
                self.init_post_process();
            }

            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;
                let lens_output_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.snapshot_lens_output"),
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
                let lens_output_view =
                    lens_output_tex.create_view(&wgpu::TextureViewDescriptor::default());

                pp.apply_from_input(
                    encoder,
                    &self.queue,
                    &out_view,
                    &lens_output_view,
                    width,
                    height,
                    lens.distortion,
                    lens.chromatic_aberration,
                    lens.vignette_strength,
                    lens.vignette_radius,
                    lens.vignette_softness,
                );
                return Some(lens_output_tex);
            }
        }

        Some(out_tex)
    }
}
