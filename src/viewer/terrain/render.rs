// src/viewer/terrain/render.rs
// Terrain rendering for the interactive viewer

use super::scene::ViewerTerrainScene;

/// Terrain uniforms for the simple terrain shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TerrainUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub terrain_params: [f32; 4],
    pub lighting: [f32; 4],
    pub background: [f32; 4],
    pub water_color: [f32; 4],
}

/// Extended uniforms for PBR terrain shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TerrainPbrUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub sun_dir: [f32; 4],
    pub terrain_params: [f32; 4],
    pub lighting: [f32; 4],
    pub background: [f32; 4],
    pub water_color: [f32; 4],
    pub pbr_params: [f32; 4],    // exposure, normal_strength, ibl_intensity, _
    pub camera_pos: [f32; 4],   // camera world position
}

impl ViewerTerrainScene {
    pub(super) fn ensure_depth(&mut self, width: u32, height: u32) {
        if self.depth_size != (width, height) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain_viewer.depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.depth_texture = Some(tex);
            self.depth_size = (width, height);
        }
    }

    /// Render terrain to the given view
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> bool {
        if self.terrain.is_none() {
            return false;
        }

        self.ensure_depth(width, height);
        
        // Pre-compute values needed for PBR bind group before borrowing terrain
        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        
        let terrain = self.terrain.as_ref().unwrap();

        let phi = terrain.cam_phi_deg.to_radians();
        let theta = terrain.cam_theta_deg.to_radians();
        let r = terrain.cam_radius;
        let (tw, th) = terrain.dimensions;
        let terrain_width = tw.max(th) as f32;
        let h_range = terrain.domain.1 - terrain.domain.0;
        let legacy_z_scale = terrain.z_scale * h_range * 1000.0 / terrain_width.max(1.0);
        let shader_z_scale = if use_pbr { terrain.z_scale } else { legacy_z_scale };
        let center_y = if use_pbr {
            h_range * terrain.z_scale * 0.5
        } else {
            terrain_width * legacy_z_scale * 0.001 * 0.5
        };
        let center = glam::Vec3::new(
            terrain_width * 0.5,
            center_y,
            terrain_width * 0.5,
        );

        let eye = glam::Vec3::new(
            center.x + r * theta.sin() * phi.cos(),
            center.y + r * theta.cos(),
            center.z + r * theta.sin() * phi.sin(),
        );

        let view_mat = glam::Mat4::look_at_rh(eye, center, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
        let view_proj = proj * view_mat;

        let sun_az = terrain.sun_azimuth_deg.to_radians();
        let sun_el = terrain.sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

        // Debug: print uniform values on first render
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            println!("[render] terrain_params: min_h={:.1}, h_range={:.1}, width={:.1}, z_scale={:.2}",
                terrain.domain.0, h_range, terrain_width, shader_z_scale);
            let max_y = if use_pbr {
                h_range * terrain.z_scale
            } else {
                terrain_width * legacy_z_scale * 0.001
            };
            println!("[render] Expected Y range: 0 to {:.1}", max_y);
            println!("[render] Camera center: ({:.1}, {:.1}, {:.1})", center.x, center.y, center.z);
        });

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
        
        if let Some((domain, z_scale, sun_intensity, ambient, shadow_intensity, water_level, background_color, water_color)) = pbr_uniforms_data {
            let pbr_uniforms = TerrainPbrUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                terrain_params: [
                    domain.0,
                    domain.1 - domain.0,
                    terrain_width,
                    z_scale,
                ],
                lighting: [
                    sun_intensity,
                    ambient,
                    shadow_intensity,
                    water_level,
                ],
                background: [
                    background_color[0],
                    background_color[1],
                    background_color[2],
                    0.0,
                ],
                water_color: [
                    water_color[0],
                    water_color[1],
                    water_color[2],
                    0.0,
                ],
                pbr_params: [
                    self.pbr_config.exposure,
                    self.pbr_config.normal_strength,
                    self.pbr_config.ibl_intensity,
                    0.0,
                ],
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
            };
            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes for heightfield AO and sun visibility before render
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);
        
        // Re-borrow terrain after mutable operations
        let terrain = self.terrain.as_ref().unwrap();
        let depth_view = self.depth_view.as_ref().unwrap();
        let bg = &terrain.background_color;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
        }

        true
    }

    /// Render terrain to an offscreen texture at the specified resolution.
    /// This allows high-resolution snapshots independent of window size.
    /// Returns the texture if successful.
    pub fn render_to_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<wgpu::Texture> {
        if self.terrain.is_none() {
            return None;
        }

        // Create offscreen color texture at requested resolution
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Calculate all the same values as render() but with custom dimensions
        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        let terrain = self.terrain.as_ref().unwrap();

        let phi = terrain.cam_phi_deg.to_radians();
        let theta = terrain.cam_theta_deg.to_radians();
        let r = terrain.cam_radius;
        let (tw, th) = terrain.dimensions;
        let terrain_width = tw.max(th) as f32;
        let h_range = terrain.domain.1 - terrain.domain.0;
        let legacy_z_scale = terrain.z_scale * h_range * 1000.0 / terrain_width.max(1.0);
        let shader_z_scale = if use_pbr { terrain.z_scale } else { legacy_z_scale };
        let center_y = if use_pbr {
            h_range * terrain.z_scale * 0.5
        } else {
            terrain_width * legacy_z_scale * 0.001 * 0.5
        };
        let center = glam::Vec3::new(
            terrain_width * 0.5,
            center_y,
            terrain_width * 0.5,
        );

        let eye = glam::Vec3::new(
            center.x + r * theta.sin() * phi.cos(),
            center.y + r * theta.cos(),
            center.z + r * theta.sin() * phi.sin(),
        );

        let view_mat = glam::Mat4::look_at_rh(eye, center, glam::Vec3::Y);
        // Use requested width/height for aspect ratio
        let proj = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            width as f32 / height as f32,
            1.0,
            r * 10.0,
        );
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

        if let Some((domain, z_scale, sun_intensity, ambient, shadow_intensity, water_level, background_color, water_color)) = pbr_uniforms_data {
            let pbr_uniforms = TerrainPbrUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                terrain_params: [
                    domain.0,
                    domain.1 - domain.0,
                    terrain_width,
                    z_scale,
                ],
                lighting: [
                    sun_intensity,
                    ambient,
                    shadow_intensity,
                    water_level,
                ],
                background: [
                    background_color[0],
                    background_color[1],
                    background_color[2],
                    0.0,
                ],
                water_color: [
                    water_color[0],
                    water_color[1],
                    water_color[2],
                    0.0,
                ],
                pbr_params: [
                    self.pbr_config.exposure,
                    self.pbr_config.normal_strength,
                    self.pbr_config.ibl_intensity,
                    0.0,
                ],
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
            };
            self.prepare_pbr_bind_group_internal(&pbr_uniforms);
        }

        // Run compute passes
        self.dispatch_heightfield_compute(encoder, terrain_width, sun_dir);

        // Re-borrow terrain
        let terrain = self.terrain.as_ref().unwrap();

        // Render to offscreen texture
        {
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
        }

        Some(color_tex)
    }

    /// Prepare PBR bind group with current uniforms (called before render pass)
    /// Gets heightmap_view internally from self.terrain to avoid borrow issues
    fn prepare_pbr_bind_group_internal(&mut self, uniforms: &TerrainPbrUniforms) {
        // Ensure fallback texture exists first (before any borrows)
        self.ensure_fallback_texture();
        
        // Early return checks
        if self.pbr_bind_group_layout.is_none() || self.terrain.is_none() {
            return;
        }

        // Create or update uniform buffer
        if self.pbr_uniform_buffer.is_none() {
            self.pbr_uniform_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain_viewer_pbr.uniform_buffer"),
                size: std::mem::size_of::<TerrainPbrUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Write uniforms
        if let Some(ref buf) = self.pbr_uniform_buffer {
            self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&[*uniforms]));
        }

        // Recreate bind group
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer_pbr.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Now borrow everything we need
        let layout = self.pbr_bind_group_layout.as_ref().unwrap();
        let terrain = self.terrain.as_ref().unwrap();
        let fallback_view = self.fallback_texture_view.as_ref().unwrap();
        let ao_view = self.height_ao_view.as_ref().unwrap_or(fallback_view);
        let sv_view = self.sun_vis_view.as_ref().unwrap_or(fallback_view);
        
        if let Some(ref buf) = self.pbr_uniform_buffer {
            self.pbr_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain_viewer_pbr.bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(sv_view),
                    },
                ],
            }));
        }
    }
    
    /// Ensure fallback 1x1 white texture exists for when AO/sun_vis are disabled
    fn ensure_fallback_texture(&mut self) {
        if self.fallback_texture.is_some() {
            return;
        }
        
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain_viewer.fallback_texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Write 1.0 (fully lit / no occlusion)
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
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        
        self.fallback_texture_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.fallback_texture = Some(texture);
    }

    /// Dispatch compute passes for heightfield AO and sun visibility
    fn dispatch_heightfield_compute(&mut self, encoder: &mut wgpu::CommandEncoder, terrain_width: f32, sun_dir: glam::Vec3) {
        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };
        let (width, height) = terrain.dimensions;
        let h_range = terrain.domain.1 - terrain.domain.0;
        let z_scale = terrain.z_scale;
        
        // Height AO compute pass
        if self.pbr_config.height_ao.enabled {
            if let (Some(ref pipeline), Some(ref layout), Some(ref uniform_buf), Some(ref ao_view), Some(ref sampler)) = (
                &self.height_ao_pipeline,
                &self.height_ao_bind_group_layout,
                &self.height_ao_uniform_buffer,
                &self.height_ao_view,
                &self.sampler_nearest,
            ) {
                let ao_width = (width as f32 * self.pbr_config.height_ao.resolution_scale) as u32;
                let ao_height = (height as f32 * self.pbr_config.height_ao.resolution_scale) as u32;
                
                // Update uniforms
                let uniforms: [f32; 16] = [
                    self.pbr_config.height_ao.directions as f32,
                    self.pbr_config.height_ao.steps as f32,
                    self.pbr_config.height_ao.max_distance,
                    self.pbr_config.height_ao.strength,
                    terrain_width / width as f32,
                    terrain_width / height as f32,
                    h_range * z_scale,
                    terrain.domain.0,
                    ao_width as f32,
                    ao_height as f32,
                    width as f32,
                    height as f32,
                    0.0, 0.0, 0.0, 0.0,
                ];
                self.queue.write_buffer(uniform_buf, 0, bytemuck::cast_slice(&uniforms));
                
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.height_ao_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(ao_view) },
                    ],
                });
                
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain_viewer.height_ao_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((ao_width + 7) / 8, (ao_height + 7) / 8, 1);
            }
        }
        
        // Sun visibility compute pass
        if self.pbr_config.sun_visibility.enabled {
            if let (Some(ref pipeline), Some(ref layout), Some(ref uniform_buf), Some(ref sv_view), Some(ref sampler)) = (
                &self.sun_vis_pipeline,
                &self.sun_vis_bind_group_layout,
                &self.sun_vis_uniform_buffer,
                &self.sun_vis_view,
                &self.sampler_nearest,
            ) {
                let sv_width = (width as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
                let sv_height = (height as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
                
                // Update uniforms - sun_dir should point toward sun (negate light direction)
                let uniforms: [f32; 16] = [
                    self.pbr_config.sun_visibility.samples as f32,
                    self.pbr_config.sun_visibility.steps as f32,
                    self.pbr_config.sun_visibility.max_distance,
                    self.pbr_config.sun_visibility.softness,
                    terrain_width / width as f32,
                    terrain_width / height as f32,
                    h_range * z_scale,
                    terrain.domain.0,
                    sv_width as f32,
                    sv_height as f32,
                    width as f32,
                    height as f32,
                    sun_dir.x,
                    sun_dir.y,
                    sun_dir.z,
                    self.pbr_config.sun_visibility.bias,
                ];
                self.queue.write_buffer(uniform_buf, 0, bytemuck::cast_slice(&uniforms));
                
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.sun_vis_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sv_view) },
                    ],
                });
                
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain_viewer.sun_vis_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((sv_width + 7) / 8, (sv_height + 7) / 8, 1);
            }
        }
    }
}
