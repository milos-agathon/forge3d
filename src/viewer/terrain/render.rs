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

    /// Prepare PBR bind group with current uniforms (called before render pass)
    /// Gets heightmap_view internally from self.terrain to avoid borrow issues
    fn prepare_pbr_bind_group_internal(&mut self, uniforms: &TerrainPbrUniforms) {
        let Some(ref layout) = self.pbr_bind_group_layout else {
            return;
        };
        let Some(ref terrain) = self.terrain else {
            return;
        };

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
                ],
            }));
        }
    }
}
