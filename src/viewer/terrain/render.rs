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
        let depth_view = self.depth_view.as_ref().unwrap();
        let terrain = self.terrain.as_ref().unwrap();

        let phi = terrain.cam_phi_deg.to_radians();
        let theta = terrain.cam_theta_deg.to_radians();
        let r = terrain.cam_radius;
        let (tw, th) = terrain.dimensions;
        let center = glam::Vec3::new(
            tw as f32 * 0.5,
            (terrain.domain.0 + terrain.domain.1) * 0.5,
            th as f32 * 0.5,
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

        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [
                terrain.domain.0,
                terrain.domain.1 - terrain.domain.0,
                tw as f32,
                terrain.z_scale,
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

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &terrain.bind_group, &[]);
            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);
        }

        true
    }
}
