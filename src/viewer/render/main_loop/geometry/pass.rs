use glam::Mat4;

use crate::core::screen_space_effects::{CameraParams, ScreenSpaceEffectsManager};
use crate::viewer::Viewer;

use super::mat4_to_array;

impl Viewer {
    pub(super) fn render_geometry_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gi: &mut ScreenSpaceEffectsManager,
        zv: &wgpu::TextureView,
    ) {
        // Skip geometry pass when no geometry is loaded (e.g. terrain-only mode)
        if self.geom_vb.is_none() {
            return;
        }
        let cam_buf = self.geom_camera_buffer.as_ref().unwrap();
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov = self.view_config.fov_deg.to_radians();
        let proj_base =
            Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);

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

        let view_mat = self.camera.view_matrix();
        let model_view = view_mat * self.object_transform;
        self.log_snapshot_geometry_transform();

        let cam_pack = [mat4_to_array(model_view), mat4_to_array(proj)];
        self.queue
            .write_buffer(cam_buf, 0, bytemuck::cast_slice(&cam_pack));

        let inv_proj = proj.inverse();
        let eye = self.camera.eye();
        let inv_model_view = model_view.inverse();
        let view_proj = proj * model_view;
        let cam = CameraParams {
            view_matrix: mat4_to_array(model_view),
            inv_view_matrix: mat4_to_array(inv_model_view),
            proj_matrix: mat4_to_array(proj),
            inv_proj_matrix: mat4_to_array(inv_proj),
            prev_view_proj_matrix: mat4_to_array(self.prev_view_proj),
            camera_pos: [eye.x, eye.y, eye.z],
            frame_index: self.frame_count as u32,
            jitter_offset: self.taa_jitter.offset_array(),
            _pad_jitter: [0.0, 0.0],
        };
        gi.update_camera(&self.queue, &cam);
        self.prev_view_proj = view_proj;
        self.taa_jitter.advance();

        self.ensure_geometry_bind_group_fallback();
        let pipe = self.geom_pipeline.as_ref().unwrap();
        let vb = self.geom_vb.as_ref().unwrap();
        let bg_ref = self.geom_bind_group.as_ref().unwrap();
        let geom_ib = self.geom_ib.as_ref();
        let geom_index_count = self.geom_index_count;
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
        if let Some(ib) = geom_ib {
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..geom_index_count, 0, 0..1);
        } else {
            pass.draw(0..geom_index_count, 0..1);
        }
    }

    fn log_snapshot_geometry_transform(&self) {
        if self.snapshot_request.is_some() {
            let is_identity = self.object_transform == glam::Mat4::IDENTITY;
            let t = self.object_translation;
            let s = self.object_scale;
            let msg = format!(
                "[D1-GEOM] frame={} transform_identity={} index_count={} trans=[{:.3},{:.3},{:.3}] scale=[{:.3},{:.3},{:.3}]\n",
                self.frame_count,
                is_identity,
                self.geom_index_count,
                t.x,
                t.y,
                t.z,
                s.x,
                s.y,
                s.z
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
    }

    fn ensure_geometry_bind_group_fallback(&mut self) {
        if self.geom_bind_group.is_none() {
            let cam_buf = self.geom_camera_buffer.as_ref().unwrap();
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
        }
    }
}
