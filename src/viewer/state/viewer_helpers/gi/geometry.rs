use crate::viewer::Viewer;
use glam::Mat4;

fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.to_cols_array();
    [
        [c[0], c[1], c[2], c[3]],
        [c[4], c[5], c[6], c[7]],
        [c[8], c[9], c[10], c[11]],
        [c[12], c[13], c[14], c[15]],
    ]
}

impl Viewer {
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
                    jitter_offset: self.taa_jitter.offset_array(),
                    _pad_jitter: [0.0, 0.0],
                };
                gi_mgr.update_camera(&self.queue, &cam);
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
