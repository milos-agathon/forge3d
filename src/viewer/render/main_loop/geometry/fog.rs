use glam::Mat4;

use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::viewer_enums::FogMode;
use crate::viewer::{CameraFrustum, FogCameraUniforms, Viewer, VolumetricUniformsStd140};

use super::mat4_to_array;

impl Viewer {
    pub(super) fn render_geometry_fog(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gi: &mut ScreenSpaceEffectsManager,
    ) {
        if !self.fog_enabled {
            return;
        }

        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov = self.view_config.fov_deg.to_radians();
        let proj = Mat4::perspective_rh(fov, aspect, self.view_config.znear, self.view_config.zfar);
        let view_mat = self.camera.view_matrix();

        if let Some(ref mut csm) = self.csm {
            let frustum = CameraFrustum::from_matrices(&view_mat, &proj);
            csm.update_cascades(&self.queue, &frustum);
        }

        if self.fog_use_shadows {
            self.render_fog_shadow_cascades(encoder);
        }

        let inv_view = view_mat.inverse();
        let inv_proj = proj.inverse();
        let eye = self.camera.eye();
        let fog_cam = FogCameraUniforms {
            view: mat4_to_array(view_mat),
            proj: mat4_to_array(proj),
            inv_view: mat4_to_array(inv_view),
            inv_proj: mat4_to_array(inv_proj),
            view_proj: mat4_to_array(proj * view_mat),
            eye_position: [eye.x, eye.y, eye.z],
            near: self.view_config.znear,
            far: self.view_config.zfar,
            _pad: [0.0; 3],
        };
        self.queue
            .write_buffer(&self.fog_camera, 0, bytemuck::bytes_of(&fog_cam));

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
        self.queue
            .write_buffer(&self.fog_params, 0, bytemuck::bytes_of(&fog_params_packed));

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
            bytemuck::bytes_of(&mat4_to_array(fog_shadow_mat)),
        );

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
            self.dispatch_raymarch_fog(encoder, gi, &bg0, &bg1, &bg2);
        } else {
            self.dispatch_froxel_fog(encoder, &bg0, &bg1, &bg2);
        }

        self.fog_frame_index = self.fog_frame_index.wrapping_add(1);
    }

    fn render_fog_shadow_cascades(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let vb = self.geom_vb.as_ref().unwrap();
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
                    self.queue
                        .write_buffer(csm_cam_buf, 0, bytemuck::cast_slice(&light_vp_arr));
                    let csm_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewer.csm.depth.bg"),
                        layout: &bgl,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: csm_cam_buf.as_entire_binding(),
                        }],
                    });
                    let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.csm.depth"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    shadow_pass.set_pipeline(csm_pipe);
                    shadow_pass.set_bind_group(0, &csm_bg, &[]);
                    shadow_pass.set_vertex_buffer(0, vb.slice(..));
                    if let Some(ib) = self.geom_ib.as_ref() {
                        shadow_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(0..self.geom_index_count, 0, 0..1);
                    } else {
                        shadow_pass.draw(0..self.geom_index_count, 0..1);
                    }
                }
            }
        }
    }
}
