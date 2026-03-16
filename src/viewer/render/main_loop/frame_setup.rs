use glam::Mat4;

use crate::viewer::{SkyUniforms, Viewer, VIEWER_SNAPSHOT_MAX_MEGAPIXELS};

impl Viewer {
    pub(super) fn prepare_render_frame(
        &mut self,
    ) -> Result<
        (
            wgpu::SurfaceTexture,
            wgpu::TextureView,
            Option<(u32, u32)>,
            wgpu::CommandEncoder,
        ),
        wgpu::SurfaceError,
    > {
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

        // Compute snapshot dimensions if requested (but don't resize yet - we'll render to offscreen)
        let _snapshot_dimensions = if self.snapshot_request.is_some() {
            let (req_w, req_h) = if let (Some(w), Some(h)) = (
                self.view_config.snapshot_width,
                self.view_config.snapshot_height,
            ) {
                (w, h)
            } else {
                (self.config.width, self.config.height)
            };

            // Apply soft megapixel clamp
            let mut snap_w = req_w;
            let mut snap_h = req_h;
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

            if snap_w != self.config.width || snap_h != self.config.height {
                eprintln!(
                    "[viewer] Snapshot requested at {}x{} (window: {}x{})",
                    snap_w, snap_h, self.config.width, self.config.height
                );
            }

            Some((snap_w, snap_h))
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Update labels with current camera (before any rendering)
        // Use terrain camera if terrain is loaded, otherwise use viewer camera
        {
            // Gather selected IDs for highlighting (labels use u64 IDs, picking uses u32)
            let selected_u32 = self.unified_picking.selection_manager().get_selection();
            let selected_ids: std::collections::HashSet<u64> =
                selected_u32.iter().map(|&id| id as u64).collect();

            let (view_proj, camera_pos) = if let Some(ref terrain_viewer) = self.terrain_viewer {
                if let Some(ref terrain) = terrain_viewer.terrain {
                    // Use terrain camera parameters
                    let phi = terrain.cam_phi_deg.to_radians();
                    let theta = terrain.cam_theta_deg.to_radians();
                    let r = terrain.cam_radius;
                    let (tw, th) = terrain.dimensions;
                    let terrain_width = tw.max(th) as f32;
                    let h_range = terrain.domain.1 - terrain.domain.0;
                    let center_y = h_range * terrain.z_scale * 0.5;
                    let center =
                        glam::Vec3::new(terrain_width * 0.5, center_y, terrain_width * 0.5);
                    let eye = glam::Vec3::new(
                        center.x + r * theta.sin() * phi.cos(),
                        center.y + r * theta.cos(),
                        center.z + r * theta.sin() * phi.sin(),
                    );
                    let view_mat = Mat4::look_at_rh(eye, center, glam::Vec3::Y);
                    let proj = Mat4::perspective_rh(
                        terrain.cam_fov_deg.to_radians(),
                        self.config.width as f32 / self.config.height as f32,
                        1.0,
                        r * 10.0,
                    );
                    (proj * view_mat, Some(eye))
                } else {
                    // No terrain data, use viewer camera
                    let aspect = self.config.width as f32 / self.config.height as f32;
                    let fov = self.view_config.fov_deg.to_radians();
                    let proj = Mat4::perspective_rh(
                        fov,
                        aspect,
                        self.view_config.znear,
                        self.view_config.zfar,
                    );
                    (proj * self.camera.view_matrix(), Some(self.camera.eye()))
                }
            } else {
                // No terrain viewer, use viewer camera
                let aspect = self.config.width as f32 / self.config.height as f32;
                let fov = self.view_config.fov_deg.to_radians();
                let proj = Mat4::perspective_rh(
                    fov,
                    aspect,
                    self.view_config.znear,
                    self.view_config.zfar,
                );
                (proj * self.camera.view_matrix(), Some(self.camera.eye()))
            };

            self.label_manager
                .update_with_camera(view_proj, camera_pos, Some(&selected_ids));
        }

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

        Ok((output, view, _snapshot_dimensions, encoder))
    }
}
