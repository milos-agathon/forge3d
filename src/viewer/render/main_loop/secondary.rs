use super::RenderAvailability;
use glam::Mat4;

use crate::viewer::Viewer;

impl Viewer {
    pub(super) fn render_secondary_paths(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _snapshot_dimensions: Option<(u32, u32)>,
        availability: RenderAvailability,
    ) -> wgpu::CommandEncoder {
        let RenderAvailability {
            have_gi,
            have_pipe,
            have_cam,
            have_vb,
            have_z,
            have_bgl,
        } = availability;
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
            if let (Some(w), Some(h)) = (
                self.view_config.snapshot_width,
                self.view_config.snapshot_height,
            ) {
                Some((w, h))
            } else {
                Some((self.config.width, self.config.height))
            }
        } else {
            None
        };

        if let Some(ref mut tv) = self.terrain_viewer {
            if tv.has_terrain() {
                eprintln!("[DEBUG main_loop] terrain_viewer path, has_terrain=true, snapshot_request={:?}, terrain_snap_size={:?}",
                self.snapshot_request.is_some(), terrain_snap_size);
                // Render to screen at window resolution
                // Note: Motion blur is too expensive for real-time rendering (N full renders per frame),
                // so we only apply it for snapshots. The interactive viewer shows a regular render.
                terrain_rendered = tv.render(
                    &mut encoder,
                    &view,
                    self.config.width,
                    self.config.height,
                    self.selected_feature_id,
                );

                // Then render to offscreen texture at snapshot resolution (if requested)
                // This must be LAST so the uniform buffer has the correct aspect ratio for the snapshot
                if let Some((snap_w, snap_h)) = terrain_snap_size {
                    // P4: Use motion blur rendering if enabled
                    if tv.pbr_config.motion_blur.enabled && tv.pbr_config.motion_blur.samples > 1 {
                        // Motion blur handles its own encoder internally
                        self.queue.submit(std::iter::once(encoder.finish()));
                        if let Some(tex) =
                            tv.render_with_motion_blur(self.config.format, snap_w, snap_h)
                        {
                            self.pending_snapshot_tex = Some(tex);
                        }
                        // Create a new encoder for any remaining work
                        encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("viewer.render.post_motion_blur"),
                                });
                    } else {
                        if let Some(tex) = tv.render_to_texture(
                            &mut encoder,
                            self.config.format,
                            snap_w,
                            snap_h,
                            self.selected_feature_id,
                        ) {
                            self.pending_snapshot_tex = Some(tex);
                        }
                    }
                }
            }
        }

        #[cfg(feature = "extension-module")]
        if !terrain_rendered {
            if let Some(ref mut scene) = self.terrain_scene {
                if scene.has_viewer_terrain() {
                    // Render terrain to swapchain view at normal resolution
                    terrain_rendered = scene.render_viewer_terrain(
                        &mut encoder,
                        &view,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                    );

                    // If snapshot requested, also render to offscreen texture at snapshot dimensions
                    if let Some((snap_w, snap_h)) = _snapshot_dimensions {
                        let snap_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("viewer.terrain.snapshot"),
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
                        let snap_view =
                            snap_tex.create_view(&wgpu::TextureViewDescriptor::default());

                        scene.render_viewer_terrain(
                            &mut encoder,
                            &snap_view,
                            self.config.format,
                            snap_w,
                            snap_h,
                        );

                        self.pending_snapshot_tex = Some(snap_tex);
                    }
                }
            }
        }

        if !terrain_rendered && !(have_gi && have_pipe && have_cam && have_vb && have_z && have_bgl)
        {
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

        // P5: Render point cloud (after terrain, before labels)
        if let Some(ref pc) = self.point_cloud {
            if pc.visible && pc.point_count > 0 && pc.instance_buffer.is_some() {
                // Points are centered at origin, compute extent for camera distance
                let extent_x = pc.bounds_max[0] - pc.bounds_min[0];
                let extent_y = pc.bounds_max[1] - pc.bounds_min[1];
                let extent_z = pc.bounds_max[2] - pc.bounds_min[2];
                let extent = extent_x.max(extent_y).max(extent_z).max(100.0);

                // Use orbit camera state from point cloud
                let base_radius = extent * 2.0;
                let r = base_radius * pc.cam_radius;
                let center = glam::Vec3::ZERO;
                // Spherical coords from point cloud camera state
                let eye = glam::Vec3::new(
                    r * pc.cam_theta.cos() * pc.cam_phi.cos(),
                    r * pc.cam_theta.sin(),
                    r * pc.cam_theta.cos() * pc.cam_phi.sin(),
                );

                let view_mat = Mat4::look_at_rh(eye, center, glam::Vec3::Y);
                let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
                let near = extent * 0.01;
                let far = extent * 10.0;
                let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, near, far);
                let view_proj = proj * view_mat;

                // Simple render pass - clear to dark background for point cloud
                {
                    let mut pc_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.pointcloud.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.1,
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

                    pc.render(
                        &mut pc_pass,
                        &self.queue,
                        view_proj.to_cols_array_2d(),
                        [self.config.width as f32, self.config.height as f32],
                    );
                }

                // Also render point cloud to snapshot texture if one exists
                if let Some(ref snap_tex) = self.pending_snapshot_tex {
                    let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let snap_w = snap_tex.width() as f32;
                    let snap_h = snap_tex.height() as f32;
                    let snap_aspect = snap_w / snap_h.max(1.0);
                    let snap_proj =
                        Mat4::perspective_rh(45.0_f32.to_radians(), snap_aspect, near, far);
                    let snap_view_proj = snap_proj * view_mat;

                    let mut pc_snap_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewer.pointcloud.pass.snapshot"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &snap_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.1,
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

                    pc.render(
                        &mut pc_snap_pass,
                        &self.queue,
                        snap_view_proj.to_cols_array_2d(),
                        [snap_w, snap_h],
                    );
                }
            }
        }

        // Render labels (screen-space text overlay) - AFTER terrain so labels appear on top
        if self.label_manager.is_enabled() && self.label_manager.visible_count() > 0 {
            self.label_manager
                .upload_to_renderer(&self.device, &self.queue, &mut self.hud);
            self.hud.set_enabled(true);
            self.hud.upload_uniforms(&self.queue);

            // Render labels to swapchain view for interactive display
            {
                let mut label_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.labels.pass"),
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
                self.hud.render(&mut label_pass);
            }

            // Render labels to snapshot texture if one exists
            if let Some(ref snap_tex) = self.pending_snapshot_tex {
                let snap_view = snap_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let mut label_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.labels.pass.snapshot"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &snap_view,
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
                self.hud.render(&mut label_pass);
            }
        }

        encoder
    }
}
