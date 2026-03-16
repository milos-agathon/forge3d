impl Scene {
    pub(super) fn render_rgba_impl<'py>(
        &mut self,
        py: pyo3::Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
        let g = crate::core::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene-encoder-rgba"),
            });
        self.encode_rgba_frame(&mut encoder)?;
        g.queue.submit(Some(encoder.finish()));

        let mut pixels = self.readback_color_pixels("scene-readback-rgba", "copy-encoder-rgba")?;
        self.apply_runtime_postfx_cpu(&mut pixels);

        let arr = Array3::from_shape_vec((self.height as usize, self.width as usize, 4), pixels)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    fn encode_rgba_frame(&mut self, encoder: &mut wgpu::CommandEncoder) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        if self.fast_softlight_only() {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };
            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });
            {
                let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene-rp-fast-clear"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.03,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: normal_target,
                            resolve_target: normal_resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Discard,
                            },
                        }),
                    ],
                    depth_stencil_attachment: depth_attachment,
                    ..Default::default()
                });
            }
            return Ok(());
        }

        self.render_reflections(encoder).map_err(reflection_err)?;
        self.render_cloud_shadows(encoder)
            .map_err(cloud_shadow_err)?;

        if let Some(ref mut renderer) = self.reflection_renderer {
            if renderer.bind_group().is_none() {
                renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
            }
        }

        {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };
            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });

            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp-rgba"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: normal_target,
                        resolve_target: normal_resolve,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Discard,
                        },
                    }),
                ],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });

            if self.ground_plane_enabled {
                if let Some(ref mut ground_renderer) = self.ground_plane_renderer {
                    let view_proj = self.scene.proj * self.scene.view;
                    ground_renderer.set_camera(view_proj);
                    ground_renderer.upload_uniforms(&g.queue);
                    ground_renderer.render(&mut rp);
                }
            }

            if self.water_surface_enabled {
                if let Some(ref mut water_renderer) = self.water_surface_renderer {
                    let view_proj = self.scene.proj * self.scene.view;
                    water_renderer.set_camera(view_proj);
                    water_renderer.upload_uniforms(&g.queue);
                    water_renderer.render(&mut rp);
                }
            }

            if self.soft_light_radius_enabled {
                if let Some(ref soft_light_renderer) = self.soft_light_radius_renderer {
                    soft_light_renderer.update_uniforms(&g.queue);
                    soft_light_renderer.render(&mut rp, false);
                }
            }

            if self.point_spot_lights_enabled {
                if let Some(ref mut lights_renderer) = self.point_spot_lights_renderer {
                    lights_renderer.set_camera(self.scene.view, self.scene.proj);
                    lights_renderer.update_buffers(&g.queue);
                    lights_renderer.render_deferred(&mut rp);
                }
            }

            if self.terrain_enabled {
                rp.set_pipeline(&self.tp.pipeline);
                rp.set_bind_group(0, &self.bg0_globals, &[]);
                rp.set_bind_group(1, &self.bg1_height, &[]);
                rp.set_bind_group(2, &self.bg2_lut, &[]);
                rp.set_bind_group(3, &self.bg3_tile, &[]);
                let max_groups = crate::core::gpu::ctx().device.limits().max_bind_groups;
                if max_groups >= 6 {
                    let cloud_bg = self
                        .bg3_cloud_shadows
                        .as_ref()
                        .unwrap_or(&self.bg4_dummy_cloud_shadows);
                    rp.set_bind_group(4, cloud_bg, &[]);
                }
                if max_groups >= 6 {
                    if let Some(ref renderer) = self.reflection_renderer {
                        if let Some(reflection_bg) = renderer.bind_group() {
                            rp.set_bind_group(5, reflection_bg, &[]);
                        }
                    }
                }

                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.nidx, 0, 0..1);
            }
        }

        if self.ssao_enabled {
            self.ssao
                .dispatch(
                    &g.device,
                    &g.queue,
                    encoder,
                    &self.normal_view,
                    &self.color,
                    &self.scene.proj,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }

        self.render_dof(encoder).map_err(dof_err)?;
        Ok(())
    }
}

