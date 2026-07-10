use super::*;

#[pymethods]
impl TerrainSpike {
    #[pyo3(signature = (path, certificate=None), text_signature = "($self, path, certificate=None)")]
    pub fn render_png(
        &mut self,
        py: pyo3::Python<'_>,
        path: String,
        certificate: Option<pyo3::Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let certificate_capture =
            crate::core::certificate::begin_render_capture("terrain_spike.render_png");
        // Encode pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain-encoder"),
            });
        // CENSOR F-04: live per-pass timing for the certificate; falls back to
        // 0.0 pass records when TIMESTAMP_QUERY is not granted. Built on this
        // spike's OWN device (not the global context).
        let mut timing = crate::core::gpu_timing::OneShotTiming::for_device(
            self.device.clone(),
            self.queue.clone(),
        );
        let main_scope = timing.begin(&mut encoder, "terrain_spike.main");
        {
            crate::core::shader_registry::record_shader_use(self.tp.shader_label);
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain-rp"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.color_view,
                        resolve_target: None,
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
                        view: &self.normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.tp.pipeline);
            // T33-BEGIN:set-bgs-0-1-2
            rp.set_bind_group(0, &self.bg0_globals, &[]);
            rp.set_bind_group(1, &self.bg1_height, &[]);
            rp.set_bind_group(2, &self.bg2_lut, &[]);
            // E2: tile uniforms (identity by default) at group(3)
            rp.set_bind_group(3, &self.bg5_tile, &[]);
            // T33-END:set-bgs-0-1-2
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }
        timing.end(&mut encoder, main_scope, 1);

        // E3: Overlay compositor pass (optional)
        if let Some(ref mut ov) = self.overlay_renderer {
            // Recreate bind group to reflect latest overlay/height views
            let overlay_view_opt = self.overlay_mosaic.as_ref().map(|m| &m.view);
            // Prefer height mosaic view if present, else None (renderer will use dummy)
            let height_view_opt = self.height_mosaic.as_ref().map(|m| &m.view);
            let pt_buf_opt = self.page_table.as_ref().map(|pt| pt.buffer.inner());
            ov.recreate_bind_group(
                &self.device,
                overlay_view_opt,
                height_view_opt,
                pt_buf_opt,
                None,
            )?;
            ov.upload_uniforms(&self.queue);

            let overlay_scope = timing.begin(&mut encoder, "terrain_spike.overlay");
            {
                let mut rp2 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("overlay-rp"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                ov.render(&mut rp2);
            }
            timing.end(&mut encoder, overlay_scope, 1);
        }
        timing.resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        // Readback → PNG
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;

        let buf_size = (padded_bpr * self.height) as wgpu::BufferAddress;

        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let readback = crate::core::resource_tracker::tracked_create_buffer(
            &self.device,
            &wgpu::BufferDescriptor {
                label: Some("terrain-readback"),
                size: buf_size,
                usage,
                mapped_at_creation: false,
            },
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Memory budget exceeded during terrain readback: {}",
                e
            ))
        })?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();

        let mut pixels = Vec::with_capacity((unpadded_bpr * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        readback.unmap();

        drop(readback);

        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Invalid image buffer"))?;
        img.save(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if !timing.record_into_certificate() {
            crate::core::certificate::record_pass("terrain_spike.main", 0.0, 1);
            if self.overlay_renderer.is_some() {
                crate::core::certificate::record_pass("terrain_spike.overlay", 0.0, 1);
            }
        }
        certificate_capture.finish();
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Ok(())
    }
}
