use super::*;

#[pymethods]
impl TerrainSpike {
    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: String) -> PyResult<()> {
        // Encode pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain-encoder"),
            });
        {
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

        // E3: Overlay compositor pass (optional)
        if let Some(ref mut ov) = self.overlay_renderer {
            // Recreate bind group to reflect latest overlay/height views
            let overlay_view_opt = self.overlay_mosaic.as_ref().map(|m| &m.view);
            // Prefer height mosaic view if present, else None (renderer will use dummy)
            let height_view_opt = self.height_mosaic.as_ref().map(|m| &m.view);
            let pt_buf_opt = self.page_table.as_ref().map(|pt| &pt.buffer);
            ov.recreate_bind_group(
                &self.device,
                overlay_view_opt,
                height_view_opt,
                pt_buf_opt,
                None,
            );
            ov.upload_uniforms(&self.queue);

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
        self.queue.submit(Some(encoder.finish()));

        // Readback → PNG
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;

        let buf_size = (padded_bpr * self.height) as wgpu::BufferAddress;

        // B15: Check memory budget before creating host-visible readback buffer
        let tracker = global_tracker();
        tracker.check_budget(buf_size).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Memory budget exceeded during terrain readback: {}",
                e
            ))
        })?;

        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-readback"),
            size: buf_size,
            usage,
            mapped_at_creation: false,
        });

        // B15: Track allocation (host-visible)
        tracker.track_buffer_allocation(buf_size, is_host_visible_usage(usage));

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

        // B15: Free allocation after use
        tracker.free_buffer_allocation(buf_size, is_host_visible_usage(usage));

        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Invalid image buffer"))?;
        img.save(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}
