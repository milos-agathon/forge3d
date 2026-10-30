use super::LabelCompositor;
use crate::core::gpu_timing::OneShotTiming;
use crate::core::resource_tracker::tracked_create_buffer;
use numpy::{ndarray::Array3, IntoPyArray, PyArray3};
use pyo3::prelude::*;

#[pymethods]
impl LabelCompositor {
    fn render_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let _allocation_scope = self.allocation_owner.activate();
        let capture = crate::core::certificate::begin_render_capture_with_resources(
            "anamnesis.label.composite",
            &[self.allocation_owner.id()],
        );
        let g = crate::core::gpu::try_ctx()?;
        let mut timing = OneShotTiming::for_current_device();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("label-compositor-encoder"),
            });
        let scope = timing.begin(&mut encoder, "label.clear");
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("label-compositor-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
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
                })],
                ..Default::default()
            });
        }
        timing.end(&mut encoder, scope, 0);
        let scope = timing.begin(&mut encoder, "label.composite");
        let mut draws = 0;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("label-compositor-composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if self.overlay_enabled {
                if let Some(overlay) = &self.overlay_renderer {
                    crate::core::shader_registry::record_shader_use("overlays_shader");
                    overlay.render(&mut pass);
                    draws += 1;
                }
            }
            if self.text_overlay_enabled && !self.text_instances.is_empty() {
                if let Some(text) = &mut self.text_overlay_renderer {
                    crate::core::shader_registry::record_shader_use("text_overlay_shader");
                    text.set_resolution(self.width, self.height);
                    text.set_alpha(self.text_overlay_alpha);
                    text.set_enabled(true);
                    text.upload_uniforms(&g.queue);
                    text.upload_instances(&g.device, &g.queue, &self.text_instances)?;
                    text.render(&mut pass);
                    draws += 1;
                }
            }
        }
        timing.end(&mut encoder, scope, draws);
        let unpadded = self.width * 4;
        let padded = crate::core::gpu::align_copy_bpr(unpadded);
        let readback = tracked_create_buffer(
            &g.device,
            &wgpu::BufferDescriptor {
                label: Some("label-compositor-readback"),
                size: u64::from(padded) * u64::from(self.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )?;
        let scope = timing.begin(&mut encoder, "label.readback_copy");
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
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        timing.end(&mut encoder, scope, 1);
        timing.submit(&g.queue, encoder);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        g.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity(unpadded as usize * self.height as usize);
        for row in 0..self.height as usize {
            let start = row * padded as usize;
            pixels.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        drop(data);
        readback.unmap();
        if !timing.record_into_certificate() {
            for (label, count) in [
                ("label.clear", 0),
                ("label.composite", draws),
                ("label.readback_copy", 1),
            ] {
                crate::core::certificate::record_pass(label, 0.0, count);
            }
        }
        capture.finish();
        let array = Array3::from_shape_vec((self.height as usize, self.width as usize, 4), pixels)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray_bound(py))
    }
}
