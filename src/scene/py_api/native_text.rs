use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // -----------------------------
    // D: Native text overlay APIs
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = true;
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_enabled(true);
            let g = crate::core::gpu::try_ctx()?;
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = false;
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_enabled(false);
            let g = crate::core::gpu::try_ctx()?;
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_native_text_alpha(&mut self, alpha: f32) -> PyResult<()> {
        self.text_overlay_alpha = alpha.clamp(0.0, 1.0);
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_alpha(self.text_overlay_alpha);
            let g = crate::core::gpu::try_ctx()?;
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, x, y, w, h, r, g, b, a)")]
    pub fn add_native_text_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) -> PyResult<()> {
        let rect_min = [x.max(0.0), y.max(0.0)];
        let rect_max = [(x + w).max(0.0), (y + h).max(0.0)];
        let uv_min = [0.0, 0.0];
        let uv_max = [1.0, 1.0];
        let color = [
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            a.clamp(0.0, 1.0),
        ];
        self.text_instances
            .push(crate::core::text_overlay::TextInstance::new(
                rect_min, rect_max, uv_min, uv_max, color,
            ));
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clear_native_text(&mut self) -> PyResult<()> {
        self.text_instances.clear();
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.instance_count = 0;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, x, y, w, h, u0, v0, u1, v1, r, g, b, a)")]
    pub fn add_native_text_rect_uv(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        u0: f32,
        v0: f32,
        u1: f32,
        v1: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) -> PyResult<()> {
        let rect_min = [x.max(0.0), y.max(0.0)];
        let rect_max = [(x + w).max(0.0), (y + h).max(0.0)];
        let uv_min = [u0, v0];
        let uv_max = [u1, v1];
        let color = [
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            a.clamp(0.0, 1.0),
        ];
        self.text_instances
            .push(crate::core::text_overlay::TextInstance::new(
                rect_min, rect_max, uv_min, uv_max, color,
            ));
        Ok(())
    }

    #[pyo3(
        text_signature = "($self, x, y, w, h, u0, v0, u1, v1, r, g, b, a, halo_r, halo_g, halo_b, halo_a, halo_width)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn add_native_text_rect_uv_halo(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        u0: f32,
        v0: f32,
        u1: f32,
        v1: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        halo_r: f32,
        halo_g: f32,
        halo_b: f32,
        halo_a: f32,
        halo_width: f32,
    ) -> PyResult<()> {
        let rect_min = [x.max(0.0), y.max(0.0)];
        let rect_max = [(x + w).max(0.0), (y + h).max(0.0)];
        let uv_min = [u0, v0];
        let uv_max = [u1, v1];
        let color = [
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            a.clamp(0.0, 1.0),
        ];
        let halo_color = [
            halo_r.clamp(0.0, 1.0),
            halo_g.clamp(0.0, 1.0),
            halo_b.clamp(0.0, 1.0),
            halo_a.clamp(0.0, 1.0),
        ];
        self.text_instances.push(
            crate::core::text_overlay::TextInstance::new(rect_min, rect_max, uv_min, uv_max, color)
                .with_halo(halo_color, halo_width),
        );
        Ok(())
    }

    #[pyo3(text_signature = "($self, atlas, channels=3, smoothing=1.0)")]
    pub fn set_native_text_atlas(
        &mut self,
        atlas: &pyo3::PyAny,
        channels: Option<u32>,
        smoothing: Option<f32>,
    ) -> PyResult<()> {
        let _allocation_scope = self.allocation_owner.activate();
        let (h, w, c, data) = if let Ok(arr) = atlas.extract::<PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "atlas must be HxWxC uint8",
                ));
            }
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2] as u32;
            if h == 0 || w == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err("atlas is empty"));
            }
            if c != 1 && c != 3 && c != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "atlas channels must be 1, 3, or 4",
                ));
            }
            (h, w, c, arr.as_array().to_owned().into_raw_vec())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected numpy uint8 array HxWxC",
            ));
        };
        let g = crate::core::gpu::try_ctx()?;
        // Convert to RGBA8
        let pixels: Vec<u32> = data
            .chunks_exact(c as usize)
            .map(|pixel| {
                let (r, g, b, a) = match c {
                    1 => (pixel[0], pixel[0], pixel[0], 255),
                    3 => (pixel[0], pixel[1], pixel[2], 255),
                    _ => (pixel[0], pixel[1], pixel[2], pixel[3]),
                };
                u32::from_le_bytes([r, g, b, a])
            })
            .collect();
        let atlas_buf = crate::core::resource_tracker::tracked_create_buffer(
            &g.device,
            &wgpu::BufferDescriptor {
                label: Some("text_msdf_atlas"),
                size: std::mem::size_of_val(pixels.as_slice()) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            },
        )?;
        atlas_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&pixels));
        atlas_buf.unmap();

        // Update text overlay renderer state
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_atlas(atlas_buf, w, h);
            tr.recreate_bind_group(&g.device, None);
            if let Some(ch) = channels {
                tr.set_channels(ch);
            }
            if let Some(sm) = smoothing {
                tr.set_smoothing(sm);
            }
            tr.upload_uniforms(&g.queue);
        }

        Ok(())
    }
}
