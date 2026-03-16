use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // -----------------------------
    // D: Native text overlay APIs (rectangle quads until MSDF is wired)
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = true;
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_enabled(true);
            let g = crate::core::gpu::ctx();
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = false;
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_enabled(false);
            let g = crate::core::gpu::ctx();
            tr.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_native_text_alpha(&mut self, alpha: f32) -> PyResult<()> {
        self.text_overlay_alpha = alpha.clamp(0.0, 1.0);
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_alpha(self.text_overlay_alpha);
            let g = crate::core::gpu::ctx();
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
            .push(crate::core::text_overlay::TextInstance {
                rect_min,
                rect_max,
                uv_min,
                uv_max,
                color,
            });
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
            .push(crate::core::text_overlay::TextInstance {
                rect_min,
                rect_max,
                uv_min,
                uv_max,
                color,
            });
        Ok(())
    }

    #[pyo3(text_signature = "($self, atlas, channels=3, smoothing=1.0)")]
    pub fn set_native_text_atlas(
        &mut self,
        atlas: &pyo3::PyAny,
        channels: Option<u32>,
        smoothing: Option<f32>,
    ) -> PyResult<()> {
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
        let g = crate::core::gpu::ctx();
        // Convert to RGBA8
        let mut rgba: Vec<u8> = Vec::with_capacity((h * w * 4) as usize);
        if c == 4 {
            rgba = data;
        } else if c == 3 {
            let mut idx = 0usize;
            while idx < data.len() {
                rgba.push(data[idx]);
                rgba.push(data[idx + 1]);
                rgba.push(data[idx + 2]);
                rgba.push(255);
                idx += 3;
            }
        } else {
            // c == 1 (SDF)
            for v in data.iter() {
                rgba.push(*v);
                rgba.push(*v);
                rgba.push(*v);
                rgba.push(255);
            }
        }
        let row_bytes = w * 4;
        let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
        }
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("text_msdf_atlas"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        g.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Update text overlay renderer state
        if let Some(ref mut tr) = self.text_overlay_renderer {
            tr.set_atlas(tex, view);
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
