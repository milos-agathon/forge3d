use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    pub fn set_raster_overlay(
        &mut self,
        image: &pyo3::types::PyAny,
        alpha: Option<f32>,
        offset_xy: Option<(i32, i32)>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        // Validate input array (HxWx3 or HxWx4, uint8)
        let (h, w, c, data) = if let Ok(arr) = image.extract::<numpy::PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay image must be HxWxC uint8",
                ));
            }
            let h = shape[0] as u32;
            let w = shape[1] as u32;
            let c = shape[2] as u32;
            if c != 3 && c != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay channels must be 3 or 4",
                ));
            }
            (h, w, c, arr.as_array().to_owned().into_raw_vec())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected numpy uint8 array HxWxC",
            ));
        };

        // Convert to RGBA8
        let mut rgba: Vec<u8> = Vec::with_capacity((h * w * 4) as usize);
        if c == 4 {
            rgba = data;
        } else {
            // c == 3
            let mut idx = 0usize;
            while idx < data.len() {
                rgba.push(data[idx]);
                rgba.push(data[idx + 1]);
                rgba.push(data[idx + 2]);
                rgba.push(255);
                idx += 3;
            }
        }

        // Create GPU texture and upload
        let g = crate::core::gpu::ctx();
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-overlay-rgba8"),
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
        let row_bytes = w * 4;
        let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
        }
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

        // Update overlay renderer state
        if let Some(ref mut ov) = self.overlay_renderer {
            // Keep GPU resources alive
            ov.set_overlay_texture(tex, view);
            // Rebind with current height view for altitude/contours
            ov.recreate_bind_group(&g.device, None, self.height_view.as_ref(), None, None);

            // Set params
            ov.set_enabled(true);
            if let Some(a) = alpha {
                ov.set_overlay_alpha(a);
            }
            let (off_x, off_y) = offset_xy.unwrap_or((0, 0));
            let scale_v = scale.unwrap_or(1.0).max(1e-3);
            let sample_s = 1.0 / scale_v;
            let uv_off_x = (off_x as f32 / self.width.max(1) as f32) * sample_s;
            let uv_off_y = (off_y as f32 / self.height.max(1) as f32) * sample_s;
            ov.set_overlay_uv(uv_off_x, uv_off_y, sample_s, sample_s);
            ov.upload_uniforms(&g.queue);
            self.overlay_enabled = true;
        }

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_overlay(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_enabled(false);
            let g = crate::core::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        self.overlay_enabled = false;
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_overlay_alpha(&mut self, alpha: f32) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_overlay_alpha(alpha);
            ov.set_enabled(true);
            let g = crate::core::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        self.overlay_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha=0.35)")]
    pub fn enable_altitude_overlay(&mut self, alpha: Option<f32>) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_enabled(true);
            if let Some(a) = alpha {
                ov.set_altitude_alpha(a);
            }
            let g = crate::core::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_altitude_overlay(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_enabled(false);
            let g = crate::core::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_terrain(&mut self) -> PyResult<()> {
        self.terrain_enabled = false;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn enable_terrain(&mut self) -> PyResult<()> {
        self.terrain_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_altitude_overlay_alpha(&mut self, alpha: f32) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_alpha(alpha);
            let g = crate::core::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    // GPU contour overlay using height texture
    #[pyo3(text_signature = "($self, interval, thickness_mul=1.0, r=0.0, g=0.0, b=0.0, a=0.75)")]
    pub fn enable_gpu_contours(
        &mut self,
        interval: f32,
        thickness_mul: Option<f32>,
        r: Option<f32>,
        g: Option<f32>,
        b: Option<f32>,
        a: Option<f32>,
    ) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_contours_enabled(true);
            ov.set_contour_interval(interval);
            ov.set_contour_thickness_mul(thickness_mul.unwrap_or(1.0));
            let cr = r.unwrap_or(0.0);
            let cg = g.unwrap_or(0.0);
            let cb = b.unwrap_or(0.0);
            let ca = a.unwrap_or(0.75);
            ov.set_contour_color(cr, cg, cb, ca);
            let gctx = crate::core::gpu::ctx();
            ov.upload_uniforms(&gctx.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_gpu_contours(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_contours_enabled(false);
            let g = crate::core::gpu::ctx();
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }
}
