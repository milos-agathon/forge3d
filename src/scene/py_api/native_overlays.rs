use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // -----------------------------
    // D: Native overlays compositor (upload overlay texture, altitude toggle)
    // -----------------------------
    #[pyo3(text_signature = "($self)")]
    pub fn enable_native_overlays(&mut self) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        self.overlay_enabled = true;
        ov.set_enabled(true);
        let g = crate::core::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_native_overlays(&mut self) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        self.overlay_enabled = false;
        ov.set_enabled(false);
        let g = crate::core::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_native_overlay_alpha(&mut self, alpha: f32) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        ov.set_overlay_alpha(alpha);
        let g = crate::core::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, enabled)")]
    pub fn set_native_altitude_overlay_enabled(&mut self, enabled: bool) -> PyResult<()> {
        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        ov.set_altitude_enabled(enabled);
        // Ensure height view is bound
        if let Some(ref hv) = self.height_view {
            let g = crate::core::gpu::ctx();
            ov.recreate_bind_group(&g.device, None, Some(hv), None, None);
        }
        let g = crate::core::gpu::ctx();
        ov.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, image)")]
    pub fn set_native_overlay_texture(&mut self, image: &pyo3::PyAny) -> PyResult<()> {
        // Accept HxWx3 or HxWx4 uint8
        let (h, w, c, data) = if let Ok(arr) = image.extract::<PyReadonlyArray3<u8>>() {
            let shape = arr.shape();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "overlay must be HxWxC",
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

        let Some(ref mut ov) = self.overlay_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Overlay renderer not available",
            ));
        };
        let g = crate::core::gpu::ctx();
        // Prepare RGBA data with row padding to COPY_BYTES_PER_ROW_ALIGNMENT
        let mut rgba: Vec<u8>;
        if c == 3 {
            rgba = Vec::with_capacity((h * w * 4) as usize);
            let mut idx = 0usize;
            for _yy in 0..h {
                for _xx in 0..w {
                    let r = data[idx];
                    let gch = data[idx + 1];
                    let b = data[idx + 2];
                    rgba.push(r);
                    rgba.push(gch);
                    rgba.push(b);
                    rgba.push(255);
                    idx += 3;
                }
            }
        } else {
            rgba = data; // already RGBA
        }
        let row_bytes = w * 4;
        let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
        let mut padded = vec![0u8; (padded_bpr * h) as usize];
        for y in 0..h as usize {
            let s = y * row_bytes as usize;
            let d = y * padded_bpr as usize;
            padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
        }

        // Create texture and upload
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native_overlay_tex"),
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
        ov.overlay_tex = Some(tex);
        ov.overlay_view = Some(view);
        // Recreate bind group with new overlay view and existing height view
        let height_view = self.height_view.as_ref();
        ov.recreate_bind_group(&g.device, None, height_view, None, None);
        Ok(())
    }
}
