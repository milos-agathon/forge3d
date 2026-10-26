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
        super::super::overlay_upload::RasterOverlayTarget {
            allocation_owner: &self.allocation_owner,
            renderer: self.overlay_renderer.as_mut(),
            width: self.width,
            height: self.height,
            height_view: self.height_view.as_ref(),
        }
        .upload(image, alpha, offset_xy, scale)?;
        if self.overlay_renderer.is_some() {
            self.overlay_enabled = true;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_overlay(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_enabled(false);
            let g = crate::core::gpu::try_ctx()?;
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
            let g = crate::core::gpu::try_ctx()?;
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
            let g = crate::core::gpu::try_ctx()?;
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_altitude_overlay(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_altitude_enabled(false);
            let g = crate::core::gpu::try_ctx()?;
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
            let g = crate::core::gpu::try_ctx()?;
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
            let gctx = crate::core::gpu::try_ctx()?;
            ov.upload_uniforms(&gctx.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_gpu_contours(&mut self) -> PyResult<()> {
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.set_contours_enabled(false);
            let g = crate::core::gpu::try_ctx()?;
            ov.upload_uniforms(&g.queue);
        }
        Ok(())
    }
}
