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
        signature = (x, y, w, h, u0, v0, u1, v1, r, g, b, a, halo_r, halo_g, halo_b, halo_a, halo_width, rotation=0.0)
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
        rotation: f32,
    ) -> PyResult<()> {
        self.text_instances
            .push(super::super::text_upload::glyph_instance(
                [x, y, w, h],
                [u0, v0, u1, v1],
                [r, g, b, a],
                [halo_r, halo_g, halo_b, halo_a],
                halo_width,
                rotation,
            )?);
        Ok(())
    }

    #[pyo3(text_signature = "($self, atlas, channels=3, smoothing=1.0)")]
    pub fn set_native_text_atlas(
        &mut self,
        atlas: &pyo3::PyAny,
        channels: Option<u32>,
        smoothing: Option<f32>,
    ) -> PyResult<()> {
        super::super::text_upload::upload_atlas(
            &self.allocation_owner,
            self.text_overlay_renderer.as_mut(),
            atlas,
            channels,
            smoothing,
        )
    }
}
