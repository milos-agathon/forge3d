use super::LabelCompositor;
use pyo3::prelude::*;

#[pymethods]
impl LabelCompositor {
    fn set_raster_overlay(
        &mut self,
        image: &pyo3::types::PyAny,
        alpha: Option<f32>,
        offset_xy: Option<(i32, i32)>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        super::super::py_api::raster_overlay::RasterOverlayTarget {
            allocation_owner: &self.allocation_owner,
            renderer: self.overlay_renderer.as_mut(),
            width: self.width,
            height: self.height,
            height_view: None,
        }
        .set_raster_overlay(image, alpha, offset_xy, scale)?;
        self.overlay_enabled = true;
        Ok(())
    }

    fn set_native_text_atlas(
        &mut self,
        atlas: &pyo3::types::PyAny,
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

    fn clear_native_text(&mut self) {
        self.text_instances.clear();
        if let Some(renderer) = &mut self.text_overlay_renderer {
            renderer.instance_count = 0;
        }
    }

    fn enable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = true;
        if let Some(renderer) = &mut self.text_overlay_renderer {
            renderer.set_enabled(true);
            renderer.upload_uniforms(&crate::core::gpu::try_ctx()?.queue);
        }
        Ok(())
    }

    fn disable_native_text(&mut self) -> PyResult<()> {
        self.text_overlay_enabled = false;
        if let Some(renderer) = &mut self.text_overlay_renderer {
            renderer.set_enabled(false);
            renderer.upload_uniforms(&crate::core::gpu::try_ctx()?.queue);
        }
        Ok(())
    }

    #[pyo3(signature = (x, y, w, h, u0, v0, u1, v1, r, g, b, a,
        halo_r, halo_g, halo_b, halo_a, halo_width, rotation=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn add_native_text_rect_uv_halo(
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
}
