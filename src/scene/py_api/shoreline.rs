use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // C3 (native): Shoreline foam controls (mirror fallback API names)
    #[pyo3(text_signature = "($self)")]
    pub fn enable_shoreline_foam(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_foam_enabled(true);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_shoreline_foam(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_foam_enabled(false);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, width_px, intensity, noise_scale)")]
    pub fn set_shoreline_foam_params(
        &mut self,
        width_px: f32,
        intensity: f32,
        noise_scale: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_foam_params(width_px, intensity, noise_scale);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    // C1 (native): Upload a water mask from a numpy array (u8 or bool)
    //  - dtype=uint8: values interpreted in [0,255]
    //  - dtype=bool : True->255, False->0
    #[pyo3(text_signature = "($self, mask)")]
    pub fn set_water_mask(&mut self, _py: pyo3::Python<'_>, mask: &pyo3::PyAny) -> PyResult<()> {
        let (height, width, data_vec_u8) =
            if let Ok(arr_u8) = mask.extract::<PyReadonlyArray2<u8>>() {
                let shape = arr_u8.shape();
                let h = shape[0] as u32;
                let w = shape[1] as u32;
                // Ensure contiguous data
                let v = arr_u8.as_array().to_owned().into_raw_vec();
                (h, w, v)
            } else if let Ok(arr_b) = mask.extract::<PyReadonlyArray2<bool>>() {
                let a = arr_b.as_array();
                let h = a.shape()[0] as u32;
                let w = a.shape()[1] as u32;
                let mut v = Vec::<u8>::with_capacity((h as usize) * (w as usize));
                for &b in a.iter() {
                    v.push(if b { 255 } else { 0 });
                }
                (h, w, v)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "mask must be a 2D numpy array of dtype uint8 or bool",
                ));
            };

        if let Some(ref mut renderer) = self.water_surface_renderer {
            let g = crate::core::gpu::ctx();
            renderer.upload_water_mask(&g.device, &g.queue, &data_vec_u8, width, height);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    /// Set water surface debug mode.
    ///
    /// - 0: Normal rendering (default)
    /// - 100: Binary water mask (blue = water, gray = land)
    /// - 101: Shore-distance scalar (falsecolor ramp with white shoreline ring)
    /// - 102: IBL specular isolation (land = black, water shows compressed HDR fresnel)
    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_water_surface_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_debug_mode(mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }
}
