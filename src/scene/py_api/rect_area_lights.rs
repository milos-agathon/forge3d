use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B14: Rect Area Lights (LTC) API
    #[pyo3(text_signature = "($self, max_lights=16)")]
    pub fn enable_ltc_rect_area_lights(&mut self, max_lights: Option<usize>) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let max_lights = max_lights.unwrap_or(16);

        let renderer = crate::core::ltc_area_lights::LTCRectAreaLightRenderer::new(
            g.device.clone(),
            max_lights,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create LTC rect area light renderer: {}",
                e
            ))
        })?;

        self.ltc_area_lights_renderer = Some(renderer);
        self.ltc_area_lights_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_ltc_rect_area_lights(&mut self) {
        self.ltc_area_lights_enabled = false;
        self.ltc_area_lights_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ltc_rect_area_lights_enabled(&self) -> bool {
        self.ltc_area_lights_enabled && self.ltc_area_lights_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, x, y, z, width, height, r, g, b, intensity)")]
    pub fn add_rect_area_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) -> PyResult<usize> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            let light = crate::core::ltc_area_lights::RectAreaLight::quad(
                glam::Vec3::new(x, y, z),
                width,
                height,
                glam::Vec3::new(r, g, b),
                intensity,
            );

            let light_id = renderer
                .add_light(light)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(light_id)
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, position, right_vec, up_vec, width, height, r, g, b, intensity, two_sided=False)"
    )]
    pub fn add_custom_rect_area_light(
        &mut self,
        position: (f32, f32, f32),
        right_vec: (f32, f32, f32),
        up_vec: (f32, f32, f32),
        width: f32,
        height: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
        two_sided: Option<bool>,
    ) -> PyResult<usize> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            let light = crate::core::ltc_area_lights::RectAreaLight::new(
                glam::Vec3::new(position.0, position.1, position.2),
                glam::Vec3::new(right_vec.0, right_vec.1, right_vec.2),
                glam::Vec3::new(up_vec.0, up_vec.1, up_vec.2),
                width,
                height,
                glam::Vec3::new(r, g, b),
                intensity,
                two_sided.unwrap_or(false),
            );

            let light_id = renderer
                .add_light(light)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(light_id)
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id)")]
    pub fn remove_rect_area_light(&mut self, light_id: usize) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            renderer
                .remove_light(light_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, x, y, z, width, height, r, g, b, intensity)")]
    pub fn update_rect_area_light(
        &mut self,
        light_id: usize,
        x: f32,
        y: f32,
        z: f32,
        width: f32,
        height: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            let light = crate::core::ltc_area_lights::RectAreaLight::quad(
                glam::Vec3::new(x, y, z),
                width,
                height,
                glam::Vec3::new(r, g, b),
                intensity,
            );

            renderer
                .update_light(light_id, light)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_rect_area_light_count(&self) -> PyResult<usize> {
        if let Some(ref renderer) = self.ltc_area_lights_renderer {
            Ok(renderer.light_count())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_ltc_global_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            renderer.set_global_intensity(intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, enabled)")]
    pub fn set_ltc_approximation_enabled(&mut self, enabled: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ltc_area_lights_renderer {
            renderer.set_ltc_enabled(enabled);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ltc_uniforms(&self) -> PyResult<(u32, f32, bool)> {
        if let Some(ref renderer) = self.ltc_area_lights_renderer {
            let uniforms = renderer.uniforms();
            Ok((
                uniforms.light_count,
                uniforms.global_intensity,
                uniforms.enable_ltc > 0.5,
            ))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LTC rect area lights not enabled. Call enable_ltc_rect_area_lights() first.",
            ))
        }
    }
}
