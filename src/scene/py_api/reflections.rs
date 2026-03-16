use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B5: Planar Reflections API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_reflections(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::reflections::ReflectionQuality::Low,
            "medium" => crate::core::reflections::ReflectionQuality::Medium,
            "high" => crate::core::reflections::ReflectionQuality::High,
            "ultra" => crate::core::reflections::ReflectionQuality::Ultra,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality '{}' . Use 'low', 'medium', 'high', or 'ultra'",
                    other
                )))
            }
        };

        let g = crate::core::gpu::ctx();
        let mut renderer =
            crate::core::reflections::PlanarReflectionRenderer::new(&g.device, quality_enum);

        if let Some(previous) = self.reflection_renderer.take() {
            let prev_uniforms = previous.uniforms;
            renderer.uniforms.reflection_plane = prev_uniforms.reflection_plane;
            renderer.set_intensity(prev_uniforms.reflection_intensity);
            renderer.set_fresnel_power(prev_uniforms.fresnel_power);
            renderer.set_distance_fade(
                prev_uniforms.distance_fade_start,
                prev_uniforms.distance_fade_end,
            );
            renderer.set_debug_mode(prev_uniforms.debug_mode);
            renderer.uniforms.camera_position = prev_uniforms.camera_position;
        }

        renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
        renderer.set_enabled(true);
        renderer.upload_uniforms(&g.queue);

        self.reflection_renderer = Some(renderer);
        self.reflections_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn disable_reflections(&mut self) {
        self.reflections_enabled = false;
        if let Some(ref mut renderer) = self.reflection_renderer {
            renderer.set_enabled(false);
            let g = crate::core::gpu::ctx();
            renderer.upload_uniforms(&g.queue);
        }
    }

    #[pyo3(text_signature = "($self, normal, point, size)")]
    pub fn set_reflection_plane(
        &mut self,
        normal: (f32, f32, f32),
        point: (f32, f32, f32),
        size: (f32, f32, f32),
    ) -> PyResult<()> {
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        let normal_v = glam::Vec3::new(normal.0, normal.1, normal.2);
        let point_v = glam::Vec3::new(point.0, point.1, point.2);
        let size_v = glam::Vec3::new(size.0, size.1, size.2);
        renderer.set_reflection_plane(normal_v, point_v, size_v);
        let g = crate::core::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_reflection_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_intensity(intensity);
        let g = crate::core::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, power)")]
    pub fn set_reflection_fresnel_power(&mut self, power: f32) -> PyResult<()> {
        if power <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Fresnel power must be positive.",
            ));
        }
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_fresnel_power(power);
        let g = crate::core::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, start, end)")]
    pub fn set_reflection_distance_fade(&mut self, start: f32, end: f32) -> PyResult<()> {
        if end <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distance_fade_end must be positive.",
            ));
        }
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_distance_fade(start, end);
        let g = crate::core::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_reflection_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if mode > 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Debug mode must be an integer in [0, 4].",
            ));
        }
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref mut renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        renderer.set_debug_mode(mode);
        let g = crate::core::gpu::ctx();
        renderer.upload_uniforms(&g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn reflection_performance_info(&self) -> PyResult<(f32, bool)> {
        if !self.reflections_enabled {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        }
        let Some(ref renderer) = self.reflection_renderer else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Reflections not enabled. Call enable_reflections() first.",
            ));
        };
        let cost = renderer.estimate_frame_cost();
        let meets_requirement = renderer.meets_performance_requirement();
        Ok((cost, meets_requirement))
    }
}
