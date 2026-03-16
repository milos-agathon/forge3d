use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B6: Depth of Field API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_dof(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::dof::DofQuality::Low,
            "medium" => crate::core::dof::DofQuality::Medium,
            "high" => crate::core::dof::DofQuality::High,
            "ultra" => crate::core::dof::DofQuality::Ultra,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality '{}'. Use 'low', 'medium', 'high', or 'ultra'",
                    other
                )))
            }
        };

        let g = crate::core::gpu::ctx();
        let renderer =
            crate::core::dof::DofRenderer::new(&g.device, self.width, self.height, quality_enum);

        self.dof_renderer = Some(renderer);
        self.dof_enabled = true;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_dof(&mut self) {
        self.dof_enabled = false;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn dof_enabled(&self) -> bool {
        self.dof_enabled
    }

    #[pyo3(text_signature = "($self, aperture, focus_distance, focal_length)")]
    pub fn set_dof_camera_params(
        &mut self,
        aperture: f32,
        focus_distance: f32,
        focal_length: f32,
    ) -> PyResult<()> {
        self.dof_params = crate::camera::create_camera_dof_params(
            aperture,
            focus_distance,
            focal_length,
            false,
            2.0,
        )?;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_camera_params(self.dof_params);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, f_stop)")]
    pub fn set_dof_f_stop(&mut self, f_stop: f32) -> PyResult<()> {
        let aperture = crate::camera::camera_f_stop_to_aperture(f_stop)?;
        self.dof_params.aperture = aperture;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_aperture(aperture);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, distance)")]
    pub fn set_dof_focus_distance(&mut self, distance: f32) -> PyResult<()> {
        if !distance.is_finite() || distance <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "focus_distance must be finite and > 0",
            ));
        }

        self.dof_params.focus_distance = distance;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_focus_distance(distance);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, focal_length)")]
    pub fn set_dof_focal_length(&mut self, focal_length: f32) -> PyResult<()> {
        if !focal_length.is_finite() || focal_length <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "focal_length must be finite and > 0",
            ));
        }

        self.dof_params.focal_length = focal_length;

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_focal_length(focal_length);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, rotation)")]
    pub fn set_dof_bokeh_rotation(&mut self, rotation: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_bokeh_rotation(rotation);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, near_range, far_range)")]
    pub fn set_dof_transition_ranges(&mut self, near_range: f32, far_range: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_transition_ranges(near_range, far_range);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, bias)")]
    pub fn set_dof_coc_bias(&mut self, bias: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_coc_bias(bias);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, method)")]
    pub fn set_dof_method(&mut self, method: &str) -> PyResult<()> {
        let method_enum = match method {
            "gather" => crate::core::dof::DofMethod::Gather,
            "separable" => crate::core::dof::DofMethod::Separable,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid method '{}'. Use 'gather' or 'separable'",
                    other
                )))
            }
        };

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_method(method_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_dof_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_debug_mode(mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, show)")]
    pub fn set_dof_show_coc(&mut self, show: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.set_show_coc(show);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DOF not enabled. Call enable_dof() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dof_params(&self) -> (f32, f32, f32) {
        (
            self.dof_params.aperture,
            self.dof_params.focus_distance,
            self.dof_params.focal_length,
        )
    }
}
