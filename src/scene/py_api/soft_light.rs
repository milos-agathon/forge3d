use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B12: Soft Light Radius (Raster) API
    #[pyo3(text_signature = "($self)")]
    pub fn enable_soft_light_radius(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let renderer = crate::core::soft_light_radius::SoftLightRadiusRenderer::new(&g.device);

        self.soft_light_radius_renderer = Some(renderer);
        self.soft_light_radius_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_soft_light_radius(&mut self) {
        self.soft_light_radius_enabled = false;
        self.soft_light_radius_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_soft_light_radius_enabled(&self) -> bool {
        self.soft_light_radius_enabled && self.soft_light_radius_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, x, y, z)")]
    pub fn set_soft_light_position(&mut self, x: f32, y: f32, z: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_light_position([x, y, z]);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_soft_light_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_light_intensity(intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b)")]
    pub fn set_soft_light_color(&mut self, r: f32, g: f32, b: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_light_color([r, g, b]);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, radius)")]
    pub fn set_light_inner_radius(&mut self, radius: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_inner_radius(radius);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, radius)")]
    pub fn set_light_outer_radius(&mut self, radius: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_outer_radius(radius);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, exponent)")]
    pub fn set_light_falloff_exponent(&mut self, exponent: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_falloff_exponent(exponent);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, softness)")]
    pub fn set_light_edge_softness(&mut self, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_edge_softness(softness);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_light_falloff_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            let mode_enum = match mode {
                "linear" => crate::core::soft_light_radius::SoftLightFalloffMode::Linear,
                "quadratic" => crate::core::soft_light_radius::SoftLightFalloffMode::Quadratic,
                "cubic" => crate::core::soft_light_radius::SoftLightFalloffMode::Cubic,
                "exponential" => crate::core::soft_light_radius::SoftLightFalloffMode::Exponential,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'linear', 'quadratic', 'cubic', 'exponential'",
                    ))
                }
            };
            renderer.set_falloff_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, softness)")]
    pub fn set_light_shadow_softness(&mut self, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            renderer.set_shadow_softness(softness);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_light_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.soft_light_radius_renderer {
            let preset_enum = match preset {
                "spotlight" => crate::core::soft_light_radius::SoftLightPreset::Spotlight,
                "area_light" => crate::core::soft_light_radius::SoftLightPreset::AreaLight,
                "ambient_light" => crate::core::soft_light_radius::SoftLightPreset::AmbientLight,
                "candle" => crate::core::soft_light_radius::SoftLightPreset::Candle,
                "street_lamp" => crate::core::soft_light_radius::SoftLightPreset::StreetLamp,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Preset must be one of: 'spotlight', 'area_light', 'ambient_light', 'candle', 'street_lamp'"
                )),
            };
            renderer.apply_preset(preset_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_light_effective_range(&self) -> PyResult<f32> {
        if let Some(ref renderer) = self.soft_light_radius_renderer {
            Ok(renderer.effective_range())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, x, y, z)")]
    pub fn light_affects_point(&self, x: f32, y: f32, z: f32) -> PyResult<bool> {
        if let Some(ref renderer) = self.soft_light_radius_renderer {
            Ok(renderer.affects_point([x, y, z]))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Soft light radius not enabled. Call enable_soft_light_radius() first.",
            ))
        }
    }
}
