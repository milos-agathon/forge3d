use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    pub fn set_light_position(&mut self, light_id: u32, x: f32, y: f32, z: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_position([x, y, z]);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, dir_x, dir_y, dir_z)")]
    pub fn set_light_direction(
        &mut self,
        light_id: u32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_direction([dir_x, dir_y, dir_z]);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, r, g, b)")]
    pub fn set_light_color(&mut self, light_id: u32, r: f32, g: f32, b: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_color([r, g, b]);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, intensity)")]
    pub fn set_light_intensity(&mut self, light_id: u32, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_intensity(intensity);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, range)")]
    pub fn set_light_range(&mut self, light_id: u32, range: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_range(range);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, inner_cone_deg, outer_cone_deg)")]
    pub fn set_spot_light_cone(
        &mut self,
        light_id: u32,
        inner_cone_deg: f32,
        outer_cone_deg: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_cone_angles(inner_cone_deg, outer_cone_deg);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, softness)")]
    pub fn set_spot_light_penumbra(&mut self, light_id: u32, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_penumbra_softness(softness);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, enabled)")]
    pub fn set_light_shadows(&mut self, light_id: u32, enabled: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light_mut(light_id) {
                light.set_shadow_enabled(enabled);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b, intensity)")]
    pub fn set_ambient_lighting(&mut self, r: f32, g: f32, b: f32, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            renderer.set_ambient([r, g, b], intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, quality)")]
    pub fn set_shadow_quality(&mut self, quality: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let quality_enum = match quality {
                "off" => crate::core::point_spot_lights::ShadowQuality::Off,
                "low" => crate::core::point_spot_lights::ShadowQuality::Low,
                "medium" => crate::core::point_spot_lights::ShadowQuality::Medium,
                "high" => crate::core::point_spot_lights::ShadowQuality::High,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Quality must be one of: 'off', 'low', 'medium', 'high'",
                    ))
                }
            };
            renderer.set_shadow_quality(quality_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_lighting_debug_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let mode_enum = match mode {
                "normal" => crate::core::point_spot_lights::DebugMode::Normal,
                "show_light_bounds" => crate::core::point_spot_lights::DebugMode::ShowLightBounds,
                "show_shadows" => crate::core::point_spot_lights::DebugMode::ShowShadows,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'normal', 'show_light_bounds', 'show_shadows'",
                    ))
                }
            };
            renderer.set_debug_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }
}
