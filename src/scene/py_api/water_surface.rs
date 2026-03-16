use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B11: Water Surface Color Toggle API
    #[pyo3(text_signature = "($self)")]
    pub fn enable_water_surface(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let renderer = crate::core::water_surface::WaterSurfaceRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(wgpu::TextureFormat::Depth32Float),
            1,
        );

        self.water_surface_renderer = Some(renderer);
        self.water_surface_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_water_surface(&mut self) {
        self.water_surface_enabled = false;
        self.water_surface_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_water_surface_enabled(&self) -> bool {
        self.water_surface_enabled && self.water_surface_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_water_surface_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            let mode_enum =
                match mode {
                    "disabled" => crate::core::water_surface::WaterSurfaceMode::Disabled,
                    "transparent" => crate::core::water_surface::WaterSurfaceMode::Transparent,
                    "reflective" => crate::core::water_surface::WaterSurfaceMode::Reflective,
                    "animated" => crate::core::water_surface::WaterSurfaceMode::Animated,
                    _ => return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'disabled', 'transparent', 'reflective', 'animated'",
                    )),
                };
            renderer.set_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, height)")]
    pub fn set_water_surface_height(&mut self, height: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_height(height);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, size)")]
    pub fn set_water_surface_size(&mut self, size: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_size(size);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b)")]
    pub fn set_water_base_color(&mut self, r: f32, g: f32, b: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_base_color(glam::Vec3::new(r, g, b));
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, hue_shift)")]
    pub fn set_water_hue_shift(&mut self, hue_shift: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_hue_shift(hue_shift);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b, strength)")]
    pub fn set_water_tint(&mut self, r: f32, g: f32, b: f32, strength: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_tint(glam::Vec3::new(r, g, b), strength);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, alpha)")]
    pub fn set_water_alpha(&mut self, alpha: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_alpha(alpha);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, amplitude, frequency, speed)")]
    pub fn set_water_wave_params(
        &mut self,
        amplitude: f32,
        frequency: f32,
        speed: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_wave_params(amplitude, frequency, speed);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, direction_x, direction_y)")]
    pub fn set_water_flow_direction(&mut self, direction_x: f32, direction_y: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_flow_direction(glam::Vec2::new(direction_x, direction_y));
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, reflection_strength, refraction_strength, fresnel_power, roughness)"
    )]
    pub fn set_water_lighting_params(
        &mut self,
        reflection_strength: f32,
        refraction_strength: f32,
        fresnel_power: f32,
        roughness: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.set_lighting_params(
                reflection_strength,
                refraction_strength,
                fresnel_power,
                roughness,
            );
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_water_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            let params = match preset {
                "ocean" => crate::core::water_surface::WaterSurfaceRenderer::create_ocean_water(),
                "lake" => crate::core::water_surface::WaterSurfaceRenderer::create_lake_water(),
                "river" => crate::core::water_surface::WaterSurfaceRenderer::create_river_water(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Preset must be one of: 'ocean', 'lake', 'river'",
                    ))
                }
            };
            renderer.update_params(params);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, delta_time)")]
    pub fn update_water_animation(&mut self, delta_time: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.water_surface_renderer {
            renderer.update(delta_time);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_water_surface_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.water_surface_renderer {
            Ok(renderer.get_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Water surface not enabled. Call enable_water_surface() first.",
            ))
        }
    }
}
