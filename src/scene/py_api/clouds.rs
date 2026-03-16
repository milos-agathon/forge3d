use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B8: Realtime Clouds API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_clouds(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::clouds::CloudQuality::Low,
            "medium" => crate::core::clouds::CloudQuality::Medium,
            "high" => crate::core::clouds::CloudQuality::High,
            "ultra" => crate::core::clouds::CloudQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Quality must be one of: 'low', 'medium', 'high', 'ultra'",
                ))
            }
        };

        let g = crate::core::gpu::ctx();
        let mut renderer = crate::core::clouds::CloudRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1, // clouds render against resolved color buffer
        );
        renderer.set_quality(quality_enum);
        renderer
            .prepare_frame(&g.device, &g.queue)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        renderer.upload_uniforms(&g.queue);
        renderer.set_enabled(true);

        self.cloud_renderer = Some(renderer);
        self.clouds_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_clouds(&mut self) {
        self.clouds_enabled = false;
        self.cloud_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_clouds_enabled(&self) -> bool {
        self.clouds_enabled && self.cloud_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, density)")]
    pub fn set_realtime_cloud_density(&mut self, density: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_density(density);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, coverage)")]
    pub fn set_realtime_cloud_coverage(&mut self, coverage: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_coverage(coverage);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, scale)")]
    pub fn set_realtime_cloud_scale(&mut self, scale: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_scale(scale);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, direction_x, direction_y, strength)")]
    pub fn set_realtime_cloud_wind(
        &mut self,
        direction_x: f32,
        direction_y: f32,
        strength: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            let direction = glam::Vec2::new(direction_x, direction_y);
            renderer.set_wind(direction, strength);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_realtime_cloud_animation_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            let preset_enum = match preset {
                "static" => crate::core::clouds::CloudAnimationPreset::Static,
                "gentle" => crate::core::clouds::CloudAnimationPreset::Gentle,
                "moderate" => crate::core::clouds::CloudAnimationPreset::Moderate,
                "stormy" => crate::core::clouds::CloudAnimationPreset::Stormy,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Preset must be one of: 'static', 'gentle', 'moderate', 'stormy'",
                    ))
                }
            };
            renderer.set_animation_preset(preset_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_cloud_render_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            let mode_enum = match mode {
                "billboard" => crate::core::clouds::CloudRenderMode::Billboard,
                "volumetric" => crate::core::clouds::CloudRenderMode::Volumetric,
                "hybrid" => crate::core::clouds::CloudRenderMode::Hybrid,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'billboard', 'volumetric', 'hybrid'",
                    ))
                }
            };
            renderer.set_render_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, delta_time)")]
    pub fn update_realtime_cloud_animation(&mut self, delta_time: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.update(delta_time);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_clouds_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.cloud_renderer {
            Ok(renderer.get_params())
        } else if let Some(ref renderer) = self.cloud_shadow_renderer {
            Ok(renderer.get_cloud_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }
}
