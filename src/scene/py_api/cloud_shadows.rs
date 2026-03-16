use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B7: Cloud Shadow API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_cloud_shadows(&mut self, quality: Option<&str>) -> PyResult<()> {
        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::cloud_shadows::CloudShadowQuality::Low,
            "medium" => crate::core::cloud_shadows::CloudShadowQuality::Medium,
            "high" => crate::core::cloud_shadows::CloudShadowQuality::High,
            "ultra" => crate::core::cloud_shadows::CloudShadowQuality::Ultra,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality '{}'. Use 'low', 'medium', 'high', or 'ultra'",
                    other
                )))
            }
        };

        let g = crate::core::gpu::ctx();
        let renderer =
            crate::core::cloud_shadows::CloudShadowRenderer::new(&g.device, quality_enum);

        self.cloud_shadow_renderer = Some(renderer);
        self.cloud_shadows_enabled = true;
        self.bg3_cloud_shadows = None; // Will be created on first render
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_cloud_shadows(&mut self) {
        self.cloud_shadows_enabled = false;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_cloud_shadows_enabled(&self) -> bool {
        self.cloud_shadows_enabled
    }

    #[pyo3(text_signature = "($self, speed_x, speed_y)")]
    pub fn set_cloud_speed(&mut self, speed_x: f32, speed_y: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_speed(glam::Vec2::new(speed_x, speed_y));
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, scale)")]
    pub fn set_cloud_scale(&mut self, scale: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_scale(scale);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_scale(scale);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, density)")]
    pub fn set_cloud_density(&mut self, density: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_density(density);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_density(density);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, coverage)")]
    pub fn set_cloud_coverage(&mut self, coverage: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_cloud_coverage(coverage);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.set_coverage(coverage);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, intensity)")]
    pub fn set_cloud_shadow_intensity(&mut self, intensity: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_shadow_intensity(intensity);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, softness)")]
    pub fn set_cloud_shadow_softness(&mut self, softness: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_shadow_softness(softness);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, direction, strength)")]
    pub fn set_cloud_wind(&mut self, direction: f32, strength: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_wind(direction, strength);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            let dir_vec = glam::Vec2::new(direction.cos(), direction.sin());
            renderer.set_wind(dir_vec, strength);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, x, y, strength)")]
    pub fn set_cloud_wind_vector(&mut self, x: f32, y: f32, strength: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            let angle = y.atan2(x);
            renderer.set_wind(angle, strength);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            let wind_vec = glam::Vec2::new(x, y);
            renderer.set_wind(wind_vec, strength);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, frequency, amplitude)")]
    pub fn set_cloud_noise_params(&mut self, frequency: f32, amplitude: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_noise_params(frequency, amplitude);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset_name)")]
    pub fn set_cloud_animation_preset(&mut self, preset_name: &str) -> PyResult<()> {
        let preset_lower = preset_name.to_ascii_lowercase();
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            let mapped = match preset_lower.as_str() {
                "static" => "calm",
                "gentle" => "calm",
                "moderate" => "windy",
                "stormy" => "stormy",
                other => other,
            };
            let params = crate::core::cloud_shadows::utils::create_animation_preset(mapped);
            renderer.set_animation_params(params);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            let preset_enum = match preset_lower.as_str() {
                "static" => crate::core::clouds::CloudAnimationPreset::Static,
                "gentle" | "calm" => crate::core::clouds::CloudAnimationPreset::Gentle,
                "moderate" => crate::core::clouds::CloudAnimationPreset::Moderate,
                "stormy" | "windy" => crate::core::clouds::CloudAnimationPreset::Stormy,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Preset must be one of: static, gentle, moderate, stormy (got '{}')",
                        other
                    )));
                }
            };
            renderer.set_animation_preset(preset_enum);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, delta_time)")]
    pub fn update_cloud_animation(&mut self, delta_time: f32) -> PyResult<()> {
        let mut updated = false;
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.update(delta_time);
            updated = true;
        }
        if let Some(ref mut renderer) = self.cloud_renderer {
            renderer.update(delta_time);
            renderer.upload_uniforms(&crate::core::gpu::ctx().queue);
            updated = true;
        }
        if updated {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Clouds not enabled. Call enable_clouds() or enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_cloud_debug_mode(&mut self, mode: u32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_debug_mode(mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, show)")]
    pub fn set_cloud_show_clouds_only(&mut self, show: bool) -> PyResult<()> {
        if let Some(ref mut renderer) = self.cloud_shadow_renderer {
            renderer.set_show_clouds_only(show);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_cloud_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.cloud_shadow_renderer {
            Ok(renderer.get_cloud_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cloud shadows not enabled. Call enable_cloud_shadows() first.",
            ))
        }
    }
}
