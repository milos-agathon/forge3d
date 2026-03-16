use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B15: Image-Based Lighting (IBL) Polish API
    #[pyo3(text_signature = "($self, quality='medium')")]
    pub fn enable_ibl(&mut self, quality: Option<&str>) -> PyResult<()> {
        let g = crate::core::gpu::ctx();

        let quality_enum = match quality.unwrap_or("medium") {
            "low" => crate::core::ibl::IBLQuality::Low,
            "medium" => crate::core::ibl::IBLQuality::Medium,
            "high" => crate::core::ibl::IBLQuality::High,
            "ultra" => crate::core::ibl::IBLQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Quality must be one of: 'low', 'medium', 'high', 'ultra'",
                ))
            }
        };

        let mut renderer = crate::core::ibl::IBLRenderer::new(&g.device, quality_enum);

        // Initialize with default environment
        renderer
            .initialize(&g.device, &g.queue)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        self.ibl_renderer = Some(renderer);
        self.ibl_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_ibl(&mut self) {
        self.ibl_enabled = false;
        self.ibl_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ibl_enabled(&self) -> bool {
        self.ibl_enabled && self.ibl_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, quality)")]
    pub fn set_ibl_quality(&mut self, quality: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ibl_renderer {
            let quality_enum = match quality {
                "low" => crate::core::ibl::IBLQuality::Low,
                "medium" => crate::core::ibl::IBLQuality::Medium,
                "high" => crate::core::ibl::IBLQuality::High,
                "ultra" => crate::core::ibl::IBLQuality::Ultra,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Quality must be one of: 'low', 'medium', 'high', 'ultra'",
                    ))
                }
            };

            renderer.set_quality(quality_enum);

            // Regenerate IBL textures with new quality
            let g = crate::core::gpu::ctx();
            renderer
                .initialize(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, hdr_data, width, height)")]
    pub fn load_environment_map(
        &mut self,
        hdr_data: Vec<f32>,
        width: u32,
        height: u32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ibl_renderer {
            let g = crate::core::gpu::ctx();
            renderer
                .load_environment_map(&g.device, &g.queue, &hdr_data, width, height)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Regenerate IBL textures with new environment
            renderer
                .initialize(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn generate_ibl_textures(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ibl_renderer {
            let g = crate::core::gpu::ctx();

            // Regenerate irradiance map
            renderer
                .generate_irradiance_map(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Regenerate specular map
            renderer
                .generate_specular_map(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Regenerate BRDF LUT
            renderer
                .generate_brdf_lut(&g.device, &g.queue)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ibl_quality(&self) -> PyResult<String> {
        if let Some(ref renderer) = self.ibl_renderer {
            let quality_str = match renderer.quality() {
                crate::core::ibl::IBLQuality::Low => "low",
                crate::core::ibl::IBLQuality::Medium => "medium",
                crate::core::ibl::IBLQuality::High => "high",
                crate::core::ibl::IBLQuality::Ultra => "ultra",
            };
            Ok(quality_str.to_string())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ibl_initialized(&self) -> PyResult<bool> {
        if let Some(ref renderer) = self.ibl_renderer {
            Ok(renderer.is_initialized())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ibl_texture_info(&self) -> PyResult<(String, String, String)> {
        if let Some(ref renderer) = self.ibl_renderer {
            let quality = renderer.quality();
            let (irr, spec, brdf) = renderer.textures();

            let irr_info = if irr.is_some() {
                format!(
                    "{}x{} (6 faces)",
                    quality.irradiance_size(),
                    quality.irradiance_size()
                )
            } else {
                "Not generated".to_string()
            };

            let spec_info = if spec.is_some() {
                format!(
                    "{}x{} (6 faces, {} mips)",
                    quality.specular_size(),
                    quality.specular_size(),
                    quality.specular_mip_levels()
                )
            } else {
                "Not generated".to_string()
            };

            let brdf_info = if brdf.is_some() {
                format!("{}x{}", quality.brdf_size(), quality.brdf_size())
            } else {
                "Not generated".to_string()
            };

            Ok((irr_info, spec_info, brdf_info))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ))
        }
    }

    // IBL Material property helpers (for future PBR integration)
    #[pyo3(text_signature = "($self, metallic, roughness, r, g, b)")]
    pub fn test_ibl_material(
        &self,
        metallic: f32,
        roughness: f32,
        r: f32,
        g: f32,
        b: f32,
    ) -> PyResult<(f32, f32, f32)> {
        if !self.is_ibl_enabled() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ));
        }

        // Test material properties for IBL rendering
        let metallic = metallic.clamp(0.0, 1.0);
        let _roughness = roughness.clamp(0.0, 1.0);

        // Calculate F0 for the material
        let dielectric_f0 = 0.04;
        let f0_r = r * metallic + dielectric_f0 * (1.0 - metallic);
        let f0_g = g * metallic + dielectric_f0 * (1.0 - metallic);
        let f0_b = b * metallic + dielectric_f0 * (1.0 - metallic);

        Ok((f0_r, f0_g, f0_b))
    }

    #[pyo3(text_signature = "($self, n_dot_v, roughness)")]
    pub fn sample_brdf_lut(&self, n_dot_v: f32, roughness: f32) -> PyResult<(f32, f32)> {
        if !self.is_ibl_enabled() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "IBL not enabled. Call enable_ibl() first.",
            ));
        }

        // Clamp inputs to valid ranges
        let n_dot_v = n_dot_v.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);

        // Simplified BRDF approximation for testing
        // In a real implementation, this would sample the actual BRDF LUT texture
        let a = 1.0 - roughness;
        let fresnel_term = a * (1.0 - n_dot_v).powf(5.0);
        let roughness_term = roughness * n_dot_v;

        Ok((fresnel_term, roughness_term))
    }
}
