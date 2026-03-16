use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B16: Dual-source blending OIT Methods

    #[pyo3(text_signature = "($self, mode, quality)")]
    pub fn enable_dual_source_oit(
        &mut self,
        mode: Option<&str>,
        quality: Option<&str>,
    ) -> PyResult<()> {
        let g = crate::core::gpu::ctx();

        // Parse mode
        let oit_mode = match mode {
            Some("dual_source") => crate::core::dual_source_oit::DualSourceOITMode::DualSource,
            Some("wboit_fallback") => {
                crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback
            }
            Some("automatic") | None => crate::core::dual_source_oit::DualSourceOITMode::Automatic,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid mode. Use 'dual_source', 'wboit_fallback', or 'automatic'.",
                ))
            }
        };

        // Parse quality
        let oit_quality = match quality {
            Some("low") => crate::core::dual_source_oit::DualSourceOITQuality::Low,
            Some("medium") | None => crate::core::dual_source_oit::DualSourceOITQuality::Medium,
            Some("high") => crate::core::dual_source_oit::DualSourceOITQuality::High,
            Some("ultra") => crate::core::dual_source_oit::DualSourceOITQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid quality. Use 'low', 'medium', 'high', or 'ultra'.",
                ))
            }
        };

        // Create dual-source OIT renderer
        let mut renderer = crate::core::dual_source_oit::DualSourceOITRenderer::new(
            &g.device,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create dual-source OIT renderer: {}",
                e
            ))
        })?;

        renderer.set_mode(oit_mode);
        renderer.set_quality(oit_quality);
        renderer.set_enabled(true);

        self.dual_source_oit_renderer = Some(renderer);
        self.dual_source_oit_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_dual_source_oit(&mut self) -> PyResult<()> {
        self.dual_source_oit_enabled = false;
        self.dual_source_oit_renderer = None;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_dual_source_oit_enabled(&self) -> bool {
        self.dual_source_oit_enabled && self.dual_source_oit_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_dual_source_oit_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dual_source_oit_renderer {
            let oit_mode = match mode {
                "dual_source" => crate::core::dual_source_oit::DualSourceOITMode::DualSource,
                "wboit_fallback" => crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback,
                "automatic" => crate::core::dual_source_oit::DualSourceOITMode::Automatic,
                "disabled" => crate::core::dual_source_oit::DualSourceOITMode::Disabled,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid mode. Use 'dual_source', 'wboit_fallback', 'automatic', or 'disabled'.",
                )),
            };

            renderer.set_mode(oit_mode);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dual_source_oit_mode(&self) -> PyResult<String> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            let mode = match renderer.get_operating_mode() {
                crate::core::dual_source_oit::DualSourceOITMode::DualSource => "dual_source",
                crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback => "wboit_fallback",
                crate::core::dual_source_oit::DualSourceOITMode::Automatic => "automatic",
                crate::core::dual_source_oit::DualSourceOITMode::Disabled => "disabled",
            };
            Ok(mode.to_string())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, quality)")]
    pub fn set_dual_source_oit_quality(&mut self, quality: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.dual_source_oit_renderer {
            let oit_quality = match quality {
                "low" => crate::core::dual_source_oit::DualSourceOITQuality::Low,
                "medium" => crate::core::dual_source_oit::DualSourceOITQuality::Medium,
                "high" => crate::core::dual_source_oit::DualSourceOITQuality::High,
                "ultra" => crate::core::dual_source_oit::DualSourceOITQuality::Ultra,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid quality. Use 'low', 'medium', 'high', or 'ultra'.",
                    ))
                }
            };

            renderer.set_quality(oit_quality);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dual_source_oit_quality(&self) -> PyResult<String> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            let quality = match renderer.quality() {
                crate::core::dual_source_oit::DualSourceOITQuality::Low => "low",
                crate::core::dual_source_oit::DualSourceOITQuality::Medium => "medium",
                crate::core::dual_source_oit::DualSourceOITQuality::High => "high",
                crate::core::dual_source_oit::DualSourceOITQuality::Ultra => "ultra",
            };
            Ok(quality.to_string())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_dual_source_supported(&self) -> PyResult<bool> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            Ok(renderer.is_dual_source_supported())
        } else {
            // Check hardware support without creating renderer
            let g = crate::core::gpu::ctx();
            let test_renderer = crate::core::dual_source_oit::DualSourceOITRenderer::new(
                &g.device,
                256,
                256,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            );
            match test_renderer {
                Ok(renderer) => Ok(renderer.is_dual_source_supported()),
                Err(_) => Ok(false),
            }
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_dual_source_oit_stats(&self) -> PyResult<(u64, u64, u64, f32, f32, f32)> {
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            let stats = renderer.get_stats();
            Ok((
                stats.frames_rendered,
                stats.dual_source_frames,
                stats.wboit_fallback_frames,
                stats.average_fragment_count,
                stats.peak_fragment_count,
                stats.quality_score,
            ))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }

    // P0.1/M1: Simple OIT API (alias for enable_dual_source_oit with sensible defaults)
    /// Enable Order-Independent Transparency for correct rendering of overlapping
    /// transparent surfaces (water, volumetrics, vector overlays).
    ///
    /// This is the recommended way to enable OIT. It automatically selects
    /// dual-source blending if hardware supports it, otherwise falls back to WBOIT.
    ///
    /// Args:
    ///     mode: Optional transparency mode ('standard', 'wboit', 'auto'). Default: 'auto'
    ///
    /// Example:
    ///     scene.enable_oit()  # automatic mode selection
    ///     scene.enable_oit('wboit')  # force weighted-blended OIT
    #[pyo3(text_signature = "($self, mode='auto')")]
    pub fn enable_oit(&mut self, mode: Option<&str>) -> PyResult<()> {
        let oit_mode = match mode {
            Some("standard") | Some("disabled") => {
                // Standard alpha blending (disable OIT)
                self.dual_source_oit_enabled = false;
                self.dual_source_oit_renderer = None;
                return Ok(());
            }
            Some("wboit") => Some("wboit_fallback"),
            Some("dual_source") => Some("dual_source"),
            Some("auto") | None => Some("automatic"),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid OIT mode '{}'. Use 'standard', 'wboit', 'dual_source', or 'auto'.",
                    other
                )))
            }
        };
        // Delegate to the full implementation with medium quality
        self.enable_dual_source_oit(oit_mode, Some("medium"))
    }

    /// Disable Order-Independent Transparency, reverting to standard alpha blending.
    #[pyo3(text_signature = "($self)")]
    pub fn disable_oit(&mut self) {
        self.dual_source_oit_enabled = false;
        self.dual_source_oit_renderer = None;
    }

    /// Check if Order-Independent Transparency is currently enabled.
    #[pyo3(text_signature = "($self)")]
    pub fn is_oit_enabled(&self) -> bool {
        self.dual_source_oit_enabled && self.dual_source_oit_renderer.is_some()
    }

    /// Get current OIT mode as a string ('auto', 'wboit', 'dual_source', or 'disabled').
    #[pyo3(text_signature = "($self)")]
    pub fn get_oit_mode(&self) -> String {
        if !self.dual_source_oit_enabled {
            return "disabled".to_string();
        }
        if let Some(ref renderer) = self.dual_source_oit_renderer {
            match renderer.get_operating_mode() {
                crate::core::dual_source_oit::DualSourceOITMode::DualSource => "dual_source",
                crate::core::dual_source_oit::DualSourceOITMode::WBOITFallback => "wboit",
                crate::core::dual_source_oit::DualSourceOITMode::Automatic => "auto",
                crate::core::dual_source_oit::DualSourceOITMode::Disabled => "disabled",
            }
            .to_string()
        } else {
            "disabled".to_string()
        }
    }

    #[pyo3(
        text_signature = "($self, alpha_correction, depth_weight_scale, max_fragments, premultiply_factor)"
    )]
    pub fn set_dual_source_oit_params(
        &mut self,
        _alpha_correction: f32,
        _depth_weight_scale: f32,
        _max_fragments: f32,
        _premultiply_factor: f32,
    ) -> PyResult<()> {
        if let Some(ref mut _renderer) = self.dual_source_oit_renderer {
            // Renderer uniforms are not exposed yet; keep this API as a no-op.
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Dual-source OIT not enabled. Call enable_dual_source_oit() first.",
            ))
        }
    }
}
