use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // ---- Bloom methods (P1.2) ----

    /// Enable bloom post-processing.
    #[pyo3(text_signature = "()")]
    pub fn enable_bloom(&mut self) -> PyResult<()> {
        self.bloom_enabled = true;
        self.bloom_config.enabled = true;
        Ok(())
    }

    /// Disable bloom post-processing.
    #[pyo3(text_signature = "()")]
    pub fn disable_bloom(&mut self) -> PyResult<()> {
        self.bloom_enabled = false;
        self.bloom_config.enabled = false;
        Ok(())
    }

    /// Query whether bloom is enabled.
    #[pyo3(text_signature = "()")]
    pub fn is_bloom_enabled(&self) -> bool {
        self.bloom_enabled
    }

    /// Apply bloom settings.
    ///
    /// Parameters
    /// ----------
    /// threshold : float
    ///     Brightness threshold for bloom extraction (>= 0).
    /// softness : float
    ///     Threshold transition softness in [0, 1].
    /// strength : float
    ///     Bloom intensity when compositing (>= 0).
    /// radius : float
    ///     Blur radius multiplier (> 0).
    #[pyo3(text_signature = "($self, threshold=1.5, softness=0.5, strength=0.3, radius=1.0)")]
    pub fn set_bloom_settings(
        &mut self,
        threshold: Option<f32>,
        softness: Option<f32>,
        strength: Option<f32>,
        radius: Option<f32>,
    ) -> PyResult<()> {
        if let Some(t) = threshold {
            if t < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "threshold must be >= 0",
                ));
            }
            self.bloom_config.threshold = t;
        }
        if let Some(s) = softness {
            if !(0.0..=1.0).contains(&s) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "softness must be in [0, 1]",
                ));
            }
            self.bloom_config.softness = s;
        }
        if let Some(st) = strength {
            if st < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "strength must be >= 0",
                ));
            }
            self.bloom_config.strength = st;
        }
        if let Some(r) = radius {
            if r <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "radius must be > 0",
                ));
            }
            self.bloom_config.radius = r;
        }
        Ok(())
    }

    /// Return current bloom settings as a dict.
    #[pyo3(text_signature = "()")]
    pub fn get_bloom_settings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("enabled", self.bloom_config.enabled)?;
        dict.set_item("threshold", self.bloom_config.threshold)?;
        dict.set_item("softness", self.bloom_config.softness)?;
        dict.set_item("strength", self.bloom_config.strength)?;
        dict.set_item("radius", self.bloom_config.radius)?;
        Ok(dict.into())
    }

    /// Render the current frame to a PNG on disk.
    ///
    /// Parameters
    /// ----------
    /// path : str | os.PathLike
    ///     Destination file path for the PNG image.
    ///
    /// Notes
    /// -----
    /// The written PNG's raw RGBA bytes will match those returned by
    /// `Scene.render_rgba()` on the same frame (row-major, C-contiguous).
    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: PathBuf) -> PyResult<()> {
        self.render_png_impl(&path)
    }
    #[pyo3(text_signature = "($self)")]
    pub fn render_rgba<'py>(
        &mut self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray3<u8>>> {
        self.render_rgba_impl(py)
    }

    #[pyo3(text_signature = "($self, samples)")]
    pub fn set_msaa_samples(&mut self, samples: u32) -> PyResult<u32> {
        const SUPPORTED: [u32; 4] = [1, 2, 4, 8];
        if !SUPPORTED.contains(&samples) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unsupported MSAA sample count: {} (allowed: {:?})",
                samples, SUPPORTED
            )));
        }
        if self.sample_count == samples {
            return Ok(samples);
        }

        let caps = DeviceCaps::from_current_device()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if samples > 1 {
            if !caps.msaa_supported {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "MSAA not supported on current device".to_string(),
                ));
            }
            if samples > caps.max_samples {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Sample count {} exceeds device limit {}",
                    samples, caps.max_samples
                )));
            }
        }

        self.sample_count = samples;
        self.rebuild_msaa_state()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(samples)
    }
    #[pyo3(text_signature = "($self)")]
    pub fn debug_uniforms_f32<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
        let bytes = bytemuck::bytes_of(&self.last_uniforms);
        let fl: &[f32] = bytemuck::cast_slice(bytes);
        Ok(numpy::PyArray1::from_vec_bound(py, fl.to_vec()))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_lut_format(&self) -> &'static str {
        self.lut_format
    }
}
