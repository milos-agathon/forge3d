use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // ---- SSGI methods ----

    /// Enable screen-space global illumination.
    #[pyo3(text_signature = "()")]
    pub fn enable_ssgi(&mut self) -> PyResult<()> {
        self.ssgi_enabled = true;
        Ok(())
    }

    /// Disable screen-space global illumination.
    #[pyo3(text_signature = "()")]
    pub fn disable_ssgi(&mut self) -> PyResult<()> {
        self.ssgi_enabled = false;
        Ok(())
    }

    /// Query whether SSGI is enabled.
    #[pyo3(text_signature = "()")]
    pub fn is_ssgi_enabled(&self) -> bool {
        self.ssgi_enabled
    }

    /// Apply SSGI settings. Validates before storing.
    ///
    /// Parameters
    /// ----------
    /// settings : SSGISettings
    ///     The SSGI configuration to apply.
    #[pyo3(text_signature = "($self, settings)")]
    pub fn set_ssgi_settings(
        &mut self,
        settings: crate::lighting::py_bindings::PySSGISettings,
    ) -> PyResult<()> {
        let native = settings.to_native()?;
        self.ssgi_settings = native;
        Ok(())
    }

    /// Return current SSGI settings as a dict.
    #[pyo3(text_signature = "()")]
    pub fn get_ssgi_settings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("ray_steps", self.ssgi_settings.ray_steps)?;
        dict.set_item("ray_radius", self.ssgi_settings.ray_radius)?;
        dict.set_item("ray_thickness", self.ssgi_settings.ray_thickness)?;
        dict.set_item("intensity", self.ssgi_settings.intensity)?;
        dict.set_item("temporal_alpha", self.ssgi_settings.temporal_alpha)?;
        dict.set_item("use_half_res", self.ssgi_settings.use_half_res != 0)?;
        dict.set_item("ibl_fallback", self.ssgi_settings.ibl_fallback)?;
        Ok(dict.into())
    }
}
