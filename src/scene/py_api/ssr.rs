use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // ---- SSR methods ----

    /// Enable screen-space reflections.
    #[pyo3(text_signature = "()")]
    pub fn enable_ssr(&mut self) -> PyResult<()> {
        self.ssr_enabled = true;
        Ok(())
    }

    /// Disable screen-space reflections.
    #[pyo3(text_signature = "()")]
    pub fn disable_ssr(&mut self) -> PyResult<()> {
        self.ssr_enabled = false;
        Ok(())
    }

    /// Query whether SSR is enabled.
    #[pyo3(text_signature = "()")]
    pub fn is_ssr_enabled(&self) -> bool {
        self.ssr_enabled
    }

    /// Apply SSR settings. Validates before storing.
    ///
    /// Parameters
    /// ----------
    /// settings : SSRSettings
    ///     The SSR configuration to apply.
    #[pyo3(text_signature = "($self, settings)")]
    pub fn set_ssr_settings(
        &mut self,
        settings: crate::lighting::py_bindings::PySSRSettings,
    ) -> PyResult<()> {
        let native = settings.to_native()?;
        self.ssr_settings = native;
        Ok(())
    }

    /// Return current SSR settings as a dict.
    #[pyo3(text_signature = "()")]
    pub fn get_ssr_settings(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("max_steps", self.ssr_settings.max_steps)?;
        dict.set_item("max_distance", self.ssr_settings.max_distance)?;
        dict.set_item("thickness", self.ssr_settings.thickness)?;
        dict.set_item("stride", self.ssr_settings.stride)?;
        dict.set_item("intensity", self.ssr_settings.intensity)?;
        dict.set_item("roughness_fade", self.ssr_settings.roughness_fade)?;
        dict.set_item("edge_fade", self.ssr_settings.edge_fade)?;
        dict.set_item("temporal_alpha", self.ssr_settings.temporal_alpha)?;
        Ok(dict.into())
    }
}
