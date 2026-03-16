use super::*;
use crate::terrain::render_params;

#[pymethods]
impl TerrainRenderer {
    #[new]
    pub fn new(session: &crate::core::session::Session) -> PyResult<Self> {
        let scene = TerrainScene::new(
            session.device.clone(),
            session.queue.clone(),
            session.adapter.clone(),
        )
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create TerrainRenderer: {:#}", e))
        })?;

        Ok(Self { scene })
    }

    #[pyo3(signature = (lights))]
    #[allow(deprecated)]
    fn set_lights(&self, py: Python, lights: &PyAny) -> PyResult<()> {
        use pyo3::types::PyList;

        let lights_list = lights
            .downcast::<PyList>()
            .map_err(|_| PyRuntimeError::new_err("lights must be a list"))?;

        let mut native_lights = Vec::new();
        for (i, light_dict) in lights_list.iter().enumerate() {
            match crate::lighting::py_bindings::parse_light_dict(py, light_dict) {
                Ok(light) => native_lights.push(light),
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Failed to parse light {}: {}",
                        i, e
                    )));
                }
            }
        }

        let mut light_buffer =
            self.scene.light_buffer.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to lock light buffer: {}", e))
            })?;

        light_buffer
            .update(&self.scene.device, &self.scene.queue, &native_lights)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update lights: {}", e)))?;

        Ok(())
    }

    #[pyo3(signature = () )]
    fn light_debug_info(&self) -> PyResult<String> {
        self.scene.light_debug_info()
    }

    #[pyo3(signature = (material_set, env_maps, params, heightmap, target=None, water_mask=None))]
    pub fn render_terrain_pbr_pom<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        target: Option<&Bound<'_, PyAny>>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Py<crate::Frame>> {
        if target.is_some() {
            return Err(PyRuntimeError::new_err(
                "Custom render targets not yet supported. Use target=None for offscreen rendering.",
            ));
        }

        let frame = self
            .scene
            .render_internal(material_set, env_maps, params, heightmap, water_mask)
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering failed: {:#}", e)))?;

        Ok(Py::new(py, frame)?)
    }

    #[pyo3(signature = (material_set, env_maps, params, heightmap, water_mask=None))]
    pub fn render_with_aov<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<(Py<crate::Frame>, Py<crate::AovFrame>)> {
        let (frame, aov_frame) = self
            .scene
            .render_internal_with_aov(material_set, env_maps, params, heightmap, water_mask)
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering with AOV failed: {:#}", e)))?;

        Ok((Py::new(py, frame)?, Py::new(py, aov_frame)?))
    }

    pub fn info(&self) -> String {
        format!(
            "TerrainRenderer(backend=wgpu, device={:?})",
            self.scene.device.features()
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainRenderer(features={:?})",
            self.scene.device.features()
        )
    }

    #[cfg(feature = "enable-renderer-config")]
    pub fn get_config(&self) -> PyResult<String> {
        let config = self
            .scene
            .config
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock config: {}", e)))?;
        serde_json::to_string_pretty(&*config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize config: {}", e)))
    }
}
