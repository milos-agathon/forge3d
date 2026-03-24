use super::*;
use crate::terrain::render_params;
use numpy::PyUntypedArrayMethods;

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

    #[pyo3(signature = (material_set, env_maps, params, heightmap, target=None, water_mask=None, time_seconds=0.0))]
    pub fn render_terrain_pbr_pom<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        target: Option<&Bound<'_, PyAny>>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
        time_seconds: f32,
    ) -> PyResult<Py<crate::Frame>> {
        if target.is_some() {
            return Err(PyRuntimeError::new_err(
                "Custom render targets not yet supported. Use target=None for offscreen rendering.",
            ));
        }

        let frame = self
            .scene
            .render_internal(material_set, env_maps, params, heightmap, water_mask, time_seconds)
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering failed: {:#}", e)))?;

        Ok(Py::new(py, frame)?)
    }

    #[pyo3(signature = (material_set, env_maps, params, heightmap, water_mask=None, time_seconds=0.0))]
    pub fn render_with_aov<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
        time_seconds: f32,
    ) -> PyResult<(Py<crate::Frame>, Py<crate::AovFrame>)> {
        let (frame, aov_frame) = self
            .scene
            .render_internal_with_aov(material_set, env_maps, params, heightmap, water_mask, time_seconds)
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering with AOV failed: {:#}", e)))?;

        Ok((Py::new(py, frame)?, Py::new(py, aov_frame)?))
    }

    pub fn info(&self) -> String {
        format!(
            "TerrainRenderer(backend=wgpu, device={:?})",
            self.scene.device.features()
        )
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = (batches))]
    pub fn set_scatter_batches(&mut self, batches: &PyAny) -> PyResult<()> {
        use pyo3::types::{PyDict, PyList};

        let batch_list = batches
            .downcast::<PyList>()
            .map_err(|_| PyRuntimeError::new_err("batches must be a list of dicts"))?;

        let mut native_batches = Vec::with_capacity(batch_list.len());
        for (batch_index, batch_any) in batch_list.iter().enumerate() {
            let batch_dict = batch_any.downcast::<PyDict>().map_err(|_| {
                PyRuntimeError::new_err(format!("scatter batch {batch_index} must be a dict"))
            })?;

            let name = batch_dict
                .get_item("name")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<String>())
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("batch {batch_index}: invalid 'name': {e}"))
                })?;

            let color = batch_dict
                .get_item("color")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<(f32, f32, f32, f32)>())
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("batch {batch_index}: invalid 'color': {e}"))
                })?
                .map(|value| [value.0, value.1, value.2, value.3])
                .unwrap_or([0.85, 0.85, 0.85, 1.0]);

            let max_draw_distance = batch_dict
                .get_item("max_draw_distance")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|value| !value.is_none())
                .map(|value| value.extract::<f32>())
                .transpose()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index}: invalid 'max_draw_distance': {e}"
                    ))
                })?;

            let transforms_any = batch_dict
                .get_item("transforms")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} missing required 'transforms'"
                    ))
                })?;
            let transforms_array: numpy::PyReadonlyArray2<'_, f32> =
                transforms_any.extract().map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index}: 'transforms' must be a float32 array: {e}"
                    ))
                })?;
            if transforms_array.ndim() != 2 || transforms_array.shape()[1] != 16 {
                return Err(PyRuntimeError::new_err(format!(
                    "batch {batch_index}: transforms must have shape (N, 16)"
                )));
            }
            let transforms = transforms_array
                .as_array()
                .rows()
                .into_iter()
                .map(|row| {
                    let mut out = [0.0f32; 16];
                    for (dst, src) in out.iter_mut().zip(row.iter()) {
                        *dst = *src;
                    }
                    out
                })
                .collect::<Vec<_>>();

            let levels_any = batch_dict
                .get_item("levels")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} missing required 'levels'"
                    ))
                })?;
            let levels_list = levels_any.downcast::<PyList>().map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "batch {batch_index}: 'levels' must be a list of dicts"
                ))
            })?;

            let mut levels = Vec::with_capacity(levels_list.len());
            for (level_index, level_any) in levels_list.iter().enumerate() {
                let level_dict = level_any.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} level {level_index} must be a dict"
                    ))
                })?;
                let mesh_any = level_dict
                    .get_item("mesh")
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index}: {e}"
                        ))
                    })?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index} missing required 'mesh'"
                        ))
                    })?;
                let mesh_dict = mesh_any.downcast::<PyDict>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} level {level_index}: 'mesh' must be a dict"
                    ))
                })?;
                let mesh = crate::geometry::mesh_from_python_dict(mesh_dict).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "batch {batch_index} level {level_index}: invalid mesh: {e}"
                    ))
                })?;
                let max_distance = level_dict
                    .get_item("max_distance")
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index}: {e}"
                        ))
                    })?
                    .filter(|value| !value.is_none())
                    .map(|value| value.extract::<f32>())
                    .transpose()
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index} level {level_index}: invalid 'max_distance': {e}"
                        ))
                    })?;
                levels
                    .push(crate::terrain::scatter::TerrainScatterLevelSpec { mesh, max_distance });
            }

            let wind = match batch_dict
                .get_item("wind")
                .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
                .filter(|v| !v.is_none())
            {
                Some(wind_any) => {
                    let wind_dict = wind_any.downcast::<PyDict>().map_err(|_| {
                        PyRuntimeError::new_err(format!(
                            "batch {batch_index}: 'wind' must be a dict"
                        ))
                    })?;

                    // Helper: extract a typed field from a dict, failing with a clear
                    // message on type mismatch (matching the fail-fast style of the
                    // rest of set_scatter_batches).  Missing keys use the default.
                    macro_rules! wind_field {
                        ($key:expr, $ty:ty, $default:expr) => {
                            match wind_dict
                                .get_item($key)
                                .map_err(|e| PyRuntimeError::new_err(format!(
                                    "batch {batch_index} wind: {e}"
                                )))?
                                .filter(|v| !v.is_none())
                            {
                                Some(v) => v.extract::<$ty>().map_err(|e| {
                                    PyRuntimeError::new_err(format!(
                                        "batch {batch_index} wind: invalid '{}': {e}", $key
                                    ))
                                })?,
                                None => $default,
                            }
                        };
                    }

                    crate::terrain::scatter::ScatterWindSettingsNative {
                        enabled: wind_field!("enabled", bool, false),
                        direction_deg: wind_field!("direction_deg", f32, 0.0),
                        speed: wind_field!("speed", f32, 1.0),
                        amplitude: wind_field!("amplitude", f32, 0.0),
                        rigidity: wind_field!("rigidity", f32, 0.5),
                        bend_start: wind_field!("bend_start", f32, 0.0),
                        bend_extent: wind_field!("bend_extent", f32, 1.0),
                        gust_strength: wind_field!("gust_strength", f32, 0.0),
                        gust_frequency: wind_field!("gust_frequency", f32, 0.3),
                        fade_start: wind_field!("fade_start", f32, 0.0),
                        fade_end: wind_field!("fade_end", f32, 0.0),
                    }
                }
                None => crate::terrain::scatter::ScatterWindSettingsNative::default(),
            };

            native_batches.push(super::scatter::TerrainScatterUploadBatch {
                name,
                color,
                max_draw_distance,
                transforms_rowmajor: transforms,
                levels,
                wind,
            });
        }

        self.scene
            .set_scatter_batches_native(native_batches)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set scatter batches: {e:#}")))
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = ())]
    pub fn clear_scatter_batches(&mut self) -> PyResult<()> {
        self.scene.clear_scatter_batches_native();
        Ok(())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = ())]
    pub fn get_scatter_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.scene.scatter_last_frame_stats();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("batch_count", stats.batch_count)?;
        dict.set_item("rendered_batches", stats.rendered_batches)?;
        dict.set_item("total_instances", stats.total_instances)?;
        dict.set_item("visible_instances", stats.visible_instances)?;
        dict.set_item("culled_instances", stats.culled_instances)?;
        dict.set_item("lod_instance_counts", stats.lod_instance_counts)?;
        Ok(dict.into())
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[pyo3(signature = ())]
    pub fn get_scatter_memory_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let report = self.scene.scatter_memory_report();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("batch_count", report.batch_count)?;
        dict.set_item("level_count", report.level_count)?;
        dict.set_item("total_instances", report.total_instances)?;
        dict.set_item("vertex_buffer_bytes", report.vertex_buffer_bytes)?;
        dict.set_item("index_buffer_bytes", report.index_buffer_bytes)?;
        dict.set_item("instance_buffer_bytes", report.instance_buffer_bytes)?;
        dict.set_item("total_buffer_bytes", report.total_buffer_bytes())?;
        Ok(dict.into())
    }

    #[pyo3(signature = ())]
    pub fn get_probe_memory_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let probe_stride = std::mem::size_of::<crate::terrain::probes::GpuProbeData>() as u64;
        let probe_count = if probe_stride > 0 {
            self.scene.probe_ssbo_bytes / probe_stride
        } else {
            0
        };
        dict.set_item("probe_count", probe_count)?;
        dict.set_item("grid_uniform_bytes", self.scene.probe_grid_uniform_bytes)?;
        dict.set_item("probe_ssbo_bytes", self.scene.probe_ssbo_bytes)?;
        dict.set_item(
            "total_bytes",
            self.scene.probe_grid_uniform_bytes + self.scene.probe_ssbo_bytes,
        )?;
        Ok(dict.into())
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
