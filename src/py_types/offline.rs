use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "OfflineBatchResult")]
pub struct OfflineBatchResult {
    total_samples: u32,
    batch_time_ms: f64,
}

#[cfg(feature = "extension-module")]
impl OfflineBatchResult {
    pub(crate) fn new(total_samples: u32, batch_time_ms: f64) -> Self {
        Self {
            total_samples,
            batch_time_ms,
        }
    }

    fn value_for_key<'py>(&self, py: Python<'py>, key: &str) -> PyResult<PyObject> {
        match key {
            "total_samples" => Ok(self.total_samples.into_py(py)),
            "batch_time_ms" => Ok(self.batch_time_ms.into_py(py)),
            _ => Err(PyValueError::new_err(format!(
                "Unknown OfflineBatchResult key: {key}"
            ))),
        }
    }

    fn as_dict_impl(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_samples", self.total_samples)?;
        dict.set_item("batch_time_ms", self.batch_time_ms)?;
        Ok(dict.into_py(py))
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl OfflineBatchResult {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "OfflineBatchResult objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn total_samples(&self) -> u32 {
        self.total_samples
    }

    #[getter]
    fn batch_time_ms(&self) -> f64 {
        self.batch_time_ms
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.value_for_key(py, key)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.as_dict_impl(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "OfflineBatchResult(total_samples={}, batch_time_ms={:.3})",
            self.total_samples, self.batch_time_ms
        )
    }
}

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "OfflineMetrics")]
pub struct OfflineMetrics {
    total_samples: u32,
    mean_delta: f32,
    p95_delta: f32,
    max_tile_delta: f32,
    converged_tile_ratio: f32,
}

#[cfg(feature = "extension-module")]
impl OfflineMetrics {
    pub(crate) fn new(
        total_samples: u32,
        mean_delta: f32,
        p95_delta: f32,
        max_tile_delta: f32,
        converged_tile_ratio: f32,
    ) -> Self {
        Self {
            total_samples,
            mean_delta,
            p95_delta,
            max_tile_delta,
            converged_tile_ratio,
        }
    }

    fn value_for_key<'py>(&self, py: Python<'py>, key: &str) -> PyResult<PyObject> {
        match key {
            "total_samples" => Ok(self.total_samples.into_py(py)),
            "mean_delta" => Ok(self.mean_delta.into_py(py)),
            "p95_delta" => Ok(self.p95_delta.into_py(py)),
            "max_tile_delta" => Ok(self.max_tile_delta.into_py(py)),
            "converged_tile_ratio" => Ok(self.converged_tile_ratio.into_py(py)),
            _ => Err(PyValueError::new_err(format!(
                "Unknown OfflineMetrics key: {key}"
            ))),
        }
    }

    fn as_dict_impl(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_samples", self.total_samples)?;
        dict.set_item("mean_delta", self.mean_delta)?;
        dict.set_item("p95_delta", self.p95_delta)?;
        dict.set_item("max_tile_delta", self.max_tile_delta)?;
        dict.set_item("converged_tile_ratio", self.converged_tile_ratio)?;
        Ok(dict.into_py(py))
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl OfflineMetrics {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "OfflineMetrics objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn total_samples(&self) -> u32 {
        self.total_samples
    }

    #[getter]
    fn mean_delta(&self) -> f32 {
        self.mean_delta
    }

    #[getter]
    fn p95_delta(&self) -> f32 {
        self.p95_delta
    }

    #[getter]
    fn max_tile_delta(&self) -> f32 {
        self.max_tile_delta
    }

    #[getter]
    fn converged_tile_ratio(&self) -> f32 {
        self.converged_tile_ratio
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.value_for_key(py, key)
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.as_dict_impl(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "OfflineMetrics(total_samples={}, mean_delta={:.6}, p95_delta={:.6}, max_tile_delta={:.6}, converged_tile_ratio={:.3})",
            self.total_samples,
            self.mean_delta,
            self.p95_delta,
            self.max_tile_delta,
            self.converged_tile_ratio
        )
    }
}
