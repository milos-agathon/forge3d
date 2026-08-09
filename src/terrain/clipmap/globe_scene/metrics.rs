use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Physical measurements for a completed ORBIS descent.
///
/// Task 7 deliberately leaves these unmeasured. Task 8 supplies GPU probes;
/// every accessor fails closed until all three values are physical evidence.
#[pyclass(module = "forge3d._forge3d", name = "GlobeMetrics")]
pub struct GlobeMetrics {
    max_vertex_jitter_px: Option<f64>,
    peak_gpu_visible_bytes: Option<u64>,
    lod_crack_pixels: Option<u64>,
}

impl GlobeMetrics {
    pub(super) fn unmeasured() -> Self {
        Self {
            max_vertex_jitter_px: None,
            peak_gpu_visible_bytes: None,
            lod_crack_pixels: None,
        }
    }

    fn require<T: Copy>(value: Option<T>, name: &str) -> PyResult<T> {
        value.ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "GlobeMetrics.{name} is unmeasured: a physical GPU probe has not completed"
            ))
        })
    }
}

#[pymethods]
impl GlobeMetrics {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "GlobeMetrics objects are produced by a completed physical GlobeScene probe",
        ))
    }

    #[getter]
    fn max_vertex_jitter_px(&self) -> PyResult<f64> {
        Self::require(self.max_vertex_jitter_px, "max_vertex_jitter_px")
    }

    #[getter]
    fn peak_gpu_visible_bytes(&self) -> PyResult<u64> {
        Self::require(self.peak_gpu_visible_bytes, "peak_gpu_visible_bytes")
    }

    #[getter]
    fn lod_crack_pixels(&self) -> PyResult<u64> {
        Self::require(self.lod_crack_pixels, "lod_crack_pixels")
    }

    fn __getitem__(&self, key: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| match key {
            "max_vertex_jitter_px" => Ok(self.max_vertex_jitter_px()?.into_py(py)),
            "peak_gpu_visible_bytes" => Ok(self.peak_gpu_visible_bytes()?.into_py(py)),
            "lod_crack_pixels" => Ok(self.lod_crack_pixels()?.into_py(py)),
            _ => Err(PyKeyError::new_err(key.to_string())),
        })
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new_bound(py);
        out.set_item("max_vertex_jitter_px", self.max_vertex_jitter_px()?)?;
        out.set_item("peak_gpu_visible_bytes", self.peak_gpu_visible_bytes()?)?;
        out.set_item("lod_crack_pixels", self.lod_crack_pixels()?)?;
        Ok(out.into())
    }
}
