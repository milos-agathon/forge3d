use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    pub fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.get_stats_impl(py)
    }
}
