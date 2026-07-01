use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyTuple};

use crate::gis::error::GisError;
use crate::gis::py_json::{json_to_py, py_to_json_strict};

use super::model::INVALID_ARGUMENT;

#[pyfunction(name = "validate_geometry")]
pub fn validate_geometry_py(py: Python<'_>, geometry: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let result = super::validate_geometry(&source)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "repair_geometry", signature = (geometry, *, method = "make_valid"))]
pub fn repair_geometry_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    method: &str,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let result = super::repair_geometry(&source, method)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "geometry_measure", signature = (geometry, *, metrics = None))]
pub fn geometry_measure_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    metrics: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let metrics = metrics_from_py(metrics)?;
    let result = super::geometry_measure(&source, &metrics)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "geometry_centroid")]
pub fn geometry_centroid_py(py: Python<'_>, geometry: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let result = super::geometry_centroid(&source)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "representative_point")]
pub fn representative_point_py(py: Python<'_>, geometry: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let result = super::representative_point(&source)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "interpolate_line", signature = (geometry, distance, *, normalized = false))]
pub fn interpolate_line_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    distance: f64,
    normalized: bool,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let result = super::interpolate_line(&source, distance, normalized)?;
    json_to_py(py, &result)
}

fn metrics_from_py(metrics: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let Some(metrics) = metrics else {
        return Ok(vec!["area".to_string(), "length".to_string()]);
    };
    if metrics.is_none() {
        return Ok(vec!["area".to_string(), "length".to_string()]);
    }
    if let Ok(metric) = metrics.extract::<String>() {
        return Ok(vec![metric]);
    }
    if let Ok(list) = metrics.downcast::<PyList>() {
        return list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<_>>>();
    }
    if let Ok(tuple) = metrics.downcast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<_>>>();
    }
    Err(GisError::InvalidArgument(format!(
        "{INVALID_ARGUMENT}: metrics must be a string or sequence of strings"
    ))
    .into())
}
