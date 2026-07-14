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

#[pyfunction(name = "geometry_measure", signature = (geometry, *, crs, metrics = None))]
pub fn geometry_measure_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    crs: &Bound<'_, PyAny>,
    metrics: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    // MENSURA: the CRS is mandatory — without it a measurement cannot know
    // whether coordinates are degrees or metres, and square degrees must
    // never be reported as a length or area.
    let source = py_to_json_strict(geometry)?;
    let metrics = metrics_from_py(metrics)?;
    let spec = crate::gis::extract_required_crs(Some(crs))?;
    let mode = super::measure_mode_for_crs(&spec)?;
    let result = super::geometry_measure(&source, &metrics, mode)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "measure_geometries", signature = (geometry, *, crs, metrics = None))]
pub fn measure_geometries_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    crs: &Bound<'_, PyAny>,
    metrics: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    geometry_measure_py(py, geometry, crs, metrics)
}

#[pyfunction(name = "geometry_centroid", signature = (geometry, *, crs = None))]
pub fn geometry_centroid_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    crs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let spec = crate::gis::extract_crs(crs)?;
    let result = super::geometry_centroid(&source, spec)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "representative_point", signature = (geometry, *, crs = None))]
pub fn representative_point_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    crs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let spec = crate::gis::extract_crs(crs)?;
    let result = super::representative_point(&source, spec)?;
    json_to_py(py, &result)
}

#[pyfunction(
    name = "interpolate_line",
    signature = (geometry, distance, *, normalized = false, crs = None)
)]
pub fn interpolate_line_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    distance: f64,
    normalized: bool,
    crs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let spec = crate::gis::extract_crs(crs)?;
    let result = super::interpolate_line(&source, distance, normalized, spec)?;
    json_to_py(py, &result)
}

#[pyfunction(name = "union_geometries", signature = (geometries, *, crs = None))]
pub fn union_geometries_py(
    py: Python<'_>,
    geometries: &Bound<'_, PyAny>,
    crs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometries)?;
    let spec = crate::gis::extract_crs(crs)?;
    let result = super::union_geometries(&source, spec)?;
    json_to_py(py, &result)
}

#[pyfunction(
    name = "buffer_geometry",
    signature = (geometry, distance, *, quad_segs = 8, crs = None)
)]
pub fn buffer_geometry_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    distance: f64,
    quad_segs: i64,
    crs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let spec = crate::gis::extract_crs(crs)?;
    let result = super::buffer_geometry(&source, distance, quad_segs, spec)?;
    json_to_py(py, &result)
}

#[pyfunction(
    name = "simplify_geometry",
    signature = (geometry, tolerance, *, preserve_topology = true, crs = None)
)]
pub fn simplify_geometry_py(
    py: Python<'_>,
    geometry: &Bound<'_, PyAny>,
    tolerance: f64,
    preserve_topology: bool,
    crs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let source = py_to_json_strict(geometry)?;
    let spec = crate::gis::extract_crs(crs)?;
    let result = super::simplify_geometry(&source, tolerance, preserve_topology, spec)?;
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
