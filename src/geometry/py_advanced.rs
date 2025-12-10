// src/geometry/py_advanced.rs
// Advanced PyO3 bindings: subdivision, displacement, curves, tangents
// RELEVANT FILES: python/forge3d/geometry.py

use super::mesh_python::{mesh_from_python, mesh_to_python};
use super::{curves, displacement, subdivision, tangents};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyfunction]
pub fn geometry_subdivide_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    levels: u32,
    creases: Option<PyReadonlyArray2<'_, u32>>,
    preserve_boundary: Option<bool>,
) -> PyResult<PyObject> {
    let mesh_in = mesh_from_python(mesh)?;
    let crease_vec: Option<Vec<(u32, u32)>> = match creases {
        Some(arr) => {
            if arr.shape()[1] != 2 {
                return Err(PyValueError::new_err("creases must have shape (K, 2)"));
            }
            let v: Vec<(u32, u32)> = arr
                .as_array()
                .outer_iter()
                .map(|row| (row[0], row[1]))
                .collect();
            Some(v)
        }
        None => None,
    };
    let pres = preserve_boundary.unwrap_or(true);
    let mesh_out = subdivision::subdivide_triangles_with_options(
        &mesh_in,
        levels,
        crease_vec.as_deref(),
        pres,
    );
    mesh_to_python(py, &mesh_out)
}

#[pyfunction]
pub fn geometry_displace_heightmap_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    heightmap: PyReadonlyArray2<'_, f32>,
    scale: f32,
    uv_space: Option<bool>,
) -> PyResult<PyObject> {
    let mut mesh_buf = mesh_from_python(mesh)?;
    let shape = heightmap.shape();
    let (h, w) = (shape[0], shape[1]);
    let hm: Vec<f32> = heightmap.as_array().iter().copied().collect();
    let uv_mode = uv_space.unwrap_or(false);
    displacement::displace_heightmap(&mut mesh_buf, &hm, w, h, scale, uv_mode);
    mesh_to_python(py, &mesh_buf)
}

#[pyfunction]
pub fn geometry_displace_procedural_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    amplitude: f32,
    frequency: f32,
) -> PyResult<PyObject> {
    let mut mesh_buf = mesh_from_python(mesh)?;
    displacement::displace_procedural(&mut mesh_buf, amplitude, frequency);
    mesh_to_python(py, &mesh_buf)
}

#[pyfunction]
pub fn geometry_generate_ribbon_py(
    py: Python<'_>,
    path: PyReadonlyArray2<'_, f32>,
    width_start: f32,
    width_end: f32,
    join_style: Option<&str>,
    miter_limit: Option<f32>,
    join_styles: Option<PyReadonlyArray1<'_, u8>>,
) -> PyResult<PyObject> {
    if path.shape()[1] != 3 {
        return Err(PyValueError::new_err("path must have shape (N, 3)"));
    }
    let pts: Vec<[f32; 3]> = path
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let style = join_style.unwrap_or("miter");
    let limit = miter_limit.unwrap_or(4.0);
    let join_vec: Option<Vec<u8>> = join_styles.map(|arr| arr.as_slice().unwrap().to_vec());
    let mesh = curves::generate_ribbon(
        &pts,
        width_start,
        width_end,
        style,
        limit,
        join_vec.as_deref(),
    );
    mesh_to_python(py, &mesh)
}

#[pyfunction]
pub fn geometry_generate_tube_py(
    py: Python<'_>,
    path: PyReadonlyArray2<'_, f32>,
    radius_start: f32,
    radius_end: f32,
    radial_segments: u32,
    cap_ends: bool,
) -> PyResult<PyObject> {
    if path.shape()[1] != 3 {
        return Err(PyValueError::new_err("path must have shape (N, 3)"));
    }
    let pts: Vec<[f32; 3]> = path
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let mesh = curves::generate_tube(&pts, radius_start, radius_end, radial_segments, cap_ends);
    mesh_to_python(py, &mesh)
}

#[pyfunction]
pub fn geometry_generate_tangents_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let mesh_buf = mesh_from_python(mesh)?;
    let tans = tangents::generate_tangents(&mesh_buf);
    let rows: Vec<Vec<f32>> = tans.iter().map(|t| t.to_vec()).collect();
    let arr = PyArray2::from_vec2_bound(py, &rows)?;
    Ok(arr.into_py(py))
}

#[pyfunction]
pub fn geometry_attach_tangents_py(py: Python<'_>, mesh: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let mut mesh_buf = mesh_from_python(mesh)?;
    let tans = tangents::generate_tangents(&mesh_buf);
    mesh_buf.tangents = tans;
    mesh_to_python(py, &mesh_buf)
}

#[pyfunction(signature = (
    mesh,
    edge_length_limit=None,
    curvature_threshold=None,
    max_levels=3,
    creases=None,
    preserve_boundary=None
))]
pub fn geometry_subdivide_adaptive_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    edge_length_limit: Option<f32>,
    curvature_threshold: Option<f32>,
    max_levels: u32,
    creases: Option<PyReadonlyArray2<'_, u32>>,
    preserve_boundary: Option<bool>,
) -> PyResult<PyObject> {
    let mesh_in = mesh_from_python(mesh)?;
    let crease_vec: Option<Vec<(u32, u32)>> = match creases {
        Some(arr) => {
            if arr.shape()[1] != 2 {
                return Err(PyValueError::new_err("creases must have shape (K, 2)"));
            }
            let v: Vec<(u32, u32)> = arr
                .as_array()
                .outer_iter()
                .map(|row| (row[0], row[1]))
                .collect();
            Some(v)
        }
        None => None,
    };
    let pres = preserve_boundary.unwrap_or(true);
    let mesh_out = subdivision::subdivide_adaptive(
        &mesh_in,
        edge_length_limit,
        curvature_threshold,
        max_levels,
        crease_vec.as_deref(),
        pres,
    );
    mesh_to_python(py, &mesh_out)
}
