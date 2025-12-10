// src/geometry/py_bindings.rs
// Core PyO3 bindings for geometry operations
// RELEVANT FILES: python/forge3d/geometry.py

use super::mesh_python::{map_geometry_err, mesh_from_python, mesh_to_python};
use super::{
    extrude_polygon_with_options, generate_primitive, transform, validate_mesh, weld_mesh,
    ExtrudeOptions, MeshBuffers, MeshValidationIssue, PrimitiveParams, PrimitiveType, WeldOptions,
};
use glam::Vec3;
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAnyMethods, PyDict, PyList},
};

#[pyfunction]
pub fn geometry_extrude_polygon_py(
    py: Python<'_>,
    polygon: PyReadonlyArray2<'_, f32>,
    height: f32,
    cap_uv_scale: Option<f32>,
) -> PyResult<PyObject> {
    if polygon.shape()[1] != 2 {
        return Err(PyValueError::new_err("polygon must have shape (N, 2)"));
    }

    let mut points = Vec::with_capacity(polygon.shape()[0]);
    for row in polygon.as_array().outer_iter() {
        points.push([row[0], row[1]]);
    }

    let mut options = ExtrudeOptions::default();
    options.height = height;
    if let Some(scale) = cap_uv_scale {
        options.cap_uv_scale = scale;
    }

    let mesh = extrude_polygon_with_options(&points, options)
        .map_err(|err| PyValueError::new_err(err.message().to_string()))?;
    mesh_to_python(py, &mesh)
}

#[pyfunction]
pub fn geometry_generate_primitive_py(
    py: Python<'_>,
    kind: &str,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let primitive_kind = match kind.to_ascii_lowercase().as_str() {
        "plane" => PrimitiveType::Plane,
        "box" | "cube" => PrimitiveType::Box,
        "sphere" => PrimitiveType::Sphere,
        "cylinder" => PrimitiveType::Cylinder,
        "cone" => PrimitiveType::Cone,
        "torus" => PrimitiveType::Torus,
        "text" | "text3d" => PrimitiveType::TextStub,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported primitive: {other}"
            )));
        }
    };

    let mut config = PrimitiveParams::default();
    if let Some(kwargs) = params {
        if let Some(value) = kwargs.get_item("resolution")? {
            let tuple: (u32, u32) = value.extract()?;
            config.resolution = tuple;
        }
        if let Some(value) = kwargs.get_item("radial_segments")? {
            config.radial_segments = value.extract()?;
        }
        if let Some(value) = kwargs.get_item("rings")? {
            config.rings = value.extract()?;
        }
        if let Some(value) = kwargs.get_item("height_segments")? {
            config.height_segments = value.extract()?;
        }
        if let Some(value) = kwargs.get_item("tube_segments")? {
            config.tube_segments = value.extract()?;
        }
        if let Some(value) = kwargs.get_item("radius")? {
            config.radius = value.extract()?;
        }
        if let Some(value) = kwargs.get_item("tube_radius")? {
            config.tube_radius = value.extract()?;
        }
        if let Some(value) = kwargs.get_item("include_caps")? {
            config.include_caps = value.extract()?;
        }
    }

    let mesh = generate_primitive(primitive_kind, config);
    mesh_to_python(py, &mesh)
}

#[pyfunction]
pub fn geometry_validate_mesh_py(
    py: Python<'_>,
    positions: PyReadonlyArray2<'_, f32>,
    indices: PyReadonlyArray2<'_, u32>,
) -> PyResult<PyObject> {
    if positions.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must have shape (N, 3)"));
    }

    let mut mesh = MeshBuffers::new();
    mesh.positions = positions
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    mesh.indices = indices.as_array().iter().copied().collect();

    let report = validate_mesh(&mesh);

    let dict = PyDict::new_bound(py);
    dict.set_item("ok", report.is_clean())?;

    let stats = PyDict::new_bound(py);
    stats.set_item("vertex_count", report.stats.vertex_count)?;
    stats.set_item("triangle_count", report.stats.triangle_count)?;
    stats.set_item("bbox_min", report.stats.bbox_min)?;
    stats.set_item("bbox_max", report.stats.bbox_max)?;
    dict.set_item("stats", stats)?;

    let issues = PyList::empty_bound(py);
    for issue in report.issues {
        let item = PyDict::new_bound(py);
        match issue {
            MeshValidationIssue::IndexOutOfBounds { index } => {
                item.set_item("type", "index_out_of_bounds")?;
                item.set_item("index", index)?;
            }
            MeshValidationIssue::DegenerateTriangle { triangle } => {
                item.set_item("type", "degenerate_triangle")?;
                item.set_item("triangle", triangle)?;
            }
            MeshValidationIssue::DuplicateVertex { first, duplicate } => {
                item.set_item("type", "duplicate_vertex")?;
                item.set_item("first", first)?;
                item.set_item("duplicate", duplicate)?;
            }
            MeshValidationIssue::NonManifoldEdge { edge, count } => {
                item.set_item("type", "non_manifold_edge")?;
                item.set_item("edge", edge)?;
                item.set_item("count", count)?;
            }
        }
        issues.append(item)?;
    }
    dict.set_item("issues", issues)?;

    Ok(dict.into_py(py))
}

#[pyfunction]
pub fn geometry_weld_mesh_py(
    py: Python<'_>,
    positions: PyReadonlyArray2<'_, f32>,
    indices: PyReadonlyArray2<'_, u32>,
    uvs: Option<PyReadonlyArray2<'_, f32>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    if positions.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must have shape (N, 3)"));
    }

    let mut mesh = MeshBuffers::new();
    mesh.positions = positions
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    mesh.indices = indices.as_array().iter().copied().collect();

    if let Some(uv_array) = uvs {
        if uv_array.shape()[1] != 2 {
            return Err(PyValueError::new_err("uvs must have shape (N, 2)"));
        }
        mesh.uvs = uv_array
            .as_array()
            .outer_iter()
            .map(|row| [row[0], row[1]])
            .collect();
    }

    let mut weld_options = WeldOptions::default();
    if let Some(opts) = options {
        if let Some(value) = opts.get_item("position_epsilon")? {
            weld_options.position_epsilon = value.extract()?;
        }
        if let Some(value) = opts.get_item("uv_epsilon")? {
            weld_options.uv_epsilon = value.extract()?;
        }
    }

    let result = weld_mesh(&mesh, weld_options);
    let dict = PyDict::new_bound(py);
    let mesh_obj = mesh_to_python(py, &result.mesh)?;
    dict.set_item("mesh", mesh_obj)?;
    let remap = PyArray1::from_vec_bound(py, result.remap);
    dict.set_item("remap", remap)?;
    dict.set_item("collapsed", result.collapsed)?;
    Ok(dict.into_py(py))
}

#[pyfunction]
pub fn geometry_transform_center_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    target: Option<(f32, f32, f32)>,
) -> PyResult<(PyObject, (f32, f32, f32))> {
    let mut mesh_buffers = mesh_from_python(mesh)?;
    let target_vec = target
        .map(|t| Vec3::new(t.0, t.1, t.2))
        .unwrap_or(Vec3::ZERO);
    let center = map_geometry_err(transform::center_to_target(&mut mesh_buffers, target_vec))?;
    let py_mesh = mesh_to_python(py, &mesh_buffers)?;
    Ok((py_mesh, (center.x, center.y, center.z)))
}

#[pyfunction]
pub fn geometry_transform_scale_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    scale: (f32, f32, f32),
    pivot: Option<(f32, f32, f32)>,
) -> PyResult<(PyObject, bool)> {
    let mut mesh_buffers = mesh_from_python(mesh)?;
    let scale_vec = Vec3::new(scale.0, scale.1, scale.2);
    let pivot_vec = pivot
        .map(|p| Vec3::new(p.0, p.1, p.2))
        .unwrap_or(Vec3::ZERO);
    let flipped = map_geometry_err(transform::scale_about_pivot(
        &mut mesh_buffers,
        scale_vec,
        pivot_vec,
    ))?;
    let py_mesh = mesh_to_python(py, &mesh_buffers)?;
    Ok((py_mesh, flipped))
}

#[pyfunction]
pub fn geometry_transform_flip_axis_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    axis: usize,
) -> PyResult<(PyObject, bool)> {
    let mut mesh_buffers = mesh_from_python(mesh)?;
    let flipped = map_geometry_err(transform::flip_axis(&mut mesh_buffers, axis))?;
    let py_mesh = mesh_to_python(py, &mesh_buffers)?;
    Ok((py_mesh, flipped))
}

#[pyfunction]
pub fn geometry_transform_swap_axes_py(
    py: Python<'_>,
    mesh: &Bound<'_, PyDict>,
    axis_a: usize,
    axis_b: usize,
) -> PyResult<(PyObject, bool)> {
    let mut mesh_buffers = mesh_from_python(mesh)?;
    let flipped = map_geometry_err(transform::swap_axes(&mut mesh_buffers, axis_a, axis_b))?;
    let py_mesh = mesh_to_python(py, &mesh_buffers)?;
    Ok((py_mesh, flipped))
}

#[pyfunction]
pub fn geometry_transform_bounds_py(
    mesh: &Bound<'_, PyDict>,
) -> PyResult<Option<((f32, f32, f32), (f32, f32, f32))>> {
    let mesh_buffers = mesh_from_python(mesh)?;
    Ok(transform::compute_bounds(&mesh_buffers)
        .map(|(min, max)| ((min.x, min.y, min.z), (max.x, max.y, max.z))))
}
