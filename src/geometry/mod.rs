// src/geometry/mod.rs
// Geometry module hub providing mesh utilities and shared data structures
// Exists to centralize generation, validation, and welding logic for workstream F
// RELEVANT FILES:src/geometry/extrude.rs,src/geometry/primitives.rs,src/geometry/validate.rs,src/geometry/weld.rs

//! Core geometry utilities for Workstream F phase 1.

mod extrude;
mod primitives;
mod validate;
mod weld;

pub use extrude::{extrude_polygon, extrude_polygon_with_options, ExtrudeOptions};
pub use primitives::{
    generate_cone, generate_cylinder, generate_plane, generate_primitive, generate_sphere,
    generate_text3d_stub, generate_torus, generate_unit_box, PrimitiveParams, PrimitiveType,
};
pub use validate::{validate_mesh, MeshStats, MeshValidationIssue, MeshValidationReport};
pub use weld::{weld_mesh, WeldOptions, WeldResult};

/// Shared mesh container used by the geometry module family.
#[derive(Debug, Clone, Default)]
pub struct MeshBuffers {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl MeshBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(vertex_capacity: usize, index_capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(vertex_capacity),
            normals: Vec::with_capacity(vertex_capacity),
            uvs: Vec::with_capacity(vertex_capacity),
            indices: Vec::with_capacity(index_capacity),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty() || self.indices.is_empty()
    }
}

/// Error type returned by geometry helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryError {
    message: String,
}

impl GeometryError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GeometryError {}

/// Convenience alias for geometry results.
pub type GeometryResult<T> = Result<T, GeometryError>;
#[cfg(feature = "extension-module")]
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};

#[cfg(feature = "extension-module")]
fn mesh_to_python<'py>(py: Python<'py>, mesh: &MeshBuffers) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    let positions = PyArray2::from_vec2(py, &to_vec3(mesh.positions.as_slice()))?;
    dict.set_item("positions", positions)?;

    let normals = if mesh.normals.len() == mesh.positions.len() {
        PyArray2::from_vec2(py, &to_vec3(mesh.normals.as_slice()))?
    } else {
        PyArray2::from_vec2(py, &Vec::<Vec<f32>>::new())?
    };
    dict.set_item("normals", normals)?;

    let uvs = if mesh.uvs.len() == mesh.positions.len() {
        PyArray2::from_vec2(py, &to_vec2(mesh.uvs.as_slice()))?
    } else {
        PyArray2::from_vec2(py, &Vec::<Vec<f32>>::new())?
    };
    dict.set_item("uvs", uvs)?;

    let indices = PyArray1::from_vec(py, mesh.indices.clone());
    dict.set_item("indices", indices)?;
    dict.set_item("vertex_count", mesh.vertex_count())?;
    dict.set_item("triangle_count", mesh.triangle_count())?;

    Ok(dict.into_py(py))
}

#[cfg(feature = "extension-module")]
fn to_vec3(data: &[[f32; 3]]) -> Vec<Vec<f32>> {
    data.iter().map(|row| row.to_vec()).collect()
}

#[cfg(feature = "extension-module")]
fn to_vec2(data: &[[f32; 2]]) -> Vec<Vec<f32>> {
    data.iter().map(|row| row.to_vec()).collect()
}

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_generate_primitive_py(
    py: Python<'_>,
    kind: &str,
    params: Option<&PyDict>,
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
        if let Ok(Some(value)) = kwargs.get_item("resolution") {
            let tuple: (u32, u32) = value.extract()?;
            config.resolution = tuple;
        }
        if let Ok(Some(value)) = kwargs.get_item("radial_segments") {
            config.radial_segments = value.extract()?;
        }
        if let Ok(Some(value)) = kwargs.get_item("rings") {
            config.rings = value.extract()?;
        }
        if let Ok(Some(value)) = kwargs.get_item("height_segments") {
            config.height_segments = value.extract()?;
        }
        if let Ok(Some(value)) = kwargs.get_item("tube_segments") {
            config.tube_segments = value.extract()?;
        }
        if let Ok(Some(value)) = kwargs.get_item("radius") {
            config.radius = value.extract()?;
        }
        if let Ok(Some(value)) = kwargs.get_item("tube_radius") {
            config.tube_radius = value.extract()?;
        }
        if let Ok(Some(value)) = kwargs.get_item("include_caps") {
            config.include_caps = value.extract()?;
        }
    }

    let mesh = generate_primitive(primitive_kind, config);
    mesh_to_python(py, &mesh)
}

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_weld_mesh_py(
    py: Python<'_>,
    positions: PyReadonlyArray2<'_, f32>,
    indices: PyReadonlyArray2<'_, u32>,
    uvs: Option<PyReadonlyArray2<'_, f32>>,
    options: Option<&PyDict>,
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
        if let Ok(Some(value)) = opts.get_item("position_epsilon") {
            weld_options.position_epsilon = value.extract()?;
        }
        if let Ok(Some(value)) = opts.get_item("uv_epsilon") {
            weld_options.uv_epsilon = value.extract()?;
        }
    }

    let result = weld_mesh(&mesh, weld_options);
    let dict = PyDict::new_bound(py);
    let mesh_obj = mesh_to_python(py, &result.mesh)?;
    dict.set_item("mesh", mesh_obj)?;
    let remap = PyArray1::from_vec(py, result.remap);
    dict.set_item("remap", remap)?;
    dict.set_item("collapsed", result.collapsed)?;
    Ok(dict.into_py(py))
}
