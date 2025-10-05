// src/geometry/mod.rs
// Geometry module hub providing mesh utilities and shared data structures
// Exists to centralize generation, validation, and welding logic for workstream F
// RELEVANT FILES:src/geometry/extrude.rs,src/geometry/primitives.rs,src/geometry/validate.rs,src/geometry/weld.rs

//! Core geometry utilities for Workstream F phase 1.

mod curves;
mod displacement;
mod extrude;
mod primitives;
mod subdivision;
mod tangents;
mod thick_polyline;
mod transform;
mod validate;
mod weld;

pub use extrude::{extrude_polygon, extrude_polygon_with_options, ExtrudeOptions};
pub use primitives::{
    generate_cone, generate_cylinder, generate_plane, generate_primitive, generate_sphere,
    generate_text3d_stub, generate_torus, generate_unit_box, PrimitiveParams, PrimitiveType,
};
pub use subdivision::subdivide_triangles;
#[cfg(feature = "extension-module")]
pub use thick_polyline::geometry_generate_thick_polyline_py;
pub use transform::{center_to_target, compute_bounds, flip_axis, scale_about_pivot, swap_axes};
pub use validate::{validate_mesh, MeshStats, MeshValidationIssue, MeshValidationReport};
pub use weld::{weld_mesh, WeldOptions, WeldResult};

/// Shared mesh container used by the geometry module family.
#[derive(Debug, Clone, Default)]
pub struct MeshBuffers {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
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
            tangents: Vec::with_capacity(vertex_capacity),
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
use glam::Vec3;
#[cfg(feature = "extension-module")]
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAnyMethods, PyDict, PyList},
};

#[cfg(feature = "extension-module")]
pub(crate) fn mesh_to_python<'py>(py: Python<'py>, mesh: &MeshBuffers) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    let positions = PyArray2::from_vec2_bound(py, &to_vec3(mesh.positions.as_slice()))?;
    dict.set_item("positions", positions)?;

    let normals = if mesh.normals.len() == mesh.positions.len() {
        PyArray2::from_vec2_bound(py, &to_vec3(mesh.normals.as_slice()))?
    } else {
        PyArray2::<f32>::zeros_bound(py, [0, 3], false)
    };
    dict.set_item("normals", normals)?;

    let uvs = if mesh.uvs.len() == mesh.positions.len() {
        PyArray2::from_vec2_bound(py, &to_vec2(mesh.uvs.as_slice()))?
    } else {
        PyArray2::<f32>::zeros_bound(py, [0, 2], false)
    };
    dict.set_item("uvs", uvs)?;

    let tangents = if mesh.tangents.len() == mesh.positions.len() {
        PyArray2::from_vec2_bound(py, &to_vec4(mesh.tangents.as_slice()))?
    } else {
        PyArray2::<f32>::zeros_bound(py, [0, 4], false)
    };
    dict.set_item("tangents", tangents)?;

    let indices = PyArray1::from_vec_bound(py, mesh.indices.clone());
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
fn to_vec4(data: &[[f32; 4]]) -> Vec<Vec<f32>> {
    data.iter().map(|row| row.to_vec()).collect()
}

#[cfg(feature = "extension-module")]
fn read_vec3_array(array: PyReadonlyArray2<'_, f32>) -> Vec<[f32; 3]> {
    array
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect()
}

#[cfg(feature = "extension-module")]
fn read_vec2_array(array: PyReadonlyArray2<'_, f32>) -> Vec<[f32; 2]> {
    array
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1]])
        .collect()
}

#[cfg(feature = "extension-module")]
fn read_vec4_array(array: PyReadonlyArray2<'_, f32>) -> Vec<[f32; 4]> {
    array
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2], row[3]])
        .collect()
}

#[cfg(feature = "extension-module")]
pub(crate) fn mesh_from_python(mesh: &Bound<'_, PyDict>) -> PyResult<MeshBuffers> {
    let positions_obj = mesh
        .get_item("positions")?
        .ok_or_else(|| PyValueError::new_err("mesh dict missing 'positions'"))?;
    let positions_array: PyReadonlyArray2<f32> = positions_obj.extract()?;
    if positions_array.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "positions array must have shape (N, 3)",
        ));
    }
    let positions = read_vec3_array(positions_array);

    let normals = match mesh.get_item("normals")? {
        Some(value) if !value.is_none() => {
            let array: PyReadonlyArray2<f32> = value.extract()?;
            if array.shape()[1] != 3 {
                return Err(PyValueError::new_err(
                    "normals array must have shape (N, 3)",
                ));
            }
            read_vec3_array(array)
        }
        _ => Vec::new(),
    };

    let uvs = match mesh.get_item("uvs")? {
        Some(value) if !value.is_none() => {
            let array: PyReadonlyArray2<f32> = value.extract()?;
            if array.shape()[1] != 2 {
                return Err(PyValueError::new_err("uvs array must have shape (N, 2)"));
            }
            read_vec2_array(array)
        }
        _ => Vec::new(),
    };

    let tangents = match mesh.get_item("tangents")? {
        Some(value) if !value.is_none() => {
            let array: PyReadonlyArray2<f32> = value.extract()?;
            if array.shape()[1] != 4 {
                return Err(PyValueError::new_err(
                    "tangents array must have shape (N, 4)",
                ));
            }
            read_vec4_array(array)
        }
        _ => Vec::new(),
    };

    let indices_obj = mesh
        .get_item("indices")?
        .ok_or_else(|| PyValueError::new_err("mesh dict missing 'indices'"))?;
    let indices: Vec<u32>;
    if let Ok(array) = indices_obj.extract::<PyReadonlyArray2<u32>>() {
        if array.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "indices array must have shape (M, 3) when 2D",
            ));
        }
        indices = array.as_array().iter().copied().collect();
    } else {
        let array: PyReadonlyArray1<u32> = indices_obj.extract()?;
        let slice = array.as_slice()?;
        if slice.len() % 3 != 0 {
            return Err(PyValueError::new_err(
                "indices length must be a multiple of 3",
            ));
        }
        indices = slice.to_vec();
    }

    if indices.len() % 3 != 0 {
        return Err(PyValueError::new_err(
            "indices length must be a multiple of 3",
        ));
    }

    Ok(MeshBuffers {
        positions,
        normals,
        uvs,
        tangents,
        indices,
    })
}

#[cfg(feature = "extension-module")]
pub(crate) fn map_geometry_err<T>(result: GeometryResult<T>) -> PyResult<T> {
    result.map_err(|err| PyValueError::new_err(err.message().to_string()))
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_transform_bounds_py(
    mesh: &Bound<'_, PyDict>,
) -> PyResult<Option<((f32, f32, f32), (f32, f32, f32))>> {
    let mesh_buffers = mesh_from_python(mesh)?;
    Ok(transform::compute_bounds(&mesh_buffers)
        .map(|(min, max)| ((min.x, min.y, min.z), (max.x, max.y, max.z))))
}

// ---------------- Phase 4: Subdivision, Displacement, Curves (F11, F12, F17) ----------------

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
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

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_attach_tangents_py(py: Python<'_>, mesh: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let mut mesh_buf = mesh_from_python(mesh)?;
    let tans = tangents::generate_tangents(&mesh_buf);
    mesh_buf.tangents = tans;
    mesh_to_python(py, &mesh_buf)
}

#[cfg(feature = "extension-module")]
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
