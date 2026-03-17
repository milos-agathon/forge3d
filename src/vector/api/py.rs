use std::sync::Mutex;

use crate::core::error::RenderError;
use glam::Vec2;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use super::core::{CrsType, GraphDef, PointDef, PolygonDef, PolylineDef, VectorApi, VectorStyle};

/// NOTE: Numpy inputs must be 2D arrays shaped (N, 2). Parameters are accepted as
/// `PyReadonlyArray2<'py, T>` (owned pyo3 handles). Do not use `&PyReadonlyArray2<T>`
/// in #[pyfunction] signatures because pyo3 cannot extract references from Python call sites.
pub fn parse_polygon_from_numpy<'py>(
    exterior: PyReadonlyArray2<'py, f64>,
) -> Result<Vec<Vec2>, RenderError> {
    if !exterior.is_contiguous() {
        return Err(RenderError::Upload(
            "Polygon exterior array must be C-contiguous (row-major); use np.ascontiguousarray()"
                .to_string(),
        ));
    }

    let exterior_arr = exterior.as_array();
    if exterior_arr.shape()[1] != 2 {
        return Err(RenderError::Upload(format!(
            "Polygon exterior must have shape (N, 2); got shape ({}, {})",
            exterior_arr.shape()[0],
            exterior_arr.shape()[1]
        )));
    }

    let mut vertices = Vec::with_capacity(exterior_arr.shape()[0]);
    for i in 0..exterior_arr.shape()[0] {
        let x = exterior_arr[[i, 0]] as f32;
        let y = exterior_arr[[i, 1]] as f32;

        if !x.is_finite() || !y.is_finite() {
            return Err(RenderError::Upload(format!(
                "Polygon vertex {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }

        vertices.push(Vec2::new(x, y));
    }

    Ok(vertices)
}

static GLOBAL_VECTOR_API: Mutex<Option<VectorApi>> = Mutex::new(None);

fn with_global_api<F, T>(f: F) -> PyResult<T>
where
    F: FnOnce(&mut VectorApi) -> Result<T, RenderError>,
{
    let mut api_guard = GLOBAL_VECTOR_API.lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("Failed to acquire vector API lock")
    })?;

    if api_guard.is_none() {
        *api_guard = Some(VectorApi::new());
    }

    let api = api_guard
        .as_mut()
        .expect("vector API must exist after initialization");
    f(api).map_err(|e| e.to_py_err())
}

#[pyfunction]
#[pyo3(
    text_signature = "(exterior_coords, holes=None, fill_color=None, stroke_color=None, stroke_width=1.0)"
)]
pub fn add_polygons_py<'py>(
    _py: Python<'py>,
    exterior_coords: PyReadonlyArray2<'py, f64>,
    holes: Option<Vec<PyReadonlyArray2<'py, f64>>>,
    fill_color: Option<[f32; 4]>,
    stroke_color: Option<[f32; 4]>,
    stroke_width: Option<f32>,
) -> PyResult<Vec<u32>> {
    let exterior = parse_polygon_from_numpy(exterior_coords).map_err(|e| e.to_py_err())?;

    let mut hole_rings = Vec::new();
    if let Some(hole_arrays) = holes {
        for hole_array in hole_arrays {
            let hole_vertices = parse_polygon_from_numpy(hole_array).map_err(|e| e.to_py_err())?;
            hole_rings.push(hole_vertices);
        }
    }

    let polygon = PolygonDef {
        exterior,
        holes: hole_rings,
        style: VectorStyle {
            fill_color: fill_color.unwrap_or([0.2, 0.4, 0.8, 1.0]),
            stroke_color: stroke_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            stroke_width: stroke_width.unwrap_or(1.0),
            point_size: 4.0,
        },
    };

    with_global_api(|api| {
        api.add_polygons(vec![polygon], CrsType::Planar)
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    })
}

#[pyfunction]
#[pyo3(text_signature = "(path_coords, stroke_color=None, stroke_width=1.0)")]
pub fn add_lines_py<'py>(
    _py: Python<'py>,
    path_coords: PyReadonlyArray2<'py, f64>,
    stroke_color: Option<[f32; 4]>,
    stroke_width: Option<f32>,
) -> PyResult<Vec<u32>> {
    if !path_coords.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Path array must be C-contiguous (row-major); use np.ascontiguousarray()",
        ));
    }

    let path_arr = path_coords.as_array();
    if path_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Path must have shape (N, 2); got shape ({}, {})",
            path_arr.shape()[0],
            path_arr.shape()[1]
        )));
    }

    let mut path = Vec::with_capacity(path_arr.shape()[0]);
    for i in 0..path_arr.shape()[0] {
        let x = path_arr[[i, 0]] as f32;
        let y = path_arr[[i, 1]] as f32;

        if !x.is_finite() || !y.is_finite() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Path vertex {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }

        path.push(Vec2::new(x, y));
    }

    let polyline = PolylineDef {
        path,
        style: VectorStyle {
            fill_color: [0.0, 0.0, 0.0, 0.0],
            stroke_color: stroke_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            stroke_width: stroke_width.unwrap_or(1.0),
            point_size: 4.0,
        },
    };

    with_global_api(|api| {
        api.add_lines(vec![polyline], CrsType::Planar)
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    })
}

#[pyfunction]
#[pyo3(text_signature = "(positions, fill_color=None, point_size=4.0)")]
pub fn add_points_py<'py>(
    _py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    fill_color: Option<[f32; 4]>,
    point_size: Option<f32>,
) -> PyResult<Vec<u32>> {
    if !positions.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Positions array must be C-contiguous (row-major); use np.ascontiguousarray()",
        ));
    }

    let pos_arr = positions.as_array();
    if pos_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Positions must have shape (N, 2); got shape ({}, {})",
            pos_arr.shape()[0],
            pos_arr.shape()[1]
        )));
    }

    let mut points = Vec::with_capacity(pos_arr.shape()[0]);
    for i in 0..pos_arr.shape()[0] {
        let x = pos_arr[[i, 0]] as f32;
        let y = pos_arr[[i, 1]] as f32;

        if !x.is_finite() || !y.is_finite() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Position {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }

        points.push(PointDef {
            position: Vec2::new(x, y),
            style: VectorStyle {
                fill_color: fill_color.unwrap_or([1.0, 0.0, 0.0, 1.0]),
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                point_size: point_size.unwrap_or(4.0),
            },
        });
    }

    with_global_api(|api| {
        api.add_points(points, CrsType::Planar)
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    })
}

#[pyfunction]
#[pyo3(
    text_signature = "(nodes, edges, node_fill_color=None, node_size=4.0, edge_stroke_color=None, edge_width=1.0)"
)]
pub fn add_graph_py<'py>(
    _py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, u32>,
    node_fill_color: Option<[f32; 4]>,
    node_size: Option<f32>,
    edge_stroke_color: Option<[f32; 4]>,
    edge_width: Option<f32>,
) -> PyResult<u32> {
    if !nodes.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Nodes array must be C-contiguous (row-major); use np.ascontiguousarray()",
        ));
    }

    let nodes_arr = nodes.as_array();
    if nodes_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Nodes must have shape (N, 2); got shape ({}, {})",
            nodes_arr.shape()[0],
            nodes_arr.shape()[1]
        )));
    }

    let mut node_positions = Vec::with_capacity(nodes_arr.shape()[0]);
    for i in 0..nodes_arr.shape()[0] {
        let x = nodes_arr[[i, 0]] as f32;
        let y = nodes_arr[[i, 1]] as f32;

        if !x.is_finite() || !y.is_finite() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Node {} has non-finite coordinates: ({}, {})",
                i, x, y
            )));
        }

        node_positions.push(Vec2::new(x, y));
    }

    if !edges.is_contiguous() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Edges array must be C-contiguous (row-major); use np.ascontiguousarray()",
        ));
    }

    let edges_arr = edges.as_array();
    if edges_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Edges must have shape (M, 2); got shape ({}, {})",
            edges_arr.shape()[0],
            edges_arr.shape()[1]
        )));
    }

    let mut edge_pairs = Vec::with_capacity(edges_arr.shape()[0]);
    for i in 0..edges_arr.shape()[0] {
        edge_pairs.push((edges_arr[[i, 0]], edges_arr[[i, 1]]));
    }

    let graph = GraphDef {
        nodes: node_positions,
        edges: edge_pairs,
        node_style: VectorStyle {
            fill_color: node_fill_color.unwrap_or([1.0, 0.0, 0.0, 1.0]),
            stroke_color: [0.0, 0.0, 0.0, 1.0],
            stroke_width: 1.0,
            point_size: node_size.unwrap_or(4.0),
        },
        edge_style: VectorStyle {
            fill_color: [0.0, 0.0, 0.0, 0.0],
            stroke_color: edge_stroke_color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            stroke_width: edge_width.unwrap_or(1.0),
            point_size: 4.0,
        },
    };

    with_global_api(|api| api.add_graph(graph, CrsType::Planar).map(|id| id.0))
}

#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn clear_vectors_py() -> PyResult<()> {
    with_global_api(|api| {
        api.clear();
        Ok(())
    })
}

#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn get_vector_counts_py() -> PyResult<(usize, usize, usize, usize)> {
    with_global_api(|api| Ok(api.get_counts()))
}
