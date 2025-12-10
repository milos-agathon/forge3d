//! Thick 3D polyline generation.
//!
//! Builds ribbon geometry with constant world-space width and optional depth offset
//! for Z-fighting mitigation when overlaying on other geometry.

use super::{curves, MeshBuffers};

/// Default miter limit for sharp corner clamping.
const DEFAULT_MITER_LIMIT: f32 = 4.0;

/// Default join style for polyline corners.
const DEFAULT_JOIN_STYLE: &str = "miter";

/// Generate a thick 3D polyline as a ribbon with constant world-space width.
///
/// - `path`: polyline points in world space (N>=2)
/// - `width_world`: width in world units
/// - `depth_offset`: small Z offset added to all vertices to mitigate Z-fighting when overlaying on geometry
/// - `join_style`: miter/bevel/round
/// - `miter_limit`: clamp on miter length for sharp corners
pub fn generate_thick_polyline(
    path: &[[f32; 3]],
    width_world: f32,
    depth_offset: f32,
    join_style: &str,
    miter_limit: f32,
) -> MeshBuffers {
    if path.len() < 2 || !width_world.is_finite() || width_world <= 0.0 {
        return MeshBuffers::new();
    }
    let mut mesh = curves::generate_ribbon(
        path,
        width_world,
        width_world,
        join_style,
        miter_limit,
        None,
    );
    if depth_offset != 0.0 {
        for p in &mut mesh.positions {
            p[2] += depth_offset;
        }
    }
    mesh
}

#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{exceptions::PyValueError, prelude::*};

/// Python binding for thick polyline generation.
#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn geometry_generate_thick_polyline_py(
    py: Python<'_>,
    path: PyReadonlyArray2<'_, f32>,
    width_world: f32,
    depth_offset: Option<f32>,
    join_style: Option<&str>,
    miter_limit: Option<f32>,
) -> PyResult<PyObject> {
    if path.shape()[1] != 3 {
        return Err(PyValueError::new_err("path must have shape (N, 3)"));
    }
    if !width_world.is_finite() || width_world <= 0.0 {
        return Err(PyValueError::new_err("width_world must be positive finite"));
    }

    let pts: Vec<[f32; 3]> = path
        .as_array()
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();

    let style = join_style.unwrap_or(DEFAULT_JOIN_STYLE);
    let limit = miter_limit.unwrap_or(DEFAULT_MITER_LIMIT);
    let z_offset = depth_offset.unwrap_or(0.0);

    let mesh = generate_thick_polyline(&pts, width_world, z_offset, style, limit);
    super::mesh_to_python(py, &mesh)
}
