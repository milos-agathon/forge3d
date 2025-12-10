// src/geometry/mesh_python.rs
// Mesh <-> Python dict conversion utilities
// RELEVANT FILES: python/forge3d/geometry.py

use super::array_convert::{read_vec2, read_vec3, read_vec4, to_vec2, to_vec3, to_vec4};
use super::{GeometryResult, MeshBuffers};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAnyMethods, PyDict},
};

/// Convert MeshBuffers to Python dict
pub fn mesh_to_python<'py>(py: Python<'py>, mesh: &MeshBuffers) -> PyResult<PyObject> {
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

/// Convert Python dict to MeshBuffers
pub fn mesh_from_python(mesh: &Bound<'_, PyDict>) -> PyResult<MeshBuffers> {
    let positions_obj = mesh
        .get_item("positions")?
        .ok_or_else(|| PyValueError::new_err("mesh dict missing 'positions'"))?;
    let positions_array: PyReadonlyArray2<f32> = positions_obj.extract()?;
    if positions_array.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "positions array must have shape (N, 3)",
        ));
    }
    let positions = read_vec3(positions_array);

    let normals = match mesh.get_item("normals")? {
        Some(value) if !value.is_none() => {
            let array: PyReadonlyArray2<f32> = value.extract()?;
            if array.shape()[1] != 3 {
                return Err(PyValueError::new_err(
                    "normals array must have shape (N, 3)",
                ));
            }
            read_vec3(array)
        }
        _ => Vec::new(),
    };

    let uvs = match mesh.get_item("uvs")? {
        Some(value) if !value.is_none() => {
            let array: PyReadonlyArray2<f32> = value.extract()?;
            if array.shape()[1] != 2 {
                return Err(PyValueError::new_err("uvs array must have shape (N, 2)"));
            }
            read_vec2(array)
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
            read_vec4(array)
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

/// Map geometry error to PyResult
pub fn map_geometry_err<T>(result: GeometryResult<T>) -> PyResult<T> {
    result.map_err(|err| PyValueError::new_err(err.message().to_string()))
}
