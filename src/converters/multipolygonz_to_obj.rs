//! F7: MultipolygonZ → OBJ mesh converter.
//!
//! Minimal triangulation (fan) per polygon ring without holes. Intended for simple demo use.

use crate::geometry::MeshBuffers;

#[inline]
fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ux = b[0] - a[0];
    let uy = b[1] - a[1];
    let uz = b[2] - a[2];
    let vx = c[0] - a[0];
    let vy = c[1] - a[1];
    let vz = c[2] - a[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 0.0 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

pub fn multipolygonz_to_mesh(polygons: &[Vec<[f32; 3]>]) -> MeshBuffers {
    let mut mesh = MeshBuffers::new();
    for ring in polygons.iter() {
        if ring.len() < 3 {
            continue;
        }
        // Append ring vertices to mesh
        let base = mesh.positions.len() as u32;
        for p in ring.iter() {
            mesh.positions.push(*p);
        }
        // Triangulate fan
        for i in 1..(ring.len() - 1) {
            mesh.indices
                .extend_from_slice(&[base, base + i as u32, base + (i as u32 + 1)]);
        }
    }
    // Compute flat-shaded normals per triangle and assign per vertex (duplicate per-vertex per triangle for simplicity)
    // Here we simply create a normal per position if not present; for flat shading we can average contributing faces
    // For simplicity, set normals to face normals averaged into vertex normals.
    mesh.normals.resize(mesh.positions.len(), [0.0, 0.0, 0.0]);
    let mut accum = vec![[0.0f32; 3]; mesh.positions.len()];
    let mut counts = vec![0u32; mesh.positions.len()];
    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];
        let n = face_normal(a, b, c);
        for &vid in tri {
            let e = &mut accum[vid as usize];
            e[0] += n[0];
            e[1] += n[1];
            e[2] += n[2];
            counts[vid as usize] += 1;
        }
    }
    for (i, e) in accum.iter().enumerate() {
        let count = counts[i].max(1) as f32;
        let nx = e[0] / count;
        let ny = e[1] / count;
        let nz = e[2] / count;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        mesh.normals[i] = if len > 0.0 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 0.0, 1.0]
        };
    }
    mesh
}

// ---------------- PyO3 bridge -----------------
#[cfg(feature = "extension-module")]
use crate::geometry::mesh_to_python;
#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyList;

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn converters_multipolygonz_to_obj_py(
    py: Python<'_>,
    polys: &Bound<'_, PyList>,
) -> PyResult<PyObject> {
    // polys: list of (N,3) float32 arrays
    let mut rings: Vec<Vec<[f32; 3]>> = Vec::with_capacity(polys.len());
    for item in polys.iter() {
        let arr: PyReadonlyArray2<f32> = item.extract()?;
        if arr.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "polygon must have shape (N,3)",
            ));
        }
        let ring: Vec<[f32; 3]> = arr
            .as_array()
            .outer_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();
        rings.push(ring);
    }
    let mesh = multipolygonz_to_mesh(&rings);
    mesh_to_python(py, &mesh)
}
