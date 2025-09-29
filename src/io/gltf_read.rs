// src/io/gltf_read.rs
// Minimal glTF 2.0 importer: loads first mesh primitive as MeshBuffers.
// Supports embedded (data URI) and external buffers via gltf::import.

use crate::geometry::MeshBuffers;
use crate::error::RenderError;

pub fn import_gltf_to_mesh(path: &str) -> Result<MeshBuffers, RenderError> {
    let (doc, buffers, _images) = gltf::import(path).map_err(|e| RenderError::io(e.to_string()))?;

    // Find first mesh primitive
    let mut out = MeshBuffers::new();
    'outer: for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buffer| {
                let idx = buffer.index();
                buffers.get(idx).map(|d| d.0.as_slice())
            });

            // Positions
            if let Some(positions) = reader.read_positions() {
                out.positions = positions.map(|p| [p[0], p[1], p[2]]).collect();
            }
            // Normals
            if let Some(normals) = reader.read_normals() {
                out.normals = normals.map(|n| [n[0], n[1], n[2]]).collect();
            }
            // UV0
            if let Some(tex0) = reader.read_tex_coords(0) {
                out.uvs = tex0.into_f32().map(|uv| [uv[0], uv[1]]).collect();
            }
            // Indices
            if let Some(indices) = reader.read_indices() {
                out.indices = indices.into_u32().collect();
            } else {
                // No indices: build a trivial triangle list if length divisible by 3
                let n = out.positions.len();
                if n % 3 == 0 { out.indices = (0u32..(n as u32)).collect(); }
            }

            break 'outer;
        }
    }

    if out.positions.is_empty() || out.indices.is_empty() {
        return Err(RenderError::Render("glTF contains no mesh primitives".into()));
    }
    Ok(out)
}

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_import_gltf_py(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let mesh = import_gltf_to_mesh(path).map_err(|e| e.to_py_err())?;
    crate::geometry::mesh_to_python(py, &mesh)
}
