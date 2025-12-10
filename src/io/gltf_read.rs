//! Minimal glTF 2.0 mesh importer.
//!
//! Loads the first mesh primitive as [`MeshBuffers`]. Supports both embedded
//! (data URI) and external buffer references via `gltf::import`.

use crate::core::error::RenderError;
use crate::geometry::MeshBuffers;

/// Import a glTF file and extract the first mesh primitive.
///
/// # Arguments
/// - `path`: Path to a `.gltf` or `.glb` file.
///
/// # Returns
/// A [`MeshBuffers`] containing positions, normals, UVs, and indices.
///
/// # Errors
/// Returns an error if the file cannot be read or contains no mesh primitives.
pub fn import_gltf_to_mesh(path: &str) -> Result<MeshBuffers, RenderError> {
    let (doc, buffers, _images) = gltf::import(path).map_err(|e| RenderError::io(e.to_string()))?;

    let mut out = MeshBuffers::new();

    'outer: for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buffer| buffers.get(buffer.index()).map(|d| d.0.as_slice()));

            read_positions(&reader, &mut out);
            read_normals(&reader, &mut out);
            read_uvs(&reader, &mut out);
            read_or_generate_indices(&reader, &mut out);

            break 'outer;
        }
    }

    if out.positions.is_empty() || out.indices.is_empty() {
        return Err(RenderError::Render(
            "glTF contains no mesh primitives".into(),
        ));
    }

    Ok(out)
}

/// Extract position data from a primitive reader.
fn read_positions<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(positions) = reader.read_positions() {
        out.positions = positions.collect();
    }
}

/// Extract normal data from a primitive reader.
fn read_normals<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(normals) = reader.read_normals() {
        out.normals = normals.collect();
    }
}

/// Extract UV coordinates from a primitive reader.
fn read_uvs<'a, 's, F>(reader: &gltf::mesh::Reader<'a, 's, F>, out: &mut MeshBuffers)
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(tex0) = reader.read_tex_coords(0) {
        out.uvs = tex0.into_f32().collect();
    }
}

/// Extract indices or generate trivial indices if not present.
fn read_or_generate_indices<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    out: &mut MeshBuffers,
) where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(indices) = reader.read_indices() {
        out.indices = indices.into_u32().collect();
    } else {
        let vertex_count = out.positions.len();
        if vertex_count % 3 == 0 {
            out.indices = (0u32..(vertex_count as u32)).collect();
        }
    }
}

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_import_gltf_py(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let mesh = import_gltf_to_mesh(path).map_err(|e| e.to_py_err())?;
    crate::geometry::mesh_to_python(py, &mesh)
}
