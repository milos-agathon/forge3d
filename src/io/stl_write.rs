//! Binary STL writer (F6) for 3D print export.
//!
//! Writes binary STL from `MeshBuffers`. Computes face normals per triangle.
//! Optional watertightness check ensures each edge is shared by exactly two triangles.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::RenderError;
use crate::geometry::MeshBuffers;

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ux = b[0] - a[0];
    let uy = b[1] - a[1];
    let uz = b[2] - a[2];
    let vx = c[0] - a[0];
    let vy = c[1] - a[1];
    let vz = c[2] - a[2];
    // cross(u, v)
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

fn watertight_edge_check(indices: &[u32]) -> bool {
    if indices.len() % 3 != 0 {
        return false;
    }
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        let tri_edges = [
            (tri[0].min(tri[1]), tri[0].max(tri[1])),
            (tri[1].min(tri[2]), tri[1].max(tri[2])),
            (tri[2].min(tri[0]), tri[2].max(tri[0])),
        ];
        for e in tri_edges.iter() {
            *edges.entry(*e).or_insert(0) += 1;
        }
    }
    edges.values().all(|&c| c == 2)
}

pub fn export_stl_binary<P: AsRef<Path>>(
    path: P,
    mesh: &MeshBuffers,
    validate_watertight: bool,
) -> Result<bool, RenderError> {
    let tri_count = mesh.indices.len() / 3;
    let file = File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);

    // 80-byte header
    let mut header = [0u8; 80];
    let tag = b"forge3d stl export";
    header[..tag.len()].copy_from_slice(tag);
    w.write_all(&header)
        .map_err(|e| RenderError::io(format!("{}", e)))?;

    // Triangle count (u32 LE)
    w.write_all(&(tri_count as u32).to_le_bytes())
        .map_err(|e| RenderError::io(format!("{}", e)))?;

    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];
        let n = face_normal(a, b, c);
        for v in [n, a, b, c].iter() {
            for comp in v.iter() {
                w.write_all(&comp.to_le_bytes())
                    .map_err(|e| RenderError::io(format!("{}", e)))?;
            }
        }
        // Attribute byte count (u16)
        w.write_all(&0u16.to_le_bytes())
            .map_err(|e| RenderError::io(format!("{}", e)))?;
    }

    let watertight = if validate_watertight {
        watertight_edge_check(&mesh.indices)
    } else {
        false
    };
    Ok(watertight)
}

// ---------------- PyO3 bridge -----------------

#[cfg(feature = "extension-module")]
use crate::geometry::mesh_from_python;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyDict;

#[cfg(feature = "extension-module")]
#[pyfunction]
pub fn io_export_stl_py(
    path: &str,
    mesh: &Bound<'_, PyDict>,
    validate: Option<bool>,
) -> PyResult<bool> {
    let mesh_buf = mesh_from_python(mesh)?;
    export_stl_binary(path, &mesh_buf, validate.unwrap_or(false)).map_err(|e| e.to_py_err())
}
