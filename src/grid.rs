// T11-BEGIN:file-header
//! Grid mesh generator for XZ plane (Y=0). Deterministic CCW winding (viewed from +Y).
//! Provides CPU-side generation for vertices (pos, normal, uv) and triangle indices.
//! This file is consumed by PyO3 wrappers in lib.rs for testing and future pipelines.

use bytemuck::{Pod, Zeroable};
use pyo3::prelude::*;
use pyo3::Bound;
use numpy::{PyArray2, PyArray1, ToPyArray};
use ndarray::Array2;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridVertex {
    pub pos: [f32; 3],   // (x, y=0, z)
    pub nrm: [f32; 3],   // +Y
    pub uv:  [f32; 2],   // [0,1]x[0,1]
}

pub struct GridMesh {
    pub vertices: Vec<GridVertex>,
    pub indices:  Vec<u32>, // triangle-list, CCW
}

#[derive(Copy, Clone, Debug)]
pub enum GridOrigin {
    Center,    // grid spans [-W/2, +W/2] x [-D/2, +D/2]
    MinCorner, // grid spans [0, W] x [0, D]
}

pub fn generate_grid(nx: u32, nz: u32, spacing: (f32, f32), origin: GridOrigin) -> GridMesh {
    assert!(nx >= 2 && nz >= 2, "nx, nz must be >= 2");
    let (dx, dz) = spacing;
    assert!(dx > 0.0 && dz > 0.0, "spacing must be > 0");

    let w = (nx - 1) as f32 * dx;
    let d = (nz - 1) as f32 * dz;

    let (x0, z0) = match origin {
        GridOrigin::Center   => (-0.5 * w, -0.5 * d),
        GridOrigin::MinCorner => (0.0, 0.0),
    };

    let mut vertices = Vec::with_capacity((nx * nz) as usize);
    let up = [0.0_f32, 1.0_f32, 0.0_f32];

    for j in 0..nz {
        let z = z0 + j as f32 * dz;
        let v = if nz > 1 { j as f32 / (nz - 1) as f32 } else { 0.0 };
        for i in 0..nx {
            let x = x0 + i as f32 * dx;
            let u = if nx > 1 { i as f32 / (nx - 1) as f32 } else { 0.0 };
            vertices.push(GridVertex { pos: [x, 0.0, z], nrm: up, uv: [u, v] });
        }
    }

    let mut indices = Vec::with_capacity(((nx - 1) * (nz - 1) * 6) as usize);
    for j in 0..(nz - 1) {
        for i in 0..(nx - 1) {
            let i0 = j * nx + i;        // bottom-left corner
            let i1 = i0 + 1;            // bottom-right corner  
            let i2 = i0 + nx;           // top-left corner
            let i3 = i2 + 1;            // top-right corner
            indices.extend_from_slice(&[
                i0, i3, i1, // Triangle 1: CCW from +Y
                i1, i3, i2, // Triangle 2: CCW from +Y
            ]);
        }
    }

    GridMesh { vertices, indices }
}

#[pyfunction]
#[pyo3(text_signature = "(nx, nz, spacing=(1.0,1.0), origin='center')")]
pub fn grid_generate(py: Python<'_>, nx: u32, nz: u32, spacing: (f32, f32), origin: Option<String>)
    -> pyo3::PyResult<(Bound<'_, PyArray2<f32>>, Bound<'_, PyArray2<f32>>, Bound<'_, PyArray1<u32>>)>
{
    if nx < 2 || nz < 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("nx and nz must be >= 2"));
    }
    let org = match origin.unwrap_or_else(|| "center".to_string()).to_lowercase().as_str() {
        "center" => GridOrigin::Center,
        "min" | "mincorner" | "origin" => GridOrigin::MinCorner,
        _ => return Err(pyo3::exceptions::PyRuntimeError::new_err("origin must be 'center' or 'min'")),
    };
    let mesh = generate_grid(nx, nz, spacing, org);

    let n_verts = mesh.vertices.len();
    let mut pos_flat = Vec::<f32>::with_capacity(n_verts * 3);
    let mut uv_flat = Vec::<f32>::with_capacity(n_verts * 2);

    for v in &mesh.vertices {
        pos_flat.extend_from_slice(&v.pos);
        uv_flat.extend_from_slice(&v.uv);
    }

    let pos_array = Array2::from_shape_vec((n_verts, 3), pos_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let uv_array = Array2::from_shape_vec((n_verts, 2), uv_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pos_arr = pos_array.to_pyarray_bound(py);
    let uv_arr  = uv_array.to_pyarray_bound(py);
    let idx_arr = PyArray1::from_vec_bound(py, mesh.indices);

    Ok((pos_arr, uv_arr, idx_arr))
}
