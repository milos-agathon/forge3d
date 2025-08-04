// T11-BEGIN:file-header
//! Grid mesh generator for XZ plane (Y=0). Deterministic CCW winding (viewed from +Y).
//! Provides CPU-side generation for vertices (pos, normal, uv) and triangle indices.
//! This file is consumed by PyO3 wrappers in lib.rs for testing and future pipelines.

use bytemuck::{Pod, Zeroable};

// T11-BEGIN:types
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
// T11-END:types

// T11-BEGIN:generator
/// Generate a regular grid of `nx` by `nz` vertices (columns x rows).
/// spacing = (dx, dz). nx>=2, nz>=2 enforced by caller.
/// Winding: CCW when looking from +Y toward origin.
/// UVs: u in [0,1] along x, v in [0,1] along z.
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

    // Indices (CCW, +Y normal):
    // tri1: (i,j) -> (i,j+1) -> (i+1,j)
    // tri2: (i+1,j) -> (i,j+1) -> (i+1,j+1)
    let mut indices = Vec::with_capacity(((nx - 1) * (nz - 1) * 6) as usize);
    for j in 0..(nz - 1) {
        for i in 0..(nx - 1) {
            let i0 = j * nx + i;
            let i1 = i0 + 1;
            let i2 = i0 + nx;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[
                i0, i2, i1, // tri1 (CCW, +Y)
                i1, i2, i3, // tri2 (CCW, +Y)
            ]);
        }
    }

    GridMesh { vertices, indices }
}
// T11-END:generator
// T11-END:file-header