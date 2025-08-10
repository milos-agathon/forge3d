// T11-BEGIN:grid-mesh
//! Grid mesh generator for regular (W,H) heightmaps.
//! Decision: positions are centered at origin (0,0) in world XY.
//!   x ∈ [-(W-1)/2 * dx, +(W-1)/2 * dx], y ∈ [-(H-1)/2 * dy, +(H-1)/2 * dy]
//! UVs cover [0,1]x[0,1]: u=x/(W-1), v=y/(H-1)
//! Indices form two CCW triangles per cell, suitable for Vulkan's CCW front face.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct GridVertex {
    pub position: [f32; 2], // world XY; Z comes from height in shader
    pub uv: [f32; 2],
}

#[derive(Debug, Clone)]
pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

#[derive(Debug, Clone)]
pub struct GridMesh {
    pub vertices: Vec<GridVertex>,
    pub indices: Indices,
}

#[inline]
fn choose_u16(count_vertices: usize) -> bool {
    count_vertices <= u16::MAX as usize
}

/// Build a (W,H) grid with spacing (dx, dy).
pub fn make_grid(w: usize, h: usize, dx: f32, dy: f32) -> GridMesh {
    assert!(w >= 2 && h >= 2, "grid must be at least 2x2");
    assert!(dx.is_finite() && dy.is_finite() && dx > 0.0 && dy > 0.0, "dx/dy must be finite and > 0");

    let n_verts = w * h;
    let n_quads = (w - 1) * (h - 1);
    let n_indices = n_quads * 6;

    let mut vertices = Vec::with_capacity(n_verts);

    // center offsets
    let cx = (w as f32 - 1.0) * 0.5 * dx;
    let cy = (h as f32 - 1.0) * 0.5 * dy;

    for y in 0..h {
        let wy = y as f32 * dy - cy;
        let v = if h > 1 { y as f32 / (h as f32 - 1.0) } else { 0.0 };
        for x in 0..w {
            let wx = x as f32 * dx - cx;
            let u = if w > 1 { x as f32 / (w as f32 - 1.0) } else { 0.0 };
            vertices.push(GridVertex {
                position: [wx, wy],
                uv: [u, v],
            });
        }
    }

    if choose_u16(n_verts) {
        let mut idx: Vec<u16> = Vec::with_capacity(n_indices);
        for y in 0..(h - 1) {
            let row = y * w;
            for x in 0..(w - 1) {
                let i0 = (row + x) as u16;
                let i1 = (row + x + 1) as u16;
                let i2 = (row + x + w) as u16;
                let i3 = (row + x + w + 1) as u16;
                // CCW: (i0, i1, i2) and (i2, i1, i3)
                idx.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
            }
        }
        GridMesh { vertices, indices: Indices::U16(idx) }
    } else {
        let mut idx: Vec<u32> = Vec::with_capacity(n_indices);
        for y in 0..(h - 1) {
            let row = y * w;
            for x in 0..(w - 1) {
                let i0 = (row + x) as u32;
                let i1 = (row + x + 1) as u32;
                let i2 = (row + x + w) as u32;
                let i3 = (row + x + w + 1) as u32;
                idx.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
            }
        }
        GridMesh { vertices, indices: Indices::U32(idx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_and_uv() {
        let g = make_grid(4, 3, 2.0, 1.0);
        assert_eq!(g.vertices.len(), 12);
        assert_eq!(match &g.indices { Indices::U16(v)=>v.len(), Indices::U32(v)=>v.len()}, (4-1)*(3-1)*6);
        // corners uv
        let w=4; let h=3;
        assert_eq!(g.vertices[0].uv, [0.0, 0.0]);
        assert_eq!(g.vertices[w-1].uv, [1.0, 0.0]);
        assert_eq!(g.vertices[(h-1)*w].uv, [0.0, 1.0]);
        assert_eq!(g.vertices[h*w-1].uv, [1.0, 1.0]);
    }
    #[test]
    fn ccw_first_cell() {
        let g = make_grid(3, 3, 1.0, 1.0);
        let verts = &g.vertices;
        let (i0,i1,i2) = match &g.indices {
            Indices::U16(v)=> (v[0] as usize, v[1] as usize, v[2] as usize),
            Indices::U32(v)=> (v[0] as usize, v[1] as usize, v[2] as usize)
        };
        let p0 = glam::Vec2::from(verts[i0].position);
        let p1 = glam::Vec2::from(verts[i1].position);
        let p2 = glam::Vec2::from(verts[i2].position);
        // z of 2D cross (p1-p0) x (p2-p0)
        let z = (p1-p0).perp_dot(p2-p0);
        assert!(z > 0.0, "first triangle should be CCW (+Z)");
    }

    #[test]
    fn index_width_switch() {
        let big = make_grid(256, 256, 1.0, 1.0); // 65536 vertices => needs u32 (> 65535)
        match big.indices { Indices::U32(_) => {}, _ => panic!("expected u32 indices") }
        let ok = make_grid(255, 255, 1.0, 1.0); // 65025 => u16
        match ok.indices { Indices::U16(_) => {}, _ => panic!("expected u16 indices") }
    }

    // Release-only perf check to avoid flakiness in debug builds
    #[test]
    #[cfg(not(debug_assertions))]
    fn perf_1024_release() {
        use std::time::Instant;
        let t0 = Instant::now();
        let _ = make_grid(1024, 1024, 1.0, 1.0);
        let ms = t0.elapsed().as_millis() as u64;
        assert!(ms <= 80, "expected ≤80ms in Release; got {}ms", ms);
    }
}
// T11-END:grid-mesh