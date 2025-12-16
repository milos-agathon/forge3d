// src/geometry/displacement.rs
// Heightmap and procedural displacement with normal recomputation and bounds growth.
#![allow(dead_code)]

use super::MeshBuffers;

fn recompute_normals(mesh: &mut MeshBuffers) {
    mesh.normals.clear();
    mesh.normals.resize(mesh.positions.len(), [0.0, 0.0, 0.0]);
    let mut counts = vec![0u32; mesh.positions.len()];
    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];
        let ux = b[0] - a[0];
        let uy = b[1] - a[1];
        let uz = b[2] - a[2];
        let vx = c[0] - a[0];
        let vy = c[1] - a[1];
        let vz = c[2] - a[2];
        let nx = uy * vz - uz * vy;
        let ny = uz * vx - ux * vz;
        let nz = ux * vy - uy * vx;
        for &vid in tri {
            let e = &mut mesh.normals[vid as usize];
            e[0] += nx;
            e[1] += ny;
            e[2] += nz;
            counts[vid as usize] += 1;
        }
    }
    for (i, n) in mesh.normals.iter_mut().enumerate() {
        let k = counts[i].max(1) as f32;
        let nx = n[0] / k;
        let ny = n[1] / k;
        let nz = n[2] / k;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        *n = if len > 0.0 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 0.0, 1.0]
        };
    }
}

fn compute_bounds(positions: &[[f32; 3]]) -> Option<([f32; 3], [f32; 3])> {
    if positions.is_empty() {
        return None;
    }
    let mut min = positions[0];
    let mut max = positions[0];
    for p in positions.iter() {
        for a in 0..3 {
            if p[a] < min[a] {
                min[a] = p[a];
            }
            if p[a] > max[a] {
                max[a] = p[a];
            }
        }
    }
    Some((min, max))
}

fn sample_height_bilinear(hm: &[f32], w: usize, h: usize, u: f32, v: f32) -> f32 {
    let uu = u.clamp(0.0, 1.0) * ((w - 1) as f32);
    let vv = v.clamp(0.0, 1.0) * ((h - 1) as f32);
    let x0 = uu.floor() as usize;
    let y0 = vv.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = uu - x0 as f32;
    let ty = vv - y0 as f32;
    let h00 = hm[y0 * w + x0];
    let h10 = hm[y0 * w + x1];
    let h01 = hm[y1 * w + x0];
    let h11 = hm[y1 * w + x1];
    let a = h00 * (1.0 - tx) + h10 * tx;
    let b = h01 * (1.0 - tx) + h11 * tx;
    a * (1.0 - ty) + b * ty
}

pub fn displace_heightmap(
    mesh: &mut MeshBuffers,
    heightmap: &[f32],
    w: usize,
    h: usize,
    scale: f32,
    uv_space: bool,
) {
    // Ensure normals present for displacement direction
    if mesh.normals.len() != mesh.positions.len() {
        recompute_normals(mesh);
    }
    let Some((min, max)) = compute_bounds(&mesh.positions) else {
        return;
    };
    let dx = (max[0] - min[0]).abs().max(1e-8);
    let dy = (max[1] - min[1]).abs().max(1e-8);
    let use_uv = uv_space && mesh.uvs.len() == mesh.positions.len();
    let uvs_copy = if use_uv { Some(mesh.uvs.clone()) } else { None };
    for (i, p) in mesh.positions.iter_mut().enumerate() {
        let (u, v) = if let (true, Some(ref uvc)) = (use_uv, &uvs_copy) {
            let uv = uvc[i];
            (uv[0], uv[1])
        } else {
            ((p[0] - min[0]) / dx, (p[1] - min[1]) / dy)
        };
        let hval = sample_height_bilinear(heightmap, w, h, u, v);
        let n = mesh.normals[i];
        p[0] += n[0] * hval * scale;
        p[1] += n[1] * hval * scale;
        p[2] += n[2] * hval * scale;
    }
    // Update normals after displacement
    recompute_normals(mesh);
}

pub fn displace_procedural(mesh: &mut MeshBuffers, amplitude: f32, frequency: f32) {
    if mesh.normals.len() != mesh.positions.len() {
        recompute_normals(mesh);
    }
    for (i, p) in mesh.positions.iter_mut().enumerate() {
        let n = mesh.normals[i];
        let s = (p[0] * frequency).sin() * (p[1] * frequency).cos() * amplitude;
        p[0] += n[0] * s;
        p[1] += n[1] * s;
        p[2] += n[2] * s;
    }
    recompute_normals(mesh);
}
