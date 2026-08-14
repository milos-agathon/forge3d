// src/geometry/displacement.rs
// Heightmap and procedural displacement with normal recomputation and bounds growth.
use super::{recompute_normals, MeshBuffers};

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
