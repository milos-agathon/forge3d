// src/geometry/curves.rs
// Curve & Tube primitives (F17)
#![allow(dead_code)]

use glam::Vec3;

use super::MeshBuffers;

fn safe_normal_from_tangent(t: Vec3, hint_up: Vec3) -> (Vec3, Vec3) {
    let mut n = hint_up.cross(t);
    if n.length_squared() < 1e-6 {
        // choose another hint
        n = Vec3::Y.cross(t);
        if n.length_squared() < 1e-6 {
            n = Vec3::X.cross(t);
        }
    }
    n = n.normalize();
    let b = t.cross(n).normalize();
    (n, b)
}

fn join_side_vector(i: usize, path: &[[f32; 3]], up: Vec3, style: &str, miter_limit: f32) -> Vec3 {
    let n = path.len();
    // Compute side vectors for prev and next segments
    let p_im1 = if i > 0 {
        Vec3::from(path[i - 1])
    } else {
        Vec3::from(path[i])
    };
    let p_i = Vec3::from(path[i]);
    let p_ip1 = if i + 1 < n {
        Vec3::from(path[i + 1])
    } else {
        Vec3::from(path[i])
    };
    let t_prev = (p_i - p_im1).normalize_or_zero();
    let t_next = (p_ip1 - p_i).normalize_or_zero();
    // Side vectors from frames (use in-plane normals to offset ribbon width)
    let (n_prev, _) = safe_normal_from_tangent(
        if t_prev.length_squared() > 0.0 {
            t_prev
        } else {
            t_next
        },
        up,
    );
    let (n_next, _) = safe_normal_from_tangent(
        if t_next.length_squared() > 0.0 {
            t_next
        } else {
            t_prev
        },
        up,
    );
    let s = (n_prev + n_next).normalize_or_zero();
    if s.length_squared() == 0.0 {
        return n_next; // straight or degenerate; fallback
    }
    match style {
        "bevel" => s, // no extra lengthening
        "round" => {
            // Use bisector length scaled modestly based on bisector alignment
            let denom = s.dot(n_next).abs().max(1e-6);
            let scale = (1.0 / denom).min(miter_limit.max(1.0));
            s * scale
        }
        "miter" => {
            // Scale by segment angle: 1 + (1 - |cos(theta)|), theta between t_prev and t_next
            // This yields >1 for sharp corners and ~1 for near-colinear segments.
            let cosang = t_prev.dot(t_next).clamp(-1.0, 1.0);
            let mut scale = 1.0 + (1.0 - cosang.abs());
            scale = scale.min(miter_limit.max(1.0));
            s * scale
        }
        _ => s,
    }
}

pub fn generate_ribbon(
    path: &[[f32; 3]],
    width_start: f32,
    width_end: f32,
    join_style: &str,
    miter_limit: f32,
    join_styles: Option<&[u8]>, // 0=miter,1=bevel,2=round per vertex
) -> MeshBuffers {
    let mut mesh = MeshBuffers::new();
    if path.len() < 2 {
        return mesh;
    }
    // Precompute cumulative length for UVs
    let mut lengths = vec![0.0f32; path.len()];
    for i in 1..path.len() {
        let a = Vec3::from(path[i - 1]);
        let b = Vec3::from(path[i]);
        lengths[i] = lengths[i - 1] + (b - a).length();
    }
    let total_len = lengths.last().copied().unwrap_or(1.0).max(1e-8);

    let up = Vec3::Z;
    // Two verts per point
    mesh.positions.reserve(path.len() * 2);
    mesh.normals.reserve(path.len() * 2);
    mesh.uvs.reserve(path.len() * 2);

    for (i, p) in path.iter().enumerate() {
        let p0 = if i == 0 { *p } else { path[i - 1] };
        let p1 = if i == path.len() - 1 { *p } else { path[i + 1] };
        let t = (Vec3::from(p1) - Vec3::from(p0)).normalize_or_zero();
        let (n, _b) = safe_normal_from_tangent(t, up);
        let style_sel: &str = if let Some(arr) = join_styles {
            if i < arr.len() {
                match arr[i] {
                    1 => "bevel",
                    2 => "round",
                    _ => join_style,
                }
            } else {
                join_style
            }
        } else {
            join_style
        };
        let s = join_side_vector(i, path, up, style_sel, miter_limit); // side vector across ribbon
        let alpha = (i as f32) / ((path.len() - 1) as f32);
        let width = width_start * (1.0 - alpha) + width_end * alpha;
        let half = 0.5 * width;
        let center = Vec3::from(*p);
        let left = center - s * half;
        let right = center + s * half;
        mesh.positions.push(left.into());
        mesh.positions.push(right.into());
        // Normal facing using computed n
        let normal = n;
        mesh.normals.push(normal.into());
        mesh.normals.push(normal.into());
        let u = lengths[i] / total_len;
        mesh.uvs.push([u, 0.0]);
        mesh.uvs.push([u, 1.0]);
    }

    // Indices (two tris per segment)
    for i in 0..(path.len() - 1) {
        let a0 = (i * 2) as u32;
        let a1 = a0 + 1;
        let b0 = a0 + 2;
        let b1 = a0 + 3;
        mesh.indices.extend_from_slice(&[a0, b0, b1]);
        mesh.indices.extend_from_slice(&[a0, b1, a1]);
    }

    mesh
}

pub fn generate_tube(
    path: &[[f32; 3]],
    radius_start: f32,
    radius_end: f32,
    radial_segments: u32,
    cap_ends: bool,
) -> MeshBuffers {
    let mut mesh = MeshBuffers::new();
    if path.len() < 2 || radial_segments < 3 {
        return mesh;
    }
    let rs = radial_segments as usize;

    // Build frames and cumulative length
    let mut tangents: Vec<Vec3> = Vec::with_capacity(path.len());
    let mut frames_n: Vec<Vec3> = Vec::with_capacity(path.len());
    let mut frames_b: Vec<Vec3> = Vec::with_capacity(path.len());
    let mut lengths = vec![0.0f32; path.len()];
    for i in 0..path.len() {
        let p = Vec3::from(path[i]);
        let t = if i + 1 < path.len() {
            (Vec3::from(path[i + 1]) - p).normalize_or_zero()
        } else {
            (p - Vec3::from(path[i - 1])).normalize_or_zero()
        };
        tangents.push(t);
        if i > 0 {
            lengths[i] = lengths[i - 1] + (Vec3::from(path[i]) - Vec3::from(path[i - 1])).length();
        }
    }
    let total_len = lengths.last().copied().unwrap_or(1.0).max(1e-8);

    // Initial frame
    let up = Vec3::Z;
    let (mut n_prev, mut b_prev) = safe_normal_from_tangent(tangents[0], up);
    frames_n.push(n_prev);
    frames_b.push(b_prev);

    // Parallel transport
    for i in 1..path.len() {
        let t_prev = tangents[i - 1];
        let t = tangents[i];
        // project previous normal onto plane orthogonal to current tangent
        let n_proj = (n_prev - t * n_prev.dot(t)).normalize_or_zero();
        if n_proj.length_squared() < 1e-6 {
            let (n_alt, b_alt) = safe_normal_from_tangent(t, up);
            n_prev = n_alt;
            b_prev = b_alt;
        } else {
            n_prev = n_proj;
            b_prev = t.cross(n_prev).normalize_or_zero();
            if b_prev.length_squared() < 1e-6 {
                b_prev = t_prev.cross(n_prev).normalize_or_zero();
            }
        }
        frames_n.push(n_prev);
        frames_b.push(b_prev);
    }

    // Generate rings
    let mut ring_base_index: Vec<u32> = Vec::with_capacity(path.len());
    for i in 0..path.len() {
        let center = Vec3::from(path[i]);
        let alpha = (i as f32) / ((path.len() - 1) as f32);
        let radius = radius_start * (1.0 - alpha) + radius_end * alpha;
        let base = mesh.positions.len() as u32;
        ring_base_index.push(base);
        for s in 0..rs {
            let phi = (s as f32) / (rs as f32) * std::f32::consts::TAU;
            let dir = frames_n[i] * phi.cos() + frames_b[i] * phi.sin();
            let pos = center + dir * radius;
            mesh.positions.push(pos.into());
            mesh.normals.push(dir.normalize().into());
            let u = lengths[i] / total_len;
            let v = (s as f32) / (rs as f32);
            mesh.uvs.push([u, v]);
        }
    }

    // Connect rings
    for i in 0..(path.len() - 1) {
        let a = ring_base_index[i] as usize;
        let b = ring_base_index[i + 1] as usize;
        for s in 0..rs {
            let s0 = s;
            let s1 = (s + 1) % rs;
            let a0 = (a + s0) as u32;
            let a1 = (a + s1) as u32;
            let b0 = (b + s0) as u32;
            let b1 = (b + s1) as u32;
            mesh.indices.extend_from_slice(&[a0, b0, b1]);
            mesh.indices.extend_from_slice(&[a0, b1, a1]);
        }
    }

    // Caps
    if cap_ends {
        // start cap
        let base_start = ring_base_index[0] as usize;
        let center_start = mesh.positions.len() as u32;
        let center_pos = Vec3::from(path[0]);
        let t0 = tangents[0];
        mesh.positions.push(center_pos.into());
        mesh.normals.push((-t0).into());
        mesh.uvs.push([0.0, 0.0]);
        for s in 0..rs {
            let s1 = (s + 1) % rs;
            let v0 = (base_start + s) as u32;
            let v1 = (base_start + s1) as u32;
            mesh.indices.extend_from_slice(&[center_start, v1, v0]);
        }
        // end cap
        let base_end = ring_base_index[ring_base_index.len() - 1] as usize;
        let center_end = mesh.positions.len() as u32;
        let center_pos_e = Vec3::from(path[path.len() - 1]);
        let t1 = tangents[tangents.len() - 1];
        mesh.positions.push(center_pos_e.into());
        mesh.normals.push(t1.into());
        mesh.uvs.push([1.0, 1.0]);
        for s in 0..rs {
            let s1 = (s + 1) % rs;
            let v0 = (base_end + s) as u32;
            let v1 = (base_end + s1) as u32;
            mesh.indices.extend_from_slice(&[center_end, v0, v1]);
        }
    }

    mesh
}
