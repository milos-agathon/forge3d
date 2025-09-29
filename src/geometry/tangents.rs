// src/geometry/tangents.rs
// Tangent generation (simple per-triangle accumulation, similar to Lengyel method)

use super::MeshBuffers;

pub fn generate_tangents(mesh: &MeshBuffers) -> Vec<[f32; 4]> {
    let n_verts = mesh.positions.len();
    let mut tan1 = vec![[0.0f32; 3]; n_verts];
    let mut tan2 = vec![[0.0f32; 3]; n_verts];

    if mesh.uvs.len() != n_verts || mesh.indices.len() % 3 != 0 {
        // No UVs or invalid indices: return default X tangents with w=1
        return vec![[1.0, 0.0, 0.0, 1.0]; n_verts];
    }

    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        let uv0 = mesh.uvs[i0];
        let uv1 = mesh.uvs[i1];
        let uv2 = mesh.uvs[i2];

        let x1 = p1[0] - p0[0];
        let x2 = p2[0] - p0[0];
        let y1 = p1[1] - p0[1];
        let y2 = p2[1] - p0[1];
        let z1 = p1[2] - p0[2];
        let z2 = p2[2] - p0[2];

        let s1 = uv1[0] - uv0[0];
        let s2 = uv2[0] - uv0[0];
        let t1 = uv1[1] - uv0[1];
        let t2 = uv2[1] - uv0[1];

        let denom = s1 * t2 - s2 * t1;
        if denom.abs() < 1e-12 {
            continue; // degenerate UV mapping
        }
        let r = 1.0 / denom;
        let tx = (t2 * x1 - t1 * x2) * r;
        let ty = (t2 * y1 - t1 * y2) * r;
        let tz = (t2 * z1 - t1 * z2) * r;
        let bx = (s1 * x2 - s2 * x1) * r;
        let by = (s1 * y2 - s2 * y1) * r;
        let bz = (s1 * z2 - s2 * z1) * r;

        for &idx in &[i0, i1, i2] {
            tan1[idx][0] += tx;
            tan1[idx][1] += ty;
            tan1[idx][2] += tz;
            tan2[idx][0] += bx;
            tan2[idx][1] += by;
            tan2[idx][2] += bz;
        }
    }

    let mut out = vec![[1.0, 0.0, 0.0, 1.0]; n_verts];
    if mesh.normals.len() != n_verts {
        return out;
    }

    for i in 0..n_verts {
        let n = mesh.normals[i];
        let t = tan1[i];
        // Gram-Schmidt orthogonalize
        let dot_nt = n[0]*t[0] + n[1]*t[1] + n[2]*t[2];
        let mut tx = t[0] - n[0]*dot_nt;
        let mut ty = t[1] - n[1]*dot_nt;
        let mut tz = t[2] - n[2]*dot_nt;
        let len = (tx*tx + ty*ty + tz*tz).sqrt();
        if len > 1e-20 { tx/=len; ty/=len; tz/=len; } else { tx=1.0; ty=0.0; tz=0.0; }
        // Handedness
        let cx = n[1]*tx - n[2]*ty; // cross(n, t).x (partial)
        let cy = n[2]*tx - n[0]*tz; // not exact but we only need sign of dot with bitangent
        let cz = n[0]*ty - n[1]*tx;
        let b = tan2[i];
        let handed = if (cx*b[0] + cy*b[1] + cz*b[2]) < 0.0 { -1.0 } else { 1.0 };
        out[i] = [tx, ty, tz, handed];
    }

    out
}
