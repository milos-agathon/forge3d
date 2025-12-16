// src/geometry/subdivision.rs
// Triangle subdivision: Loop refinement with crease and boundary preservation (basic)
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use super::MeshBuffers;

#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

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

fn beta_loop(n: usize) -> f32 {
    if n == 3 {
        3.0 / 16.0
    } else {
        let n_f = n as f32;
        let theta = std::f32::consts::TAU / n_f;
        let val = (5.0 / 8.0) - ((3.0 / 8.0) + (0.25 * theta.cos())).powi(2);
        val / n_f
    }
}

pub fn subdivide_triangles(input: &MeshBuffers, levels: u32) -> MeshBuffers {
    subdivide_triangles_with_options(input, levels, None, true)
}

pub fn subdivide_triangles_with_options(
    input: &MeshBuffers,
    levels: u32,
    creases: Option<&[(u32, u32)]>,
    preserve_boundary: bool,
) -> MeshBuffers {
    let mut mesh = input.clone();
    if levels == 0 {
        return mesh;
    }

    // Ensure UVs length matches if present, otherwise keep empty
    let has_uv = mesh.uvs.len() == mesh.positions.len();

    let crease_set: HashSet<(u32, u32)> = match creases {
        Some(list) => list.iter().map(|&(a, b)| edge_key(a, b)).collect(),
        None => HashSet::new(),
    };

    for _ in 0..levels {
        let old_vertex_count = mesh.positions.len();

        // Build edge opposite vertices and valence/boundary sets
        let mut opposites: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let mut edge_use_count: HashMap<(u32, u32), u32> = HashMap::new();
        let mut neighbors: Vec<HashSet<u32>> = vec![HashSet::new(); old_vertex_count];
        for tri in mesh.indices.chunks_exact(3) {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];
            let e01 = edge_key(v0, v1);
            let e12 = edge_key(v1, v2);
            let e20 = edge_key(v2, v0);
            opposites.entry(e01).or_default().push(v2);
            opposites.entry(e12).or_default().push(v0);
            opposites.entry(e20).or_default().push(v1);
            *edge_use_count.entry(e01).or_insert(0) += 1;
            *edge_use_count.entry(e12).or_insert(0) += 1;
            *edge_use_count.entry(e20).or_insert(0) += 1;
            neighbors[v0 as usize].insert(v1);
            neighbors[v0 as usize].insert(v2);
            neighbors[v1 as usize].insert(v0);
            neighbors[v1 as usize].insert(v2);
            neighbors[v2 as usize].insert(v0);
            neighbors[v2 as usize].insert(v1);
        }
        let boundary_edges: HashSet<(u32, u32)> = edge_use_count
            .iter()
            .filter_map(|(e, &c)| if c == 1 { Some(*e) } else { None })
            .collect();

        // Helper: is crease edge
        let is_crease_edge = |e: (u32, u32)| -> bool {
            crease_set.contains(&e) || (preserve_boundary && boundary_edges.contains(&e))
        };

        // Map to new edge vertex index
        let mut edge_new_idx: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_positions = mesh.positions.clone();
        let mut new_uvs = if has_uv { mesh.uvs.clone() } else { Vec::new() };

        // Create edge points
        for (e, opps) in opposites.iter() {
            let (a, b) = *e;
            let pa = mesh.positions[a as usize];
            let pb = mesh.positions[b as usize];
            let pos = if is_crease_edge(*e) || opps.len() < 2 {
                [
                    (pa[0] + pb[0]) * 0.5,
                    (pa[1] + pb[1]) * 0.5,
                    (pa[2] + pb[2]) * 0.5,
                ]
            } else {
                let c = mesh.positions[opps[0] as usize];
                let d = mesh.positions[opps[1] as usize];
                [
                    0.375 * (pa[0] + pb[0]) + 0.125 * (c[0] + d[0]),
                    0.375 * (pa[1] + pb[1]) + 0.125 * (c[1] + d[1]),
                    0.375 * (pa[2] + pb[2]) + 0.125 * (c[2] + d[2]),
                ]
            };
            let idx = new_positions.len() as u32;
            new_positions.push(pos);
            if has_uv {
                let ua = mesh.uvs[a as usize];
                let ub = mesh.uvs[b as usize];
                let uv = if is_crease_edge(*e) || opps.len() < 2 {
                    [(ua[0] + ub[0]) * 0.5, (ua[1] + ub[1]) * 0.5]
                } else {
                    // Use same weights for UVs
                    let uc = mesh.uvs[opps[0] as usize];
                    let ud = mesh.uvs[opps[1] as usize];
                    [
                        0.375 * (ua[0] + ub[0]) + 0.125 * (uc[0] + ud[0]),
                        0.375 * (ua[1] + ub[1]) + 0.125 * (uc[1] + ud[1]),
                    ]
                };
                new_uvs.push(uv);
            }
            edge_new_idx.insert(*e, idx);
        }

        // Re-index triangles -> 4 each
        let mut new_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len() * 4);
        for tri in mesh.indices.chunks_exact(3) {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];
            let m01 = *edge_new_idx.get(&edge_key(v0, v1)).unwrap();
            let m12 = *edge_new_idx.get(&edge_key(v1, v2)).unwrap();
            let m20 = *edge_new_idx.get(&edge_key(v2, v0)).unwrap();
            new_indices.extend_from_slice(&[v0, m01, m20]);
            new_indices.extend_from_slice(&[v1, m12, m01]);
            new_indices.extend_from_slice(&[v2, m20, m12]);
            new_indices.extend_from_slice(&[m01, m12, m20]);
        }

        // Smooth original vertices (Loop)
        let mut smoothed = vec![[0.0f32; 3]; old_vertex_count];
        let mut smoothed_uv = if has_uv {
            vec![[0.0f32; 2]; old_vertex_count]
        } else {
            Vec::new()
        };

        // Identify crease vertices (two or more crease edges incident)
        let mut crease_count: Vec<u32> = vec![0; old_vertex_count];
        for e in crease_set.iter() {
            if (e.0 as usize) < old_vertex_count && (e.1 as usize) < old_vertex_count {
                crease_count[e.0 as usize] += 1;
                crease_count[e.1 as usize] += 1;
            }
        }
        if preserve_boundary {
            for e in boundary_edges.iter() {
                crease_count[e.0 as usize] += 1;
                crease_count[e.1 as usize] += 1;
            }
        }

        for vi in 0..old_vertex_count {
            let nbrs: Vec<u32> = neighbors[vi].iter().copied().collect();
            let n = nbrs.len();
            let v = mesh.positions[vi];
            let crease_n = crease_count[vi];
            if preserve_boundary && crease_n >= 2 {
                // Crease/boundary rule: (6/8)*v + (1/8)*(v_prev+v_next)
                // Find two crease neighbors if possible, else pick first two neighbors
                let mut candidates: Vec<u32> = Vec::new();
                for &nb in nbrs.iter() {
                    if crease_set.contains(&edge_key(vi as u32, nb))
                        || boundary_edges.contains(&edge_key(vi as u32, nb))
                    {
                        candidates.push(nb);
                        if candidates.len() == 2 {
                            break;
                        }
                    }
                }
                if candidates.len() < 2 {
                    candidates.extend(nbrs.iter().copied().take(2 - candidates.len()));
                }
                let p1 = mesh.positions[candidates.get(0).copied().unwrap_or(vi as u32) as usize];
                let p2 = mesh.positions[candidates.get(1).copied().unwrap_or(vi as u32) as usize];
                smoothed[vi] = [
                    0.75 * v[0] + 0.125 * (p1[0] + p2[0]),
                    0.75 * v[1] + 0.125 * (p1[1] + p2[1]),
                    0.75 * v[2] + 0.125 * (p1[2] + p2[2]),
                ];
                if has_uv {
                    let uv = mesh.uvs[vi];
                    let uv1 = mesh.uvs[candidates.get(0).copied().unwrap_or(vi as u32) as usize];
                    let uv2 = mesh.uvs[candidates.get(1).copied().unwrap_or(vi as u32) as usize];
                    smoothed_uv[vi] = [
                        0.75 * uv[0] + 0.125 * (uv1[0] + uv2[0]),
                        0.75 * uv[1] + 0.125 * (uv1[1] + uv2[1]),
                    ];
                }
            } else if n >= 3 {
                // Interior Loop rule
                let beta = beta_loop(n);
                let mut sum = [0.0f32; 3];
                let mut sum_uv = [0.0f32; 2];
                for &nb in nbrs.iter() {
                    let p = mesh.positions[nb as usize];
                    sum[0] += p[0];
                    sum[1] += p[1];
                    sum[2] += p[2];
                    if has_uv {
                        let uv = mesh.uvs[nb as usize];
                        sum_uv[0] += uv[0];
                        sum_uv[1] += uv[1];
                    }
                }
                smoothed[vi] = [
                    (1.0 - (n as f32) * beta) * v[0] + beta * sum[0],
                    (1.0 - (n as f32) * beta) * v[1] + beta * sum[1],
                    (1.0 - (n as f32) * beta) * v[2] + beta * sum[2],
                ];
                if has_uv {
                    let uv = mesh.uvs[vi];
                    smoothed_uv[vi] = [
                        (1.0 - (n as f32) * beta) * uv[0] + beta * sum_uv[0],
                        (1.0 - (n as f32) * beta) * uv[1] + beta * sum_uv[1],
                    ];
                }
            } else {
                smoothed[vi] = v;
                if has_uv {
                    smoothed_uv[vi] = mesh.uvs[vi];
                }
            }
        }

        // Apply smoothed old vertices + keep newly added edge vertices
        for vi in 0..old_vertex_count {
            new_positions[vi] = smoothed[vi];
            if has_uv {
                new_uvs[vi] = smoothed_uv[vi];
            }
        }

        mesh.positions = new_positions;
        mesh.indices = new_indices;
        if has_uv {
            mesh.uvs = new_uvs;
        }

        // Recompute normals for smooth shading
        recompute_normals(&mut mesh);
    }
    mesh
}

fn compute_max_edge_length(mesh: &MeshBuffers) -> f32 {
    let mut max_len = 0.0f32;
    for tri in mesh.indices.chunks_exact(3) {
        let v = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p0 = mesh.positions[v[0]];
        let p1 = mesh.positions[v[1]];
        let p2 = mesh.positions[v[2]];
        let e01 =
            ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        let e12 =
            ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
        let e20 =
            ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2) + (p0[2] - p2[2]).powi(2)).sqrt();
        max_len = max_len.max(e01.max(e12.max(e20)));
    }
    max_len
}

fn face_normal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let ux = p1[0] - p0[0];
    let uy = p1[1] - p0[1];
    let uz = p1[2] - p0[2];
    let vx = p2[0] - p0[0];
    let vy = p2[1] - p0[1];
    let vz = p2[2] - p0[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 0.0 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn max_dihedral_angle(mesh: &MeshBuffers) -> f32 {
    let mut edge_to_normals: HashMap<(u32, u32), Vec<[f32; 3]>> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let v0 = tri[0] as usize;
        let v1 = tri[1] as usize;
        let v2 = tri[2] as usize;
        let n = face_normal(mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]);
        let e01 = edge_key(tri[0], tri[1]);
        let e12 = edge_key(tri[1], tri[2]);
        let e20 = edge_key(tri[2], tri[0]);
        edge_to_normals.entry(e01).or_default().push(n);
        edge_to_normals.entry(e12).or_default().push(n);
        edge_to_normals.entry(e20).or_default().push(n);
    }
    let mut max_angle = 0.0f32;
    for normals in edge_to_normals.values() {
        if normals.len() >= 2 {
            let a = normals[0];
            let b = normals[1];
            let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
            let angle = dot.acos();
            max_angle = max_angle.max(angle);
        }
    }
    // Convert to degrees for easier thresholding by callers? Keep radians
    max_angle
}

pub fn subdivide_adaptive(
    input: &MeshBuffers,
    edge_length_limit: Option<f32>,
    curvature_threshold: Option<f32>,
    max_levels: u32,
    creases: Option<&[(u32, u32)]>,
    preserve_boundary: bool,
) -> MeshBuffers {
    let mut levels = 0u32;
    if let Some(th) = edge_length_limit {
        if th > 0.0 {
            let orig = compute_max_edge_length(input);
            if orig > th {
                let ratio = orig / th;
                let l = ratio.log2().ceil() as u32;
                levels = levels.max(l);
            }
        }
    }
    if let Some(curv_th) = curvature_threshold {
        if curv_th > 0.0 {
            let max_d = max_dihedral_angle(input);
            if max_d > curv_th {
                // Add levels proportional to how much we exceed the threshold
                let ratio = (max_d / curv_th).ceil();
                let l = if ratio > 1.0 {
                    (ratio as u32).saturating_sub(1)
                } else {
                    0
                };
                levels = levels.max(l);
            }
        }
    }
    if levels == 0 {
        return input.clone();
    }
    let levels = levels.min(max_levels);
    subdivide_triangles_with_options(input, levels, creases, preserve_boundary)
}
