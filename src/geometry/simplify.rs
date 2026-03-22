// src/geometry/simplify.rs
// QEM (Quadric Error Metrics) edge-collapse mesh simplification
// Exists to reduce triangle counts for LOD generation in terrain population pipelines
// RELEVANT FILES:src/geometry/mod.rs,src/geometry/primitives.rs,src/geometry/weld.rs

use std::collections::{BinaryHeap, HashMap, HashSet};

use super::{GeometryError, GeometryResult, MeshBuffers};

/// A symmetric 4x4 quadric matrix stored as 10 unique coefficients (f64 for precision).
#[derive(Debug, Clone, Copy)]
struct Quadric {
    // Upper-triangular entries of the symmetric 4x4 matrix:
    // | a  b  c  d |
    // |    e  f  g |
    // |       h  i |
    // |          j |
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    i: f64,
    j: f64,
}

impl Quadric {
    fn zero() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
            g: 0.0,
            h: 0.0,
            i: 0.0,
            j: 0.0,
        }
    }

    /// Build quadric from a plane equation ax + by + cz + d = 0.
    fn from_plane(nx: f64, ny: f64, nz: f64, d: f64) -> Self {
        Self {
            a: nx * nx,
            b: nx * ny,
            c: nx * nz,
            d: nx * d,
            e: ny * ny,
            f: ny * nz,
            g: ny * d,
            h: nz * nz,
            i: nz * d,
            j: d * d,
        }
    }

    fn add(&self, other: &Quadric) -> Self {
        Self {
            a: self.a + other.a,
            b: self.b + other.b,
            c: self.c + other.c,
            d: self.d + other.d,
            e: self.e + other.e,
            f: self.f + other.f,
            g: self.g + other.g,
            h: self.h + other.h,
            i: self.i + other.i,
            j: self.j + other.j,
        }
    }

    /// Evaluate quadric error for a point [x, y, z].
    fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let r = self.a * x * x
            + 2.0 * self.b * x * y
            + 2.0 * self.c * x * z
            + 2.0 * self.d * x
            + self.e * y * y
            + 2.0 * self.f * y * z
            + 2.0 * self.g * y
            + self.h * z * z
            + 2.0 * self.i * z
            + self.j;
        r.max(0.0)
    }
}

/// Canonical edge key with smaller vertex first.
#[inline]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Entry in the collapse priority queue. Ordered by cost (min-heap via Reverse ordering).
#[derive(Debug, Clone)]
struct CollapseCandidate {
    cost: f64,
    v0: u32,
    v1: u32,
    /// Generation counter to detect stale entries.
    generation: u32,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering so BinaryHeap gives us min-cost first.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Simplify a triangle mesh using Quadric Error Metrics (QEM) edge-collapse.
///
/// `target_ratio` must be in (0.0, 1.0] — the fraction of original triangles to keep.
/// Returns a new `MeshBuffers` with reduced geometry:
/// - Normals are recomputed (area-weighted vertex normals).
/// - UVs and tangents are carried through best-effort (copied from surviving vertex).
pub fn simplify_mesh(mesh: &MeshBuffers, target_ratio: f32) -> GeometryResult<MeshBuffers> {
    // --- Validation ---
    if target_ratio <= 0.0 || target_ratio > 1.0 {
        return Err(GeometryError::new(format!(
            "target_ratio must be in (0.0, 1.0], got {}",
            target_ratio
        )));
    }
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        return Err(GeometryError::new("Cannot simplify an empty mesh"));
    }
    let original_tri_count = mesh.indices.len() / 3;
    if original_tri_count == 0 {
        return Err(GeometryError::new("Mesh has no triangles to simplify"));
    }

    // Ratio 1.0 → just clone
    if (target_ratio - 1.0).abs() < 1e-7 {
        return Ok(mesh.clone());
    }

    let target_tris = ((original_tri_count as f64 * target_ratio as f64).ceil() as usize).max(1);
    let vertex_count = mesh.positions.len();
    let has_uvs = mesh.uvs.len() == vertex_count;
    let has_tangents = mesh.tangents.len() == vertex_count;

    // --- Working copies ---
    let mut positions: Vec<[f64; 3]> = mesh
        .positions
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let mut uvs: Vec<[f32; 2]> = if has_uvs {
        mesh.uvs.clone()
    } else {
        Vec::new()
    };
    let tangents: Vec<[f32; 4]> = if has_tangents {
        mesh.tangents.clone()
    } else {
        Vec::new()
    };

    // Triangle storage: each triangle is [v0, v1, v2], u32::MAX marks deleted
    let mut triangles: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    // Union-find for vertex merging
    let mut parent: Vec<u32> = (0..vertex_count as u32).collect();

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize]; // path compression
            x = parent[x as usize];
        }
        x
    }

    // --- Build vertex-to-triangle adjacency ---
    let mut vert_tris: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
    for (ti, tri) in triangles.iter().enumerate() {
        for &v in tri {
            vert_tris[v as usize].push(ti);
        }
    }

    // --- Build per-vertex quadrics from incident face planes ---
    let mut quadrics: Vec<Quadric> = vec![Quadric::zero(); vertex_count];
    for tri in &triangles {
        let p0 = positions[tri[0] as usize];
        let p1 = positions[tri[1] as usize];
        let p2 = positions[tri[2] as usize];
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
        if len < 1e-14 {
            continue;
        }
        let nx = nx / len;
        let ny = ny / len;
        let nz = nz / len;
        let d = -(nx * p0[0] + ny * p0[1] + nz * p0[2]);
        let q = Quadric::from_plane(nx, ny, nz, d);
        for &v in tri {
            quadrics[v as usize] = quadrics[v as usize].add(&q);
        }
    }

    // --- Detect boundary edges (edges with only 1 adjacent triangle) ---
    let mut edge_tri_count: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in &triangles {
        for pair in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let ek = edge_key(pair.0, pair.1);
            *edge_tri_count.entry(ek).or_insert(0) += 1;
        }
    }
    let boundary_edges: HashSet<(u32, u32)> = edge_tri_count
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&ek, _)| ek)
        .collect();

    // --- Build initial priority queue ---
    let mut generation: Vec<u32> = vec![0; vertex_count];
    let mut heap: BinaryHeap<CollapseCandidate> = BinaryHeap::new();

    // Collect unique edges
    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();
    for tri in &triangles {
        for pair in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let ek = edge_key(pair.0, pair.1);
            if seen_edges.insert(ek) {
                let cost = compute_collapse_cost(
                    &positions,
                    &quadrics,
                    ek.0,
                    ek.1,
                    boundary_edges.contains(&ek),
                );
                heap.push(CollapseCandidate {
                    cost,
                    v0: ek.0,
                    v1: ek.1,
                    generation: 0,
                });
            }
        }
    }

    // --- Greedy edge collapse ---
    let mut live_tri_count = original_tri_count;

    while live_tri_count > target_tris {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let rv0 = find(&mut parent, candidate.v0);
        let rv1 = find(&mut parent, candidate.v1);

        // Skip stale or self-referencing
        if rv0 == rv1 {
            continue;
        }
        if candidate.generation != generation[rv0 as usize]
            && candidate.generation != generation[rv1 as usize]
        {
            // Both endpoints have been updated since this candidate was enqueued — stale
            // Re-check: if either generation matches, we allow it (one side is current)
            continue;
        }

        // Collapse rv1 into rv0: merge at midpoint
        let p0 = positions[rv0 as usize];
        let p1 = positions[rv1 as usize];
        let mid = [
            (p0[0] + p1[0]) * 0.5,
            (p0[1] + p1[1]) * 0.5,
            (p0[2] + p1[2]) * 0.5,
        ];
        positions[rv0 as usize] = mid;

        // Merge UVs (average)
        if has_uvs {
            let uv0 = uvs[rv0 as usize];
            let uv1 = uvs[rv1 as usize];
            uvs[rv0 as usize] = [
                (uv0[0] + uv1[0]) * 0.5,
                (uv0[1] + uv1[1]) * 0.5,
            ];
        }

        // Tangents: keep from rv0 (best-effort)

        // Merge quadrics
        quadrics[rv0 as usize] = quadrics[rv0 as usize].add(&quadrics[rv1 as usize]);

        // Union-find: rv1 -> rv0
        parent[rv1 as usize] = rv0;

        // Update triangles: replace rv1 with rv0 and remove degenerate triangles
        // Collect triangle indices to update from both vertices
        let tris_to_check: Vec<usize> = {
            let mut set: HashSet<usize> = HashSet::new();
            for &ti in &vert_tris[rv0 as usize] {
                set.insert(ti);
            }
            for &ti in &vert_tris[rv1 as usize] {
                set.insert(ti);
            }
            set.into_iter().collect()
        };

        let mut neighbor_verts: HashSet<u32> = HashSet::new();
        let mut surviving_tris: Vec<usize> = Vec::new();

        for ti in tris_to_check {
            let tri = &mut triangles[ti];
            if tri[0] == u32::MAX {
                continue; // already deleted
            }
            // Resolve all vertices through union-find
            for v in tri.iter_mut() {
                *v = find(&mut parent, *v);
            }
            // Check degenerate (two or more same vertices)
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                // Mark deleted
                live_tri_count -= 1;
                tri[0] = u32::MAX;
                continue;
            }
            surviving_tris.push(ti);
            for &v in tri.iter() {
                if v != rv0 {
                    neighbor_verts.insert(v);
                }
            }
        }

        // Update vert_tris for rv0
        vert_tris[rv0 as usize] = surviving_tris;

        // Bump generation for rv0
        generation[rv0 as usize] += 1;

        // Re-enqueue edges from rv0 to its neighbors
        for &nv in &neighbor_verts {
            let ek = edge_key(rv0, nv);
            let is_boundary = boundary_edges.contains(&ek);
            let cost =
                compute_collapse_cost(&positions, &quadrics, rv0, nv, is_boundary);
            heap.push(CollapseCandidate {
                cost,
                v0: ek.0,
                v1: ek.1,
                generation: generation[rv0 as usize],
            });
        }

        if live_tri_count <= target_tris {
            break;
        }
    }

    // --- Compact: build new mesh ---
    // Collect live vertices and remap
    let mut used_verts: HashSet<u32> = HashSet::new();
    let mut live_indices: Vec<[u32; 3]> = Vec::new();
    for tri in &triangles {
        if tri[0] == u32::MAX {
            continue;
        }
        let t = [
            find(&mut parent, tri[0]),
            find(&mut parent, tri[1]),
            find(&mut parent, tri[2]),
        ];
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            continue;
        }
        used_verts.insert(t[0]);
        used_verts.insert(t[1]);
        used_verts.insert(t[2]);
        live_indices.push(t);
    }

    let mut sorted_verts: Vec<u32> = used_verts.into_iter().collect();
    sorted_verts.sort_unstable();
    let mut vert_remap: HashMap<u32, u32> = HashMap::new();
    for (new_idx, &old_idx) in sorted_verts.iter().enumerate() {
        vert_remap.insert(old_idx, new_idx as u32);
    }

    let new_vertex_count = sorted_verts.len();
    let mut result = MeshBuffers::with_capacity(new_vertex_count, live_indices.len() * 3);

    for &old_idx in &sorted_verts {
        let p = positions[old_idx as usize];
        result.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        if has_uvs {
            result.uvs.push(uvs[old_idx as usize]);
        }
        if has_tangents {
            result.tangents.push(tangents[old_idx as usize]);
        }
    }

    for tri in &live_indices {
        result.indices.push(vert_remap[&tri[0]]);
        result.indices.push(vert_remap[&tri[1]]);
        result.indices.push(vert_remap[&tri[2]]);
    }

    // --- Recompute area-weighted normals ---
    recompute_area_weighted_normals(&mut result);

    Ok(result)
}

fn compute_collapse_cost(
    positions: &[[f64; 3]],
    quadrics: &[Quadric],
    v0: u32,
    v1: u32,
    is_boundary: bool,
) -> f64 {
    let p0 = positions[v0 as usize];
    let p1 = positions[v1 as usize];
    let mid = [
        (p0[0] + p1[0]) * 0.5,
        (p0[1] + p1[1]) * 0.5,
        (p0[2] + p1[2]) * 0.5,
    ];
    let q = quadrics[v0 as usize].add(&quadrics[v1 as usize]);
    let mut cost = q.evaluate(mid[0], mid[1], mid[2]);
    if is_boundary {
        cost *= 10.0;
    }
    cost
}

fn recompute_area_weighted_normals(mesh: &mut MeshBuffers) {
    let n = mesh.positions.len();
    mesh.normals.clear();
    mesh.normals.resize(n, [0.0, 0.0, 0.0]);

    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];
        // Cross product (unnormalized = area-weighted normal)
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
        }
    }

    // Normalize
    for n in mesh.normals.iter_mut() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-10 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            *n = [0.0, 0.0, 1.0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{
        generate_primitive, generate_unit_box, PrimitiveParams, PrimitiveType,
    };

    #[test]
    fn simplify_box_reduces_triangles() {
        let mesh = generate_unit_box();
        assert_eq!(mesh.triangle_count(), 12);
        let result = simplify_mesh(&mesh, 0.5).unwrap();
        assert!(
            result.triangle_count() <= 6,
            "Expected ≤6 tris, got {}",
            result.triangle_count()
        );
        assert!(
            result.triangle_count() >= 1,
            "Should have at least 1 triangle"
        );
        assert_eq!(
            result.normals.len(),
            result.positions.len(),
            "Normals count must match vertex count"
        );
    }

    #[test]
    fn simplify_sphere_reduces_geometry() {
        let mesh = generate_primitive(
            PrimitiveType::Sphere,
            PrimitiveParams {
                rings: 16,
                radial_segments: 32,
                ..Default::default()
            },
        );
        let orig_tris = mesh.triangle_count();
        let orig_verts = mesh.vertex_count();
        let result = simplify_mesh(&mesh, 0.25).unwrap();
        assert!(
            result.triangle_count() < orig_tris,
            "Triangle count should decrease"
        );
        assert!(
            result.vertex_count() < orig_verts,
            "Vertex count should decrease"
        );
    }

    #[test]
    fn ratio_one_returns_clone() {
        let mesh = generate_unit_box();
        let result = simplify_mesh(&mesh, 1.0).unwrap();
        assert_eq!(result.triangle_count(), mesh.triangle_count());
        assert_eq!(result.vertex_count(), mesh.vertex_count());
    }

    #[test]
    fn very_low_ratio_on_small_mesh_returns_nonempty() {
        let mesh = generate_unit_box();
        let result = simplify_mesh(&mesh, 0.01).unwrap();
        assert!(
            result.triangle_count() >= 1,
            "Should have at least 1 triangle, got {}",
            result.triangle_count()
        );
    }

    #[test]
    fn rejects_empty_mesh() {
        let mesh = MeshBuffers::default();
        let err = simplify_mesh(&mesh, 0.5).unwrap_err();
        assert!(
            err.message().to_lowercase().contains("empty"),
            "Error should mention 'empty', got: {}",
            err.message()
        );
    }

    #[test]
    fn rejects_invalid_ratio() {
        let mesh = generate_unit_box();
        assert!(simplify_mesh(&mesh, 0.0).is_err());
        assert!(simplify_mesh(&mesh, -0.5).is_err());
        assert!(simplify_mesh(&mesh, 1.5).is_err());
    }

    #[test]
    fn preserves_uvs_when_present() {
        let mesh = generate_primitive(
            PrimitiveType::Sphere,
            PrimitiveParams {
                rings: 16,
                radial_segments: 32,
                ..Default::default()
            },
        );
        assert!(
            !mesh.uvs.is_empty(),
            "Sphere should have UVs"
        );
        let result = simplify_mesh(&mesh, 0.5).unwrap();
        assert_eq!(
            result.uvs.len(),
            result.positions.len(),
            "UV count must match vertex count after simplification"
        );
    }

    #[test]
    fn boundary_preservation_on_open_mesh() {
        let mesh = generate_primitive(
            PrimitiveType::Plane,
            PrimitiveParams {
                resolution: (8, 8),
                ..Default::default()
            },
        );
        let orig_tris = mesh.triangle_count();
        let result = simplify_mesh(&mesh, 0.5).unwrap();
        assert!(
            result.triangle_count() < orig_tris,
            "Triangle count should decrease"
        );
        assert!(
            result.triangle_count() >= 1,
            "Should have at least 1 triangle"
        );
        // Verify it's a valid mesh
        assert_eq!(result.indices.len() % 3, 0);
        assert_eq!(result.normals.len(), result.positions.len());
    }
}
