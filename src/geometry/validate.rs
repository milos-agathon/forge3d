// src/geometry/validate.rs
// Mesh validation utilities for Forge3D geometry module
// Exists to provide diagnostics covering stats, degenerates, and topology issues
// RELEVANT FILES:src/geometry/mod.rs,src/geometry/weld.rs,tests/test_f15_validate.py,python/forge3d/geometry.py

use std::collections::BTreeMap;

use glam::Vec3;

use super::MeshBuffers;

/// Basic statistics collected for a mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

impl Default for MeshStats {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            triangle_count: 0,
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
        }
    }
}

/// Issue categories detected by validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshValidationIssue {
    IndexOutOfBounds { index: u32 },
    DegenerateTriangle { triangle: usize },
    DuplicateVertex { first: usize, duplicate: usize },
    NonManifoldEdge { edge: (u32, u32), count: u32 },
}

/// Complete validation report.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshValidationReport {
    pub stats: MeshStats,
    pub issues: Vec<MeshValidationIssue>,
}

impl MeshValidationReport {
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Run validation across topology, geometry, and statistics.
pub fn validate_mesh(mesh: &MeshBuffers) -> MeshValidationReport {
    let stats = compute_stats(mesh);
    let mut issues = Vec::new();

    let vertex_count = mesh.vertex_count();

    let mut invalid_indices = Vec::new();
    for &idx in &mesh.indices {
        if idx as usize >= vertex_count {
            issues.push(MeshValidationIssue::IndexOutOfBounds { index: idx });
            invalid_indices.push(idx);
        }
    }

    detect_degenerate_triangles(mesh, &mut issues);
    detect_duplicate_vertices(mesh, &mut issues);
    detect_non_manifold_edges(mesh, &invalid_indices, &mut issues);

    MeshValidationReport { stats, issues }
}

fn compute_stats(mesh: &MeshBuffers) -> MeshStats {
    if mesh.positions.is_empty() {
        return MeshStats::default();
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for pos in &mesh.positions {
        for i in 0..3 {
            if pos[i] < min[i] {
                min[i] = pos[i];
            }
            if pos[i] > max[i] {
                max[i] = pos[i];
            }
        }
    }

    MeshStats {
        vertex_count: mesh.vertex_count(),
        triangle_count: mesh.triangle_count(),
        bbox_min: min,
        bbox_max: max,
    }
}

fn detect_degenerate_triangles(mesh: &MeshBuffers, issues: &mut Vec<MeshValidationIssue>) {
    let mut triangle_index = 0usize;
    for chunk in mesh.indices.chunks_exact(3) {
        if chunk.len() < 3 {
            break;
        }
        let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
        if i0 == i1 || i1 == i2 || i0 == i2 {
            issues.push(MeshValidationIssue::DegenerateTriangle {
                triangle: triangle_index,
            });
            triangle_index += 1;
            continue;
        }

        let p0 = Vec3::from(mesh.positions[i0 as usize]);
        let p1 = Vec3::from(mesh.positions[i1 as usize]);
        let p2 = Vec3::from(mesh.positions[i2 as usize]);
        let cross = (p1 - p0).cross(p2 - p0);
        if cross.length_squared() <= 1e-12 {
            issues.push(MeshValidationIssue::DegenerateTriangle {
                triangle: triangle_index,
            });
        }
        triangle_index += 1;
    }
}

fn detect_duplicate_vertices(mesh: &MeshBuffers, issues: &mut Vec<MeshValidationIssue>) {
    if mesh.positions.is_empty() {
        return;
    }
    let mut seen = BTreeMap::new();
    let eps = 1e-6;
    for (idx, pos) in mesh.positions.iter().enumerate() {
        let key = (
            quantize_scalar(pos[0], eps),
            quantize_scalar(pos[1], eps),
            quantize_scalar(pos[2], eps),
        );
        if let Some(&first) = seen.get(&key) {
            issues.push(MeshValidationIssue::DuplicateVertex {
                first,
                duplicate: idx,
            });
        } else {
            seen.insert(key, idx);
        }
    }
}

fn detect_non_manifold_edges(
    mesh: &MeshBuffers,
    invalid_indices: &[u32],
    issues: &mut Vec<MeshValidationIssue>,
) {
    if mesh.indices.len() < 3 {
        return;
    }

    let mut invalid = BTreeMap::new();
    for &idx in invalid_indices {
        invalid.insert(idx, ());
    }

    let mut edge_counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();

    for chunk in mesh.indices.chunks_exact(3) {
        if chunk.len() < 3 {
            continue;
        }
        let tri = (chunk[0], chunk[1], chunk[2]);
        if tri.0 == tri.1 || tri.1 == tri.2 || tri.0 == tri.2 {
            continue;
        }
        if invalid.contains_key(&tri.0)
            || invalid.contains_key(&tri.1)
            || invalid.contains_key(&tri.2)
        {
            continue;
        }

        add_edge(&mut edge_counts, tri.0, tri.1);
        add_edge(&mut edge_counts, tri.1, tri.2);
        add_edge(&mut edge_counts, tri.2, tri.0);
    }

    for (edge, count) in edge_counts {
        if count > 2 {
            issues.push(MeshValidationIssue::NonManifoldEdge { edge, count });
        }
    }
}

fn add_edge(edge_counts: &mut BTreeMap<(u32, u32), u32>, a: u32, b: u32) {
    let key = if a < b { (a, b) } else { (b, a) };
    *edge_counts.entry(key).or_insert(0) += 1;
}

fn quantize_scalar(value: f32, eps: f32) -> i64 {
    (value / eps).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle_mesh() -> MeshBuffers {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let uvs = vec![[0.0, 0.0]; 3];
        let indices = vec![0, 1, 2];
        MeshBuffers {
            positions,
            normals,
            uvs,
            indices,
        }
    }

    #[test]
    fn detects_clean_mesh() {
        let mesh = make_triangle_mesh();
        let report = validate_mesh(&mesh);
        assert!(report.is_clean());
        assert_eq!(report.stats.vertex_count, 3);
        assert_eq!(report.stats.triangle_count, 1);
    }

    #[test]
    fn detects_duplicate_vertices() {
        let mut mesh = make_triangle_mesh();
        mesh.positions.push([0.0, 0.0, 0.0]);
        let report = validate_mesh(&mesh);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, MeshValidationIssue::DuplicateVertex { .. })));
    }

    #[test]
    fn detects_non_manifold_edges() {
        let mut mesh = MeshBuffers::default();
        mesh.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        mesh.indices = vec![0, 1, 2, 2, 1, 3, 0, 2, 1];
        let report = validate_mesh(&mesh);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, MeshValidationIssue::NonManifoldEdge { .. })));
    }
}
