use super::types::MeshStats;
use crate::accel::cpu_bvh::{Aabb, MeshCPU};
use anyhow::Result;

/// Reusable mesh builder for creating common test meshes
pub struct MeshBuilder;

impl MeshBuilder {
    /// Create a simple triangle mesh
    pub fn triangle() -> MeshCPU {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let indices = vec![[0, 1, 2]];
        MeshCPU::new(vertices, indices)
    }

    /// Create a unit cube mesh (12 triangles)
    pub fn cube() -> MeshCPU {
        let vertices = vec![
            // Front face
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            // Back face
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let indices = vec![
            // Front face
            [0, 1, 2],
            [0, 2, 3],
            // Right face
            [1, 5, 6],
            [1, 6, 2],
            // Back face
            [5, 4, 7],
            [5, 7, 6],
            // Left face
            [4, 0, 3],
            [4, 3, 7],
            // Top face
            [3, 2, 6],
            [3, 6, 7],
            // Bottom face
            [4, 5, 1],
            [4, 1, 0],
        ];
        MeshCPU::new(vertices, indices)
    }

    /// Create a quad mesh (2 triangles)
    pub fn quad() -> MeshCPU {
        let vertices = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let indices = vec![
            [0, 1, 2], // First triangle
            [0, 2, 3], // Second triangle
        ];
        MeshCPU::new(vertices, indices)
    }
}

/// Mesh validation utilities
pub fn validate_mesh(mesh: &MeshCPU) -> Result<()> {
    if mesh.vertices.is_empty() {
        anyhow::bail!("Mesh has no vertices");
    }

    if mesh.indices.is_empty() {
        anyhow::bail!("Mesh has no triangles");
    }

    // Check that all indices are valid
    let vertex_count = mesh.vertices.len();
    for (tri_idx, triangle) in mesh.indices.iter().enumerate() {
        for (corner_idx, &vertex_idx) in triangle.iter().enumerate() {
            if vertex_idx as usize >= vertex_count {
                anyhow::bail!(
                    "Triangle {} corner {} references invalid vertex {} (max {})",
                    tri_idx,
                    corner_idx,
                    vertex_idx,
                    vertex_count - 1
                );
            }
        }
    }

    // Check for degenerate triangles
    let mut degenerate_count = 0;
    for (tri_idx, _) in mesh.indices.iter().enumerate() {
        if let Some((v0, v1, v2)) = mesh.get_triangle(tri_idx) {
            let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            // Cross product magnitude
            let cross = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            let area = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

            if area < 1e-6 {
                degenerate_count += 1;
            }
        }
    }

    if degenerate_count > 0 {
        log::warn!("Mesh contains {} degenerate triangles", degenerate_count);
    }

    Ok(())
}

pub fn compute_mesh_stats(mesh: &MeshCPU) -> MeshStats {
    let mut world_aabb = Aabb::empty();
    let mut total_area = 0.0;

    for i in 0..mesh.triangle_count() {
        if let Some(aabb) = mesh.triangle_aabb(i as usize) {
            world_aabb.expand_aabb(&aabb);
        }

        if let Some((v0, v1, v2)) = mesh.get_triangle(i as usize) {
            let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            let area =
                (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt() * 0.5;
            total_area += area;
        }
    }

    let avg_area = if mesh.triangle_count() > 0 {
        total_area / mesh.triangle_count() as f32
    } else {
        0.0
    };

    let memory_usage = (mesh.vertices.len() * std::mem::size_of::<[f32; 3]>()
        + mesh.indices.len() * std::mem::size_of::<[u32; 3]>()) as u64;

    MeshStats {
        vertex_count: mesh.vertex_count(),
        triangle_count: mesh.triangle_count(),
        world_aabb,
        average_triangle_area: avg_area,
        memory_usage_bytes: memory_usage,
    }
}
