//! Tangent, Bitangent, Normal (TBN) generation for indexed meshes
//!
//! Provides MikkTSpace-compatible TBN generation for normal mapping and PBR shading.
//! Calculates per-vertex tangents and bitangents from positions, normals, and UV coordinates.

use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// Vertex data required for TBN calculation
#[derive(Debug, Clone)]
pub struct TbnVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

/// Generated TBN data for a vertex
#[derive(Debug, Clone, Copy)]
pub struct TbnData {
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub normal: Vec3,
    pub handedness: f32, // +1.0 or -1.0
}

impl TbnData {
    pub fn new(tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Self {
        // Calculate handedness using cross product
        let cross = tangent.cross(bitangent);
        let handedness = if cross.dot(normal) >= 0.0 { 1.0 } else { -1.0 };

        Self {
            tangent,
            bitangent,
            normal,
            handedness,
        }
    }

    /// Validate TBN orthogonality and handedness
    pub fn is_valid(&self) -> bool {
        let t_len = self.tangent.length();
        let b_len = self.bitangent.length();
        let n_len = self.normal.length();

        // Check unit length (within tolerance)
        if (t_len - 1.0).abs() > 1e-3 || (b_len - 1.0).abs() > 1e-3 || (n_len - 1.0).abs() > 1e-3 {
            return false;
        }

        // Check orthogonality
        if self.tangent.dot(self.normal).abs() > 1e-3 {
            return false;
        }

        // Check handedness consistency
        let cross = self.tangent.cross(self.bitangent);
        let computed_handedness = if cross.dot(self.normal) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        if (computed_handedness - self.handedness).abs() > 1e-3 {
            return false;
        }

        true
    }
}

/// Triangle for TBN calculation
struct Triangle {
    indices: [u32; 3],
    positions: [Vec3; 3],
    normals: [Vec3; 3],
    uvs: [Vec2; 3],
}

impl Triangle {
    fn new(vertices: &[TbnVertex], indices: [u32; 3]) -> Self {
        let positions = [
            vertices[indices[0] as usize].position,
            vertices[indices[1] as usize].position,
            vertices[indices[2] as usize].position,
        ];
        let normals = [
            vertices[indices[0] as usize].normal,
            vertices[indices[1] as usize].normal,
            vertices[indices[2] as usize].normal,
        ];
        let uvs = [
            vertices[indices[0] as usize].uv,
            vertices[indices[1] as usize].uv,
            vertices[indices[2] as usize].uv,
        ];

        Self {
            indices,
            positions,
            normals,
            uvs,
        }
    }

    /// Calculate face tangent and bitangent using the MikkTSpace method
    fn calculate_face_tangent(&self) -> (Vec3, Vec3) {
        // Edge vectors
        let edge1 = self.positions[1] - self.positions[0];
        let edge2 = self.positions[2] - self.positions[0];

        // UV deltas
        let delta_uv1 = self.uvs[1] - self.uvs[0];
        let delta_uv2 = self.uvs[2] - self.uvs[0];

        // Calculate determinant
        let det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;

        if det.abs() < 1e-6 {
            // Degenerate UV mapping, use arbitrary tangent
            let face_normal = edge1.cross(edge2).normalize();
            let tangent = Vec3::new(1.0, 0.0, 0.0);
            let bitangent = face_normal.cross(tangent);
            return (tangent, bitangent);
        }

        let inv_det = 1.0 / det;

        let tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * inv_det;
        let bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * inv_det;

        (tangent, bitangent)
    }
}

/// Generate TBN data for an indexed mesh using MikkTSpace-compatible algorithm
pub fn generate_tbn(vertices: &[TbnVertex], indices: &[u32]) -> Vec<TbnData> {
    assert_eq!(indices.len() % 3, 0, "Indices must form triangles");

    let vertex_count = vertices.len();
    let mut vertex_tangents = vec![Vec3::ZERO; vertex_count];
    let mut vertex_bitangents = vec![Vec3::ZERO; vertex_count];
    let mut vertex_counts = vec![0u32; vertex_count];

    // Process each triangle
    for triangle_indices in indices.chunks_exact(3) {
        let tri_indices = [
            triangle_indices[0],
            triangle_indices[1],
            triangle_indices[2],
        ];
        let triangle = Triangle::new(vertices, tri_indices);

        let (face_tangent, face_bitangent) = triangle.calculate_face_tangent();

        // Calculate triangle area for weighting
        let edge1 = triangle.positions[1] - triangle.positions[0];
        let edge2 = triangle.positions[2] - triangle.positions[0];
        let area = edge1.cross(edge2).length() * 0.5;

        // Accumulate weighted tangents and bitangents
        for &vertex_idx in &tri_indices {
            let idx = vertex_idx as usize;
            vertex_tangents[idx] += face_tangent * area;
            vertex_bitangents[idx] += face_bitangent * area;
            vertex_counts[idx] += 1;
        }
    }

    // Generate final TBN data for each vertex
    let mut result = Vec::with_capacity(vertex_count);

    for (i, vertex) in vertices.iter().enumerate() {
        if vertex_counts[i] == 0 {
            // Isolated vertex, create arbitrary TBN
            let normal = vertex.normal.normalize();
            let tangent = Vec3::new(1.0, 0.0, 0.0);
            let bitangent = normal.cross(tangent).normalize();
            let orthogonal_tangent = bitangent.cross(normal).normalize();

            result.push(TbnData::new(orthogonal_tangent, bitangent, normal));
            continue;
        }

        let normal = vertex.normal.normalize();
        let tangent = vertex_tangents[i].normalize();

        // Gram-Schmidt orthogonalization: T = T - (T·N)N
        let orthogonal_tangent = (tangent - normal * tangent.dot(normal)).normalize();

        // Ensure we have a valid tangent
        let final_tangent = if orthogonal_tangent.length_squared() < 1e-6 {
            // Tangent parallel to normal, construct perpendicular
            let up = if normal.y.abs() > 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            up.cross(normal).normalize()
        } else {
            orthogonal_tangent
        };

        // Calculate bitangent
        let bitangent = normal.cross(final_tangent);

        result.push(TbnData::new(final_tangent, bitangent, normal));
    }

    result
}

/// Generate TBN for a simple plane (useful for testing)
pub fn generate_plane_tbn(width: u32, height: u32) -> (Vec<TbnVertex>, Vec<u32>, Vec<TbnData>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate grid vertices
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1) as f32;
            let v = y as f32 / (height - 1) as f32;

            vertices.push(TbnVertex {
                position: Vec3::new(u * 2.0 - 1.0, 0.0, v * 2.0 - 1.0),
                normal: Vec3::Y,
                uv: Vec2::new(u, v),
            });
        }
    }

    // Generate indices for triangles
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let base = y * width + x;

            // First triangle (top-left)
            indices.extend_from_slice(&[base, base + width, base + 1]);

            // Second triangle (bottom-right)
            indices.extend_from_slice(&[base + 1, base + width, base + width + 1]);
        }
    }

    let tbn_data = generate_tbn(&vertices, &indices);
    (vertices, indices, tbn_data)
}

/// Generate TBN for a unit cube (useful for testing)
pub fn generate_cube_tbn() -> (Vec<TbnVertex>, Vec<u32>, Vec<TbnData>) {
    let vertices = vec![
        // Front face
        TbnVertex {
            position: Vec3::new(-1.0, -1.0, 1.0),
            normal: Vec3::Z,
            uv: Vec2::new(0.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, -1.0, 1.0),
            normal: Vec3::Z,
            uv: Vec2::new(1.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, 1.0, 1.0),
            normal: Vec3::Z,
            uv: Vec2::new(1.0, 1.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, 1.0, 1.0),
            normal: Vec3::Z,
            uv: Vec2::new(0.0, 1.0),
        },
        // Back face
        TbnVertex {
            position: Vec3::new(1.0, -1.0, -1.0),
            normal: Vec3::NEG_Z,
            uv: Vec2::new(0.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, -1.0, -1.0),
            normal: Vec3::NEG_Z,
            uv: Vec2::new(1.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            normal: Vec3::NEG_Z,
            uv: Vec2::new(1.0, 1.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, 1.0, -1.0),
            normal: Vec3::NEG_Z,
            uv: Vec2::new(0.0, 1.0),
        },
        // Left face
        TbnVertex {
            position: Vec3::new(-1.0, -1.0, -1.0),
            normal: Vec3::NEG_X,
            uv: Vec2::new(0.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, -1.0, 1.0),
            normal: Vec3::NEG_X,
            uv: Vec2::new(1.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, 1.0, 1.0),
            normal: Vec3::NEG_X,
            uv: Vec2::new(1.0, 1.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            normal: Vec3::NEG_X,
            uv: Vec2::new(0.0, 1.0),
        },
        // Right face
        TbnVertex {
            position: Vec3::new(1.0, -1.0, 1.0),
            normal: Vec3::X,
            uv: Vec2::new(0.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, -1.0, -1.0),
            normal: Vec3::X,
            uv: Vec2::new(1.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, 1.0, -1.0),
            normal: Vec3::X,
            uv: Vec2::new(1.0, 1.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, 1.0, 1.0),
            normal: Vec3::X,
            uv: Vec2::new(0.0, 1.0),
        },
        // Bottom face
        TbnVertex {
            position: Vec3::new(-1.0, -1.0, -1.0),
            normal: Vec3::NEG_Y,
            uv: Vec2::new(0.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, -1.0, -1.0),
            normal: Vec3::NEG_Y,
            uv: Vec2::new(1.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, -1.0, 1.0),
            normal: Vec3::NEG_Y,
            uv: Vec2::new(1.0, 1.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, -1.0, 1.0),
            normal: Vec3::NEG_Y,
            uv: Vec2::new(0.0, 1.0),
        },
        // Top face
        TbnVertex {
            position: Vec3::new(-1.0, 1.0, 1.0),
            normal: Vec3::Y,
            uv: Vec2::new(0.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, 1.0, 1.0),
            normal: Vec3::Y,
            uv: Vec2::new(1.0, 0.0),
        },
        TbnVertex {
            position: Vec3::new(1.0, 1.0, -1.0),
            normal: Vec3::Y,
            uv: Vec2::new(1.0, 1.0),
        },
        TbnVertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            normal: Vec3::Y,
            uv: Vec2::new(0.0, 1.0),
        },
    ];

    let indices = vec![
        // Front
        0, 1, 2, 2, 3, 0, // Back
        4, 5, 6, 6, 7, 4, // Left
        8, 9, 10, 10, 11, 8, // Right
        12, 13, 14, 14, 15, 12, // Bottom
        16, 17, 18, 18, 19, 16, // Top
        20, 21, 22, 22, 23, 20,
    ];

    let tbn_data = generate_tbn(&vertices, &indices);
    (vertices, indices, tbn_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_tbn_generation() {
        let (vertices, indices, tbn_data) = generate_plane_tbn(3, 3);

        // Verify we have expected counts
        assert_eq!(vertices.len(), 9);
        assert_eq!(indices.len(), 24); // 4 quads * 2 triangles * 3 vertices
        assert_eq!(tbn_data.len(), 9);

        // Verify TBN validity
        for tbn in &tbn_data {
            assert!(tbn.is_valid(), "TBN data should be valid: {:?}", tbn);
        }

        // For a flat plane, all normals should be Y-up
        for tbn in &tbn_data {
            assert!(
                (tbn.normal - Vec3::Y).length() < 1e-3,
                "Normal should be Y-up"
            );
        }
    }

    #[test]
    fn test_cube_tbn_generation() {
        let (vertices, indices, tbn_data) = generate_cube_tbn();

        // Verify we have expected counts
        assert_eq!(vertices.len(), 24); // 6 faces * 4 vertices
        assert_eq!(indices.len(), 36); // 6 faces * 2 triangles * 3 vertices
        assert_eq!(tbn_data.len(), 24);

        // Verify TBN validity
        for tbn in &tbn_data {
            assert!(tbn.is_valid(), "TBN data should be valid: {:?}", tbn);
        }
    }

    #[test]
    fn test_tbn_orthogonality() {
        let (_, _, tbn_data) = generate_cube_tbn();

        for tbn in &tbn_data {
            // Check unit length
            assert!((tbn.tangent.length() - 1.0).abs() < 1e-3);
            assert!((tbn.bitangent.length() - 1.0).abs() < 1e-3);
            assert!((tbn.normal.length() - 1.0).abs() < 1e-3);

            // Check orthogonality
            assert!(tbn.tangent.dot(tbn.normal).abs() < 1e-3);
            assert!(tbn.bitangent.dot(tbn.normal).abs() < 1e-3);
        }
    }
}
