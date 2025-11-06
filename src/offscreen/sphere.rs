// src/offscreen/sphere.rs
// P7-02: UV-sphere mesh generator for BRDF tile rendering
// Generates a parametric UV-sphere with normals, tangents, and texture coordinates
// RELEVANT FILES: src/mesh/vertex.rs, src/offscreen/brdf_tile.rs

use bytemuck::{Pod, Zeroable};

/// Vertex with full TBN (Tangent, Bitangent, Normal) attributes for PBR rendering
/// Matches the layout in src/mesh/vertex.rs::TbnVertex
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct TbnVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl TbnVertex {
    pub fn new(
        position: [f32; 3],
        uv: [f32; 2],
        normal: [f32; 3],
        tangent: [f32; 3],
        bitangent: [f32; 3],
    ) -> Self {
        Self {
            position,
            uv,
            normal,
            tangent,
            bitangent,
        }
    }
}

/// Generate a UV-sphere mesh with TBN (tangent-bitangent-normal) attributes.
///
/// # Arguments
/// * `sectors` - Number of longitudinal divisions (e.g., 64)
/// * `stacks` - Number of latitudinal divisions (e.g., 32)
/// * `radius` - Sphere radius (typically 1.0)
///
/// # Returns
/// Tuple of (vertices, indices) where:
/// - `vertices`: Vec of TbnVertex with position, UV, normal, tangent, bitangent
/// - `indices`: Vec of u32 indices in CCW winding (counter-clockwise front faces)
///
/// # UV Mapping
/// - U maps to longitude [0, 1] wrapping around
/// - V maps to latitude [0, 1] from south pole to north pole
///
/// # Coordinate System
/// - Y-up: north pole at (0, 1, 0), south pole at (0, -1, 0)
/// - X-right, Z-forward
/// - CCW winding for Y-up right-handed system
pub fn generate_uv_sphere(sectors: u32, stacks: u32, radius: f32) -> (Vec<TbnVertex>, Vec<u32>) {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);
    
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let sector_step = 2.0 * std::f32::consts::PI / sectors as f32;
    let stack_step = std::f32::consts::PI / stacks as f32;

    // Generate vertices
    for i in 0..=stacks {
        let stack_angle = std::f32::consts::PI / 2.0 - i as f32 * stack_step; // From π/2 to -π/2
        let xz = radius * stack_angle.cos(); // Horizontal radius at this latitude
        let y = radius * stack_angle.sin();   // Y (height)

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step; // From 0 to 2π

            // Position (Y-up, X-right, Z-forward)
            let x = xz * sector_angle.cos();
            let z = xz * sector_angle.sin();
            let position = [x, y, z];

            // Normal (normalized position for unit sphere)
            let normal = [x / radius, y / radius, z / radius];

            // UV coordinates
            let u = j as f32 / sectors as f32;
            let v = i as f32 / stacks as f32;
            let uv = [u, v];

            // Tangent: derivative with respect to longitude (azimuthal direction)
            // d/dθ of (r*cos(φ)*cos(θ), r*sin(φ), r*cos(φ)*sin(θ))
            // = (-r*cos(φ)*sin(θ), 0, r*cos(φ)*cos(θ))
            let tx = -xz * sector_angle.sin();
            let ty = 0.0;
            let tz = xz * sector_angle.cos();
            let tangent = normalize([tx, ty, tz]);

            // Bitangent: cross(normal, tangent) for right-handed system
            let bitangent = normalize(cross(normal, tangent));

            vertices.push(TbnVertex::new(position, uv, normal, tangent, bitangent));
        }
    }

    // Generate indices with CCW winding for Y-up right-handed system
    // For each quad (viewed from outside), split into two triangles with CCW winding
    // Quad layout:
    //   k1+1 ---- k2+1
    //    |          |
    //    |          |
    //   k1   ---- k2
    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;

        for j in 0..sectors {
            // Triangle 1: (k1, k1+1, k2) - CCW from outside
            if i != 0 {
                indices.push(k1 + j);
                indices.push(k1 + j + 1);
                indices.push(k2 + j);
            }

            // Triangle 2: (k1+1, k2+1, k2) - CCW from outside
            if i != stacks - 1 {
                indices.push(k1 + j + 1);
                indices.push(k2 + j + 1);
                indices.push(k2 + j);
            }
        }
    }

    (vertices, indices)
}

/// Compute cross product of two 3D vectors
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3D vector
#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_sphere_generation() {
        let (vertices, indices) = generate_uv_sphere(64, 32, 1.0);
        
        // Verify vertex count: (stacks+1) * (sectors+1)
        assert_eq!(vertices.len(), 33 * 65);
        
        // Verify we have indices (triangles)
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0, "indices must be divisible by 3");
        
        // Verify all normals are roughly unit length
        for v in &vertices {
            let len = (v.normal[0] * v.normal[0] 
                     + v.normal[1] * v.normal[1] 
                     + v.normal[2] * v.normal[2]).sqrt();
            assert!((len - 1.0).abs() < 0.01, "normal must be unit length, got {}", len);
        }
        
        // Verify UV coordinates are in [0, 1]
        for v in &vertices {
            assert!(v.uv[0] >= 0.0 && v.uv[0] <= 1.0, "U must be in [0,1]");
            assert!(v.uv[1] >= 0.0 && v.uv[1] <= 1.0, "V must be in [0,1]");
        }
        
        // Verify indices are valid
        let max_index = vertices.len() as u32;
        for &idx in &indices {
            assert!(idx < max_index, "index {} out of bounds (max {})", idx, max_index);
        }
    }

    #[test]
    fn test_sphere_topology() {
        let (vertices, indices) = generate_uv_sphere(8, 4, 1.0);
        
        // Basic sanity checks for closed mesh topology
        let v = vertices.len();
        let f = indices.len() / 3;
        
        // UV-sphere with s sectors and t stacks has:
        // V = (s+1)*(t+1) vertices
        // F = 2*s*t triangles (after handling poles)
        assert_eq!(v, 9 * 5, "vertex count should be (sectors+1)*(stacks+1)");
        
        // Triangle count should be reasonable for 8x4 sphere
        assert!(f > 0 && f <= 2 * 8 * 4, "face count should be reasonable");
        
        // All indices should be valid
        for &idx in &indices {
            assert!((idx as usize) < v, "index out of bounds");
        }
    }

    #[test]
    fn test_sphere_ccw_winding() {
        let (vertices, indices) = generate_uv_sphere(8, 4, 1.0);
        
        // Check first non-degenerate triangle has CCW winding
        // For Y-up, CCW means when viewing from outside, vertices go counter-clockwise
        if indices.len() >= 3 {
            let i0 = indices[0] as usize;
            let i1 = indices[1] as usize;
            let i2 = indices[2] as usize;
            
            let v0 = vertices[i0].position;
            let v1 = vertices[i1].position;
            let v2 = vertices[i2].position;
            
            // Compute face normal via cross product
            let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let face_normal = cross(edge1, edge2);
            
            // Face center (approximate)
            let center = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ];
            
            // For CCW winding on a sphere, face normal should point outward (same direction as center)
            let dot = face_normal[0] * center[0] 
                    + face_normal[1] * center[1] 
                    + face_normal[2] * center[2];
            
            assert!(dot > 0.0, "CCW winding should have outward-pointing normals");
        }
    }
}
