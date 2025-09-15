//! Extended vertex structures with TBN attributes
//!
//! Provides vertex layouts that include tangent and bitangent data for normal mapping

use bytemuck::{Pod, Zeroable};

/// Vertex with full TBN (Tangent, Bitangent, Normal) attributes for PBR rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct TbnVertex {
    /// World space position [x, y, z]
    pub position: [f32; 3],
    /// Texture coordinates [u, v]  
    pub uv: [f32; 2],
    /// Surface normal [x, y, z]
    pub normal: [f32; 3],
    /// Tangent vector [x, y, z]
    pub tangent: [f32; 3],
    /// Bitangent vector [x, y, z]
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

    /// Get the vertex buffer layout for wgpu
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TbnVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // UV
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Tangent
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Bitangent
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Compact vertex with handedness-encoded tangent for memory efficiency
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct CompactTbnVertex {
    /// World space position [x, y, z]
    pub position: [f32; 3],
    /// Texture coordinates [u, v]
    pub uv: [f32; 2],
    /// Surface normal [x, y, z]
    pub normal: [f32; 3],
    /// Tangent vector with handedness [x, y, z, handedness]
    pub tangent: [f32; 4],
}

impl CompactTbnVertex {
    pub fn new(
        position: [f32; 3],
        uv: [f32; 2],
        normal: [f32; 3],
        tangent: [f32; 3],
        handedness: f32,
    ) -> Self {
        Self {
            position,
            uv,
            normal,
            tangent: [tangent[0], tangent[1], tangent[2], handedness],
        }
    }

    /// Get the vertex buffer layout for wgpu
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CompactTbnVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // UV
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Tangent (includes handedness in w component)
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Convert TBN mesh data to GPU vertex buffers
pub fn create_tbn_vertices_from_mesh(
    vertices: &[crate::mesh::tbn::TbnVertex],
    tbn_data: &[crate::mesh::tbn::TbnData],
) -> Vec<TbnVertex> {
    assert_eq!(vertices.len(), tbn_data.len());

    vertices
        .iter()
        .zip(tbn_data.iter())
        .map(|(vertex, tbn)| {
            TbnVertex::new(
                vertex.position.to_array(),
                vertex.uv.to_array(),
                tbn.normal.to_array(),
                tbn.tangent.to_array(),
                tbn.bitangent.to_array(),
            )
        })
        .collect()
}

/// Convert TBN mesh data to compact GPU vertex buffers
pub fn create_compact_tbn_vertices_from_mesh(
    vertices: &[crate::mesh::tbn::TbnVertex],
    tbn_data: &[crate::mesh::tbn::TbnData],
) -> Vec<CompactTbnVertex> {
    assert_eq!(vertices.len(), tbn_data.len());

    vertices
        .iter()
        .zip(tbn_data.iter())
        .map(|(vertex, tbn)| {
            CompactTbnVertex::new(
                vertex.position.to_array(),
                vertex.uv.to_array(),
                tbn.normal.to_array(),
                tbn.tangent.to_array(),
                tbn.handedness,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tbn_vertex_layout() {
        let layout = TbnVertex::buffer_layout();
        assert_eq!(layout.array_stride, std::mem::size_of::<TbnVertex>() as u64);
        assert_eq!(layout.attributes.len(), 5); // position, uv, normal, tangent, bitangent
    }

    #[test]
    fn test_compact_tbn_vertex_layout() {
        let layout = CompactTbnVertex::buffer_layout();
        assert_eq!(
            layout.array_stride,
            std::mem::size_of::<CompactTbnVertex>() as u64
        );
        assert_eq!(layout.attributes.len(), 4); // position, uv, normal, tangent+handedness

        // Compact should be smaller
        assert!(std::mem::size_of::<CompactTbnVertex>() < std::mem::size_of::<TbnVertex>());
    }
}
