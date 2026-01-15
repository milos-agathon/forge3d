//! P2.1/M5: Clipmap vertex format with geo-morphing support.

use bytemuck::{Pod, Zeroable};

/// Clipmap vertex with geo-morph data for seamless LOD transitions.
///
/// Layout: 24 bytes total, compatible with wgpu vertex buffers.
/// - position: XZ world-space coordinates (Y computed from heightmap in shader)
/// - uv: Heightmap texture coordinates [0,1]
/// - morph_data: [morph_weight, ring_index] for geo-morphing
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClipmapVertex {
    /// World-space XZ position (Y comes from heightmap).
    pub position: [f32; 2],
    /// Heightmap UV coordinates [0,1].
    pub uv: [f32; 2],
    /// Geo-morph data: x=morph_weight [0,1], y=ring_index (as float).
    pub morph_data: [f32; 2],
}

impl ClipmapVertex {
    /// Create a new clipmap vertex.
    pub fn new(x: f32, z: f32, u: f32, v: f32, morph_weight: f32, ring_index: u32) -> Self {
        Self {
            position: [x, z],
            uv: [u, v],
            morph_data: [morph_weight, ring_index as f32],
        }
    }

    /// Create a center block vertex (no morphing, ring_index = 0).
    pub fn center(x: f32, z: f32, u: f32, v: f32) -> Self {
        Self::new(x, z, u, v, 0.0, 0)
    }

    /// Create a skirt vertex (same as source but marked for depth offset in shader).
    /// Skirt vertices use negative morph_weight as a flag.
    pub fn skirt(x: f32, z: f32, u: f32, v: f32, ring_index: u32) -> Self {
        Self {
            position: [x, z],
            uv: [u, v],
            morph_data: [-1.0, ring_index as f32], // Negative = skirt flag
        }
    }

    /// Check if this is a skirt vertex.
    pub fn is_skirt(&self) -> bool {
        self.morph_data[0] < 0.0
    }

    /// Get the morph weight (0.0 for skirt vertices).
    pub fn morph_weight(&self) -> f32 {
        self.morph_data[0].max(0.0)
    }

    /// Get the ring index.
    pub fn ring_index(&self) -> u32 {
        self.morph_data[1] as u32
    }

    /// wgpu vertex buffer layout descriptor.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec2<f32>
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // uv: vec2<f32>
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // morph_data: vec2<f32>
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_size() {
        assert_eq!(std::mem::size_of::<ClipmapVertex>(), 24);
    }

    #[test]
    fn test_center_vertex() {
        let v = ClipmapVertex::center(10.0, 20.0, 0.5, 0.5);
        assert_eq!(v.position, [10.0, 20.0]);
        assert_eq!(v.uv, [0.5, 0.5]);
        assert_eq!(v.morph_weight(), 0.0);
        assert_eq!(v.ring_index(), 0);
        assert!(!v.is_skirt());
    }

    #[test]
    fn test_ring_vertex() {
        let v = ClipmapVertex::new(100.0, 200.0, 0.8, 0.2, 0.5, 2);
        assert_eq!(v.position, [100.0, 200.0]);
        assert_eq!(v.morph_weight(), 0.5);
        assert_eq!(v.ring_index(), 2);
        assert!(!v.is_skirt());
    }

    #[test]
    fn test_skirt_vertex() {
        let v = ClipmapVertex::skirt(50.0, 50.0, 0.25, 0.75, 1);
        assert!(v.is_skirt());
        assert_eq!(v.morph_weight(), 0.0); // Clamped to 0
        assert_eq!(v.ring_index(), 1);
    }

    #[test]
    fn test_vertex_layout() {
        let layout = ClipmapVertex::desc();
        assert_eq!(layout.array_stride, 24);
        assert_eq!(layout.attributes.len(), 3);
    }
}
