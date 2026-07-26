//! P2.1/M5: Clipmap vertex format with geo-morphing support.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use wgpu::vertex_attr_array;

/// Clipmap vertex with geo-morph data for seamless LOD transitions.
///
/// Layout: 36 bytes total, compatible with wgpu vertex buffers.
/// - position: camera-relative XYZ (flat mode uses Z=0)
/// - uv: height-mosaic texture coordinates [0,1]
/// - morph_data: [morph_weight, signed ring_index] for geo-morphing; a
///   negative encoded ring marks globe-space normals without another attribute
/// - normal_oct: octahedral-encoded geodetic up direction
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClipmapVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub morph_data: [f32; 2],
    pub normal_oct: [f32; 2],
}

impl ClipmapVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x2,
        3 => Float32x2
    ];

    pub fn new(x: f32, z: f32, u: f32, v: f32, morph_weight: f32, ring_index: u32) -> Self {
        Self::with_position(
            Vec3::new(x, z, 0.0),
            Vec3::Z,
            Vec2::new(u, v),
            morph_weight,
            ring_index,
        )
    }

    pub fn with_position(
        position: Vec3,
        geodetic_up: Vec3,
        uv: Vec2,
        morph_weight: f32,
        ring_index: u32,
    ) -> Self {
        Self {
            position: position.to_array(),
            uv: uv.to_array(),
            morph_data: [morph_weight, ring_index as f32],
            normal_oct: encode_octahedral(geodetic_up),
        }
    }

    pub fn center(x: f32, z: f32, u: f32, v: f32) -> Self {
        Self::new(x, z, u, v, 0.0, 0)
    }

    pub fn skirt(x: f32, z: f32, u: f32, v: f32, ring_index: u32) -> Self {
        let mut skirt = Self::new(x, z, u, v, 0.0, ring_index);
        skirt.morph_data[0] = -1.0;
        skirt
    }

    pub fn skirt_from(source: &Self, ring_index: u32) -> Self {
        let mut skirt = *source;
        skirt.morph_data = [-1.0, ring_index as f32];
        skirt
    }

    pub fn set_globe_position(&mut self, position: Vec3, geodetic_up: Vec3) {
        self.position = position.to_array();
        self.normal_oct = encode_octahedral(geodetic_up);
        if self.morph_data[1] >= 0.0 {
            self.morph_data[1] = -self.morph_data[1] - 1.0;
        }
    }

    pub fn is_skirt(&self) -> bool {
        self.morph_data[0] < 0.0
    }

    pub fn morph_weight(&self) -> f32 {
        self.morph_data[0].max(0.0)
    }

    pub fn ring_index(&self) -> u32 {
        if self.is_globe() {
            (-self.morph_data[1] - 1.0) as u32
        } else {
            self.morph_data[1] as u32
        }
    }

    pub fn is_globe(&self) -> bool {
        self.morph_data[1] < 0.0
    }

    pub fn geodetic_up(&self) -> Vec3 {
        decode_octahedral(Vec2::from(self.normal_oct))
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

fn sign_not_zero(value: f32) -> f32 {
    if value >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn encode_octahedral(normal: Vec3) -> [f32; 2] {
    if !normal.is_finite() || normal.length_squared() == 0.0 {
        return [0.0, 0.0];
    }
    let normal = normal.normalize();
    let projected = normal / (normal.x.abs() + normal.y.abs() + normal.z.abs());
    let encoded = if projected.z < 0.0 {
        Vec2::new(
            (1.0 - projected.y.abs()) * sign_not_zero(projected.x),
            (1.0 - projected.x.abs()) * sign_not_zero(projected.y),
        )
    } else {
        projected.truncate()
    };
    encoded.to_array()
}

fn decode_octahedral(encoded: Vec2) -> Vec3 {
    let mut normal = Vec3::new(
        encoded.x,
        encoded.y,
        1.0 - encoded.x.abs() - encoded.y.abs(),
    );
    if normal.z < 0.0 {
        let old_x = normal.x;
        normal.x = (1.0 - normal.y.abs()) * sign_not_zero(old_x);
        normal.y = (1.0 - old_x.abs()) * sign_not_zero(normal.y);
    }
    normal.normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_size() {
        assert_eq!(std::mem::size_of::<ClipmapVertex>(), 36);
    }

    #[test]
    fn test_center_vertex() {
        let vertex = ClipmapVertex::center(10.0, 20.0, 0.5, 0.5);
        assert_eq!(vertex.position, [10.0, 20.0, 0.0]);
        assert_eq!(vertex.geodetic_up(), Vec3::Z);
        assert_eq!(vertex.uv, [0.5, 0.5]);
        assert_eq!(vertex.morph_weight(), 0.0);
        assert_eq!(vertex.ring_index(), 0);
        assert!(!vertex.is_skirt());
    }

    #[test]
    fn test_ring_vertex() {
        let vertex = ClipmapVertex::new(100.0, 200.0, 0.8, 0.2, 0.5, 2);
        assert_eq!(vertex.position, [100.0, 200.0, 0.0]);
        assert_eq!(vertex.morph_weight(), 0.5);
        assert_eq!(vertex.ring_index(), 2);
        assert!(!vertex.is_skirt());
    }

    #[test]
    fn test_skirt_vertex() {
        let vertex = ClipmapVertex::skirt(50.0, 50.0, 0.25, 0.75, 1);
        assert!(vertex.is_skirt());
        assert_eq!(vertex.morph_weight(), 0.0);
        assert_eq!(vertex.ring_index(), 1);
    }

    #[test]
    fn test_vertex_layout() {
        let layout = ClipmapVertex::desc();
        assert_eq!(layout.array_stride, 36);
        assert_eq!(layout.attributes.len(), 4);
    }

    #[test]
    fn globe_flag_preserves_ring_index() {
        let mut vertex = ClipmapVertex::new(1.0, 2.0, 0.5, 0.5, 0.25, 3);
        vertex.set_globe_position(Vec3::new(1.0, 2.0, 3.0), Vec3::Y);
        assert!(vertex.is_globe());
        assert_eq!(vertex.ring_index(), 3);
    }

    #[test]
    fn octahedral_normal_round_trip_covers_both_hemispheres() {
        for normal in [
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(0.3, -0.4, 0.866_025_4),
            Vec3::new(-0.2, 0.7, -0.685_565_5),
        ] {
            let decoded = decode_octahedral(Vec2::from(encode_octahedral(normal)));
            assert!((decoded - normal.normalize()).length() < 1.0e-6);
        }
    }
}
