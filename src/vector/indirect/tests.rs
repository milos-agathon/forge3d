use super::*;
use crate::vector::batch::{Frustum, PrimitiveType, AABB};
use glam::{Mat4, Vec2, Vec3};

#[test]
fn test_indirect_draw_command_size() {
    assert_eq!(std::mem::size_of::<IndirectDrawCommand>(), 16);
    assert_eq!(std::mem::size_of::<IndirectDrawIndexedCommand>(), 20);
}

#[test]
fn test_cullable_instance_creation() {
    let aabb = AABB {
        min: Vec2::new(-1.0, -1.0),
        max: Vec2::new(1.0, 1.0),
    };
    let transform = Mat4::IDENTITY;

    let instance = create_cullable_instance(&aabb, transform, PrimitiveType::Triangle, 0);

    assert_eq!(instance.aabb_min, [-1.0, -1.0, 0.0]);
    assert_eq!(instance.aabb_max, [1.0, 1.0, 1.0]);
    assert_eq!(instance.primitive_type, PrimitiveType::Triangle as u32);
}

#[test]
fn test_vertex_count_for_types() {
    let Some(device) = crate::core::gpu::create_device_for_test() else {
        return;
    };
    let renderer = IndirectRenderer::new(&device).unwrap();

    assert_eq!(renderer.get_vertex_count_for_type(0), 3);
    assert_eq!(renderer.get_vertex_count_for_type(1), 4);
    assert_eq!(renderer.get_vertex_count_for_type(2), 1);
    assert_eq!(renderer.get_vertex_count_for_type(3), 2);
}

#[test]
fn test_cpu_culling_distance() {
    let Some(device) = crate::core::gpu::create_device_for_test() else {
        return;
    };
    let renderer = IndirectRenderer::new(&device).unwrap();

    let instance = CullableInstance {
        aabb_min: [-1.0, -1.0, -1.0],
        aabb_max: [1.0, 1.0, 1.0],
        transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -100.0)).to_cols_array_2d(),
        primitive_type: PrimitiveType::Triangle as u32,
        draw_command_index: 0,
    };

    let frustum = Frustum::from_view_proj(&Mat4::IDENTITY);
    let camera_pos = Vec3::ZERO;
    let max_distance = 50.0;

    let visible = renderer.cull_cpu(&[instance], &frustum, camera_pos, max_distance);
    assert_eq!(visible.len(), 0);
}
