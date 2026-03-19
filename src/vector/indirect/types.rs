use crate::vector::batch::{PrimitiveType, AABB};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// Indirect draw command for GPU execution
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndirectDrawCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Indexed indirect draw command
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndirectDrawIndexedCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

/// Object instance data for culling
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CullableInstance {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub transform: [[f32; 4]; 4],
    pub primitive_type: u32,
    pub draw_command_index: u32,
}

/// GPU culling uniforms
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct CullingUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) frustum_plane_0: [f32; 4],
    pub(super) frustum_plane_1: [f32; 4],
    pub(super) frustum_plane_2: [f32; 4],
    pub(super) frustum_plane_3: [f32; 4],
    pub(super) frustum_plane_4: [f32; 4],
    pub(super) frustum_plane_5: [f32; 4],
    pub(super) camera_position: [f32; 3],
    pub(super) _pad0: f32,
    pub(super) cull_distance: f32,
    pub(super) enable_frustum_cull: u32,
    pub(super) enable_distance_cull: u32,
    pub(super) enable_occlusion_cull: u32,
}

/// Culling statistics
#[derive(Debug, Clone, Default)]
pub struct CullingStats {
    pub total_objects: u32,
    pub visible_objects: u32,
    pub frustum_culled: u32,
    pub distance_culled: u32,
    pub occlusion_culled: u32,
    pub gpu_time_ms: f32,
}

pub fn create_cullable_instance(
    aabb: &AABB,
    transform: Mat4,
    primitive_type: PrimitiveType,
    draw_command_index: u32,
) -> CullableInstance {
    CullableInstance {
        aabb_min: [aabb.min.x, aabb.min.y, 0.0],
        aabb_max: [aabb.max.x, aabb.max.y, 1.0],
        transform: transform.to_cols_array_2d(),
        primitive_type: primitive_type as u32,
        draw_command_index,
    }
}
