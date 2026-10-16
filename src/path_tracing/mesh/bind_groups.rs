use super::types::GpuVertex;
use crate::accel::cpu_bvh::BvhNode;
use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use wgpu::{BufferUsages, Device};

/// Helper to create empty buffers when no mesh is provided
/// This allows the path tracer to work with or without mesh data
pub fn create_empty_mesh_buffers(
    device: &Device,
) -> Result<(TrackedBuffer, TrackedBuffer, TrackedBuffer), RenderError> {
    // Create minimal empty buffers to satisfy bind group requirements
    let empty_vertex_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("Empty Vertex Buffer"),
            size: std::mem::size_of::<GpuVertex>() as u64, // One vertex minimum
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        },
    )?;

    let empty_index_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("Empty Index Buffer"),
            size: std::mem::size_of::<u32>() as u64, // One index minimum
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        },
    )?;

    let empty_bvh_buffer = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("Empty BVH Buffer"),
            size: std::mem::size_of::<BvhNode>() as u64, // One node minimum
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        },
    )?;

    Ok((empty_vertex_buffer, empty_index_buffer, empty_bvh_buffer))
}
