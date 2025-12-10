use super::types::{GpuMesh, GpuVertex};
use crate::accel::cpu_bvh::BvhNode;
use wgpu::{Buffer, BufferUsages, Device};

/// Create bind group for mesh data (Group 1 in pt_kernel.wgsl)
/// This binds the mesh buffers for use in the path tracing compute shader
pub fn create_mesh_bind_group(
    device: &Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    gpu_mesh: &GpuMesh,
    sphere_buffer: &Buffer, // Existing sphere buffer from Group 1 binding 0
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh Data Bind Group"),
        layout: bind_group_layout,
        entries: &[
            // Binding 0: Spheres (existing)
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sphere_buffer.as_entire_binding(),
            },
            // Binding 1: Mesh vertices
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_mesh.vertex_buffer.as_entire_binding(),
            },
            // Binding 2: Mesh indices
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu_mesh.index_buffer.as_entire_binding(),
            },
            // Binding 3: BVH nodes
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gpu_mesh.bvh_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Create bind group layout for mesh data (Group 1)
/// Defines the layout expected by pt_kernel.wgsl
pub fn create_mesh_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mesh Data Bind Group Layout"),
        entries: &[
            // Binding 0: Spheres (readonly storage buffer)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Mesh vertices (readonly storage buffer)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Mesh indices (readonly storage buffer)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 3: BVH nodes (readonly storage buffer)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Helper to create empty buffers when no mesh is provided
/// This allows the path tracer to work with or without mesh data
pub fn create_empty_mesh_buffers(device: &Device) -> (Buffer, Buffer, Buffer) {
    // Create minimal empty buffers to satisfy bind group requirements
    let empty_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Empty Vertex Buffer"),
        size: std::mem::size_of::<GpuVertex>() as u64, // One vertex minimum
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let empty_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Empty Index Buffer"),
        size: std::mem::size_of::<u32>() as u64, // One index minimum
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let empty_bvh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Empty BVH Buffer"),
        size: std::mem::size_of::<BvhNode>() as u64, // One node minimum
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    (empty_vertex_buffer, empty_index_buffer, empty_bvh_buffer)
}
