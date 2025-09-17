// src/path_tracing/mesh.rs
// GPU mesh upload and buffer management for triangle mesh path tracing (Task A3).
// This file handles uploading CPU BVH and mesh data to GPU buffers for path tracing integration.
// RELEVANT FILES:src/accel/cpu_bvh.rs,src/shaders/pt_intersect_mesh.wgsl,python/forge3d/mesh.py

use crate::accel::cpu_bvh::{MeshCPU, BvhCPU, BvhNode, Aabb};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::{Device, Queue, Buffer, BufferUsages};
use wgpu::util::DeviceExt;

/// GPU-compatible vertex layout (matches WGSL Vertex struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub _pad: f32,
}

impl From<[f32; 3]> for GpuVertex {
    fn from(position: [f32; 3]) -> Self {
        Self {
            position,
            _pad: 0.0,
        }
    }
}

/// GPU mesh handle containing all necessary buffers for path tracing
#[derive(Debug)]
pub struct GpuMesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub bvh_buffer: Buffer,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub node_count: u32,
    pub world_aabb: Aabb,
    pub build_stats: crate::accel::cpu_bvh::BuildStats,
}

impl GpuMesh {
    /// Get the size in bytes of all GPU buffers
    pub fn gpu_memory_usage(&self) -> u64 {
        self.vertex_buffer.size() + self.index_buffer.size() + self.bvh_buffer.size()
    }

    /// Get triangle density (triangles per BVH node)
    pub fn triangle_density(&self) -> f32 {
        if self.node_count > 0 {
            self.triangle_count as f32 / self.node_count as f32
        } else {
            0.0
        }
    }
}

/// Upload mesh and BVH data to GPU buffers
/// This is the main integration point for getting CPU mesh data into GPU format
pub fn upload_mesh_and_bvh(
    device: &Device,
    _queue: &Queue,
    mesh: &MeshCPU,
    bvh: &BvhCPU,
) -> Result<GpuMesh> {
    // Validate input data
    if mesh.vertices.is_empty() {
        anyhow::bail!("Cannot upload empty mesh");
    }
    if mesh.indices.is_empty() {
        anyhow::bail!("Cannot upload mesh with no triangles");
    }
    if bvh.nodes.is_empty() {
        anyhow::bail!("Cannot upload empty BVH");
    }

    // Convert vertices to GPU format
    let gpu_vertices: Vec<GpuVertex> = mesh.vertices.iter()
        .map(|&pos| GpuVertex::from(pos))
        .collect();

    // Flatten triangle indices to u32 array
    let flat_indices: Vec<u32> = mesh.indices.iter()
        .flat_map(|tri| tri.iter().copied())
        .collect();

    // Create vertex buffer
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh Vertex Buffer"),
        contents: bytemuck::cast_slice(&gpu_vertices),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Create index buffer
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh Index Buffer"),
        contents: bytemuck::cast_slice(&flat_indices),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Create BVH buffer
    let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mesh BVH Buffer"),
        contents: bytemuck::cast_slice(&bvh.nodes),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    log::info!("Uploaded mesh to GPU: {} vertices, {} triangles, {} BVH nodes",
              gpu_vertices.len(), mesh.indices.len(), bvh.nodes.len());
    log::info!("GPU memory usage: vertices={} bytes, indices={} bytes, BVH={} bytes",
              vertex_buffer.size(), index_buffer.size(), bvh_buffer.size());

    Ok(GpuMesh {
        vertex_buffer,
        index_buffer,
        bvh_buffer,
        vertex_count: gpu_vertices.len() as u32,
        triangle_count: mesh.triangle_count(),
        node_count: bvh.node_count(),
        world_aabb: bvh.world_aabb,
        build_stats: bvh.build_stats.clone(),
    })
}

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

/// Reusable mesh builder for creating common test meshes
pub struct MeshBuilder;

impl MeshBuilder {
    /// Create a simple triangle mesh
    pub fn triangle() -> MeshCPU {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2]];
        MeshCPU::new(vertices, indices)
    }

    /// Create a unit cube mesh (12 triangles)
    pub fn cube() -> MeshCPU {
        let vertices = vec![
            // Front face
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            // Back face
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let indices = vec![
            // Front face
            [0, 1, 2], [0, 2, 3],
            // Right face
            [1, 5, 6], [1, 6, 2],
            // Back face
            [5, 4, 7], [5, 7, 6],
            // Left face
            [4, 0, 3], [4, 3, 7],
            // Top face
            [3, 2, 6], [3, 6, 7],
            // Bottom face
            [4, 5, 1], [4, 1, 0],
        ];
        MeshCPU::new(vertices, indices)
    }

    /// Create a quad mesh (2 triangles)
    pub fn quad() -> MeshCPU {
        let vertices = vec![
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 1.0,  1.0, 0.0],
            [-1.0,  1.0, 0.0],
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
                    tri_idx, corner_idx, vertex_idx, vertex_count - 1
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

/// Mesh statistics for debugging and optimization
#[derive(Debug, Clone)]
pub struct MeshStats {
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub world_aabb: Aabb,
    pub average_triangle_area: f32,
    pub memory_usage_bytes: u64,
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
            let area = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt() * 0.5;
            total_area += area;
        }
    }

    let avg_area = if mesh.triangle_count() > 0 {
        total_area / mesh.triangle_count() as f32
    } else {
        0.0
    };

    let memory_usage = (mesh.vertices.len() * std::mem::size_of::<[f32; 3]>() +
                       mesh.indices.len() * std::mem::size_of::<[u32; 3]>()) as u64;

    MeshStats {
        vertex_count: mesh.vertex_count(),
        triangle_count: mesh.triangle_count(),
        world_aabb,
        average_triangle_area: avg_area,
        memory_usage_bytes: memory_usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_vertex_layout() {
        // Verify GPU vertex layout
        assert_eq!(std::mem::size_of::<GpuVertex>(), 16); // 4 floats
        assert_eq!(std::mem::align_of::<GpuVertex>(), 4);
    }

    #[test]
    fn test_mesh_builder_triangle() {
        let mesh = MeshBuilder::triangle();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        validate_mesh(&mesh).unwrap();
    }

    #[test]
    fn test_mesh_builder_cube() {
        let mesh = MeshBuilder::cube();
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.triangle_count(), 12);
        validate_mesh(&mesh).unwrap();
    }

    #[test]
    fn test_mesh_validation() {
        // Valid mesh
        let valid_mesh = MeshBuilder::triangle();
        assert!(validate_mesh(&valid_mesh).is_ok());

        // Empty mesh
        let empty_mesh = MeshCPU::new(vec![], vec![]);
        assert!(validate_mesh(&empty_mesh).is_err());

        // Invalid indices
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let indices = vec![[0, 1, 2]]; // Index 2 is out of bounds
        let invalid_mesh = MeshCPU::new(vertices, indices);
        assert!(validate_mesh(&invalid_mesh).is_err());
    }

    #[test]
    fn test_mesh_stats() {
        let mesh = MeshBuilder::cube();
        let stats = compute_mesh_stats(&mesh);

        assert_eq!(stats.vertex_count, 8);
        assert_eq!(stats.triangle_count, 12);
        assert!(stats.world_aabb.is_valid());
        assert!(stats.average_triangle_area > 0.0);
        assert!(stats.memory_usage_bytes > 0);
    }
}