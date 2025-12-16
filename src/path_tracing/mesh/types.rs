use crate::accel::cpu_bvh::{Aabb, BuildStats};
use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

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

/// BLAS descriptor matching WGSL `BlasDesc` layout
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlasDesc {
    pub node_offset: u32,
    pub node_count: u32,
    pub tri_offset: u32,
    pub tri_count: u32,
    pub vtx_offset: u32,
    pub vtx_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Mesh atlas buffers (concatenated vertices/indices/BVH) + descriptor table
#[derive(Debug)]
pub struct MeshAtlas {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub bvh_buffer: Buffer,
    pub descs_buffer: Buffer,
    pub desc_count: u32,
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
    pub build_stats: BuildStats,
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

/// Mesh statistics for debugging and optimization
#[derive(Debug, Clone)]
pub struct MeshStats {
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub world_aabb: Aabb,
    pub average_triangle_area: f32,
    pub memory_usage_bytes: u64,
}
