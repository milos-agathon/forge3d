use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PolygonMeta {
    pub base_vertex_offset: u32,
    pub base_vertex_count: u32,
    pub base_index_offset: u32,
    pub base_index_count: u32,
    pub ring_offset: u32,
    pub ring_count: u32,
    pub output_vertex_offset: u32,
    pub output_index_offset: u32,
    pub bbox_min: [f32; 2],
    pub bbox_scale: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RingVertexPacked {
    pub position: [f32; 2],
    pub u_coord: f32,
    pub _pad: f32,
}

/// Buffers returned from a GPU extrusion dispatch.
pub struct GpuExtrusionOutput {
    pub positions: crate::core::resource_tracker::TrackedBuffer,
    pub indices: crate::core::resource_tracker::TrackedBuffer,
    pub normals: crate::core::resource_tracker::TrackedBuffer,
    pub uvs: crate::core::resource_tracker::TrackedBuffer,
    pub vertex_count: u32,
    pub index_count: u32,
}
