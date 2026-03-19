use super::*;
use wgpu::{BufferDescriptor, BufferUsages};

impl CloudRenderer {
    pub fn create_cloud_quad_geometry(device: &Device) -> (Buffer, Buffer, u32) {
        let vertices: &[f32] = &[
            -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let indices: &[u16] = &[0, 1, 2, 2, 3, 0];

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_vertex_buffer"),
            size: (vertices.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(vertices));
        vertex_buffer.unmap();

        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_index_buffer"),
            size: (indices.len() * std::mem::size_of::<u16>()) as wgpu::BufferAddress,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(indices));
        index_buffer.unmap();

        (vertex_buffer, index_buffer, indices.len() as u32)
    }
}
