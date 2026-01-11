// src/viewer/state/mesh_upload.rs
// Mesh upload helpers for the Viewer
// Extracted from mod.rs as part of the viewer refactoring

use std::path::Path;
use wgpu::util::DeviceExt;

use crate::viewer::event_loop::update_ipc_stats;
use crate::viewer::viewer_types::PackedVertex;
use crate::viewer::Viewer;

impl Viewer {
    /// Upload mesh geometry to GPU buffers for rendering
    pub(crate) fn upload_mesh(&mut self, mesh: &crate::geometry::MeshBuffers) -> anyhow::Result<()> {
        if mesh.positions.is_empty() || mesh.indices.is_empty() {
            anyhow::bail!("Mesh is empty (no vertices or indices)");
        }

        // Store original mesh data for CPU-side transform workaround
        self.original_mesh_positions = mesh.positions.clone();
        self.original_mesh_normals = mesh.normals.clone();
        self.original_mesh_uvs = mesh.uvs.clone();
        self.original_mesh_indices = mesh.indices.clone();
        // Reset transform when new mesh is loaded
        self.object_transform = glam::Mat4::IDENTITY;
        self.object_translation = glam::Vec3::ZERO;
        self.object_rotation = glam::Quat::IDENTITY;
        self.object_scale = glam::Vec3::ONE;

        // Convert MeshBuffers to PackedVertex format
        let vertex_count = mesh.positions.len();
        let mut vertices: Vec<PackedVertex> = Vec::with_capacity(vertex_count);
        
        for i in 0..vertex_count {
            let pos = mesh.positions[i];
            let normal = if i < mesh.normals.len() {
                mesh.normals[i]
            } else {
                [0.0, 1.0, 0.0]
            };
            let uv = if i < mesh.uvs.len() {
                mesh.uvs[i]
            } else {
                [0.0, 0.0]
            };
            vertices.push(PackedVertex {
                position: pos,
                normal,
                uv,
                rough_metal: [0.5, 0.0],
            });
        }

        let vertex_data = bytemuck::cast_slice(&vertices);
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ipc.mesh.vb"),
                contents: vertex_data,
                usage: wgpu::BufferUsages::VERTEX,
            });
        
        let ib = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("viewer.ipc.mesh.ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        
        self.geom_vb = Some(vb);
        self.geom_ib = Some(ib);
        self.geom_index_count = mesh.indices.len() as u32;
        self.geom_bind_group = None;
        
        update_ipc_stats(
            true,
            vertices.len() as u32,
            mesh.indices.len() as u32,
            true,
        );
        
        println!(
            "[viewer] Uploaded mesh: {} vertices, {} indices",
            vertices.len(),
            mesh.indices.len()
        );
        Ok(())
    }

    /// No-op loader until albedo textures are wired in.
    pub(crate) fn load_albedo_texture(&mut self, _path: &Path) -> anyhow::Result<()> {
        Ok(())
    }
}
