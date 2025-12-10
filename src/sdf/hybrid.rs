// src/sdf/hybrid.rs
// Hybrid traversal combining SDF raymarching with BVH mesh traversal
// Provides unified intersection testing for both analytic SDFs and polygonal meshes

use once_cell::sync::OnceCell;

use crate::accel::BvhHandle;
use crate::core::error::RenderError;
use crate::core::gpu::ctx;
use crate::sdf::SdfScene;

// Re-export types from hybrid_types
pub use super::hybrid_types::{HybridHitResult, HybridMetrics, MeshBuffers, Ray, SdfBuffers, Vertex};

/// Hybrid scene containing both SDF and mesh geometry
#[derive(Debug)]
pub struct HybridScene {
    /// SDF scene for analytic geometry
    pub sdf_scene: SdfScene,
    /// BVH for mesh geometry
    pub bvh: Option<BvhHandle>,
    /// Mesh vertices
    pub vertices: Vec<Vertex>,
    /// Mesh indices (triangles)
    pub indices: Vec<u32>,
    /// GPU buffers for SDF data
    pub sdf_buffers: Option<SdfBuffers>,
    /// GPU buffers for mesh data
    pub mesh_buffers: Option<MeshBuffers>,
}

/// Provide a process-lifetime dummy storage buffer for bind group layout requirements
fn dummy_storage_buffer() -> &'static wgpu::Buffer {
    static DUMMY: OnceCell<wgpu::Buffer> = OnceCell::new();
    DUMMY.get_or_init(|| {
        ctx().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-dummy-storage"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    })
}

impl HybridScene {
    /// Create a new hybrid scene
    pub fn new() -> Self {
        Self {
            sdf_scene: SdfScene::new(),
            bvh: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            sdf_buffers: None,
            mesh_buffers: None,
        }
    }

    /// Create a hybrid scene with only SDF geometry
    pub fn sdf_only(sdf_scene: SdfScene) -> Self {
        Self {
            sdf_scene,
            bvh: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            sdf_buffers: None,
            mesh_buffers: None,
        }
    }

    /// Create a hybrid scene with only mesh geometry
    pub fn mesh_only(vertices: Vec<Vertex>, indices: Vec<u32>, bvh: BvhHandle) -> Self {
        Self {
            sdf_scene: SdfScene::new(),
            bvh: Some(bvh),
            vertices,
            indices,
            sdf_buffers: None,
            mesh_buffers: None,
        }
    }

    /// Add mesh geometry to the scene
    pub fn add_mesh(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        bvh: BvhHandle,
    ) -> Result<(), RenderError> {
        self.vertices = vertices;
        self.indices = indices;
        self.bvh = Some(bvh);

        // Invalidate GPU buffers so they get recreated
        self.mesh_buffers = None;

        Ok(())
    }

    /// Set SDF scene
    pub fn set_sdf_scene(&mut self, sdf_scene: SdfScene) {
        self.sdf_scene = sdf_scene;

        // Invalidate GPU buffers so they get recreated
        self.sdf_buffers = None;
    }

    /// Check if scene has SDF geometry
    pub fn has_sdf(&self) -> bool {
        self.sdf_scene.primitive_count() > 0
    }

    /// Check if scene has mesh geometry
    pub fn has_mesh(&self) -> bool {
        self.bvh.is_some() && !self.vertices.is_empty()
    }

    /// Intersect ray with the hybrid scene (CPU implementation)
    pub fn intersect(&self, ray: Ray) -> HybridHitResult {
        let mut result = HybridHitResult {
            t: ray.tmax,
            point: glam::Vec3::ZERO,
            normal: glam::Vec3::Z,
            material_id: 0,
            hit_type: 0,
            hit: false,
            triangle_info: None,
            sdf_distance: None,
        };

        // Test SDF geometry using raymarching
        if self.has_sdf() {
            if let Some(sdf_hit) = self.raymarch_sdf(ray) {
                if sdf_hit.t < result.t {
                    result = sdf_hit;
                }
            }
        }

        // Test mesh geometry using BVH traversal
        if self.has_mesh() {
            if let Some(mesh_hit) = self.intersect_mesh(ray) {
                if mesh_hit.t < result.t {
                    result = mesh_hit;
                }
            }
        }

        result
    }

    /// Raymarch against SDF geometry
    fn raymarch_sdf(&self, ray: Ray) -> Option<HybridHitResult> {
        const MAX_STEPS: u32 = 128;
        const MIN_DISTANCE: f32 = 0.001;
        const MAX_DISTANCE: f32 = 100.0;

        let mut t = ray.tmin;
        let mut steps = 0;

        while steps < MAX_STEPS && t < ray.tmax.min(MAX_DISTANCE) {
            let point = ray.origin + ray.direction * t;

            // Evaluate SDF at current point
            let sdf_result = self.sdf_scene.evaluate(point);

            if sdf_result.distance < MIN_DISTANCE {
                // Hit! Calculate normal using finite differences
                let eps = 0.001;
                let normal = glam::Vec3::new(
                    self.sdf_scene
                        .evaluate(point + glam::Vec3::X * eps)
                        .distance
                        - self
                            .sdf_scene
                            .evaluate(point - glam::Vec3::X * eps)
                            .distance,
                    self.sdf_scene
                        .evaluate(point + glam::Vec3::Y * eps)
                        .distance
                        - self
                            .sdf_scene
                            .evaluate(point - glam::Vec3::Y * eps)
                            .distance,
                    self.sdf_scene
                        .evaluate(point + glam::Vec3::Z * eps)
                        .distance
                        - self
                            .sdf_scene
                            .evaluate(point - glam::Vec3::Z * eps)
                            .distance,
                )
                .normalize();

                return Some(HybridHitResult {
                    t,
                    point,
                    normal,
                    material_id: sdf_result.material_id,
                    hit_type: 1, // SDF hit
                    hit: true,
                    triangle_info: None,
                    sdf_distance: Some(sdf_result.distance),
                });
            }

            // March forward by the distance to the nearest surface
            t += sdf_result.distance.abs().max(MIN_DISTANCE * 0.1);
            steps += 1;
        }

        None
    }

    /// Intersect with mesh geometry using BVH
    fn intersect_mesh(&self, _ray: Ray) -> Option<HybridHitResult> {
        // This would use the existing BVH traversal code
        // For now, return None as a placeholder
        // In a full implementation, this would traverse the BVH and test triangles
        None
    }

    /// Upload SDF data to GPU buffers
    pub fn upload_sdf_to_gpu(&mut self) -> Result<(), RenderError> {
        if !self.has_sdf() {
            return Ok(());
        }

        let device = &ctx().device;

        // Convert SDF data to GPU-compatible format
        // Note: primitives may not be Pod; defer actual upload and create minimal buffers.
        let primitives_data: Vec<u8> = Vec::new();
        let nodes_data: Vec<u8> = Vec::new();

        // Create buffers
        let primitives_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-sdf-primitives"),
            size: primitives_data.len().max(4) as u64, // Ensure minimum size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-sdf-nodes"),
            size: nodes_data.len().max(4) as u64, // Ensure minimum size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload data if not empty
        if !primitives_data.is_empty() {
            ctx()
                .queue
                .write_buffer(&primitives_buffer, 0, &primitives_data);
        }
        if !nodes_data.is_empty() {
            ctx().queue.write_buffer(&nodes_buffer, 0, &nodes_data);
        }

        self.sdf_buffers = Some(SdfBuffers {
            primitives_buffer,
            nodes_buffer,
            primitive_count: self.sdf_scene.primitive_count() as u32,
            node_count: self.sdf_scene.node_count() as u32,
        });

        Ok(())
    }

    /// Upload mesh data to GPU buffers
    pub fn upload_mesh_to_gpu(&mut self) -> Result<(), RenderError> {
        if !self.has_mesh() {
            return Ok(());
        }

        let device = &ctx().device;

        // Convert vertices to bytes
        let vertices_data = bytemuck::cast_slice(&self.vertices);
        let indices_data = bytemuck::cast_slice(&self.indices);

        // Get BVH data
        let (bvh_data, bvh_node_count) = match &self.bvh {
            Some(bvh) => {
                match &bvh.backend {
                    crate::accel::BvhBackend::Gpu(_gpu_data) => {
                        // For GPU BVH, we already have the buffer
                        return Ok(()); // GPU BVH manages its own buffers
                    }
                    crate::accel::BvhBackend::Cpu(cpu_data) => {
                        let bvh_bytes = bytemuck::cast_slice(&cpu_data.nodes);
                        (bvh_bytes, cpu_data.nodes.len())
                    }
                }
            }
            None => return Err(RenderError::Upload("No BVH available".into())),
        };

        // Create buffers
        let vertices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-mesh-vertices"),
            size: vertices_data.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-mesh-indices"),
            size: indices_data.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bvh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-mesh-bvh"),
            size: bvh_data.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload data
        ctx().queue.write_buffer(&vertices_buffer, 0, vertices_data);
        ctx().queue.write_buffer(&indices_buffer, 0, indices_data);
        ctx().queue.write_buffer(&bvh_buffer, 0, bvh_data);

        self.mesh_buffers = Some(MeshBuffers {
            vertices_buffer,
            indices_buffer,
            bvh_buffer,
            vertex_count: self.vertices.len() as u32,
            index_count: self.indices.len() as u32,
            bvh_node_count: bvh_node_count as u32,
        });

        Ok(())
    }

    /// Prepare all GPU resources for rendering
    pub fn prepare_gpu_resources(&mut self) -> Result<(), RenderError> {
        self.upload_sdf_to_gpu()?;
        self.upload_mesh_to_gpu()?;
        Ok(())
    }

    /// Get bind group entries for SDF buffers
    pub fn get_sdf_bind_entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
        if let Some(sdf_buffers) = &self.sdf_buffers {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0, // SDF primitives
                    resource: sdf_buffers.primitives_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1, // SDF nodes
                    resource: sdf_buffers.nodes_buffer.as_entire_binding(),
                },
            ]
        } else {
            let dummy = dummy_storage_buffer();
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy.as_entire_binding(),
                },
            ]
        }
    }

    /// Get bind group entries for mesh buffers
    pub fn get_mesh_bind_entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
        if let Some(mesh_buffers) = &self.mesh_buffers {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0, // Mesh vertices
                    resource: mesh_buffers.vertices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1, // Mesh indices
                    resource: mesh_buffers.indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2, // BVH nodes
                    resource: mesh_buffers.bvh_buffer.as_entire_binding(),
                },
            ]
        } else {
            let dummy = dummy_storage_buffer();
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy.as_entire_binding(),
                },
            ]
        }
    }
}

impl Default for HybridScene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::{SdfPrimitive, SdfSceneBuilder};
    use glam::Vec3;

    #[test]
    fn test_hybrid_scene_creation() {
        let scene = HybridScene::new();
        assert!(!scene.has_sdf());
        assert!(!scene.has_mesh());
    }

    #[test]
    fn test_sdf_only_scene() {
        let (builder, _) = SdfSceneBuilder::new().add_sphere(Vec3::ZERO, 1.0, 1);
        let sdf_scene = builder.build();

        let hybrid = HybridScene::sdf_only(sdf_scene);
        assert!(hybrid.has_sdf());
        assert!(!hybrid.has_mesh());
        assert_eq!(hybrid.sdf_scene.primitive_count(), 1);
    }

    #[test]
    fn test_sdf_raymarching() {
        let (builder, _) = SdfSceneBuilder::new().add_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 1);
        let sdf_scene = builder.build();

        let hybrid = HybridScene::sdf_only(sdf_scene);

        let ray = Ray {
            origin: Vec3::ZERO,
            direction: Vec3::new(0.0, 0.0, -1.0),
            tmin: 0.001,
            tmax: 100.0,
        };

        let result = hybrid.intersect(ray);
        assert!(result.hit, "Ray should hit the sphere");
        assert_eq!(result.hit_type, 1, "Should be SDF hit");
        assert_eq!(result.material_id, 1, "Should have correct material ID");
    }
}
