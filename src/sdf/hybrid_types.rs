// src/sdf/hybrid_types.rs
// Type definitions for hybrid SDF/mesh traversal system

use wgpu::Buffer;

/// Simplified vertex type for hybrid scene mesh geometry
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub _pad: f32,
}

/// GPU buffers for SDF data
#[derive(Debug)]
pub struct SdfBuffers {
    pub primitives_buffer: Buffer,
    pub nodes_buffer: Buffer,
    pub primitive_count: u32,
    pub node_count: u32,
}

/// GPU buffers for mesh data
#[derive(Debug)]
pub struct MeshBuffers {
    pub vertices_buffer: Buffer,
    pub indices_buffer: Buffer,
    pub bvh_buffer: Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub bvh_node_count: u32,
}

/// Hybrid intersection result containing both SDF and mesh data
#[derive(Clone, Copy, Debug)]
pub struct HybridHitResult {
    /// Distance from ray origin
    pub t: f32,
    /// Hit point in world space
    pub point: glam::Vec3,
    /// Surface normal at hit point
    pub normal: glam::Vec3,
    /// Material ID
    pub material_id: u32,
    /// Intersection type (0 = mesh, 1 = SDF)
    pub hit_type: u32,
    /// Whether any intersection occurred
    pub hit: bool,
    /// For mesh hits: triangle index and barycentric coordinates
    pub triangle_info: Option<(u32, glam::Vec2)>,
    /// For SDF hits: signed distance at surface
    pub sdf_distance: Option<f32>,
}

impl Default for HybridHitResult {
    fn default() -> Self {
        Self {
            t: f32::MAX,
            point: glam::Vec3::ZERO,
            normal: glam::Vec3::Z,
            material_id: 0,
            hit_type: 0,
            hit: false,
            triangle_info: None,
            sdf_distance: None,
        }
    }
}

/// Ray representation for hybrid traversal
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: glam::Vec3,
    pub direction: glam::Vec3,
    pub tmin: f32,
    pub tmax: f32,
}

impl Ray {
    /// Create a new ray with the given parameters
    pub fn new(origin: glam::Vec3, direction: glam::Vec3, tmin: f32, tmax: f32) -> Self {
        Self {
            origin,
            direction,
            tmin,
            tmax,
        }
    }
}

/// Performance metrics for hybrid traversal
#[derive(Clone, Copy, Debug, Default)]
pub struct HybridMetrics {
    /// Number of SDF raymarching steps
    pub sdf_steps: u32,
    /// Number of BVH nodes traversed
    pub bvh_nodes_visited: u32,
    /// Number of triangle tests performed
    pub triangle_tests: u32,
    /// Total rays cast
    pub total_rays: u32,
    /// Rays that hit SDF geometry
    pub sdf_hits: u32,
    /// Rays that hit mesh geometry
    pub mesh_hits: u32,
}

impl HybridMetrics {
    /// Calculate performance overhead compared to mesh-only rendering
    pub fn performance_overhead(&self) -> f32 {
        if self.total_rays == 0 {
            return 0.0;
        }

        // Estimate cost: SDF steps are more expensive than BVH traversal
        let sdf_cost = self.sdf_steps as f32 * 2.0;
        let bvh_cost = self.bvh_nodes_visited as f32;
        let triangle_cost = self.triangle_tests as f32 * 3.0;

        let total_cost = sdf_cost + bvh_cost + triangle_cost;
        let mesh_only_cost = self.bvh_nodes_visited as f32 + self.triangle_tests as f32 * 3.0;

        if mesh_only_cost == 0.0 {
            return if total_cost > 0.0 { f32::INFINITY } else { 0.0 };
        }

        (total_cost - mesh_only_cost) / mesh_only_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = HybridMetrics::default();
        metrics.total_rays = 100;
        metrics.sdf_steps = 500;
        metrics.bvh_nodes_visited = 200;
        metrics.triangle_tests = 50;

        let overhead = metrics.performance_overhead();
        assert!(overhead >= 0.0, "Overhead should be non-negative");
    }

    #[test]
    fn test_ray_creation() {
        let ray = Ray::new(glam::Vec3::ZERO, glam::Vec3::Z, 0.001, 100.0);
        assert_eq!(ray.origin, glam::Vec3::ZERO);
        assert_eq!(ray.direction, glam::Vec3::Z);
    }

    #[test]
    fn test_hit_result_default() {
        let hit = HybridHitResult::default();
        assert!(!hit.hit);
        assert_eq!(hit.hit_type, 0);
    }
}
