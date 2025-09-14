// src/path_tracing/accel.rs
// Integration layer between BVH acceleration structures and path tracing system.
// This file exists to provide BVH traversal utilities and integration with the path tracing renderer.
// RELEVANT FILES:src/accel/mod.rs,src/path_tracing/mod.rs,python/forge3d/path_tracing.py

use crate::accel::{BvhHandle, BvhBackend, GpuBvhData, CpuBvhData, Triangle};
use crate::accel::types::{BvhNode, Aabb};
use anyhow::{Result, Context};
use bytemuck::{Pod, Zeroable};

/// Ray structure for traversal
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Ray {
    pub origin: [f32; 3],
    pub t_min: f32,
    pub direction: [f32; 3],
    pub t_max: f32,
}

impl Ray {
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self {
            origin,
            t_min: 1e-4,
            direction,
            t_max: f32::INFINITY,
        }
    }

    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }
}

/// Hit information from ray-triangle intersection
#[derive(Debug, Clone, Copy)]
pub struct HitInfo {
    pub t: f32,
    pub triangle_idx: u32,
    pub barycentric: [f32; 2], // u, v coordinates (w = 1-u-v)
    pub normal: [f32; 3],
    pub hit_point: [f32; 3],
}

/// BVH traversal interface for path tracing
pub struct BvhTraverser {
    // CPU traversal state
    cpu_stack: Vec<u32>,
}

impl BvhTraverser {
    pub fn new() -> Self {
        Self {
            cpu_stack: Vec::with_capacity(64), // Typical max depth
        }
    }

    /// Traverse BVH to find closest ray-triangle intersection
    pub fn intersect(
        &mut self,
        bvh: &BvhHandle,
        triangles: &[Triangle],
        ray: &Ray,
    ) -> Result<Option<HitInfo>> {
        match &bvh.backend {
            BvhBackend::Cpu(cpu_data) => {
                self.intersect_cpu(cpu_data, triangles, ray)
            }
            BvhBackend::Gpu(_gpu_data) => {
                // For GPU BVH, we'd need to copy data back to CPU for traversal
                // or implement GPU-based traversal. For now, fall back to CPU.
                anyhow::bail!("GPU BVH traversal not yet implemented - use CPU fallback")
            }
        }
    }

    /// Test ray against AABB
    fn ray_aabb_intersect(&self, ray: &Ray, aabb: &Aabb) -> bool {
        let mut t_min = ray.t_min;
        let mut t_max = ray.t_max;

        for i in 0..3 {
            let inv_dir = 1.0 / ray.direction[i];
            let mut t0 = (aabb.min[i] - ray.origin[i]) * inv_dir;
            let mut t1 = (aabb.max[i] - ray.origin[i]) * inv_dir;

            if inv_dir < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            t_min = t_min.max(t0);
            t_max = t_max.min(t1);

            if t_min > t_max {
                return false;
            }
        }

        true
    }

    /// Ray-triangle intersection using Möller-Trumbore algorithm
    fn ray_triangle_intersect(&self, ray: &Ray, triangle: &Triangle) -> Option<HitInfo> {
        let edge1 = [
            triangle.v1[0] - triangle.v0[0],
            triangle.v1[1] - triangle.v0[1],
            triangle.v1[2] - triangle.v0[2],
        ];
        let edge2 = [
            triangle.v2[0] - triangle.v0[0],
            triangle.v2[1] - triangle.v0[1], 
            triangle.v2[2] - triangle.v0[2],
        ];

        // Cross product: direction × edge2
        let h = [
            ray.direction[1] * edge2[2] - ray.direction[2] * edge2[1],
            ray.direction[2] * edge2[0] - ray.direction[0] * edge2[2],
            ray.direction[0] * edge2[1] - ray.direction[1] * edge2[0],
        ];

        // Dot product: edge1 · h
        let a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
        
        if a > -1e-7 && a < 1e-7 {
            return None; // Ray is parallel to triangle
        }

        let f = 1.0 / a;
        let s = [
            ray.origin[0] - triangle.v0[0],
            ray.origin[1] - triangle.v0[1],
            ray.origin[2] - triangle.v0[2],
        ];

        // u parameter
        let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
        if u < 0.0 || u > 1.0 {
            return None;
        }

        // Cross product: s × edge1
        let q = [
            s[1] * edge1[2] - s[2] * edge1[1],
            s[2] * edge1[0] - s[0] * edge1[2],
            s[0] * edge1[1] - s[1] * edge1[0],
        ];

        // v parameter
        let v = f * (ray.direction[0] * q[0] + ray.direction[1] * q[1] + ray.direction[2] * q[2]);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        // t parameter
        let t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
        if t > ray.t_min && t < ray.t_max {
            // Compute normal (not normalized)
            let normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            
            // Normalize normal
            let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            let normal = if length > 1e-6 {
                [normal[0] / length, normal[1] / length, normal[2] / length]
            } else {
                [0.0, 1.0, 0.0] // Degenerate triangle
            };

            let hit_point = ray.at(t);

            Some(HitInfo {
                t,
                triangle_idx: 0, // Will be set by caller
                barycentric: [u, v],
                normal,
                hit_point,
            })
        } else {
            None
        }
    }

    /// CPU BVH traversal using stack-based iteration
    fn intersect_cpu(
        &mut self,
        cpu_data: &CpuBvhData,
        triangles: &[Triangle],
        ray: &Ray,
    ) -> Result<Option<HitInfo>> {
        if cpu_data.nodes.is_empty() {
            return Ok(None);
        }

        let mut closest_hit: Option<HitInfo> = None;
        let mut closest_t = ray.t_max;
        let mut current_ray = *ray;

        self.cpu_stack.clear();
        self.cpu_stack.push(0); // Start with root node

        while let Some(node_idx) = self.cpu_stack.pop() {
            if node_idx as usize >= cpu_data.nodes.len() {
                continue;
            }

            let node = &cpu_data.nodes[node_idx as usize];

            // Test ray against node AABB
            if !self.ray_aabb_intersect(&current_ray, &node.aabb) {
                continue;
            }

            if node.is_leaf() {
                // Test ray against all triangles in leaf
                let (first_prim, prim_count) = node.primitives().unwrap();
                
                for i in 0..prim_count {
                    let prim_idx = cpu_data.indices[(first_prim + i) as usize] as usize;
                    if prim_idx >= triangles.len() {
                        continue;
                    }

                    if let Some(mut hit) = self.ray_triangle_intersect(&current_ray, &triangles[prim_idx]) {
                        if hit.t < closest_t {
                            hit.triangle_idx = prim_idx as u32;
                            closest_hit = Some(hit);
                            closest_t = hit.t;
                            current_ray.t_max = hit.t; // Shrink ray for early termination
                        }
                    }
                }
            } else {
                // Internal node - add children to stack
                let (left_idx, right_idx) = node.children().unwrap();
                
                // Add children in order that may improve traversal efficiency
                // (closer node first, but requires more computation)
                self.cpu_stack.push(right_idx);
                self.cpu_stack.push(left_idx);
            }
        }

        Ok(closest_hit)
    }

    /// Test ray against scene bounds for early rejection
    pub fn intersect_bounds(&self, bvh: &BvhHandle, ray: &Ray) -> bool {
        self.ray_aabb_intersect(ray, &bvh.world_aabb)
    }

    /// Get BVH statistics for debugging
    pub fn get_stats(&self, bvh: &BvhHandle) -> TraversalStats {
        match &bvh.backend {
            BvhBackend::Cpu(cpu_data) => {
                let mut stats = TraversalStats::default();
                stats.node_count = cpu_data.nodes.len() as u32;
                stats.leaf_count = cpu_data.nodes.iter()
                    .filter(|n| n.is_leaf())
                    .count() as u32;
                stats.internal_count = stats.node_count - stats.leaf_count;
                stats.max_depth = self.compute_max_depth(&cpu_data.nodes, 0, 0);
                stats
            }
            BvhBackend::Gpu(gpu_data) => {
                TraversalStats {
                    node_count: gpu_data.node_count,
                    leaf_count: gpu_data.primitive_count,
                    internal_count: gpu_data.node_count - gpu_data.primitive_count,
                    max_depth: 0, // Would need to compute from GPU data
                    ..Default::default()
                }
            }
        }
    }

    fn compute_max_depth(&self, nodes: &[BvhNode], node_idx: usize, current_depth: u32) -> u32 {
        if node_idx >= nodes.len() {
            return current_depth;
        }

        let node = &nodes[node_idx];
        if node.is_leaf() {
            return current_depth;
        }

        let (left_idx, right_idx) = node.children().unwrap();
        let left_depth = self.compute_max_depth(nodes, left_idx as usize, current_depth + 1);
        let right_depth = self.compute_max_depth(nodes, right_idx as usize, current_depth + 1);
        
        left_depth.max(right_depth)
    }
}

impl Default for BvhTraverser {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for BVH traversal
#[derive(Debug, Clone, Default)]
pub struct TraversalStats {
    pub node_count: u32,
    pub leaf_count: u32,
    pub internal_count: u32,
    pub max_depth: u32,
    pub nodes_tested: u32,
    pub leaves_tested: u32,
    pub triangles_tested: u32,
}

/// Utility functions for path tracing integration
pub fn create_camera_ray(
    origin: [f32; 3],
    direction: [f32; 3],
    t_min: f32,
    t_max: f32,
) -> Ray {
    Ray {
        origin,
        t_min,
        direction,
        t_max,
    }
}

pub fn create_shadow_ray(from: [f32; 3], to: [f32; 3]) -> Ray {
    let direction = [
        to[0] - from[0],
        to[1] - from[1],
        to[2] - from[2],
    ];
    
    let length = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
    
    if length < 1e-6 {
        return Ray::new(from, [0.0, 1.0, 0.0]);
    }
    
    let normalized_direction = [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ];
    
    Ray {
        origin: from,
        t_min: 1e-4,
        direction: normalized_direction,
        t_max: length - 1e-4,
    }
}