// src/accel/types.rs
// Core types for BVH acceleration structures - AABB, nodes, triangles, and build options.
// This file exists to provide GPU-compatible data structures and packing utilities for LBVH construction.
// RELEVANT FILES:src/accel/mod.rs,src/shaders/lbvh_link.wgsl,src/shaders/bvh_refit.wgsl

use bytemuck::{Pod, Zeroable};
use std::fmt;

/// Axis-aligned bounding box - GPU compatible layout
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Aabb {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3], 
    pub _pad1: f32,
}

impl Aabb {
    /// Create empty AABB (inverted bounds for union operations)
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            _pad0: 0.0,
            max: [f32::NEG_INFINITY; 3],
            _pad1: 0.0,
        }
    }

    /// Create AABB from min/max points
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min,
            _pad0: 0.0,
            max,
            _pad1: 0.0,
        }
    }

    /// Expand AABB to include a point
    pub fn expand_point(&mut self, point: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    /// Expand AABB to include another AABB
    pub fn expand_aabb(&mut self, other: &Aabb) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    /// Get AABB center
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get AABB extent (max - min)
    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Check if AABB is valid (min <= max)
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && 
        self.min[1] <= self.max[1] && 
        self.min[2] <= self.max[2]
    }

    /// Get surface area for SAH calculations
    pub fn surface_area(&self) -> f32 {
        let extent = self.extent();
        if extent[0] < 0.0 || extent[1] < 0.0 || extent[2] < 0.0 {
            return 0.0;
        }
        2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::empty()
    }
}

/// BVH node - GPU compatible layout matching WGSL struct
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub kind: u32,      // 0 = internal, 1 = leaf
    pub left_idx: u32,  // for internal: left child; for leaf: first primitive
    pub right_idx: u32, // for internal: right child; for leaf: primitive count  
    pub parent_idx: u32,
}

impl BvhNode {
    /// Create internal node
    pub fn internal(aabb: Aabb, left: u32, right: u32) -> Self {
        Self {
            aabb,
            kind: 0,
            left_idx: left,
            right_idx: right,
            parent_idx: u32::MAX,
        }
    }

    /// Create leaf node
    pub fn leaf(aabb: Aabb, first_prim: u32, prim_count: u32) -> Self {
        Self {
            aabb,
            kind: 1,
            left_idx: first_prim,
            right_idx: prim_count,
            parent_idx: u32::MAX,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.kind == 1
    }

    /// Check if this is an internal node
    pub fn is_internal(&self) -> bool {
        self.kind == 0
    }

    /// Get child indices for internal nodes
    pub fn children(&self) -> Option<(u32, u32)> {
        if self.is_internal() {
            Some((self.left_idx, self.right_idx))
        } else {
            None
        }
    }

    /// Get primitive range for leaf nodes (first_idx, count)
    pub fn primitives(&self) -> Option<(u32, u32)> {
        if self.is_leaf() {
            Some((self.left_idx, self.right_idx))
        } else {
            None
        }
    }
}

/// Triangle primitive for BVH construction
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3], 
    pub v2: [f32; 3],
}

impl Triangle {
    /// Create triangle from vertices
    pub fn new(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Self {
        Self { v0, v1, v2 }
    }

    /// Get triangle centroid
    pub fn centroid(&self) -> [f32; 3] {
        [
            (self.v0[0] + self.v1[0] + self.v2[0]) / 3.0,
            (self.v0[1] + self.v1[1] + self.v2[1]) / 3.0,
            (self.v0[2] + self.v1[2] + self.v2[2]) / 3.0,
        ]
    }

    /// Get triangle AABB
    pub fn aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();
        aabb.expand_point(self.v0);
        aabb.expand_point(self.v1);
        aabb.expand_point(self.v2);
        aabb
    }

    /// Get triangle normal (not normalized)
    pub fn normal(&self) -> [f32; 3] {
        let e1 = [
            self.v1[0] - self.v0[0],
            self.v1[1] - self.v0[1], 
            self.v1[2] - self.v0[2],
        ];
        let e2 = [
            self.v2[0] - self.v0[0],
            self.v2[1] - self.v0[1],
            self.v2[2] - self.v0[2],
        ];
        
        [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
    }

    /// Get triangle area
    pub fn area(&self) -> f32 {
        let normal = self.normal();
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        length * 0.5
    }
}

/// Build options for BVH construction
#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// Random seed for deterministic builds
    pub seed: u32,
    /// Maximum primitives per leaf node
    pub max_leaf_size: u32,
    /// Use GPU if available (true) or force CPU (false)
    pub prefer_gpu: bool,
    /// SAH cost parameters
    pub traversal_cost: f32,
    pub intersection_cost: f32,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            seed: 1,
            max_leaf_size: 4,
            prefer_gpu: true,
            traversal_cost: 1.0,
            intersection_cost: 1.0,
        }
    }
}

/// Handle to a built BVH
pub struct BvhHandle {
    pub backend: crate::accel::BvhBackend,
    pub triangle_count: u32,
    pub node_count: u32,
    pub world_aabb: Aabb,
    pub build_stats: BuildStats,
}

impl BvhHandle {
    /// Check if this BVH uses GPU backend
    pub fn is_gpu(&self) -> bool {
        matches!(self.backend, crate::accel::BvhBackend::Gpu(_))
    }

    /// Check if this BVH uses CPU backend
    pub fn is_cpu(&self) -> bool {
        matches!(self.backend, crate::accel::BvhBackend::Cpu(_))
    }
}

impl fmt::Debug for BvhHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BvhHandle")
            .field("backend", match &self.backend {
                crate::accel::BvhBackend::Gpu(_) => &"GPU",
                crate::accel::BvhBackend::Cpu(_) => &"CPU",
            })
            .field("triangle_count", &self.triangle_count)
            .field("node_count", &self.node_count)
            .field("world_aabb", &self.world_aabb)
            .field("build_stats", &self.build_stats)
            .finish()
    }
}

/// Statistics from BVH construction
#[derive(Debug, Clone, Default)]
pub struct BuildStats {
    pub build_time_ms: f32,
    pub morton_time_ms: f32,
    pub sort_time_ms: f32,
    pub link_time_ms: f32,
    pub memory_usage_bytes: u64,
    pub leaf_count: u32,
    pub internal_count: u32,
    pub max_depth: u32,
    pub avg_leaf_size: f32,
}

/// Utility functions for working with triangles and AABBs
pub fn compute_scene_aabb(triangles: &[Triangle]) -> Aabb {
    let mut aabb = Aabb::empty();
    for triangle in triangles {
        let tri_aabb = triangle.aabb();
        aabb.expand_aabb(&tri_aabb);
    }
    aabb
}

pub fn compute_triangle_centroids(triangles: &[Triangle]) -> Vec<[f32; 3]> {
    triangles.iter().map(|t| t.centroid()).collect()
}

pub fn compute_triangle_aabbs(triangles: &[Triangle]) -> Vec<Aabb> {
    triangles.iter().map(|t| t.aabb()).collect()
}