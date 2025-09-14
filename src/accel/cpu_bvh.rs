// src/accel/cpu_bvh.rs
// CPU BVH builder with GPU-compatible layout for triangle mesh path tracing (Task A3).
// This file provides a median-split BVH builder that produces a flattened layout suitable for GPU traversal.
// RELEVANT FILES:src/path_tracing/mesh.rs,src/shaders/pt_intersect_mesh.wgsl,python/forge3d/mesh.py

use anyhow::{Result, Context};
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

/// Triangle mesh for BVH construction - simple vertex/index representation
#[derive(Debug, Clone)]
pub struct MeshCPU {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>, // triangle indices (CCW winding)
}

impl MeshCPU {
    pub fn new(vertices: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> Self {
        Self { vertices, indices }
    }

    pub fn triangle_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Get triangle vertices by index
    pub fn get_triangle(&self, tri_idx: usize) -> Option<([f32; 3], [f32; 3], [f32; 3])> {
        if tri_idx >= self.indices.len() {
            return None;
        }
        let indices = self.indices[tri_idx];
        let v0 = *self.vertices.get(indices[0] as usize)?;
        let v1 = *self.vertices.get(indices[1] as usize)?;
        let v2 = *self.vertices.get(indices[2] as usize)?;
        Some((v0, v1, v2))
    }

    /// Compute triangle centroid
    pub fn triangle_centroid(&self, tri_idx: usize) -> Option<[f32; 3]> {
        let (v0, v1, v2) = self.get_triangle(tri_idx)?;
        Some([
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ])
    }

    /// Compute triangle AABB
    pub fn triangle_aabb(&self, tri_idx: usize) -> Option<Aabb> {
        let (v0, v1, v2) = self.get_triangle(tri_idx)?;
        let mut aabb = Aabb::empty();
        aabb.expand_point(v0);
        aabb.expand_point(v1);
        aabb.expand_point(v2);
        Some(aabb)
    }
}

/// GPU-compatible AABB layout (16-byte aligned)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Aabb {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            _pad0: 0.0,
            max: [f32::NEG_INFINITY; 3],
            _pad1: 0.0,
        }
    }

    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min,
            _pad0: 0.0,
            max,
            _pad1: 0.0,
        }
    }

    pub fn expand_point(&mut self, point: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    pub fn expand_aabb(&mut self, other: &Aabb) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn surface_area(&self) -> f32 {
        let extent = self.extent();
        if extent[0] < 0.0 || extent[1] < 0.0 || extent[2] < 0.0 {
            return 0.0;
        }
        2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] &&
        self.min[1] <= self.max[1] &&
        self.min[2] <= self.max[2]
    }
}

/// GPU-compatible BVH node layout as specified in task A3
/// Matches the WGSL struct layout exactly for GPU traversal
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb_min: [f32; 3],
    pub left: u32,      // if internal: left child index; if leaf: first triangle index
    pub aabb_max: [f32; 3],
    pub right: u32,     // if internal: right child index; if leaf: triangle count
    pub flags: u32,     // bit 0: leaf flag (1 = leaf, 0 = internal)
    pub _pad: u32,      // padding for 16-byte alignment
}

impl BvhNode {
    /// Create internal node
    pub fn internal(aabb: Aabb, left_idx: u32, right_idx: u32) -> Self {
        Self {
            aabb_min: aabb.min,
            left: left_idx,
            aabb_max: aabb.max,
            right: right_idx,
            flags: 0, // internal node
            _pad: 0,
        }
    }

    /// Create leaf node
    pub fn leaf(aabb: Aabb, first_tri: u32, tri_count: u32) -> Self {
        Self {
            aabb_min: aabb.min,
            left: first_tri,
            aabb_max: aabb.max,
            right: tri_count,
            flags: 1, // leaf node (bit 0 set)
            _pad: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        (self.flags & 1) != 0
    }

    pub fn is_internal(&self) -> bool {
        (self.flags & 1) == 0
    }

    pub fn aabb(&self) -> Aabb {
        Aabb {
            min: self.aabb_min,
            _pad0: 0.0,
            max: self.aabb_max,
            _pad1: 0.0,
        }
    }

    /// Get child indices for internal nodes
    pub fn children(&self) -> Option<(u32, u32)> {
        if self.is_internal() {
            Some((self.left, self.right))
        } else {
            None
        }
    }

    /// Get triangle range for leaf nodes
    pub fn triangles(&self) -> Option<(u32, u32)> {
        if self.is_leaf() {
            Some((self.left, self.right))
        } else {
            None
        }
    }
}

// Verify the struct layout matches expected GPU layout at compile time
const _: () = {
    assert!(std::mem::size_of::<BvhNode>() == 32); // 8 * 4 bytes
    assert!(std::mem::align_of::<BvhNode>() == 4);
};

/// Build options for BVH construction
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub max_leaf_size: u32,
    pub method: BuildMethod,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            max_leaf_size: 4,
            method: BuildMethod::MedianSplit,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuildMethod {
    MedianSplit,
}

/// CPU BVH data with flattened layout suitable for GPU upload
#[derive(Debug, Clone)]
pub struct BvhCPU {
    pub nodes: Vec<BvhNode>,
    pub tri_indices: Vec<u32>, // reordered triangle indices
    pub world_aabb: Aabb,
    pub build_stats: BuildStats,
}

impl BvhCPU {
    pub fn node_count(&self) -> u32 {
        self.nodes.len() as u32
    }

    pub fn triangle_count(&self) -> u32 {
        self.build_stats.triangle_count
    }
}

/// Build statistics
#[derive(Debug, Clone, Default)]
pub struct BuildStats {
    pub build_time_ms: f32,
    pub triangle_count: u32,
    pub node_count: u32,
    pub leaf_count: u32,
    pub internal_count: u32,
    pub max_depth: u32,
    pub avg_leaf_size: f32,
    pub memory_usage_bytes: u64,
}

/// Build BVH from mesh using specified method
pub fn build_bvh_cpu(mesh: &MeshCPU, options: &BuildOptions) -> Result<BvhCPU> {
    let start_time = Instant::now();

    if mesh.indices.is_empty() {
        anyhow::bail!("Cannot build BVH from empty mesh");
    }

    let triangle_count = mesh.triangle_count();

    // Compute triangle AABBs and centroids
    let mut tri_aabbs = Vec::with_capacity(triangle_count as usize);
    let mut tri_centroids = Vec::with_capacity(triangle_count as usize);

    for i in 0..triangle_count {
        let aabb = mesh.triangle_aabb(i as usize)
            .context("Failed to compute triangle AABB")?;
        let centroid = mesh.triangle_centroid(i as usize)
            .context("Failed to compute triangle centroid")?;
        tri_aabbs.push(aabb);
        tri_centroids.push(centroid);
    }

    // Compute world AABB
    let mut world_aabb = Aabb::empty();
    for aabb in &tri_aabbs {
        world_aabb.expand_aabb(aabb);
    }

    // Initialize triangle indices (will be reordered during build)
    let mut tri_indices: Vec<u32> = (0..triangle_count).collect();

    let mut nodes = Vec::new();
    let mut stats = BuildStats {
        triangle_count,
        ..Default::default()
    };

    // Build BVH recursively
    let build_info = BuildInfo {
        aabb: world_aabb,
        first: 0,
        count: triangle_count,
        depth: 0,
    };

    let _root_idx = build_recursive(
        &tri_aabbs,
        &tri_centroids,
        &mut tri_indices,
        &mut nodes,
        build_info,
        options,
        &mut stats,
    )?;

    // Calculate final statistics
    stats.build_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
    stats.node_count = nodes.len() as u32;
    stats.internal_count = stats.node_count - stats.leaf_count;
    stats.memory_usage_bytes = (nodes.len() * std::mem::size_of::<BvhNode>() +
                               tri_indices.len() * std::mem::size_of::<u32>()) as u64;

    if stats.leaf_count > 0 {
        stats.avg_leaf_size = triangle_count as f32 / stats.leaf_count as f32;
    }

    Ok(BvhCPU {
        nodes,
        tri_indices,
        world_aabb,
        build_stats: stats,
    })
}

/// Recursive build information
struct BuildInfo {
    aabb: Aabb,
    first: u32,
    count: u32,
    depth: u32,
}

/// Build BVH recursively using median split
fn build_recursive(
    tri_aabbs: &[Aabb],
    tri_centroids: &[[f32; 3]],
    tri_indices: &mut [u32],
    nodes: &mut Vec<BvhNode>,
    info: BuildInfo,
    options: &BuildOptions,
    stats: &mut BuildStats,
) -> Result<u32> {
    stats.max_depth = stats.max_depth.max(info.depth);

    // Check if we should create a leaf
    if info.count <= options.max_leaf_size || info.depth > 64 {
        stats.leaf_count += 1;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::leaf(info.aabb, info.first, info.count));
        return Ok(node_idx);
    }

    // Find split axis and position using median split
    let split_result = find_median_split(tri_centroids, &tri_indices[info.first as usize..(info.first + info.count) as usize], &info.aabb);

    if split_result.is_none() {
        // Cannot split, create leaf
        stats.leaf_count += 1;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::leaf(info.aabb, info.first, info.count));
        return Ok(node_idx);
    }

    let (split_axis, split_pos) = split_result.unwrap();

    // Partition triangles around split
    let split_index = partition_triangles(
        tri_indices,
        info.first,
        info.count,
        split_axis,
        split_pos,
        tri_centroids,
    )?;

    let left_count = split_index - info.first;
    let right_count = info.count - left_count;

    if left_count == 0 || right_count == 0 {
        // Degenerate split, create leaf
        stats.leaf_count += 1;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::leaf(info.aabb, info.first, info.count));
        return Ok(node_idx);
    }

    // Compute child AABBs
    let left_aabb = compute_bounds(tri_aabbs, &tri_indices[info.first as usize..split_index as usize]);
    let right_aabb = compute_bounds(tri_aabbs, &tri_indices[split_index as usize..(info.first + info.count) as usize]);

    // Build left and right subtrees
    let left_info = BuildInfo {
        aabb: left_aabb,
        first: info.first,
        count: left_count,
        depth: info.depth + 1,
    };

    let right_info = BuildInfo {
        aabb: right_aabb,
        first: split_index,
        count: right_count,
        depth: info.depth + 1,
    };

    let left_child_idx = build_recursive(
        tri_aabbs, tri_centroids, tri_indices, nodes, left_info, options, stats
    )?;

    let right_child_idx = build_recursive(
        tri_aabbs, tri_centroids, tri_indices, nodes, right_info, options, stats
    )?;

    // Create internal node
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode::internal(info.aabb, left_child_idx, right_child_idx));

    Ok(node_idx)
}

/// Find median split position
fn find_median_split(
    tri_centroids: &[[f32; 3]],
    indices: &[u32],
    parent_aabb: &Aabb,
) -> Option<(usize, f32)> {
    if indices.len() < 2 {
        return None;
    }

    let extent = parent_aabb.extent();

    // Find axis with largest extent
    let axis = if extent[0] > extent[1] && extent[0] > extent[2] {
        0
    } else if extent[1] > extent[2] {
        1
    } else {
        2
    };

    // Compute median centroid position along axis
    let mut centroids: Vec<f32> = indices.iter()
        .map(|&idx| tri_centroids[idx as usize][axis])
        .collect();
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_idx = centroids.len() / 2;
    let split_pos = centroids[median_idx];

    Some((axis, split_pos))
}

/// Partition triangles around split position
fn partition_triangles(
    indices: &mut [u32],
    first: u32,
    count: u32,
    axis: usize,
    split_pos: f32,
    tri_centroids: &[[f32; 3]],
) -> Result<u32> {
    let range = &mut indices[first as usize..(first + count) as usize];

    let mut left = 0;
    let mut right = range.len();

    while left < right {
        let centroid = tri_centroids[range[left] as usize];
        if centroid[axis] < split_pos {
            left += 1;
        } else {
            right -= 1;
            range.swap(left, right);
        }
    }

    Ok(first + left as u32)
}

/// Compute bounding box for a set of triangles
fn compute_bounds(tri_aabbs: &[Aabb], indices: &[u32]) -> Aabb {
    let mut aabb = Aabb::empty();
    for &idx in indices {
        aabb.expand_aabb(&tri_aabbs[idx as usize]);
    }
    aabb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh_node_layout() {
        // Verify GPU-compatible layout
        assert_eq!(std::mem::size_of::<BvhNode>(), 32);
        assert_eq!(std::mem::align_of::<BvhNode>(), 4);

        // Test node creation
        let aabb = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let leaf = BvhNode::leaf(aabb, 0, 4);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.triangles(), Some((0, 4)));

        let internal = BvhNode::internal(aabb, 1, 2);
        assert!(internal.is_internal());
        assert_eq!(internal.children(), Some((1, 2)));
    }

    #[test]
    fn test_mesh_simple() {
        // Simple triangle mesh (single triangle)
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2]];

        let mesh = MeshCPU::new(vertices, indices);
        assert_eq!(mesh.triangle_count(), 1);

        let (v0, v1, v2) = mesh.get_triangle(0).unwrap();
        assert_eq!(v0, [0.0, 0.0, 0.0]);
        assert_eq!(v1, [1.0, 0.0, 0.0]);
        assert_eq!(v2, [0.5, 1.0, 0.0]);

        let centroid = mesh.triangle_centroid(0).unwrap();
        assert!((centroid[0] - 0.5).abs() < 1e-6);
        assert!((centroid[1] - 1.0/3.0).abs() < 1e-6);
        assert_eq!(centroid[2], 0.0);
    }

    #[test]
    fn test_bvh_build_single_triangle() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2]];
        let mesh = MeshCPU::new(vertices, indices);

        let options = BuildOptions::default();
        let bvh = build_bvh_cpu(&mesh, &options).unwrap();

        assert_eq!(bvh.triangle_count(), 1);
        assert!(bvh.node_count() >= 1);
        assert_eq!(bvh.build_stats.leaf_count, 1);
        assert!(bvh.world_aabb.is_valid());
    }

    #[test]
    fn test_bvh_build_cube() {
        // Simple cube mesh (12 triangles, 8 vertices)
        let vertices = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
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

        let mesh = MeshCPU::new(vertices, indices);
        let options = BuildOptions::default();
        let bvh = build_bvh_cpu(&mesh, &options).unwrap();

        assert_eq!(bvh.triangle_count(), 12);
        assert!(bvh.node_count() >= 1);
        assert!(bvh.build_stats.leaf_count > 0);
        assert!(bvh.build_stats.max_depth > 0);
        assert!(bvh.world_aabb.is_valid());

        // Verify world AABB covers all vertices
        assert!(bvh.world_aabb.min[0] <= 0.0);
        assert!(bvh.world_aabb.max[0] >= 1.0);
        assert!(bvh.world_aabb.min[1] <= 0.0);
        assert!(bvh.world_aabb.max[1] >= 1.0);
        assert!(bvh.world_aabb.min[2] <= 0.0);
        assert!(bvh.world_aabb.max[2] >= 1.0);
    }
}