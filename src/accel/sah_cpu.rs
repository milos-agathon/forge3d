// src/accel/sah_cpu.rs
// CPU Surface Area Heuristic BVH builder as fallback for GPU LBVH construction.
// This file exists to provide API-compatible CPU BVH building when GPU is unavailable or for small scenes.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/accel/types.rs,src/accel/mod.rs

use crate::accel::types::{Aabb, BuildOptions, BuildStats, BvhHandle, BvhNode, Triangle};
use crate::accel::{BvhBackend, CpuBvhData};
use anyhow::{Context, Result};
use std::time::Instant;

/// CPU SAH-based BVH builder
pub struct CpuSahBuilder {
    // No persistent state needed for CPU builder
}

impl CpuSahBuilder {
    /// Create new CPU SAH builder
    pub fn new() -> Self {
        Self {}
    }

    /// Build BVH from triangles using Surface Area Heuristic
    pub fn build(&mut self, triangles: &[Triangle], options: &BuildOptions) -> Result<BvhHandle> {
        let start_time = Instant::now();

        if triangles.is_empty() {
            anyhow::bail!("Cannot build BVH from empty triangle list");
        }

        let triangle_count = triangles.len() as u32;

        // Compute scene bounds and primitive information
        let world_aabb = crate::accel::types::compute_scene_aabb(triangles);
        let primitive_aabbs = crate::accel::types::compute_triangle_aabbs(triangles);
        let mut primitive_indices: Vec<u32> = (0..triangle_count).collect();

        let mut nodes = Vec::new();
        let mut stats = BuildStats {
            build_time_ms: 0.0,
            morton_time_ms: 0.0, // N/A for SAH
            sort_time_ms: 0.0,   // N/A for SAH
            link_time_ms: 0.0,   // N/A for SAH
            memory_usage_bytes: 0,
            leaf_count: 0,
            internal_count: 0,
            max_depth: 0,
            avg_leaf_size: 0.0,
        };

        // Build BVH recursively using SAH
        let root_info = BuildInfo {
            aabb: world_aabb,
            first: 0,
            count: triangle_count,
            depth: 0,
        };

        let root_node = self.build_recursive(
            &triangles,
            &primitive_aabbs,
            &mut primitive_indices,
            &mut nodes,
            root_info,
            options,
            &mut stats,
        )?;

        // Add root node
        nodes.push(root_node);
        let node_count = nodes.len() as u32;

        // Calculate final statistics
        stats.build_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        stats.memory_usage_bytes = (nodes.len() * std::mem::size_of::<BvhNode>()
            + primitive_indices.len() * std::mem::size_of::<u32>())
            as u64;
        stats.internal_count = node_count - stats.leaf_count;

        if stats.leaf_count > 0 {
            stats.avg_leaf_size = triangle_count as f32 / stats.leaf_count as f32;
        }

        let cpu_data = CpuBvhData {
            nodes,
            indices: primitive_indices,
            world_aabb,
        };

        Ok(BvhHandle {
            backend: BvhBackend::Cpu(cpu_data),
            triangle_count,
            node_count,
            world_aabb,
            build_stats: stats,
        })
    }

    /// Refit existing BVH with updated triangle data
    pub fn refit(&mut self, handle: &mut BvhHandle, triangles: &[Triangle]) -> Result<()> {
        let cpu_data = match &mut handle.backend {
            BvhBackend::Cpu(data) => data,
            BvhBackend::Gpu(_) => anyhow::bail!("Cannot refit GPU BVH with CPU builder"),
        };

        if triangles.len() as u32 != handle.triangle_count {
            anyhow::bail!(
                "Triangle count mismatch: expected {}, got {}",
                handle.triangle_count,
                triangles.len()
            );
        }

        // Update primitive AABBs
        let primitive_aabbs = crate::accel::types::compute_triangle_aabbs(triangles);

        // Refit nodes bottom-up
        self.refit_recursive(&mut cpu_data.nodes, &cpu_data.indices, &primitive_aabbs, 0)?;

        // Update world AABB
        if !cpu_data.nodes.is_empty() {
            cpu_data.world_aabb = cpu_data.nodes[0].aabb;
            handle.world_aabb = cpu_data.world_aabb;
        }

        Ok(())
    }

    /// Recursively build BVH using SAH
    fn build_recursive(
        &self,
        triangles: &[Triangle],
        primitive_aabbs: &[Aabb],
        primitive_indices: &mut [u32],
        nodes: &mut Vec<BvhNode>,
        info: BuildInfo,
        options: &BuildOptions,
        stats: &mut BuildStats,
    ) -> Result<BvhNode> {
        stats.max_depth = stats.max_depth.max(info.depth);

        // Check if we should create a leaf
        if info.count <= options.max_leaf_size || info.depth > 64 {
            stats.leaf_count += 1;
            return Ok(BvhNode::leaf(info.aabb, info.first, info.count));
        }

        // Find best split using SAH
        let split_result = self.find_best_split(
            triangles,
            primitive_aabbs,
            &primitive_indices[info.first as usize..(info.first + info.count) as usize],
            &info.aabb,
            options,
        )?;

        // If no good split found, create leaf
        if split_result.is_none() {
            stats.leaf_count += 1;
            return Ok(BvhNode::leaf(info.aabb, info.first, info.count));
        }

        let split = split_result.unwrap();

        // Partition primitives
        let split_index = self.partition_primitives(
            primitive_indices,
            info.first,
            info.count,
            split.axis,
            split.position,
            primitive_aabbs,
        )?;

        let left_count = split_index - info.first;
        let right_count = info.count - left_count;

        if left_count == 0 || right_count == 0 {
            stats.leaf_count += 1;
            return Ok(BvhNode::leaf(info.aabb, info.first, info.count));
        }

        // Build left and right subtrees
        let left_aabb = self.compute_bounds(
            primitive_aabbs,
            &primitive_indices[info.first as usize..split_index as usize],
        );
        let right_aabb = self.compute_bounds(
            primitive_aabbs,
            &primitive_indices[split_index as usize..(info.first + info.count) as usize],
        );

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

        let left_child = self.build_recursive(
            triangles,
            primitive_aabbs,
            primitive_indices,
            nodes,
            left_info,
            options,
            stats,
        )?;

        let right_child = self.build_recursive(
            triangles,
            primitive_aabbs,
            primitive_indices,
            nodes,
            right_info,
            options,
            stats,
        )?;

        let left_idx = nodes.len() as u32;
        nodes.push(left_child);
        let right_idx = nodes.len() as u32;
        nodes.push(right_child);

        Ok(BvhNode::internal(info.aabb, left_idx, right_idx))
    }

    /// Find best split using Surface Area Heuristic
    fn find_best_split(
        &self,
        _triangles: &[Triangle],
        primitive_aabbs: &[Aabb],
        indices: &[u32],
        parent_aabb: &Aabb,
        options: &BuildOptions,
    ) -> Result<Option<SplitInfo>> {
        if indices.len() < 2 {
            return Ok(None);
        }

        let mut best_split: Option<SplitInfo> = None;
        let mut best_cost = f32::INFINITY;

        // Try splitting on each axis
        for axis in 0..3 {
            // Collect primitive centroids for this axis
            let mut centroids: Vec<f32> = indices
                .iter()
                .map(|&idx| primitive_aabbs[idx as usize].center()[axis])
                .collect();
            centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Try splits between unique centroid values
            for i in 1..centroids.len() {
                if centroids[i] == centroids[i - 1] {
                    continue; // Skip duplicate positions
                }

                let split_pos = (centroids[i] + centroids[i - 1]) * 0.5;

                // Count primitives and compute AABBs for each side
                let (left_aabb, right_aabb, left_count, right_count) =
                    self.evaluate_split(primitive_aabbs, indices, axis, split_pos);

                if left_count == 0 || right_count == 0 {
                    continue;
                }

                // Compute SAH cost
                let parent_sa = parent_aabb.surface_area();
                if parent_sa <= 0.0 {
                    continue;
                }

                let left_sa = left_aabb.surface_area();
                let right_sa = right_aabb.surface_area();

                let cost = options.traversal_cost
                    + options.intersection_cost
                        * ((left_sa / parent_sa) * left_count as f32
                            + (right_sa / parent_sa) * right_count as f32);

                if cost < best_cost {
                    best_cost = cost;
                    best_split = Some(SplitInfo {
                        axis,
                        position: split_pos,
                        cost,
                    });
                }
            }
        }

        // Only return split if it's better than leaf cost
        let leaf_cost = options.intersection_cost * indices.len() as f32;
        if best_cost < leaf_cost {
            Ok(best_split)
        } else {
            Ok(None)
        }
    }

    /// Evaluate a potential split
    fn evaluate_split(
        &self,
        primitive_aabbs: &[Aabb],
        indices: &[u32],
        axis: usize,
        split_pos: f32,
    ) -> (Aabb, Aabb, u32, u32) {
        let mut left_aabb = Aabb::empty();
        let mut right_aabb = Aabb::empty();
        let mut left_count = 0;
        let mut right_count = 0;

        for &idx in indices {
            let aabb = &primitive_aabbs[idx as usize];
            let centroid = aabb.center();

            if centroid[axis] < split_pos {
                left_aabb.expand_aabb(aabb);
                left_count += 1;
            } else {
                right_aabb.expand_aabb(aabb);
                right_count += 1;
            }
        }

        (left_aabb, right_aabb, left_count, right_count)
    }

    /// Partition primitives around split
    fn partition_primitives(
        &self,
        indices: &mut [u32],
        first: u32,
        count: u32,
        axis: usize,
        split_pos: f32,
        primitive_aabbs: &[Aabb],
    ) -> Result<u32> {
        let range = &mut indices[first as usize..(first + count) as usize];

        let mut left = 0;
        let mut right = range.len();

        while left < right {
            let centroid = primitive_aabbs[range[left] as usize].center();
            if centroid[axis] < split_pos {
                left += 1;
            } else {
                right -= 1;
                range.swap(left, right);
            }
        }

        Ok(first + left as u32)
    }

    /// Compute bounding box for a set of primitives
    fn compute_bounds(&self, primitive_aabbs: &[Aabb], indices: &[u32]) -> Aabb {
        let mut aabb = Aabb::empty();
        for &idx in indices {
            aabb.expand_aabb(&primitive_aabbs[idx as usize]);
        }
        aabb
    }

    /// Recursively refit BVH nodes bottom-up
    fn refit_recursive(
        &self,
        nodes: &mut [BvhNode],
        indices: &[u32],
        primitive_aabbs: &[Aabb],
        node_idx: usize,
    ) -> Result<()> {
        if node_idx >= nodes.len() {
            return Ok(());
        }

        let node = nodes[node_idx];

        if node.is_leaf() {
            // Update leaf AABB from primitives
            let (first, count) = node.primitives().unwrap();
            let mut aabb = Aabb::empty();

            for i in 0..count {
                let prim_idx = indices[(first + i) as usize];
                aabb.expand_aabb(&primitive_aabbs[prim_idx as usize]);
            }

            nodes[node_idx].aabb = aabb;
        } else {
            // Update internal node from children
            let (left_idx, right_idx) = node.children().unwrap();

            // Recursively refit children first
            self.refit_recursive(nodes, indices, primitive_aabbs, left_idx as usize)?;
            self.refit_recursive(nodes, indices, primitive_aabbs, right_idx as usize)?;

            // Update this node's AABB from children
            let mut aabb = Aabb::empty();
            aabb.expand_aabb(&nodes[left_idx as usize].aabb);
            aabb.expand_aabb(&nodes[right_idx as usize].aabb);

            nodes[node_idx].aabb = aabb;
        }

        Ok(())
    }
}

impl Default for CpuSahBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Information for recursive BVH construction
struct BuildInfo {
    aabb: Aabb,
    first: u32,
    count: u32,
    depth: u32,
}

/// Information about a potential split
struct SplitInfo {
    axis: usize,
    position: f32,
    cost: f32,
}
