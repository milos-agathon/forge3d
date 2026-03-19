use super::{Aabb, BuildOptions, BuildStats, BvhCPU, BvhNode, MeshCPU};
use anyhow::{Context, Result};
use std::time::Instant;

/// Build BVH from mesh using specified method
pub fn build_bvh_cpu(mesh: &MeshCPU, options: &BuildOptions) -> Result<BvhCPU> {
    let start_time = Instant::now();

    if mesh.indices.is_empty() {
        anyhow::bail!("Cannot build BVH from empty mesh");
    }

    let triangle_count = mesh.triangle_count();

    let mut tri_aabbs = Vec::with_capacity(triangle_count as usize);
    let mut tri_centroids = Vec::with_capacity(triangle_count as usize);

    for i in 0..triangle_count {
        let aabb = mesh
            .triangle_aabb(i as usize)
            .context("Failed to compute triangle AABB")?;
        let centroid = mesh
            .triangle_centroid(i as usize)
            .context("Failed to compute triangle centroid")?;
        tri_aabbs.push(aabb);
        tri_centroids.push(centroid);
    }

    let mut world_aabb = Aabb::empty();
    for aabb in &tri_aabbs {
        world_aabb.expand_aabb(aabb);
    }

    let mut tri_indices: Vec<u32> = (0..triangle_count).collect();
    let mut nodes = Vec::new();
    let mut stats = BuildStats {
        triangle_count,
        ..Default::default()
    };

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

    stats.build_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
    stats.node_count = nodes.len() as u32;
    stats.internal_count = stats.node_count - stats.leaf_count;
    stats.memory_usage_bytes = (nodes.len() * std::mem::size_of::<BvhNode>()
        + tri_indices.len() * std::mem::size_of::<u32>()) as u64;

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

struct BuildInfo {
    aabb: Aabb,
    first: u32,
    count: u32,
    depth: u32,
}

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

    if info.count <= options.max_leaf_size || info.depth > 64 {
        stats.leaf_count += 1;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::leaf(info.aabb, info.first, info.count));
        return Ok(node_idx);
    }

    let Some((split_axis, split_pos)) = find_median_split(
        tri_centroids,
        &tri_indices[info.first as usize..(info.first + info.count) as usize],
        &info.aabb,
    ) else {
        stats.leaf_count += 1;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::leaf(info.aabb, info.first, info.count));
        return Ok(node_idx);
    };

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
        stats.leaf_count += 1;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::leaf(info.aabb, info.first, info.count));
        return Ok(node_idx);
    }

    let left_aabb = compute_bounds(
        tri_aabbs,
        &tri_indices[info.first as usize..split_index as usize],
    );
    let right_aabb = compute_bounds(
        tri_aabbs,
        &tri_indices[split_index as usize..(info.first + info.count) as usize],
    );

    let left_child_idx = build_recursive(
        tri_aabbs,
        tri_centroids,
        tri_indices,
        nodes,
        BuildInfo {
            aabb: left_aabb,
            first: info.first,
            count: left_count,
            depth: info.depth + 1,
        },
        options,
        stats,
    )?;

    let right_child_idx = build_recursive(
        tri_aabbs,
        tri_centroids,
        tri_indices,
        nodes,
        BuildInfo {
            aabb: right_aabb,
            first: split_index,
            count: right_count,
            depth: info.depth + 1,
        },
        options,
        stats,
    )?;

    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode::internal(
        info.aabb,
        left_child_idx,
        right_child_idx,
    ));

    Ok(node_idx)
}

fn find_median_split(
    tri_centroids: &[[f32; 3]],
    indices: &[u32],
    parent_aabb: &Aabb,
) -> Option<(usize, f32)> {
    if indices.len() < 2 {
        return None;
    }

    let extent = parent_aabb.extent();
    let axis = if extent[0] > extent[1] && extent[0] > extent[2] {
        0
    } else if extent[1] > extent[2] {
        1
    } else {
        2
    };

    let mut centroids: Vec<f32> = indices
        .iter()
        .map(|&idx| tri_centroids[idx as usize][axis])
        .collect();
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_idx = centroids.len() / 2;
    Some((axis, centroids[median_idx]))
}

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

fn compute_bounds(tri_aabbs: &[Aabb], indices: &[u32]) -> Aabb {
    let mut aabb = Aabb::empty();
    for &idx in indices {
        aabb.expand_aabb(&tri_aabbs[idx as usize]);
    }
    aabb
}
