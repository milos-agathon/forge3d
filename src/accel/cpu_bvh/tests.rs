use super::*;

#[test]
fn test_bvh_node_layout() {
    assert_eq!(std::mem::size_of::<BvhNode>(), 40);
    assert_eq!(std::mem::align_of::<BvhNode>(), 4);

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
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
    let indices = vec![[0, 1, 2]];

    let mesh = MeshCPU::new(vertices, indices);
    assert_eq!(mesh.triangle_count(), 1);

    let (v0, v1, v2) = mesh.get_triangle(0).unwrap();
    assert_eq!(v0, [0.0, 0.0, 0.0]);
    assert_eq!(v1, [1.0, 0.0, 0.0]);
    assert_eq!(v2, [0.5, 1.0, 0.0]);

    let centroid = mesh.triangle_centroid(0).unwrap();
    assert!((centroid[0] - 0.5).abs() < 1e-6);
    assert!((centroid[1] - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(centroid[2], 0.0);
}

#[test]
fn test_bvh_build_single_triangle() {
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
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
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let indices = vec![
        [0, 1, 2],
        [0, 2, 3],
        [1, 5, 6],
        [1, 6, 2],
        [5, 4, 7],
        [5, 7, 6],
        [4, 0, 3],
        [4, 3, 7],
        [3, 2, 6],
        [3, 6, 7],
        [4, 5, 1],
        [4, 1, 0],
    ];

    let mesh = MeshCPU::new(vertices, indices);
    let options = BuildOptions::default();
    let bvh = build_bvh_cpu(&mesh, &options).unwrap();

    assert_eq!(bvh.triangle_count(), 12);
    assert!(bvh.node_count() >= 1);
    assert!(bvh.build_stats.leaf_count > 0);
    assert!(bvh.build_stats.max_depth > 0);
    assert!(bvh.world_aabb.is_valid());
    assert!(bvh.world_aabb.min[0] <= 0.0);
    assert!(bvh.world_aabb.max[0] >= 1.0);
    assert!(bvh.world_aabb.min[1] <= 0.0);
    assert!(bvh.world_aabb.max[1] >= 1.0);
    assert!(bvh.world_aabb.min[2] <= 0.0);
    assert!(bvh.world_aabb.max[2] >= 1.0);
}

/// The packed node layout is the `#[repr(C)]` `Pod` layout: writing a node to
/// bytes and reading it back through the same layout must be lossless, and the
/// stride must stay 40 bytes because `forge3d.mesh` round-trips it to Python.
#[test]
fn node_bytes_round_trip_through_pod_layout() {
    let internal = BvhNode::internal(Aabb::new([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]), 7, 9);
    let leaf = BvhNode::leaf(Aabb::new([-1.0, -2.0, -3.0], [0.5, 0.25, 0.125]), 2, 4);
    for node in [internal, leaf] {
        let bytes = bytemuck::bytes_of(&node);
        assert_eq!(bytes.len(), std::mem::size_of::<BvhNode>());
        let read = bytemuck::pod_read_unaligned::<BvhNode>(bytes);
        assert_eq!(read.aabb_min, node.aabb_min);
        assert_eq!(read.aabb_max, node.aabb_max);
        assert_eq!(read.left, node.left);
        assert_eq!(read.right, node.right);
        assert_eq!(read.flags, node.flags);
    }
    assert_eq!(std::mem::size_of::<BvhNode>(), 40);
}

/// A clustered mesh must produce a measured tree, not the retired
/// `(triangle_count + 3) // 4 * 2 - 1` estimate, and every triangle must sit in
/// exactly one leaf. This is the invariant `forge3d.mesh` now reports.
#[test]
fn clustered_mesh_build_is_measured_and_covers_every_triangle() {
    let mut vertices = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    // 12 well-separated clusters: a 6/6 median split, then 3/3, so four leaves
    // of three triangles each. The retired formula predicts ceil(12/4) = 3.
    // The x offset is accumulated in f32 so the test adds no narrowing cast.
    let mut x = 0.0f32;
    for _ in 0..12u32 {
        let base = vertices.len() as u32;
        for corner in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            vertices.push([x + corner[0], corner[1], corner[2]]);
        }
        indices.push([base, base + 1, base + 2]);
        x += 1000.0;
    }

    let mesh = MeshCPU::new(vertices, indices);
    let bvh = build_bvh_cpu(&mesh, &BuildOptions::default()).unwrap();
    let stats = &bvh.build_stats;
    let triangle_count = stats.triangle_count;

    // A full binary tree: internal = leaves - 1.
    assert_eq!(stats.node_count, 2 * stats.leaf_count - 1);
    assert_eq!(stats.internal_count, stats.leaf_count - 1);
    assert!(stats.max_depth > 1);

    let retired_nodes = {
        let retired_leaves = ((triangle_count + 3) / 4).max(1);
        retired_leaves * 2 - 1
    };
    assert_ne!(
        stats.node_count, retired_nodes,
        "node count matched the retired estimate {retired_nodes}"
    );

    // Every triangle is covered exactly once across the leaves.
    let mut coverage = vec![0u32; triangle_count as usize];
    for node in &bvh.nodes {
        if !node.is_leaf() {
            continue;
        }
        let first = node.left;
        let count = node.right;
        assert!(count > 0 && count <= 4, "leaf size {count} out of range");
        for offset in 0..count {
            coverage[(first + offset) as usize] += 1;
        }
    }
    assert!(coverage.iter().all(|&seen| seen == 1), "triangle coverage: {coverage:?}");

    // The triangle order is a permutation of the triangle indices.
    let mut order = bvh.tri_indices.clone();
    order.sort_unstable();
    assert_eq!(order, (0..triangle_count).collect::<Vec<_>>());
}

