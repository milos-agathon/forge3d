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
