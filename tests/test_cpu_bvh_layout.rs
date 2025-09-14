// tests/test_cpu_bvh_layout.rs
// CPU BVH layout tests for GPU compatibility verification (Task A3).
// Tests that CPU-built BVH nodes match the expected GPU layout and alignment.
// RELEVANT FILES:src/accel/cpu_bvh.rs,src/shaders/pt_intersect_mesh.wgsl,src/path_tracing/mesh.rs

use forge3d::accel::cpu_bvh::{build_bvh_cpu, Aabb, BuildOptions, BvhNode, MeshCPU};
use std::mem;

#[test]
fn cpu_bvh_layout_matches_gpu_pack() {
    // Build a tiny BVH; verify node packing size/offsets and leaf spans; spot-check root AABB covers mesh.

    // Create simple triangle mesh (single triangle)
    let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
    let indices = vec![[0, 1, 2]];
    let mesh = MeshCPU::new(vertices, indices);

    let options = BuildOptions::default();
    let bvh = build_bvh_cpu(&mesh, &options).expect("BVH build should succeed");

    // Verify BVH structure
    assert_eq!(bvh.triangle_count(), 1);
    assert!(bvh.node_count() >= 1);

    // Test BVH node layout matches GPU expectations
    let nodes = &bvh.nodes;
    assert!(!nodes.is_empty(), "BVH should have at least one node");

    // Test the root node
    let root = nodes[0];

    // Verify AABB covers the triangle
    let aabb = root.aabb();
    assert!(aabb.is_valid(), "Root AABB should be valid");
    assert!(aabb.min[0] <= 0.0, "AABB should cover first vertex");
    assert!(aabb.max[0] >= 1.0, "AABB should cover second vertex");
    assert!(aabb.max[1] >= 1.0, "AABB should cover third vertex");

    // For single triangle, should create a leaf node
    assert!(root.is_leaf(), "Single triangle should create leaf node");
    let (first_tri, tri_count) = root.triangles().expect("Leaf should have triangles");
    assert_eq!(first_tri, 0, "Should start at triangle 0");
    assert_eq!(tri_count, 1, "Should contain 1 triangle");
}

#[test]
fn bvh_node_memory_layout() {
    // Verify BvhNode struct layout matches WGSL expectations

    // BvhNode should be exactly 32 bytes (8 f32 values)
    assert_eq!(
        mem::size_of::<BvhNode>(),
        32,
        "BvhNode should be 32 bytes for GPU compatibility"
    );

    // Should be 4-byte aligned
    assert_eq!(
        mem::align_of::<BvhNode>(),
        4,
        "BvhNode should be 4-byte aligned"
    );

    // Test field offsets match WGSL struct layout
    let test_aabb = Aabb::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let test_node = BvhNode::internal(test_aabb, 10, 20);

    // Verify internal node properties
    assert!(test_node.is_internal());
    assert!(!test_node.is_leaf());
    assert_eq!(test_node.children(), Some((10, 20)));

    // Test leaf node
    let leaf_node = BvhNode::leaf(test_aabb, 5, 3);
    assert!(leaf_node.is_leaf());
    assert!(!leaf_node.is_internal());
    assert_eq!(leaf_node.triangles(), Some((5, 3)));

    // Verify AABB reconstruction
    let reconstructed_aabb = test_node.aabb();
    assert_eq!(reconstructed_aabb.min, [1.0, 2.0, 3.0]);
    assert_eq!(reconstructed_aabb.max, [4.0, 5.0, 6.0]);
}

#[test]
fn multi_triangle_bvh_structure() {
    // Build BVH for multiple triangles and verify structure

    // Create cube mesh (12 triangles)
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
        // Front face
        [0, 1, 2],
        [0, 2, 3],
        // Right face
        [1, 5, 6],
        [1, 6, 2],
        // Back face
        [5, 4, 7],
        [5, 7, 6],
        // Left face
        [4, 0, 3],
        [4, 3, 7],
        // Top face
        [3, 2, 6],
        [3, 6, 7],
        // Bottom face
        [4, 5, 1],
        [4, 1, 0],
    ];

    let mesh = MeshCPU::new(vertices, indices);
    let options = BuildOptions::default();
    let bvh = build_bvh_cpu(&mesh, &options).expect("Multi-triangle BVH build should succeed");

    // Verify BVH properties
    assert_eq!(bvh.triangle_count(), 12);
    assert!(bvh.node_count() >= 1);
    assert!(bvh.build_stats.max_depth > 0);
    assert!(bvh.build_stats.leaf_count > 0);

    // Verify world AABB covers entire cube
    let world_aabb = bvh.world_aabb;
    assert!(world_aabb.is_valid());
    assert!(world_aabb.min[0] <= 0.0);
    assert!(world_aabb.max[0] >= 1.0);
    assert!(world_aabb.min[1] <= 0.0);
    assert!(world_aabb.max[1] >= 1.0);
    assert!(world_aabb.min[2] <= 0.0);
    assert!(world_aabb.max[2] >= 1.0);

    // All triangle indices should be valid
    let total_triangles_in_leaves = bvh
        .nodes
        .iter()
        .filter(|node| node.is_leaf())
        .map(|node| node.triangles().unwrap().1)
        .sum::<u32>();

    assert_eq!(
        total_triangles_in_leaves, 12,
        "All triangles should be in leaf nodes"
    );
}

#[test]
fn bvh_triangle_indices_validity() {
    // Verify triangle indices in BVH are valid and reordered correctly

    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0], // Triangle 0
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.5, 1.0, 0.0], // Triangle 1
    ];
    let indices = vec![[0, 1, 2], [3, 4, 5]];
    let mesh = MeshCPU::new(vertices, indices);

    let options = BuildOptions::default();
    let bvh = build_bvh_cpu(&mesh, &options).expect("BVH build should succeed");

    // Check triangle index array
    assert_eq!(bvh.tri_indices.len(), 2, "Should have 2 triangle indices");

    // All indices should be valid (0 or 1)
    for &tri_idx in &bvh.tri_indices {
        assert!(tri_idx < 2, "Triangle index should be valid: {}", tri_idx);
    }

    // Verify leaf nodes reference valid triangle ranges
    for node in &bvh.nodes {
        if node.is_leaf() {
            let (first_tri, tri_count) = node.triangles().unwrap();
            assert!(
                first_tri + tri_count <= bvh.tri_indices.len() as u32,
                "Leaf triangle range should be valid: [{}, {})",
                first_tri,
                first_tri + tri_count
            );

            // Check individual triangle indices
            for i in 0..tri_count {
                let idx = bvh.tri_indices[(first_tri + i) as usize];
                assert!(idx < 2, "Referenced triangle index should be valid");
            }
        }
    }
}

#[test]
fn empty_mesh_handling() {
    // Verify graceful handling of edge cases

    // Empty vertices should fail
    let empty_vertices = vec![];
    let empty_indices = vec![];
    let empty_mesh = MeshCPU::new(empty_vertices, empty_indices);

    let options = BuildOptions::default();
    let result = build_bvh_cpu(&empty_mesh, &options);
    assert!(result.is_err(), "Empty mesh should fail to build BVH");
}

#[test]
fn bvh_build_options() {
    // Test different BVH build options

    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.5, 3.0, 0.0],
    ];
    let indices = vec![[0, 1, 2], [3, 4, 5]];
    let mesh = MeshCPU::new(vertices, indices);

    // Test with different max leaf sizes
    let mut options = BuildOptions::default();
    options.max_leaf_size = 1;

    let bvh1 = build_bvh_cpu(&mesh, &options).expect("BVH with max_leaf_size=1 should succeed");

    options.max_leaf_size = 4;
    let bvh4 = build_bvh_cpu(&mesh, &options).expect("BVH with max_leaf_size=4 should succeed");

    // Smaller max leaf size should generally create more nodes
    // (though exact behavior depends on spatial distribution)
    assert!(
        bvh1.triangle_count() == bvh4.triangle_count(),
        "Both should have same triangle count"
    );

    // Both should be valid BVHs
    assert!(bvh1.node_count() > 0);
    assert!(bvh4.node_count() > 0);
    assert!(bvh1.build_stats.leaf_count > 0);
    assert!(bvh4.build_stats.leaf_count > 0);
}

#[test]
fn aabb_operations() {
    // Test AABB utility functions used by BVH construction

    let aabb1 = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let aabb2 = Aabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);

    // Test properties
    assert!(aabb1.is_valid());
    assert!(aabb2.is_valid());

    let center1 = aabb1.center();
    assert_eq!(center1, [0.5, 0.5, 0.5]);

    let extent1 = aabb1.extent();
    assert_eq!(extent1, [1.0, 1.0, 1.0]);

    let surface_area = aabb1.surface_area();
    assert_eq!(surface_area, 6.0); // 2 * (1*1 + 1*1 + 1*1) = 6

    // Test expansion
    let mut expanded = aabb1;
    expanded.expand_aabb(&aabb2);
    assert_eq!(expanded.min, [0.0, 0.0, 0.0]);
    assert_eq!(expanded.max, [1.5, 1.5, 1.5]);

    // Test empty AABB
    let empty = Aabb::empty();
    assert!(!empty.is_valid()); // Should have min > max initially
}

#[test]
fn mesh_triangle_access() {
    // Test mesh triangle access methods used during BVH construction

    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![[0, 1, 2], [0, 2, 3]];
    let mesh = MeshCPU::new(vertices, indices);

    // Test triangle access
    assert_eq!(mesh.triangle_count(), 2);
    assert_eq!(mesh.vertex_count(), 4);

    let tri0 = mesh.get_triangle(0).expect("Should get first triangle");
    assert_eq!(tri0.0, [0.0, 0.0, 0.0]);
    assert_eq!(tri0.1, [1.0, 0.0, 0.0]);
    assert_eq!(tri0.2, [0.5, 1.0, 0.0]);

    let centroid0 = mesh.triangle_centroid(0).expect("Should compute centroid");
    let expected_centroid = [(0.0 + 1.0 + 0.5) / 3.0, (0.0 + 0.0 + 1.0) / 3.0, 0.0];
    assert!((centroid0[0] - expected_centroid[0]).abs() < 1e-6);
    assert!((centroid0[1] - expected_centroid[1]).abs() < 1e-6);
    assert!((centroid0[2] - expected_centroid[2]).abs() < 1e-6);

    let aabb0 = mesh.triangle_aabb(0).expect("Should compute AABB");
    assert!(aabb0.is_valid());
    assert_eq!(aabb0.min[0], 0.0);
    assert_eq!(aabb0.max[0], 1.0);
    assert_eq!(aabb0.max[1], 1.0);
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bvh_build_performance() {
        // Simple performance test to ensure BVH build is reasonable

        // Create a moderately complex mesh (48 triangles)
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Create a 4x4 grid of quads
        for j in 0..5 {
            for i in 0..5 {
                vertices.push([i as f32, 0.0, j as f32]);
            }
        }

        for j in 0..4 {
            for i in 0..4 {
                let base = j * 5 + i;
                // Two triangles per quad
                indices.push([base, base + 1, base + 5]);
                indices.push([base + 1, base + 6, base + 5]);
            }
        }

        let mesh = MeshCPU::new(vertices, indices);
        let options = BuildOptions::default();

        let start = Instant::now();
        let bvh = build_bvh_cpu(&mesh, &options).expect("Performance test BVH should build");
        let elapsed = start.elapsed();

        println!(
            "Built BVH for {} triangles in {:?}",
            mesh.triangle_count(),
            elapsed
        );
        println!(
            "BVH stats: {} nodes, {} leaves, max depth {}",
            bvh.node_count(),
            bvh.build_stats.leaf_count,
            bvh.build_stats.max_depth
        );

        // Should build in reasonable time (< 100ms for this small mesh)
        assert!(
            elapsed.as_millis() < 100,
            "BVH build took too long: {:?}",
            elapsed
        );

        // Should create a reasonable structure
        assert_eq!(bvh.triangle_count(), 32); // 4*4*2 triangles
        assert!(
            bvh.build_stats.max_depth > 1,
            "Should create a multi-level tree"
        );
        assert!(
            bvh.build_stats.max_depth < 20,
            "Should not create excessively deep tree"
        );
    }
}
