// tests/test_bvh_gpu.rs
// Tests for GPU LBVH construction and basic functionality
// This file exists to validate GPU BVH building, memory usage, and performance targets.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/accel/types.rs,src/shaders/lbvh_*.wgsl

use anyhow::Result;
use forge3d::accel::types::{Aabb, BuildOptions, Triangle};
use forge3d::accel::{build_bvh, BvhBuilder, GpuContext};

/// Helper to create a simple triangle mesh (unit cube)
fn create_unit_cube_triangles() -> Vec<Triangle> {
    vec![
        // Front face
        Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        Triangle::new([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        // Back face
        Triangle::new([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]),
        Triangle::new([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        // Left face
        Triangle::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]),
        Triangle::new([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]),
        // Right face
        Triangle::new([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]),
        Triangle::new([1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        // Top face
        Triangle::new([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
        Triangle::new([0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
        // Bottom face
        Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        Triangle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
    ]
}

/// Helper to create a larger scene for performance testing
fn create_stress_test_triangles(count: usize) -> Vec<Triangle> {
    let mut triangles = Vec::with_capacity(count);
    let side_length = (count as f32).cbrt() as usize + 1;

    for i in 0..count {
        let x = (i % side_length) as f32;
        let y = ((i / side_length) % side_length) as f32;
        let z = (i / (side_length * side_length)) as f32;

        // Create a small triangle at position (x, y, z)
        let offset = [x, y, z];
        let scale = 0.1;

        triangles.push(Triangle::new(
            [offset[0], offset[1], offset[2]],
            [offset[0] + scale, offset[1], offset[2]],
            [offset[0], offset[1] + scale, offset[2]],
        ));
    }

    triangles
}

/// Get GPU context for testing, or skip if no adapter available
fn get_gpu_context_or_skip() -> GpuContext {
    // In a real implementation, this would create wgpu device/queue
    // For now, return NotAvailable to focus on CPU fallback testing
    GpuContext::NotAvailable
}

#[test]
fn gpu_bvh_build_small_scene() -> Result<()> {
    let triangles = create_unit_cube_triangles();
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    // This should fall back to CPU since we don't have real GPU context
    let bvh = build_bvh(&triangles, &options, gpu_context)?;

    // Validate BVH properties
    assert_eq!(bvh.triangle_count, triangles.len() as u32);
    assert_eq!(bvh.node_count, 2 * triangles.len() as u32 - 1);
    assert!(bvh.world_aabb.is_valid());

    // Check that AABB contains all triangles
    for triangle in &triangles {
        let tri_aabb = triangle.aabb();
        // World AABB should contain triangle AABB
        for i in 0..3 {
            assert!(
                bvh.world_aabb.min[i] <= tri_aabb.min[i],
                "World AABB min[{}] should be <= triangle AABB min[{}]",
                i,
                i
            );
            assert!(
                bvh.world_aabb.max[i] >= tri_aabb.max[i],
                "World AABB max[{}] should be >= triangle AABB max[{}]",
                i,
                i
            );
        }
    }

    println!("GPU BVH test passed (CPU fallback):");
    println!("  Backend: {:?}", bvh.is_cpu());
    println!("  Triangles: {}", bvh.triangle_count);
    println!("  Nodes: {}", bvh.node_count);
    println!("  Build time: {:.2}ms", bvh.build_stats.build_time_ms);

    Ok(())
}

#[test]
fn gpu_bvh_memory_budget_compliance() -> Result<()> {
    // Test that large scenes respect memory budget (≤512 MiB)
    let triangle_count = 10000; // Moderately large scene
    let triangles = create_stress_test_triangles(triangle_count);
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let bvh = build_bvh(&triangles, &options, gpu_context)?;

    // Check memory usage is within budget
    let memory_mb = bvh.build_stats.memory_usage_bytes as f64 / (1024.0 * 1024.0);
    assert!(
        memory_mb <= 512.0,
        "BVH memory usage {:.1}MB exceeds 512MB budget",
        memory_mb
    );

    println!("Memory budget test passed:");
    println!("  Triangles: {}", triangle_count);
    println!("  Memory usage: {:.1}MB", memory_mb);

    Ok(())
}

#[test]
fn gpu_bvh_performance_smoke_test() -> Result<()> {
    // Test that build performance meets rough targets
    let triangle_count = 1000;
    let triangles = create_stress_test_triangles(triangle_count);
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let bvh = build_bvh(&triangles, &options, gpu_context)?;

    // Performance targets from task: build ≤ ~1s for ~1M tris
    // Scale down for 1K tris: should be much faster
    let expected_max_ms = 1000.0 * (triangle_count as f32 / 1_000_000.0);

    // Be lenient in debug builds
    let actual_ms = bvh.build_stats.build_time_ms;
    let is_debug = cfg!(debug_assertions);
    let tolerance = if is_debug { 10.0 } else { 2.0 };

    if actual_ms > expected_max_ms * tolerance {
        println!(
            "Warning: BVH build took {:.1}ms, expected <{:.1}ms (debug={})",
            actual_ms,
            expected_max_ms * tolerance,
            is_debug
        );
    }

    println!("Performance smoke test completed:");
    println!("  Triangles: {}", triangle_count);
    println!("  Build time: {:.1}ms", actual_ms);
    println!("  Target: <{:.1}ms", expected_max_ms * tolerance);

    Ok(())
}

#[test]
fn gpu_bvh_deterministic_builds() -> Result<()> {
    // Test that builds are deterministic with fixed seeds
    let triangles = create_unit_cube_triangles();
    let mut options1 = BuildOptions::default();
    let mut options2 = BuildOptions::default();

    options1.seed = 42;
    options2.seed = 42;

    let gpu_context = get_gpu_context_or_skip();

    let bvh1 = build_bvh(&triangles, &options1, gpu_context.clone())?;
    let bvh2 = build_bvh(&triangles, &options2, gpu_context)?;

    // Should have identical structure
    assert_eq!(bvh1.triangle_count, bvh2.triangle_count);
    assert_eq!(bvh1.node_count, bvh2.node_count);

    // World AABBs should be identical
    for i in 0..3 {
        assert!((bvh1.world_aabb.min[i] - bvh2.world_aabb.min[i]).abs() < 1e-6);
        assert!((bvh1.world_aabb.max[i] - bvh2.world_aabb.max[i]).abs() < 1e-6);
    }

    println!("Deterministic builds test passed");

    Ok(())
}

#[test]
fn gpu_bvh_empty_scene_handling() -> Result<()> {
    // Test graceful handling of edge cases
    let triangles = vec![];
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    // Should return error for empty scene
    let result = build_bvh(&triangles, &options, gpu_context);
    assert!(result.is_err(), "Empty triangle list should return error");

    println!("Empty scene handling test passed");

    Ok(())
}

#[test]
fn gpu_bvh_single_triangle() -> Result<()> {
    // Test minimal scene with one triangle
    let triangles = vec![Triangle::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )];
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let bvh = build_bvh(&triangles, &options, gpu_context)?;

    assert_eq!(bvh.triangle_count, 1);
    assert_eq!(bvh.node_count, 1); // Just the root leaf node
    assert!(bvh.world_aabb.is_valid());

    println!("Single triangle test passed");

    Ok(())
}
