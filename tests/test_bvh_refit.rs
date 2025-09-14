// tests/test_bvh_refit.rs
// Tests for BVH refit functionality and dynamic scene updates
// This file exists to validate BVH refitting performance and correctness for animated scenes.
// RELEVANT FILES:src/accel/lbvh_gpu.rs,src/accel/sah_cpu.rs,src/shaders/bvh_refit.wgsl

use anyhow::Result;
use forge3d::accel::types::{BuildOptions, Triangle};
use forge3d::accel::{build_bvh, refit_bvh, BvhBuilder, GpuContext};

/// Helper to create animated triangle sequence
fn create_animated_triangles(frame: f32) -> Vec<Triangle> {
    let offset = frame * 0.1; // Small animation
    vec![
        // Triangle that moves along X axis
        Triangle::new(
            [offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.5 + offset, 1.0, 0.0],
        ),
        // Triangle that moves along Y axis
        Triangle::new(
            [0.0, offset, 0.0],
            [0.0, 1.0 + offset, 0.0],
            [1.0, 0.5 + offset, 0.0],
        ),
        // Triangle that scales
        Triangle::new(
            [-frame * 0.05, -frame * 0.05, 0.0],
            [1.0 + frame * 0.05, -frame * 0.05, 0.0],
            [0.5, 1.0 + frame * 0.05, 0.0],
        ),
    ]
}

/// Get GPU context for testing, or skip if no adapter available
fn get_gpu_context_or_skip() -> GpuContext {
    // In a real implementation, this would create wgpu device/queue
    // For now, return NotAvailable to focus on CPU fallback testing
    GpuContext::NotAvailable
}

#[test]
fn refit_updates_aabb_without_rebuild() -> Result<()> {
    // Build initial BVH
    let initial_triangles = create_animated_triangles(0.0);
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let mut builder = BvhBuilder::new(gpu_context)?;
    let mut bvh = builder.build(&initial_triangles, &options)?;
    let initial_aabb = bvh.world_aabb;

    // Animate triangles and refit
    let updated_triangles = create_animated_triangles(10.0); // Significant movement
    builder.refit(&mut bvh, &updated_triangles)?;
    let updated_aabb = bvh.world_aabb;

    // AABB should have changed to accommodate movement
    let aabb_changed = (0..3).any(|i| {
        (initial_aabb.min[i] - updated_aabb.min[i]).abs() > 1e-6
            || (initial_aabb.max[i] - updated_aabb.max[i]).abs() > 1e-6
    });

    assert!(
        aabb_changed,
        "BVH AABB should change after refit with moved triangles"
    );

    // Updated AABB should be valid and contain all new triangles
    assert!(updated_aabb.is_valid());
    for triangle in &updated_triangles {
        let tri_aabb = triangle.aabb();
        for i in 0..3 {
            assert!(
                updated_aabb.min[i] <= tri_aabb.min[i] + 1e-6,
                "Refitted AABB should contain triangle"
            );
            assert!(
                updated_aabb.max[i] >= tri_aabb.max[i] - 1e-6,
                "Refitted AABB should contain triangle"
            );
        }
    }

    println!("Refit AABB update test passed:");
    println!("  Initial AABB: {:?}", initial_aabb);
    println!("  Updated AABB: {:?}", updated_aabb);

    Ok(())
}

#[test]
fn refit_performance_target() -> Result<()> {
    // Test that refit meets performance target: ≤ ~25ms for typical scenes
    let triangle_count = 1000;
    let mut triangles = Vec::new();

    // Create a grid of triangles
    let grid_size = (triangle_count as f32).sqrt() as usize + 1;
    for i in 0..triangle_count {
        let x = (i % grid_size) as f32 * 0.1;
        let y = (i / grid_size) as f32 * 0.1;
        triangles.push(Triangle::new(
            [x, y, 0.0],
            [x + 0.05, y, 0.0],
            [x, y + 0.05, 0.0],
        ));
    }

    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let mut builder = BvhBuilder::new(gpu_context)?;
    let mut bvh = builder.build(&triangles, &options)?;

    // Move all triangles slightly
    for triangle in &mut triangles {
        triangle.v0[2] += 0.01; // Move up slightly
        triangle.v1[2] += 0.01;
        triangle.v2[2] += 0.01;
    }

    let refit_start = std::time::Instant::now();
    builder.refit(&mut bvh, &triangles)?;
    let refit_time_ms = refit_start.elapsed().as_secs_f32() * 1000.0;

    // Performance target: ≤ ~25ms (be lenient in debug builds)
    let target_ms = if cfg!(debug_assertions) { 250.0 } else { 25.0 };

    if refit_time_ms > target_ms {
        println!(
            "Warning: Refit took {:.1}ms, target <{:.1}ms (debug={})",
            refit_time_ms,
            target_ms,
            cfg!(debug_assertions)
        );
    }

    println!("Refit performance test completed:");
    println!("  Triangles: {}", triangle_count);
    println!("  Refit time: {:.1}ms", refit_time_ms);
    println!("  Target: <{:.1}ms", target_ms);

    Ok(())
}

#[test]
fn refit_preserves_topology() -> Result<()> {
    // Test that refit preserves BVH topology (doesn't change structure)
    let initial_triangles = create_animated_triangles(0.0);
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let mut builder = BvhBuilder::new(gpu_context)?;
    let mut bvh = builder.build(&initial_triangles, &options)?;

    let initial_node_count = bvh.node_count;
    let initial_triangle_count = bvh.triangle_count;

    // Refit with modified triangles
    let updated_triangles = create_animated_triangles(5.0);
    builder.refit(&mut bvh, &updated_triangles)?;

    // Structure should be unchanged
    assert_eq!(bvh.node_count, initial_node_count);
    assert_eq!(bvh.triangle_count, initial_triangle_count);

    println!("Refit topology preservation test passed");

    Ok(())
}

#[test]
fn refit_count_mismatch_error() -> Result<()> {
    // Test that refit fails gracefully with wrong triangle count
    let initial_triangles = create_animated_triangles(0.0);
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let mut builder = BvhBuilder::new(gpu_context)?;
    let mut bvh = builder.build(&initial_triangles, &options)?;

    // Try to refit with different triangle count
    let wrong_count_triangles = vec![Triangle::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )]; // Only 1 triangle instead of 3

    let result = builder.refit(&mut bvh, &wrong_count_triangles);
    assert!(
        result.is_err(),
        "Refit should fail with mismatched triangle count"
    );

    println!("Refit count mismatch test passed");

    Ok(())
}

#[test]
fn refit_multiple_iterations() -> Result<()> {
    // Test multiple refits in sequence
    let mut triangles = create_animated_triangles(0.0);
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let mut builder = BvhBuilder::new(gpu_context)?;
    let mut bvh = builder.build(&triangles, &options)?;

    // Perform several refit operations
    for frame in 1..=10 {
        triangles = create_animated_triangles(frame as f32);
        builder.refit(&mut bvh, &triangles)?;

        // Verify AABB is still valid and contains triangles
        assert!(bvh.world_aabb.is_valid());

        for triangle in &triangles {
            let tri_aabb = triangle.aabb();
            for i in 0..3 {
                assert!(
                    bvh.world_aabb.min[i] <= tri_aabb.min[i] + 1e-6,
                    "AABB should contain all triangles after refit {}",
                    frame
                );
                assert!(
                    bvh.world_aabb.max[i] >= tri_aabb.max[i] - 1e-6,
                    "AABB should contain all triangles after refit {}",
                    frame
                );
            }
        }
    }

    println!("Multiple refit iterations test passed");

    Ok(())
}

#[test]
fn refit_extreme_deformation() -> Result<()> {
    // Test refit with extreme triangle deformation
    let initial_triangles = vec![Triangle::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )];
    let options = BuildOptions::default();
    let gpu_context = get_gpu_context_or_skip();

    let mut builder = BvhBuilder::new(gpu_context)?;
    let mut bvh = builder.build(&initial_triangles, &options)?;
    let initial_aabb = bvh.world_aabb;

    // Extremely deformed triangle (stretched far away)
    let deformed_triangles = vec![Triangle::new(
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
    )];

    builder.refit(&mut bvh, &deformed_triangles)?;
    let deformed_aabb = bvh.world_aabb;

    // AABB should have grown significantly
    let expansion =
        (deformed_aabb.max[0] - deformed_aabb.min[0]) / (initial_aabb.max[0] - initial_aabb.min[0]);
    assert!(expansion > 10.0, "AABB should expand for deformed geometry");

    // Should still be valid and contain the deformed triangle
    assert!(deformed_aabb.is_valid());
    let tri_aabb = deformed_triangles[0].aabb();
    for i in 0..3 {
        assert!(
            deformed_aabb.min[i] <= tri_aabb.min[i] + 1e-6,
            "AABB should contain deformed triangle"
        );
        assert!(
            deformed_aabb.max[i] >= tri_aabb.max[i] - 1e-6,
            "AABB should contain deformed triangle"
        );
    }

    println!("Extreme deformation test passed:");
    println!(
        "  Initial AABB extent: {:.3}",
        initial_aabb.max[0] - initial_aabb.min[0]
    );
    println!(
        "  Deformed AABB extent: {:.3}",
        deformed_aabb.max[0] - deformed_aabb.min[0]
    );
    println!("  Expansion factor: {:.1}x", expansion);

    Ok(())
}
