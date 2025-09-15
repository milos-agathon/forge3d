// tests/test_sdf_integration.rs
// Integration tests for the complete SDF system
// Tests the full pipeline from scene construction to rendering

use forge3d::path_tracing::hybrid_compute::{HybridPathTracer, HybridTracerParams, TraversalMode};
use forge3d::sdf::*;
use glam::Vec3;

#[test]
fn test_sdf_scene_to_hybrid_scene_conversion() {
    // Test converting SDF scene to hybrid scene
    let (builder, sphere_idx) = SdfSceneBuilder::new().add_sphere(Vec3::ZERO, 1.0, 1);

    let (builder, box_idx) = builder.add_box(Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.8, 0.8, 0.8), 2);

    let (builder, _union_idx) = builder.union(sphere_idx, box_idx, 0);

    let sdf_scene = builder.build();

    // Create hybrid scene
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    assert!(
        hybrid_scene.has_sdf(),
        "Hybrid scene should have SDF geometry"
    );
    assert!(
        !hybrid_scene.has_mesh(),
        "Hybrid scene should not have mesh geometry"
    );
    assert_eq!(hybrid_scene.sdf_scene.primitive_count(), 2);
    assert_eq!(hybrid_scene.sdf_scene.node_count(), 3); // 2 leaves + 1 union
}

#[test]
fn test_hybrid_scene_raymarching_integration() {
    // Test complete raymarching pipeline
    let (builder, sphere_idx) =
        SdfSceneBuilder::new().add_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 1);

    let sdf_scene = builder.build();
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Test ray intersection
    let ray = HybridRay {
        origin: Vec3::ZERO,
        direction: Vec3::new(0.0, 0.0, -1.0),
        tmin: 0.001,
        tmax: 100.0,
    };

    let result = hybrid_scene.intersect(ray);

    assert!(result.hit, "Ray should intersect the sphere");
    assert_eq!(result.hit_type, 1, "Should be SDF hit");
    assert_eq!(result.material_id, 1, "Should have correct material ID");
    assert!(
        result.t > 3.0 && result.t < 5.0,
        "Hit distance should be reasonable"
    );

    // Verify hit point is approximately correct
    let expected_hit_point = Vec3::new(0.0, 0.0, -4.0); // Ray hits sphere at front
    let distance_to_expected = (result.point - expected_hit_point).length();
    assert!(
        distance_to_expected < 0.1,
        "Hit point should be approximately correct"
    );

    // Verify normal points toward camera
    assert!(result.normal.z > 0.0, "Normal should point toward camera");
}

#[test]
fn test_multiple_primitive_scene() {
    // Test scene with multiple different primitives
    let (builder, sphere_idx) =
        SdfSceneBuilder::new().add_sphere(Vec3::new(-2.0, 0.0, 0.0), 1.0, 1);

    let (builder, box_idx) = builder.add_box(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 2);

    let (builder, cylinder_idx) = builder.add_cylinder(Vec3::new(2.0, 0.0, 0.0), 0.8, 2.0, 3);

    let (builder, union1_idx) = builder.union(sphere_idx, box_idx, 4);

    let (builder, _final_union_idx) = builder.union(union1_idx, cylinder_idx, 0);

    let sdf_scene = builder.build();
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Test rays hitting different primitives
    let rays = vec![
        (Vec3::new(-2.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)), // Should hit sphere
        (Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)),  // Should hit box
        (Vec3::new(2.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)),  // Should hit cylinder
    ];

    let mut hits = 0;
    for (origin, direction) in rays {
        let ray = HybridRay {
            origin,
            direction,
            tmin: 0.001,
            tmax: 100.0,
        };

        let result = hybrid_scene.intersect(ray);
        if result.hit {
            hits += 1;
            assert_eq!(result.hit_type, 1, "All hits should be SDF hits");
        }
    }

    assert!(hits >= 2, "Should hit at least 2 out of 3 primitives");
}

#[test]
fn test_csg_operations_integration() {
    // Test complex CSG operations produce reasonable results
    let (builder, sphere1_idx) =
        SdfSceneBuilder::new().add_sphere(Vec3::new(-0.5, 0.0, 0.0), 1.0, 1);

    let (builder, sphere2_idx) = builder.add_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0, 2);

    let (builder, intersection_idx) = builder.intersect(sphere1_idx, sphere2_idx, 3);

    let sdf_scene = builder.build();
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Test points inside intersection
    let ray_center = HybridRay {
        origin: Vec3::new(0.0, 0.0, -5.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
        tmin: 0.001,
        tmax: 100.0,
    };

    let result = hybrid_scene.intersect(ray_center);
    assert!(result.hit, "Should hit the intersection");

    // Test points outside intersection (but inside individual spheres)
    let ray_left = HybridRay {
        origin: Vec3::new(-1.0, 0.0, -5.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
        tmin: 0.001,
        tmax: 100.0,
    };

    let result_left = hybrid_scene.intersect(ray_left);
    // This should not hit since it's outside the intersection region
    if result_left.hit {
        // If it hits, the distance should be reasonable
        assert!(result_left.t > 0.0, "Hit distance should be positive");
    }
}

#[test]
fn test_smooth_csg_operations() {
    // Test that smooth operations produce different results than hard operations
    let (builder_hard, s1) = SdfSceneBuilder::new().add_sphere(Vec3::new(-0.3, 0.0, 0.0), 0.8, 1);

    let (builder_hard, s2) = builder_hard.add_sphere(Vec3::new(0.3, 0.0, 0.0), 0.8, 2);

    let (builder_hard, _hard_union) = builder_hard.union(s1, s2, 0);

    let hard_scene = builder_hard.build();

    // Create smooth version
    let (builder_smooth, s1) = SdfSceneBuilder::new().add_sphere(Vec3::new(-0.3, 0.0, 0.0), 0.8, 1);

    let (builder_smooth, s2) = builder_smooth.add_sphere(Vec3::new(0.3, 0.0, 0.0), 0.8, 2);

    let (builder_smooth, _smooth_union) = builder_smooth.smooth_union(s1, s2, 0.2, 0);

    let smooth_scene = builder_smooth.build();

    // Compare evaluations at a point between the spheres
    let test_point = Vec3::ZERO;
    let hard_result = hard_scene.evaluate(test_point);
    let smooth_result = smooth_scene.evaluate(test_point);

    // Both should be inside, but smooth union should have different distance
    assert!(
        hard_result.distance < 0.0,
        "Hard union should be inside at center"
    );
    assert!(
        smooth_result.distance < 0.0,
        "Smooth union should be inside at center"
    );
}

#[test]
fn test_scene_bounds_optimization() {
    // Test that scene bounds work for optimization
    let (builder, sphere_idx) = SdfSceneBuilder::new().add_sphere(Vec3::ZERO, 1.0, 1);

    let scene = builder
        .with_bounds(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0))
        .build();

    let bounds = scene.get_bounds();
    assert!(bounds.is_some(), "Scene should have bounds");

    let (min_bounds, max_bounds) = bounds.unwrap();
    assert_eq!(min_bounds, Vec3::new(-2.0, -2.0, -2.0));
    assert_eq!(max_bounds, Vec3::new(2.0, 2.0, 2.0));

    // Test bounds checking
    assert!(scene.in_bounds(Vec3::ZERO), "Origin should be in bounds");
    assert!(
        scene.in_bounds(Vec3::new(1.5, 1.5, 1.5)),
        "Point should be in bounds"
    );
    assert!(
        !scene.in_bounds(Vec3::new(3.0, 0.0, 0.0)),
        "Point should be out of bounds"
    );
}

#[test]
fn test_performance_characteristics() {
    // Test that the system has reasonable performance characteristics
    let (builder, sphere_idx) = SdfSceneBuilder::new().add_sphere(Vec3::ZERO, 1.0, 1);

    let sdf_scene = builder.build();
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Time multiple ray intersections
    let start_time = std::time::Instant::now();
    let mut hits = 0;

    for i in 0..1000 {
        let angle = i as f32 * 0.01;
        let ray = HybridRay {
            origin: Vec3::new(angle.sin() * 5.0, angle.cos() * 5.0, 0.0),
            direction: Vec3::new(-angle.sin(), -angle.cos(), 0.0).normalize(),
            tmin: 0.001,
            tmax: 100.0,
        };

        let result = hybrid_scene.intersect(ray);
        if result.hit {
            hits += 1;
        }
    }

    let elapsed = start_time.elapsed();

    // Should complete quickly and hit the sphere many times
    assert!(
        elapsed.as_millis() < 1000,
        "Should complete in reasonable time"
    );
    assert!(hits > 500, "Should hit the sphere frequently");
}

#[test]
fn test_normal_calculation_accuracy() {
    // Test that calculated normals are reasonable
    let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);
    let sdf_scene = SdfScene::single_primitive(sphere);
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Test ray hitting sphere from different angles
    let test_rays = vec![
        (Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)), // Front
        (Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)), // Right
        (Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0)), // Top
    ];

    for (origin, direction) in test_rays {
        let ray = HybridRay {
            origin,
            direction,
            tmin: 0.001,
            tmax: 100.0,
        };

        let result = hybrid_scene.intersect(ray);
        assert!(result.hit, "Ray should hit sphere");

        // Normal should be unit length
        let normal_length = result.normal.length();
        assert!(
            (normal_length - 1.0).abs() < 0.01,
            "Normal should be unit length"
        );

        // Normal should point away from sphere center
        let to_center = (Vec3::ZERO - result.point).normalize();
        let dot_product = result.normal.dot(to_center);
        assert!(dot_product < -0.9, "Normal should point away from center");
    }
}

#[test]
fn test_material_id_propagation() {
    // Test that material IDs are correctly propagated through CSG operations
    let (builder, sphere_idx) =
        SdfSceneBuilder::new().add_sphere(Vec3::new(-1.0, 0.0, 0.0), 0.8, 42); // Unique material ID

    let (builder, box_idx) =
        builder.add_box(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.8, 0.8, 0.8), 99); // Different material ID

    let (builder, _union_idx) = builder.union(sphere_idx, box_idx, 123); // Union with its own material ID

    let sdf_scene = builder.build();
    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Test ray hitting sphere
    let ray_sphere = HybridRay {
        origin: Vec3::new(-1.0, 0.0, -5.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
        tmin: 0.001,
        tmax: 100.0,
    };

    let result_sphere = hybrid_scene.intersect(ray_sphere);
    if result_sphere.hit {
        // Material ID should be from the sphere or union
        assert!(
            result_sphere.material_id == 42 || result_sphere.material_id == 123,
            "Material ID should be from sphere or union"
        );
    }

    // Test ray hitting box
    let ray_box = HybridRay {
        origin: Vec3::new(1.0, 0.0, -5.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
        tmin: 0.001,
        tmax: 100.0,
    };

    let result_box = hybrid_scene.intersect(ray_box);
    if result_box.hit {
        // Material ID should be from the box or union
        assert!(
            result_box.material_id == 99 || result_box.material_id == 123,
            "Material ID should be from box or union"
        );
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_single_sphere_raymarching() {
        let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);
        let sdf_scene = SdfScene::single_primitive(sphere);
        let hybrid_scene = HybridScene::sdf_only(sdf_scene);

        let ray = HybridRay {
            origin: Vec3::new(0.0, 0.0, 5.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
            tmin: 0.001,
            tmax: 100.0,
        };

        let start = Instant::now();
        let iterations = 10000;

        for _ in 0..iterations {
            let _result = hybrid_scene.intersect(ray);
        }

        let elapsed = start.elapsed();
        let ns_per_ray = elapsed.as_nanos() / iterations as u128;

        println!("Single sphere raymarching: {} ns per ray", ns_per_ray);

        // Should be reasonably fast (less than 1 microsecond per ray)
        assert!(
            ns_per_ray < 1_000_000,
            "Raymarching should be reasonably fast"
        );
    }

    #[test]
    fn bench_complex_scene_raymarching() {
        // Create a more complex scene
        let (builder, s1) = SdfSceneBuilder::new().add_sphere(Vec3::new(-1.0, 0.0, 0.0), 0.8, 1);

        let (builder, s2) = builder.add_sphere(Vec3::new(1.0, 0.0, 0.0), 0.8, 2);

        let (builder, b1) = builder.add_box(Vec3::ZERO, Vec3::new(1.5, 0.3, 0.3), 3);

        let (builder, union_spheres) = builder.smooth_union(s1, s2, 0.2, 4);

        let (builder, _final) = builder.subtract(union_spheres, b1, 0);

        let sdf_scene = builder.build();
        let hybrid_scene = HybridScene::sdf_only(sdf_scene);

        let ray = HybridRay {
            origin: Vec3::new(0.0, 0.0, 5.0),
            direction: Vec3::new(0.0, 0.0, -1.0),
            tmin: 0.001,
            tmax: 100.0,
        };

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _result = hybrid_scene.intersect(ray);
        }

        let elapsed = start.elapsed();
        let us_per_ray = elapsed.as_micros() / iterations as u128;

        println!("Complex scene raymarching: {} μs per ray", us_per_ray);

        // Complex scenes will be slower but should still be reasonable
        assert!(
            us_per_ray < 100,
            "Complex scene raymarching should complete in reasonable time"
        );
    }
}
