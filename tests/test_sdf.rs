// tests/test_sdf.rs
// Comprehensive tests for SDF primitives and CSG operations
// Tests both CPU evaluation and hybrid scene construction

use forge3d::sdf::*;
use glam::Vec3;
use std::f32::consts::PI;

#[test]
fn test_sdf_primitive_sphere() {
    let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);

    // Test CPU evaluation
    let distance_center = primitives::cpu_eval::evaluate_primitive(Vec3::ZERO, &sphere);
    assert!(distance_center < 0.0, "Center should be inside sphere");
    assert!(
        (distance_center + 1.0).abs() < 0.001,
        "Distance at center should be -radius"
    );

    let distance_surface =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(1.0, 0.0, 0.0), &sphere);
    assert!(
        distance_surface.abs() < 0.001,
        "Point on surface should have distance ~0"
    );

    let distance_outside =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(2.0, 0.0, 0.0), &sphere);
    assert!(
        distance_outside > 0.0,
        "Point outside should have positive distance"
    );
    assert!(
        (distance_outside - 1.0).abs() < 0.001,
        "Distance should be 1.0"
    );
}

#[test]
fn test_sdf_primitive_box() {
    let box_prim = SdfPrimitive::box_primitive(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), 2);

    // Test CPU evaluation
    let distance_center = primitives::cpu_eval::evaluate_primitive(Vec3::ZERO, &box_prim);
    assert!(distance_center < 0.0, "Center should be inside box");

    let distance_corner =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(1.0, 1.0, 1.0), &box_prim);
    assert!(distance_corner.abs() < 0.001, "Corner should be on surface");

    let distance_outside =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(2.0, 0.0, 0.0), &box_prim);
    assert!(
        distance_outside > 0.0,
        "Point outside should have positive distance"
    );
}

#[test]
fn test_sdf_primitive_cylinder() {
    let cylinder = SdfPrimitive::cylinder(Vec3::ZERO, 1.0, 2.0, 3);

    // Test CPU evaluation
    let distance_center = primitives::cpu_eval::evaluate_primitive(Vec3::ZERO, &cylinder);
    assert!(distance_center < 0.0, "Center should be inside cylinder");

    // Point on side surface
    let distance_side =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(1.0, 0.0, 0.0), &cylinder);
    assert!(
        distance_side.abs() < 0.001,
        "Point on side should be on surface"
    );

    // Point on top surface
    let distance_top =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(0.0, 1.0, 0.0), &cylinder);
    assert!(
        distance_top.abs() < 0.001,
        "Point on top should be on surface"
    );
}

#[test]
fn test_sdf_primitive_plane() {
    let plane = SdfPrimitive::plane(Vec3::new(0.0, 1.0, 0.0), 0.0, 4);

    // Test CPU evaluation
    let distance_on_plane = primitives::cpu_eval::evaluate_primitive(Vec3::ZERO, &plane);
    assert!(
        distance_on_plane.abs() < 0.001,
        "Origin should be on XZ plane"
    );

    let distance_above = primitives::cpu_eval::evaluate_primitive(Vec3::new(0.0, 1.0, 0.0), &plane);
    assert!(
        (distance_above - 1.0).abs() < 0.001,
        "Point above should have distance 1.0"
    );

    let distance_below =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(0.0, -1.0, 0.0), &plane);
    assert!(
        (distance_below + 1.0).abs() < 0.001,
        "Point below should have distance -1.0"
    );
}

#[test]
fn test_sdf_primitive_torus() {
    let torus = SdfPrimitive::torus(Vec3::ZERO, 2.0, 0.5, 5);

    // Test CPU evaluation at points on the torus
    let distance_center = primitives::cpu_eval::evaluate_primitive(Vec3::ZERO, &torus);
    assert!(
        (distance_center - 1.5).abs() < 0.001,
        "Distance at center should be major_radius - minor_radius"
    );

    // Point on torus surface
    let distance_surface =
        primitives::cpu_eval::evaluate_primitive(Vec3::new(2.5, 0.0, 0.0), &torus);
    assert!(
        distance_surface.abs() < 0.001,
        "Point on surface should have distance ~0"
    );
}

#[test]
fn test_sdf_primitive_capsule() {
    let point_a = Vec3::new(-1.0, 0.0, 0.0);
    let point_b = Vec3::new(1.0, 0.0, 0.0);
    let capsule = SdfPrimitive::capsule(point_a, point_b, 0.5, 6);

    // Test CPU evaluation
    let distance_center = primitives::cpu_eval::evaluate_primitive(Vec3::ZERO, &capsule);
    assert!(
        (distance_center + 0.5).abs() < 0.001,
        "Distance at center should be -radius"
    );

    let distance_end = primitives::cpu_eval::evaluate_primitive(Vec3::new(1.5, 0.0, 0.0), &capsule);
    assert!(
        distance_end.abs() < 0.001,
        "Point at end should be on surface"
    );
}

#[test]
fn test_csg_operations() {
    let result_a = CsgResult {
        distance: 0.5,
        material_id: 1,
    };
    let result_b = CsgResult {
        distance: 1.0,
        material_id: 2,
    };

    // Test union
    let union_result = operations::cpu_eval::union(result_a, result_b);
    assert_eq!(
        union_result.distance, 0.5,
        "Union should take minimum distance"
    );
    assert_eq!(
        union_result.material_id, 1,
        "Union should take material from closer surface"
    );

    // Test intersection
    let intersection_result = operations::cpu_eval::intersection(result_a, result_b);
    assert_eq!(
        intersection_result.distance, 1.0,
        "Intersection should take maximum distance"
    );
    assert_eq!(
        intersection_result.material_id, 2,
        "Intersection should take material from farther surface"
    );

    // Test subtraction
    let subtraction_result = operations::cpu_eval::subtraction(result_a, result_b);
    assert_eq!(
        subtraction_result.distance, 0.5,
        "Subtraction result distance"
    );
    assert_eq!(
        subtraction_result.material_id, 1,
        "Subtraction should keep left material"
    );

    // Test smooth union
    let smooth_union_result = operations::cpu_eval::smooth_union(result_a, result_b, 0.1);
    assert!(
        smooth_union_result.distance <= 0.5,
        "Smooth union should be <= minimum"
    );
    assert!(
        smooth_union_result.distance >= 0.4,
        "Smooth union should be reasonable"
    );
}

#[test]
fn test_csg_tree_construction() {
    let mut tree = CsgTree::new();

    // Add two primitives
    let sphere = SdfPrimitive::sphere(Vec3::new(-0.5, 0.0, 0.0), 1.0, 1);
    let box_prim =
        SdfPrimitive::box_primitive(Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.8, 0.8, 0.8), 2);

    let prim1_idx = tree.add_primitive(sphere);
    let prim2_idx = tree.add_primitive(box_prim);

    // Add leaf nodes
    let leaf1_idx = tree.add_leaf(prim1_idx, 1);
    let leaf2_idx = tree.add_leaf(prim2_idx, 2);

    // Add union operation
    let union_idx = tree.add_operation(CsgOperation::Union, leaf1_idx, leaf2_idx, 0.0, 0);

    // Test tree structure
    assert_eq!(tree.primitives.len(), 2, "Should have 2 primitives");
    assert_eq!(
        tree.nodes.len(),
        3,
        "Should have 3 nodes (2 leaves + 1 operation)"
    );

    // Test evaluation at a point that should be inside the sphere
    let result = tree.evaluate(Vec3::new(-0.5, 0.0, 0.0), union_idx);
    assert!(result.distance < 0.0, "Point should be inside the union");
}

#[test]
fn test_sdf_scene_builder() {
    let (builder, sphere_idx) =
        SdfSceneBuilder::new().add_sphere(Vec3::new(-1.0, 0.0, 0.0), 0.8, 1);

    let (builder, box_idx) = builder.add_box(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.7, 0.7, 0.7), 2);

    let (builder, union_idx) = builder.union(sphere_idx, box_idx, 0);

    let scene = builder.build();

    // Test scene properties
    assert_eq!(scene.primitive_count(), 2, "Scene should have 2 primitives");
    assert_eq!(scene.node_count(), 3, "Scene should have 3 nodes");

    // Test scene evaluation
    let result_sphere = scene.evaluate(Vec3::new(-1.0, 0.0, 0.0));
    assert!(
        result_sphere.distance < 0.0,
        "Point should be inside sphere"
    );
    assert_eq!(result_sphere.material_id, 1, "Should have sphere material");

    let result_box = scene.evaluate(Vec3::new(1.0, 0.0, 0.0));
    assert!(result_box.distance < 0.0, "Point should be inside box");

    let result_outside = scene.evaluate(Vec3::new(0.0, 0.0, 3.0));
    assert!(
        result_outside.distance > 0.0,
        "Point should be outside union"
    );
}

#[test]
fn test_sdf_scene_with_bounds() {
    let sphere = SdfPrimitive::sphere(Vec3::ZERO, 1.0, 1);
    let scene = SdfScene::single_primitive(sphere)
        .with_bounds(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));

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
fn test_hybrid_scene_creation() {
    let mut hybrid_scene = HybridScene::new();
    assert!(!hybrid_scene.has_sdf(), "New scene should not have SDF");
    assert!(!hybrid_scene.has_mesh(), "New scene should not have mesh");

    // Add SDF scene
    let (builder, _) = SdfSceneBuilder::new().add_sphere(Vec3::ZERO, 1.0, 1);
    let sdf_scene = builder.build();

    hybrid_scene.set_sdf_scene(sdf_scene);
    assert!(hybrid_scene.has_sdf(), "Scene should now have SDF");
    assert!(!hybrid_scene.has_mesh(), "Scene should still not have mesh");
}

#[test]
fn test_hybrid_sdf_raymarching() {
    // Create a simple SDF scene
    let (builder, _) = SdfSceneBuilder::new().add_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0, 1);
    let sdf_scene = builder.build();

    let hybrid_scene = HybridScene::sdf_only(sdf_scene);

    // Create a ray that should hit the sphere
    let ray = HybridRay {
        origin: Vec3::ZERO,
        direction: Vec3::new(0.0, 0.0, -1.0),
        tmin: 0.001,
        tmax: 100.0,
    };

    let result = hybrid_scene.intersect(ray);

    assert!(result.hit, "Ray should hit the sphere");
    assert_eq!(result.hit_type, 1, "Should be SDF hit");
    assert_eq!(result.material_id, 1, "Should have correct material ID");
    assert!(
        result.t > 3.9 && result.t < 4.1,
        "Hit distance should be approximately 4.0"
    );
    assert!(
        result.sdf_distance.is_some(),
        "SDF hit should have SDF distance"
    );
}

#[test]
fn test_complex_csg_scene() {
    let (builder, sphere1) = SdfSceneBuilder::new().add_sphere(Vec3::new(-0.5, 0.0, 0.0), 0.8, 1);

    let (builder, sphere2) = builder.add_sphere(Vec3::new(0.5, 0.0, 0.0), 0.8, 2);

    let (builder, box1) = builder.add_box(Vec3::ZERO, Vec3::new(1.5, 0.3, 0.3), 3);

    // Create intersection of the two spheres
    let (builder, intersection) = builder.intersect(sphere1, sphere2, 4);

    // Subtract the box from the intersection
    let (builder, _result) = builder.subtract(intersection, box1, 0);

    let scene = builder.build();

    // Test that the scene was built correctly
    assert_eq!(scene.primitive_count(), 3, "Should have 3 primitives");
    assert_eq!(
        scene.node_count(),
        6,
        "Should have 6 nodes (3 leaves + 3 operations)"
    );

    // Test some strategic points
    let center_result = scene.evaluate(Vec3::ZERO);
    // Center should be positive (outside) due to box subtraction
    assert!(
        center_result.distance > 0.0,
        "Center should be outside due to subtraction"
    );

    // Point in intersection but outside box should be inside
    let intersection_result = scene.evaluate(Vec3::new(0.0, 0.6, 0.0));
    assert!(
        intersection_result.distance < 0.0,
        "Point in intersection should be inside"
    );
}

#[test]
fn test_smooth_csg_operations() {
    let (builder, sphere1) = SdfSceneBuilder::new().add_sphere(Vec3::new(-0.5, 0.0, 0.0), 1.0, 1);

    let (builder, sphere2) = builder.add_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0, 2);

    // Create smooth union
    let (builder, _smooth_union) = builder.smooth_union(sphere1, sphere2, 0.2, 0);

    let scene = builder.build();

    // Test that smooth union creates smoother transitions
    let result_center = scene.evaluate(Vec3::ZERO);
    let result_midpoint = scene.evaluate(Vec3::new(0.25, 0.0, 0.0));

    // Both points should be inside the smooth union
    assert!(
        result_center.distance < 0.0,
        "Center should be inside smooth union"
    );
    assert!(
        result_midpoint.distance < 0.0,
        "Midpoint should be inside smooth union"
    );
}

#[test]
fn test_performance_metrics() {
    let metrics = HybridMetrics {
        sdf_steps: 1000,
        bvh_nodes_visited: 500,
        triangle_tests: 100,
        total_rays: 1000,
        sdf_hits: 600,
        mesh_hits: 200,
    };

    let overhead = metrics.performance_overhead();
    assert!(
        overhead >= 0.0,
        "Performance overhead should be non-negative"
    );
    assert!(overhead < 10.0, "Performance overhead should be reasonable");
}
