# tests/test_sdf_python.py
# Python tests for SDF primitives and CSG operations
# Tests the Python API and CPU fallback implementations

import pytest
import numpy as np
from forge3d.sdf import (
    SdfPrimitive, SdfScene, SdfSceneBuilder, HybridRenderer,
    SdfPrimitiveType, CsgOperation, TraversalMode,
    create_sphere, create_box, create_simple_scene, render_simple_scene
)


class TestSdfPrimitives:
    """Test SDF primitive creation and evaluation"""

    def test_sphere_creation(self):
        """Test sphere primitive creation"""
        sphere = SdfPrimitive.sphere((0, 0, 0), 1.0, 1)

        assert sphere.primitive_type == SdfPrimitiveType.SPHERE
        assert np.allclose(sphere.center, [0, 0, 0])
        assert sphere.material_id == 1
        assert sphere.params['radius'] == 1.0

    def test_sphere_evaluation(self):
        """Test sphere SDF evaluation"""
        sphere = SdfPrimitive.sphere((0, 0, 0), 1.0, 1)

        # Test center (should be inside)
        distance = sphere.evaluate((0, 0, 0))
        assert distance < 0, "Center should be inside sphere"
        assert abs(distance + 1.0) < 0.001, "Distance at center should be -radius"

        # Test surface point
        distance = sphere.evaluate((1, 0, 0))
        assert abs(distance) < 0.001, "Point on surface should have distance ~0"

        # Test outside point
        distance = sphere.evaluate((2, 0, 0))
        assert distance > 0, "Point outside should have positive distance"
        assert abs(distance - 1.0) < 0.001, "Distance should be 1.0"

    def test_box_creation(self):
        """Test box primitive creation"""
        box = SdfPrimitive.box((0, 0, 0), (1, 1, 1), 2)

        assert box.primitive_type == SdfPrimitiveType.BOX
        assert np.allclose(box.center, [0, 0, 0])
        assert box.material_id == 2
        assert box.params['extents'] == (1, 1, 1)

    def test_box_evaluation(self):
        """Test box SDF evaluation"""
        box = SdfPrimitive.box((0, 0, 0), (1, 1, 1), 2)

        # Test center (should be inside)
        distance = box.evaluate((0, 0, 0))
        assert distance < 0, "Center should be inside box"

        # Test corner (should be on surface)
        distance = box.evaluate((1, 1, 1))
        assert abs(distance) < 0.01, "Corner should be on surface"

        # Test outside point
        distance = box.evaluate((2, 0, 0))
        assert distance > 0, "Point outside should have positive distance"

    def test_cylinder_creation(self):
        """Test cylinder primitive creation"""
        cylinder = SdfPrimitive.cylinder((0, 0, 0), 1.0, 2.0, 3)

        assert cylinder.primitive_type == SdfPrimitiveType.CYLINDER
        assert np.allclose(cylinder.center, [0, 0, 0])
        assert cylinder.material_id == 3
        assert cylinder.params['radius'] == 1.0
        assert cylinder.params['height'] == 2.0

    def test_plane_creation(self):
        """Test plane primitive creation"""
        plane = SdfPrimitive.plane((0, 1, 0), 0.0, 4)

        assert plane.primitive_type == SdfPrimitiveType.PLANE
        assert plane.material_id == 4
        assert plane.params['normal'] == (0, 1, 0)
        assert plane.params['distance'] == 0.0

    def test_plane_evaluation(self):
        """Test plane SDF evaluation"""
        plane = SdfPrimitive.plane((0, 1, 0), 0.0, 4)

        # Test point on plane
        distance = plane.evaluate((0, 0, 0))
        assert abs(distance) < 0.001, "Point on plane should have distance ~0"

        # Test point above plane
        distance = plane.evaluate((0, 1, 0))
        assert abs(distance - 1.0) < 0.001, "Point above should have distance 1.0"

        # Test point below plane
        distance = plane.evaluate((0, -1, 0))
        assert abs(distance + 1.0) < 0.001, "Point below should have distance -1.0"

    def test_torus_creation(self):
        """Test torus primitive creation"""
        torus = SdfPrimitive.torus((0, 0, 0), 2.0, 0.5, 5)

        assert torus.primitive_type == SdfPrimitiveType.TORUS
        assert torus.material_id == 5
        assert torus.params['major_radius'] == 2.0
        assert torus.params['minor_radius'] == 0.5

    def test_capsule_creation(self):
        """Test capsule primitive creation"""
        capsule = SdfPrimitive.capsule((-1, 0, 0), (1, 0, 0), 0.5, 6)

        assert capsule.primitive_type == SdfPrimitiveType.CAPSULE
        assert capsule.material_id == 6
        assert capsule.params['point_a'] == (-1, 0, 0)
        assert capsule.params['point_b'] == (1, 0, 0)
        assert capsule.params['radius'] == 0.5

    def test_convenience_functions(self):
        """Test convenience functions"""
        sphere = create_sphere((0, 0, 0), 1.0, 1)
        assert isinstance(sphere, SdfPrimitive)
        assert sphere.primitive_type == SdfPrimitiveType.SPHERE

        box = create_box((0, 0, 0), (1, 1, 1), 2)
        assert isinstance(box, SdfPrimitive)
        assert box.primitive_type == SdfPrimitiveType.BOX


class TestSdfScene:
    """Test SDF scene functionality"""

    def test_empty_scene(self):
        """Test empty scene creation"""
        scene = SdfScene()

        assert scene.primitive_count() == 0
        assert scene.operation_count() == 0
        assert scene.get_bounds() is None

        # Test evaluation of empty scene
        distance, material = scene.evaluate((0, 0, 0))
        assert distance == float('inf')
        assert material == 0

    def test_single_primitive_scene(self):
        """Test scene with single primitive"""
        scene = SdfScene()
        sphere = SdfPrimitive.sphere((0, 0, 0), 1.0, 1)
        scene.add_primitive(sphere)

        assert scene.primitive_count() == 1
        assert scene.operation_count() == 0

        # Test evaluation
        distance, material = scene.evaluate((0, 0, 0))
        assert distance < 0, "Should be inside sphere"
        assert material == 1

    def test_scene_with_bounds(self):
        """Test scene bounds functionality"""
        scene = SdfScene()
        scene.set_bounds((-2, -2, -2), (2, 2, 2))

        bounds = scene.get_bounds()
        assert bounds is not None
        assert bounds[0] == (-2, -2, -2)
        assert bounds[1] == (2, 2, 2)

    def test_csg_operations(self):
        """Test CSG operations"""
        scene = SdfScene()

        sphere = SdfPrimitive.sphere((0, 0, 0), 1.0, 1)
        box = SdfPrimitive.box((0, 0, 0), (0.8, 0.8, 0.8), 2)

        sphere_idx = scene.add_primitive(sphere)
        box_idx = scene.add_primitive(box)

        # Add union operation
        union_idx = scene.add_operation(CsgOperation.UNION, sphere_idx, box_idx)

        assert scene.operation_count() == 1
        assert union_idx == 2  # 2 primitives + 1 operation - 1 (0-indexed)


class TestSdfSceneBuilder:
    """Test SDF scene builder pattern"""

    def test_builder_creation(self):
        """Test builder pattern"""
        builder = SdfSceneBuilder()
        assert isinstance(builder.scene, SdfScene)

    def test_builder_add_primitives(self):
        """Test adding primitives through builder"""
        builder, sphere_idx = SdfSceneBuilder().add_sphere((0, 0, 0), 1.0, 1)
        builder, box_idx = builder.add_box((1, 0, 0), (0.5, 0.5, 0.5), 2)

        scene = builder.build()
        assert scene.primitive_count() == 2
        assert sphere_idx == 0
        assert box_idx == 1

    def test_builder_csg_operations(self):
        """Test CSG operations through builder"""
        builder, sphere_idx = SdfSceneBuilder().add_sphere((-0.5, 0, 0), 0.8, 1)
        builder, box_idx = builder.add_box((0.5, 0, 0), (0.8, 0.8, 0.8), 2)
        builder, union_idx = builder.union(sphere_idx, box_idx, 0)

        scene = builder.build()
        assert scene.primitive_count() == 2
        assert scene.operation_count() == 1

    def test_builder_smooth_operations(self):
        """Test smooth CSG operations"""
        builder, s1 = SdfSceneBuilder().add_sphere((-0.5, 0, 0), 1.0, 1)
        builder, s2 = builder.add_sphere((0.5, 0, 0), 1.0, 2)
        builder, smooth_union = builder.smooth_union(s1, s2, 0.2, 0)

        scene = builder.build()
        assert scene.operation_count() == 1

    def test_builder_with_bounds(self):
        """Test builder bounds setting"""
        builder = SdfSceneBuilder().with_bounds((-1, -1, -1), (1, 1, 1))
        scene = builder.build()

        bounds = scene.get_bounds()
        assert bounds is not None
        assert bounds[0] == (-1, -1, -1)
        assert bounds[1] == (1, 1, 1)

    def test_complex_scene_construction(self):
        """Test construction of complex scene"""
        # Create a scene with multiple operations
        builder, sphere1 = SdfSceneBuilder().add_sphere((-1, 0, 0), 0.8, 1)
        builder, sphere2 = builder.add_sphere((1, 0, 0), 0.8, 2)
        builder, box1 = builder.add_box((0, 0, 0), (1.5, 0.3, 0.3), 3)

        # Union the spheres
        builder, union_spheres = builder.union(sphere1, sphere2, 4)

        # Subtract the box from the union
        builder, final_result = builder.subtract(union_spheres, box1, 0)

        scene = builder.build()

        assert scene.primitive_count() == 3
        assert scene.operation_count() == 2  # union + subtraction

        # Test some strategic points
        distance_center, _ = scene.evaluate((0, 0, 0))
        # Should be outside due to box subtraction
        assert distance_center > 0, "Center should be outside due to subtraction"


class TestHybridRenderer:
    """Test hybrid renderer functionality"""

    def test_renderer_creation(self):
        """Test renderer creation"""
        renderer = HybridRenderer(256, 256)

        assert renderer.width == 256
        assert renderer.height == 256
        assert renderer.traversal_mode == TraversalMode.HYBRID

    def test_camera_settings(self):
        """Test camera parameter setting"""
        renderer = HybridRenderer()
        renderer.set_camera((0, 0, 5), (0, 0, 0), (0, 1, 0), 45.0)

        assert np.allclose(renderer.camera_origin, [0, 0, 5])
        assert np.allclose(renderer.camera_target, [0, 0, 0])
        assert np.allclose(renderer.camera_up, [0, 1, 0])
        assert renderer.fov_degrees == 45.0

    def test_traversal_mode_setting(self):
        """Test traversal mode setting"""
        renderer = HybridRenderer()

        renderer.set_traversal_mode(TraversalMode.SDF_ONLY)
        assert renderer.traversal_mode == TraversalMode.SDF_ONLY

        renderer.set_traversal_mode(TraversalMode.MESH_ONLY)
        assert renderer.traversal_mode == TraversalMode.MESH_ONLY

        renderer.set_traversal_mode(TraversalMode.HYBRID)
        assert renderer.traversal_mode == TraversalMode.HYBRID

    def test_performance_params(self):
        """Test performance parameter setting"""
        renderer = HybridRenderer()
        renderer.set_performance_params(0.005, 8.0)

        assert renderer.early_exit_distance == 0.005
        assert renderer.shadow_softness == 8.0

    def test_cpu_fallback_rendering(self):
        """Test CPU fallback rendering"""
        scene = SdfScene()
        sphere = SdfPrimitive.sphere((0, 0, -5), 1.0, 1)
        scene.add_primitive(sphere)

        renderer = HybridRenderer(64, 64)  # Small size for test
        image = renderer.render_sdf_scene(scene)

        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 4)
        assert image.dtype == np.uint8

        # Check that some pixels are not sky color (should hit the sphere)
        sky_color = np.array([153, 178, 229], dtype=np.uint8)  # Sky color * 255
        non_sky_pixels = np.any(image[:, :, :3] != sky_color, axis=2)
        assert np.any(non_sky_pixels), "Should have some non-sky pixels"

    def test_performance_metrics(self):
        """Test performance metrics"""
        renderer = HybridRenderer()
        metrics = renderer.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'sdf_steps' in metrics
        assert 'bvh_nodes_visited' in metrics
        assert 'triangle_tests' in metrics
        assert 'total_rays' in metrics
        assert 'performance_overhead' in metrics


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_simple_scene(self):
        """Test simple scene creation"""
        scene = create_simple_scene()

        assert isinstance(scene, SdfScene)
        assert scene.primitive_count() >= 2  # Should have sphere and box
        assert scene.operation_count() >= 1  # Should have union

    def test_render_simple_scene(self):
        """Test simple scene rendering"""
        image = render_simple_scene(128, 128)

        assert isinstance(image, np.ndarray)
        assert image.shape == (128, 128, 4)
        assert image.dtype == np.uint8

    def test_enum_values(self):
        """Test enum value consistency"""
        # Test SDF primitive types
        assert SdfPrimitiveType.SPHERE.value == 0
        assert SdfPrimitiveType.BOX.value == 1
        assert SdfPrimitiveType.CYLINDER.value == 2
        assert SdfPrimitiveType.PLANE.value == 3
        assert SdfPrimitiveType.TORUS.value == 4
        assert SdfPrimitiveType.CAPSULE.value == 5

        # Test CSG operations
        assert CsgOperation.UNION.value == 0
        assert CsgOperation.INTERSECTION.value == 1
        assert CsgOperation.SUBTRACTION.value == 2
        assert CsgOperation.SMOOTH_UNION.value == 3
        assert CsgOperation.SMOOTH_INTERSECTION.value == 4
        assert CsgOperation.SMOOTH_SUBTRACTION.value == 5

        # Test traversal modes
        assert TraversalMode.HYBRID.value == 0
        assert TraversalMode.SDF_ONLY.value == 1
        assert TraversalMode.MESH_ONLY.value == 2


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_invalid_primitive_evaluation(self):
        """Test evaluation of unknown primitive type"""
        # Create a primitive with unknown type by directly setting the enum
        primitive = SdfPrimitive(SdfPrimitiveType.SPHERE, (0, 0, 0), 1)
        primitive.primitive_type = 999  # Invalid type

        distance = primitive.evaluate((0, 0, 0))
        assert distance == float('inf'), "Unknown primitive should return infinite distance"

    def test_zero_radius_sphere(self):
        """Test zero radius sphere"""
        sphere = SdfPrimitive.sphere((0, 0, 0), 0.0, 1)
        distance = sphere.evaluate((0, 0, 0))
        assert distance == 0.0, "Zero radius sphere at center should have distance 0"

    def test_degenerate_box(self):
        """Test degenerate box with zero extents"""
        box = SdfPrimitive.box((0, 0, 0), (0, 0, 0), 1)
        distance = box.evaluate((0, 0, 0))
        # Should behave like a point at the center
        assert distance <= 0.0, "Point at center of degenerate box should be inside or on surface"

    def test_very_distant_evaluation(self):
        """Test evaluation at very distant points"""
        sphere = SdfPrimitive.sphere((0, 0, 0), 1.0, 1)
        distance = sphere.evaluate((1000, 1000, 1000))
        expected_distance = np.sqrt(3 * 1000*1000) - 1.0  # Distance to center minus radius
        assert abs(distance - expected_distance) < 1.0, "Distance should be approximately correct"


if __name__ == "__main__":
    # Run a simple test when executed directly
    print("Running basic SDF tests...")

    # Test primitive creation
    sphere = create_sphere((0, 0, 0), 1.0, 1)
    print(f"Created sphere: {sphere.primitive_type}")

    # Test scene creation
    scene = create_simple_scene()
    print(f"Created scene with {scene.primitive_count()} primitives")

    # Test evaluation
    distance, material = scene.evaluate((0, 0, 0))
    print(f"Distance at origin: {distance}, material: {material}")

    # Test rendering
    print("Testing CPU fallback rendering...")
    image = render_simple_scene(64, 64)
    print(f"Rendered image shape: {image.shape}")

    print("Basic SDF tests completed successfully!")