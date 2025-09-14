"""
tests/test_mesh_tracing_gpu.py
GPU mesh rendering tests for triangle mesh path tracing (Task A3).
Tests GPU triangle intersection, BVH traversal, and mesh rendering functionality.
RELEVANT FILES:src/accel/cpu_bvh.rs,src/shaders/pt_intersect_mesh.wgsl,python/forge3d/mesh.py
"""

import numpy as np
import pytest
from forge3d.mesh import make_mesh, build_bvh_cpu, upload_mesh, create_triangle_mesh, create_cube_mesh, create_quad_mesh
from forge3d.path_tracing import render_rgba, render_aovs
import forge3d


def make_camera(origin, look_at, up, fov_y, aspect, exposure):
    """Create camera parameters dict for path tracing."""
    return {
        'origin': origin,
        'look_at': look_at,
        'up': up,
        'fov_y': fov_y,
        'aspect': aspect,
        'exposure': exposure,
    }


def tiny_cube():
    """Create a simple cube mesh for testing (12 tris, 8 verts)."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [0, 2, 3],  # Front
        [1, 5, 6], [1, 6, 2],  # Right
        [5, 4, 7], [5, 7, 6],  # Back
        [4, 0, 3], [4, 3, 7],  # Left
        [3, 2, 6], [3, 6, 7],  # Top
        [4, 5, 1], [4, 1, 0],  # Bottom
    ], dtype=np.uint32)
    return v, f


@pytest.mark.skipif(not forge3d.enumerate_adapters(), reason="No GPU adapters available")
def test_gpu_mesh_render_64x64():
    """Test basic GPU mesh rendering with small cube."""
    v, f = tiny_cube()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    cam = make_camera(
        origin=(2, 2, 2),
        look_at=(0.5, 0.5, 0.5),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0
    )

    # Test with GPU enabled
    img = render_rgba(64, 64, scene=[], camera=cam, seed=7, frames=1, use_gpu=True, mesh=handle)

    assert img.shape == (64, 64, 4)
    assert img.dtype == np.uint8
    assert img.mean() > 0.0, "Image should not be completely black"


@pytest.mark.skipif(not forge3d.enumerate_adapters(), reason="No GPU adapters available")
def test_mesh_vs_cpu_fallback():
    """Test that GPU and CPU mesh rendering produce similar results."""
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    cam = make_camera(
        origin=(0, 0, 2),
        look_at=(0.5, 0.5, 0),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0
    )

    # Render with GPU
    img_gpu = render_rgba(32, 32, scene=[], camera=cam, seed=42, frames=1, use_gpu=True, mesh=handle)

    # Render with CPU fallback
    img_cpu = render_rgba(32, 32, scene=[], camera=cam, seed=42, frames=1, use_gpu=False, mesh=handle)

    # Both should produce valid images
    assert img_gpu.shape == (32, 32, 4)
    assert img_cpu.shape == (32, 32, 4)
    assert img_gpu.mean() > 0
    assert img_cpu.mean() > 0

    # Results should be reasonably similar (allowing for implementation differences)
    # This is just a sanity check, not exact matching
    gpu_mean = img_gpu.astype(np.float32).mean()
    cpu_mean = img_cpu.astype(np.float32).mean()
    relative_diff = abs(gpu_mean - cpu_mean) / max(gpu_mean, cpu_mean, 1.0)
    assert relative_diff < 0.5, f"GPU/CPU results too different: {gpu_mean} vs {cpu_mean}"


def test_mesh_creation_and_validation():
    """Test mesh creation and validation functions."""
    # Valid triangle mesh
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    assert m['vertex_count'] == 3
    assert m['triangle_count'] == 1

    # Valid cube mesh
    v, f = create_cube_mesh()
    m = make_mesh(v, f)
    assert m['vertex_count'] == 8
    assert m['triangle_count'] == 12

    # Invalid vertex shape
    with pytest.raises(ValueError, match="vertices must have shape"):
        make_mesh(np.array([[0, 0]], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32))

    # Invalid index range
    v = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    f = np.array([[0, 1, 5]], dtype=np.uint32)  # Index 5 is out of bounds
    with pytest.raises(RuntimeError, match="indices reference vertex"):
        make_mesh(v, f)


def test_bvh_construction():
    """Test BVH construction with different mesh sizes."""
    # Single triangle
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")

    assert b['triangle_count'] == 1
    assert b['node_count'] >= 1
    assert b['leaf_count'] >= 1
    assert b['method'] == "median"
    assert 'build_time_ms' in b
    assert 'world_aabb_min' in b
    assert 'world_aabb_max' in b

    # Multiple triangles
    v, f = create_cube_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")

    assert b['triangle_count'] == 12
    assert b['node_count'] > 1
    assert b['max_depth'] > 0

    # Unsupported method
    with pytest.raises(ValueError, match="Unsupported BVH method"):
        build_bvh_cpu(m, method="unsupported")


def test_mesh_upload():
    """Test mesh and BVH upload functionality."""
    v, f = create_quad_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    assert handle.triangle_count == 2
    assert handle.vertex_count == 4
    assert handle.node_count >= 1

    # Test handle properties
    stats = handle.build_stats
    assert 'triangle_count' in stats
    assert 'build_time_ms' in stats

    aabb_min, aabb_max = handle.world_aabb
    assert len(aabb_min) == 3
    assert len(aabb_max) == 3

    # Test incompatible mesh and BVH
    v2, f2 = create_triangle_mesh()  # Different triangle count
    m2 = make_mesh(v2, f2)
    with pytest.raises(ValueError, match="triangle counts don't match"):
        upload_mesh(m2, b)  # BVH from quad mesh, mesh from triangle


@pytest.mark.skipif(not forge3d.enumerate_adapters(), reason="No GPU adapters available")
def test_mesh_aov_rendering():
    """Test AOV rendering with mesh data."""
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    cam = make_camera(
        origin=(0, 0, 2),
        look_at=(0.5, 0.3, 0),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0
    )

    aovs = render_aovs(
        32, 32, scene=[], camera=cam,
        aovs=("albedo", "normal", "depth", "visibility"),
        seed=123, frames=1, use_gpu=True, mesh=handle
    )

    # Check AOV outputs
    assert "albedo" in aovs
    assert "normal" in aovs
    assert "depth" in aovs
    assert "visibility" in aovs

    assert aovs["albedo"].shape == (32, 32, 3)
    assert aovs["normal"].shape == (32, 32, 3)
    assert aovs["depth"].shape == (32, 32)
    assert aovs["visibility"].shape == (32, 32)

    # Should have some hits
    assert np.any(aovs["visibility"] > 0), "Should have visible geometry"
    assert np.any(aovs["albedo"] > 0), "Should have non-zero albedo"


def test_empty_scene_with_mesh():
    """Test rendering with empty sphere scene but with mesh."""
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    cam = make_camera(
        origin=(0, 0, 1.5),
        look_at=(0.5, 0.3, 0),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0
    )

    # Empty scene but with mesh should still render
    img = render_rgba(32, 32, scene=[], camera=cam, seed=1, frames=1, use_gpu=False, mesh=handle)

    assert img.shape == (32, 32, 4)
    assert img.mean() > 0, "Should render mesh even with empty sphere scene"


def test_scene_and_mesh_combined():
    """Test rendering with both spheres and mesh."""
    # Create a simple sphere scene
    scene = [
        {'center': (0, 0, -1), 'radius': 0.3, 'albedo': (1.0, 0.0, 0.0)}
    ]

    # Create mesh
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    cam = make_camera(
        origin=(0, 0, 2),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0
    )

    # Render with both spheres and mesh
    img = render_rgba(32, 32, scene=scene, camera=cam, seed=1, frames=1, use_gpu=False, mesh=handle)

    assert img.shape == (32, 32, 4)
    assert img.mean() > 0, "Should render both spheres and mesh"


def test_deterministic_rendering():
    """Test that mesh rendering is deterministic with fixed seed."""
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m, method="median")
    handle = upload_mesh(m, b)

    cam = make_camera(
        origin=(0, 0, 1.5),
        look_at=(0.5, 0.3, 0),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0
    )

    # Render twice with same seed
    img1 = render_rgba(24, 24, scene=[], camera=cam, seed=42, frames=1, use_gpu=False, mesh=handle)
    img2 = render_rgba(24, 24, scene=[], camera=cam, seed=42, frames=1, use_gpu=False, mesh=handle)

    # Results should be identical
    np.testing.assert_array_equal(img1, img2, "Rendering should be deterministic with fixed seed")


if __name__ == "__main__":
    # Simple test runner for standalone execution
    print("forge3d mesh tracing GPU tests")

    # Test basic functionality without pytest
    print("Testing mesh creation...")
    v, f = create_triangle_mesh()
    m = make_mesh(v, f)
    b = build_bvh_cpu(m)
    handle = upload_mesh(m, b)
    print(f"Created mesh handle: {handle}")

    print("Testing CPU mesh rendering...")
    cam = make_camera((0, 0, 2), (0.5, 0.3, 0), (0, 1, 0), 45.0, 1.0, 1.0)
    img = render_rgba(16, 16, [], cam, 1, 1, False, mesh=handle)
    print(f"Rendered image shape: {img.shape}, mean: {img.mean():.1f}")

    if forge3d.enumerate_adapters():
        print("Testing GPU mesh rendering...")
        img_gpu = render_rgba(16, 16, [], cam, 1, 1, True, mesh=handle)
        print(f"GPU image shape: {img_gpu.shape}, mean: {img_gpu.mean():.1f}")
    else:
        print("No GPU adapters available, skipping GPU tests")

    print("All tests completed successfully!")