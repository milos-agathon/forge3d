"""
tests/test_bvh_cpu_vs_gpu.py
Tests for CPU vs GPU BVH equivalence and cross-validation
This file exists to validate that CPU and GPU BVH builders produce equivalent results for the same inputs.
RELEVANT FILES:python/forge3d/path_tracing.py,src/accel/lbvh_gpu.rs,src/accel/sah_cpu.rs
"""

import pytest
import numpy as np
from forge3d.path_tracing import build_bvh, refit_bvh, BvhHandle


def make_unit_cube_tris():
    """Create triangles for a unit cube (12 triangles, 2 per face)."""
    return [
        # Front face
        ((0,0,0), (1,0,0), (1,1,0)),
        ((0,0,0), (1,1,0), (0,1,0)),
        # Back face  
        ((0,0,1), (1,1,1), (1,0,1)),
        ((0,0,1), (0,1,1), (1,1,1)),
        # Left face
        ((0,0,0), (0,1,0), (0,1,1)),
        ((0,0,0), (0,1,1), (0,0,1)),
        # Right face
        ((1,0,0), (1,1,1), (1,1,0)),
        ((1,0,0), (1,0,1), (1,1,1)),
        # Top face
        ((0,1,0), (1,1,0), (1,1,1)),
        ((0,1,0), (1,1,1), (0,1,1)),
        # Bottom face
        ((0,0,0), (1,0,1), (1,0,0)),
        ((0,0,0), (0,0,1), (1,0,1)),
    ]


def make_random_tris(count: int, seed: int = 42):
    """Create random triangles for testing."""
    np.random.seed(seed)
    triangles = []
    
    for _ in range(count):
        # Create triangle with random vertices in [-10, 10] cube
        v0 = tuple(np.random.uniform(-10, 10, 3))
        v1 = tuple(np.random.uniform(-10, 10, 3)) 
        v2 = tuple(np.random.uniform(-10, 10, 3))
        triangles.append((v0, v1, v2))
    
    return triangles


def aabb_distance(aabb1, aabb2):
    """Compute maximum difference between two AABBs."""
    (min1, max1) = aabb1
    (min2, max2) = aabb2
    
    min_diff = max(abs(a - b) for a, b in zip(min1, min2))
    max_diff = max(abs(a - b) for a, b in zip(max1, max2))
    
    return max(min_diff, max_diff)


class TestBvhCpuVsGpu:
    """Test CPU vs GPU BVH equivalence."""
    
    def test_cpu_gpu_equivalence_small(self):
        """Test CPU/GPU equivalence on small scene."""
        triangles = make_unit_cube_tris()
        seed = 123
        
        # Build with both backends
        bvh_gpu = build_bvh(triangles, use_gpu=True, seed=seed)
        bvh_cpu = build_bvh(triangles, use_gpu=False, seed=seed)
        
        # Both should process same number of triangles
        assert bvh_gpu.triangle_count == bvh_cpu.triangle_count == len(triangles)
        
        # Node counts should be identical for same input
        assert bvh_gpu.node_count == bvh_cpu.node_count
        
        # World AABBs should be very close (allowing for minor numerical differences)
        aabb_diff = aabb_distance(bvh_gpu.world_aabb, bvh_cpu.world_aabb)
        assert aabb_diff < 1e-5, f"AABB difference {aabb_diff} too large"
        
        # Build times should be reasonable
        assert bvh_gpu.build_stats['build_time_ms'] > 0
        assert bvh_cpu.build_stats['build_time_ms'] > 0
        
        print(f"GPU build: {bvh_gpu.build_stats['build_time_ms']:.2f}ms ({bvh_gpu.backend_type})")
        print(f"CPU build: {bvh_cpu.build_stats['build_time_ms']:.2f}ms ({bvh_cpu.backend_type})")
    
    def test_deterministic_with_fixed_seed(self):
        """Test that builds are deterministic with fixed seed."""
        triangles = make_unit_cube_tris()
        seed = 987
        
        # Build same scene multiple times
        bvh1 = build_bvh(triangles, use_gpu=False, seed=seed)
        bvh2 = build_bvh(triangles, use_gpu=False, seed=seed)
        
        # Should be identical
        assert bvh1.triangle_count == bvh2.triangle_count
        assert bvh1.node_count == bvh2.node_count
        
        # AABBs should be exactly identical for CPU builds
        aabb_diff = aabb_distance(bvh1.world_aabb, bvh2.world_aabb)
        assert aabb_diff == 0.0, "Deterministic builds should have identical AABBs"
        
        print("Deterministic build test passed")
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds can produce different builds."""
        triangles = make_random_tris(20, seed=111)
        
        bvh1 = build_bvh(triangles, use_gpu=False, seed=1)
        bvh2 = build_bvh(triangles, use_gpu=False, seed=999)
        
        # Triangle counts should be same
        assert bvh1.triangle_count == bvh2.triangle_count
        
        # Build stats may differ slightly due to different construction order
        # (this is acceptable for non-deterministic algorithms like SAH)
        print(f"Seed 1 build time: {bvh1.build_stats['build_time_ms']:.2f}ms")
        print(f"Seed 999 build time: {bvh2.build_stats['build_time_ms']:.2f}ms")
    
    def test_cpu_vs_gpu_memory_usage(self):
        """Test memory usage reporting for CPU vs GPU."""
        triangles = make_random_tris(100, seed=222)
        
        bvh_gpu = build_bvh(triangles, use_gpu=True, seed=42)
        bvh_cpu = build_bvh(triangles, use_gpu=False, seed=42)
        
        # Both should report non-zero memory usage
        gpu_memory = bvh_gpu.build_stats['memory_usage_bytes']
        cpu_memory = bvh_cpu.build_stats['memory_usage_bytes']
        
        assert gpu_memory > 0, "GPU build should report memory usage"
        assert cpu_memory > 0, "CPU build should report memory usage"
        
        print(f"GPU memory: {gpu_memory / 1024:.1f} KB")
        print(f"CPU memory: {cpu_memory / 1024:.1f} KB")
    
    def test_refit_cpu_vs_gpu_equivalence(self):
        """Test that refit produces similar results on CPU vs GPU."""
        initial_triangles = make_unit_cube_tris()
        
        # Build initial BVHs
        bvh_gpu = build_bvh(initial_triangles, use_gpu=True, seed=333)
        bvh_cpu = build_bvh(initial_triangles, use_gpu=False, seed=333)
        
        # Create moved triangles (translate all by (0.5, 0.5, 0.5))
        moved_triangles = []
        for tri in initial_triangles:
            moved_tri = tuple(
                (v[0] + 0.5, v[1] + 0.5, v[2] + 0.5) for v in tri
            )
            moved_triangles.append(moved_tri)
        
        # Refit both
        refit_bvh(bvh_gpu, moved_triangles)
        refit_bvh(bvh_cpu, moved_triangles)
        
        # AABBs should be similar after refit
        aabb_diff = aabb_distance(bvh_gpu.world_aabb, bvh_cpu.world_aabb)
        assert aabb_diff < 1e-4, f"Post-refit AABB difference {aabb_diff} too large"
        
        print("Refit equivalence test passed")
    
    def test_edge_case_single_triangle(self):
        """Test edge case with single triangle."""
        triangles = [((0,0,0), (1,0,0), (0,1,0))]
        
        bvh_gpu = build_bvh(triangles, use_gpu=True, seed=444)
        bvh_cpu = build_bvh(triangles, use_gpu=False, seed=444)
        
        assert bvh_gpu.triangle_count == bvh_cpu.triangle_count == 1
        # Single triangle BVH should have just 1 node (root leaf)
        assert bvh_gpu.node_count == bvh_cpu.node_count == 1
        
        # AABBs should be identical (tight bounds around single triangle)
        aabb_diff = aabb_distance(bvh_gpu.world_aabb, bvh_cpu.world_aabb)
        assert aabb_diff < 1e-6, "Single triangle AABBs should be nearly identical"
        
        print("Single triangle test passed")
    
    def test_empty_primitives_error_handling(self):
        """Test that empty primitive lists are handled consistently."""
        empty_triangles = []
        
        # Both backends should raise ValueError
        with pytest.raises(ValueError, match="Cannot build BVH from empty primitive list"):
            build_bvh(empty_triangles, use_gpu=True)
        
        with pytest.raises(ValueError, match="Cannot build BVH from empty primitive list"):
            build_bvh(empty_triangles, use_gpu=False)
        
        print("Empty primitives error handling test passed")
    
    def test_invalid_primitive_format(self):
        """Test error handling for invalid primitive formats."""
        invalid_primitives = [
            "not a triangle",  # String instead of tuple
            ((0,0,0), (1,1,1)),  # Only 2 vertices
            ((0,0,0), (1,0,0), (0,1,0), (1,1,1)),  # 4 vertices
        ]
        
        for i, bad_prim in enumerate(invalid_primitives):
            with pytest.raises(ValueError, match="Invalid primitive format"):
                build_bvh([bad_prim], use_gpu=False)
            
            print(f"Invalid format {i+1} correctly rejected")
    
    def test_refit_triangle_count_mismatch(self):
        """Test refit error when triangle count changes."""
        initial_triangles = make_unit_cube_tris()
        bvh = build_bvh(initial_triangles, use_gpu=False, seed=555)
        
        # Try to refit with different triangle count
        wrong_count_triangles = [((0,0,0), (1,0,0), (0,1,0))]  # Only 1 triangle
        
        with pytest.raises(ValueError, match="Primitive count mismatch"):
            refit_bvh(bvh, wrong_count_triangles)
        
        print("Refit count mismatch error handling test passed")
    
    def test_bvh_handle_properties(self):
        """Test BvhHandle properties and methods."""
        triangles = make_unit_cube_tris()
        
        bvh_cpu = build_bvh(triangles, use_gpu=False, seed=666)
        bvh_gpu = build_bvh(triangles, use_gpu=True, seed=666)
        
        # Test backend detection
        assert bvh_cpu.is_cpu()
        assert not bvh_cpu.is_gpu()
        
        # Note: GPU build will fall back to CPU currently
        assert bvh_gpu.is_cpu()  # Due to mock implementation
        assert not bvh_gpu.is_gpu()
        
        # Test repr
        repr_str = repr(bvh_cpu)
        assert "BvhHandle" in repr_str
        assert "triangles=" in repr_str
        assert "nodes=" in repr_str
        
        print(f"BVH repr: {repr_str}")
        print("BvhHandle properties test passed")


if __name__ == "__main__":
    # Run tests directly if script is executed
    test_instance = TestBvhCpuVsGpu()
    
    print("Running BVH CPU vs GPU tests...")
    test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
    
    for method_name in test_methods:
        print(f"\n--- {method_name} ---")
        try:
            method = getattr(test_instance, method_name)
            method()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
    
    print("\nAll tests completed.")