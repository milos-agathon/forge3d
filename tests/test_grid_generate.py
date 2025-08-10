"""
Tests for grid_generate function.
"""
import pytest
import numpy as np
from vulkan_forge import grid_generate


def test_grid_generate_basic():
    """Test grid (4,3) with spacing (2,1): assert XY=(12,2) float32, UV=(12,2) float32, IDX=36 uint32"""
    xy, uv, indices = grid_generate(4, 3, spacing=(2.0, 1.0))
    
    # Check shapes and dtypes
    assert xy.shape == (12, 2), f"Expected XY shape (12, 2), got {xy.shape}"
    assert uv.shape == (12, 2), f"Expected UV shape (12, 2), got {uv.shape}" 
    assert indices.shape == (36,), f"Expected indices shape (36,), got {indices.shape}"
    
    assert xy.dtype == np.float32, f"Expected XY dtype float32, got {xy.dtype}"
    assert uv.dtype == np.float32, f"Expected UV dtype float32, got {uv.dtype}"
    assert indices.dtype == np.uint32, f"Expected indices dtype uint32, got {indices.dtype}"


def test_uv_corners():
    """Check UV corners: (0,0), (1,0), (0,1), (1,1)"""
    xy, uv, indices = grid_generate(4, 3, spacing=(2.0, 1.0))
    
    # For a 4x3 grid (nx=4, nz=3), vertices are arranged as:
    # Row 0: (0,0), (1,0), (2,0), (3,0)  - indices 0,1,2,3
    # Row 1: (0,1), (1,1), (2,1), (3,1)  - indices 4,5,6,7  
    # Row 2: (0,2), (1,2), (2,2), (3,2)  - indices 8,9,10,11
    
    # UV corners should be:
    # Bottom-left (0,0): index 0
    # Bottom-right (1,0): index 3  
    # Top-left (0,1): index 8
    # Top-right (1,1): index 11
    
    np.testing.assert_array_almost_equal(uv[0], [0.0, 0.0], err_msg="UV corner (0,0) mismatch")
    np.testing.assert_array_almost_equal(uv[3], [1.0, 0.0], err_msg="UV corner (1,0) mismatch")
    np.testing.assert_array_almost_equal(uv[8], [0.0, 1.0], err_msg="UV corner (0,1) mismatch")  
    np.testing.assert_array_almost_equal(uv[11], [1.0, 1.0], err_msg="UV corner (1,1) mismatch")


def test_ccw_winding_first_triangle():
    """Check CCW winding on first triangle using XY positions"""
    xy, uv, indices = grid_generate(3, 3, spacing=(1.0, 1.0))
    
    # Get first triangle indices
    i0, i1, i2 = indices[0], indices[1], indices[2]
    
    # Get positions
    p0 = xy[i0]
    p1 = xy[i1] 
    p2 = xy[i2]
    
    # Compute 2D cross product (z component of 3D cross product)
    # For CCW winding, (p1-p0) × (p2-p0) should have positive z
    edge1 = p1 - p0
    edge2 = p2 - p0
    cross_z = edge1[0] * edge2[1] - edge1[1] * edge2[0]
    
    assert cross_z > 0, f"First triangle should be CCW, but cross_z = {cross_z} <= 0"


def test_large_grid_uint32():
    """Check large grid (256,256) returns uint32 indices"""
    xy, uv, indices = grid_generate(256, 256, spacing=(1.0, 1.0))
    
    # Check that indices are uint32
    assert indices.dtype == np.uint32, f"Expected indices dtype uint32, got {indices.dtype}"
    
    # Check shapes
    assert xy.shape == (256 * 256, 2), f"Expected XY shape ({256*256}, 2), got {xy.shape}"
    assert uv.shape == (256 * 256, 2), f"Expected UV shape ({256*256}, 2), got {uv.shape}"
    
    # Number of triangles = (nx-1) * (nz-1) * 2 = 255 * 255 * 2 = 130050
    # Number of indices = 130050 * 3 = 390150
    expected_indices = 255 * 255 * 2 * 3
    assert indices.shape == (expected_indices,), f"Expected {expected_indices} indices, got {indices.shape[0]}"


def test_validation_errors():
    """Test validation errors"""
    # Test nx < 2
    with pytest.raises(ValueError, match="nx and nz must be >= 2"):
        grid_generate(1, 3)
    
    # Test nz < 2  
    with pytest.raises(ValueError, match="nx and nz must be >= 2"):
        grid_generate(3, 1)
    
    # Test invalid spacing
    with pytest.raises(ValueError, match="spacing components must be finite and > 0"):
        grid_generate(3, 3, spacing=(0.0, 1.0))
    
    with pytest.raises(ValueError, match="spacing components must be finite and > 0"):
        grid_generate(3, 3, spacing=(1.0, -1.0))
    
    with pytest.raises(ValueError, match="spacing components must be finite and > 0"):
        grid_generate(3, 3, spacing=(float('inf'), 1.0))
    
    # Test invalid origin
    with pytest.raises(ValueError, match="origin must be 'center'"):
        grid_generate(3, 3, origin="corner")


def test_centered_grid():
    """Test that grid is properly centered at origin"""
    xy, uv, indices = grid_generate(3, 3, spacing=(2.0, 2.0))
    
    # For centered 3x3 grid with spacing (2,2):
    # Grid spans from -2 to +2 in both X and Y
    # Positions should be: (-2,-2), (0,-2), (2,-2), (-2,0), (0,0), (2,0), (-2,2), (0,2), (2,2)
    
    expected_positions = [
        [-2.0, -2.0], [0.0, -2.0], [2.0, -2.0],  # Row 0
        [-2.0,  0.0], [0.0,  0.0], [2.0,  0.0],  # Row 1
        [-2.0,  2.0], [0.0,  2.0], [2.0,  2.0]   # Row 2
    ]
    
    np.testing.assert_array_almost_equal(xy, expected_positions, err_msg="Grid not properly centered")


if __name__ == "__main__":
    pytest.main([__file__])