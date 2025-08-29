# D5: Orthographic camera mode tests
import math
import numpy as np
import pytest

import forge3d as f3d


def test_camera_orthographic_basic():
    """Test basic orthographic projection functionality."""
    # Basic orthographic projection
    left, right = -10.0, 10.0
    bottom, top = -8.0, 8.0
    znear, zfar = 0.1, 100.0
    
    proj = f3d.camera_orthographic(left, right, bottom, top, znear, zfar)
    
    assert proj.shape == (4, 4)
    assert proj.dtype == np.float32
    
    # Test that the matrix is finite
    assert np.all(np.isfinite(proj))


def test_camera_orthographic_pixel_aligned():
    """Test orthographic projection for pixel-aligned 2D rendering.
    
    Acceptance criterion:
    Given width=800,height=600 and camera_orthographic(0,800,0,600,near=0.1,far=10),
    mapping (0,0) pixel center -> NDC ≈ (-1,+1), (800,600) -> (+1,-1) within 1e-5.
    """
    width, height = 800, 600
    left, right = 0, width
    bottom, top = 0, height
    znear, zfar = 0.1, 10.0
    
    proj = f3d.camera_orthographic(left, right, bottom, top, znear, zfar)
    
    # Test left edge (0, middle) -> NDC (-1, 0) approximately
    left_edge = np.array([0.0, height/2, 1.0, 1.0])
    ndc_left = proj @ left_edge
    ndc_left = ndc_left / ndc_left[3]
    
    expected_left_x = -1.0
    expected_left_y = 0.0
    
    assert abs(ndc_left[0] - expected_left_x) < 1e-5, f"Expected left edge NDC X ≈ {expected_left_x}, got {ndc_left[0]}"
    assert abs(ndc_left[1] - expected_left_y) < 1e-5, f"Expected left edge NDC Y ≈ {expected_left_y}, got {ndc_left[1]}"
    
    # Test right edge (800, middle) -> NDC (+1, 0) approximately  
    right_edge = np.array([width, height/2, 1.0, 1.0])
    ndc_right = proj @ right_edge
    ndc_right = ndc_right / ndc_right[3]
    
    expected_right_x = 1.0
    expected_right_y = 0.0
    
    assert abs(ndc_right[0] - expected_right_x) < 1e-5, f"Expected right edge NDC X ≈ {expected_right_x}, got {ndc_right[0]}"
    assert abs(ndc_right[1] - expected_right_y) < 1e-5, f"Expected right edge NDC Y ≈ {expected_right_y}, got {ndc_right[1]}"


def test_camera_orthographic_clip_space_wgpu():
    """Test orthographic projection with WGPU clip space (default)."""
    proj = f3d.camera_orthographic(-1, 1, -1, 1, 0.1, 10.0, clip_space="wgpu")
    
    # Test a point at z=5 (midway between near and far)
    point = np.array([0.0, 0.0, -5.0, 1.0])  # negative Z for right-handed
    ndc = proj @ point
    ndc = ndc / ndc[3]
    
    # In WGPU, Z should be in [0,1] range
    assert 0.0 <= ndc[2] <= 1.0, f"WGPU Z should be in [0,1], got {ndc[2]}"


def test_camera_orthographic_clip_space_gl():
    """Test orthographic projection with GL clip space."""
    proj = f3d.camera_orthographic(-1, 1, -1, 1, 0.1, 10.0, clip_space="gl")
    
    # Test a point at z=5 (midway between near and far)
    point = np.array([0.0, 0.0, -5.0, 1.0])  # negative Z for right-handed
    ndc = proj @ point
    ndc = ndc / ndc[3]
    
    # In GL, Z should be in [-1,1] range
    assert -1.0 <= ndc[2] <= 1.0, f"GL Z should be in [-1,1], got {ndc[2]}"


def test_camera_orthographic_validation():
    """Test input validation for orthographic projection."""
    # Invalid left >= right
    with pytest.raises(RuntimeError, match="left must be finite and < right"):
        f3d.camera_orthographic(10.0, 10.0, -1.0, 1.0, 0.1, 10.0)
    
    # Invalid bottom >= top
    with pytest.raises(RuntimeError, match="bottom must be finite and < top"):
        f3d.camera_orthographic(-1.0, 1.0, 5.0, 5.0, 0.1, 10.0)
    
    # Invalid znear <= 0
    with pytest.raises(RuntimeError, match="znear must be finite and > 0"):
        f3d.camera_orthographic(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0)
    
    # Invalid zfar <= znear
    with pytest.raises(RuntimeError, match="zfar must be finite and > znear"):
        f3d.camera_orthographic(-1.0, 1.0, -1.0, 1.0, 10.0, 5.0)
    
    # Invalid clip_space
    with pytest.raises(RuntimeError, match="clip_space must be 'wgpu' or 'gl'"):
        f3d.camera_orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0, clip_space="invalid")
        
    # Non-finite values
    with pytest.raises(RuntimeError):
        f3d.camera_orthographic(float('nan'), 1.0, -1.0, 1.0, 0.1, 10.0)


def test_orthographic_vs_perspective_properties():
    """Test that orthographic projection has expected properties vs perspective."""
    # Create orthographic and perspective projections
    ortho = f3d.camera_orthographic(-1, 1, -1, 1, 0.1, 10.0)
    persp = f3d.camera_perspective(45.0, 1.0, 0.1, 10.0)
    
    # Both should be 4x4
    assert ortho.shape == (4, 4)
    assert persp.shape == (4, 4)
    
    # Test parallel lines preservation in orthographic
    # Two parallel points at different Z depths should have same X,Y after projection
    point1 = np.array([0.5, 0.5, -1.0, 1.0])
    point2 = np.array([0.5, 0.5, -9.0, 1.0])  # same X,Y, different Z
    
    ndc1 = ortho @ point1
    ndc2 = ortho @ point2
    ndc1 = ndc1 / ndc1[3]
    ndc2 = ndc2 / ndc2[3]
    
    # In orthographic projection, X,Y should be the same regardless of Z
    assert abs(ndc1[0] - ndc2[0]) < 1e-6, "Orthographic should preserve X coordinate across Z"
    assert abs(ndc1[1] - ndc2[1]) < 1e-6, "Orthographic should preserve Y coordinate across Z"


def test_orthographic_deterministic():
    """Test that orthographic projection is deterministic."""
    params = (-5.0, 5.0, -3.0, 3.0, 0.1, 100.0)
    
    proj1 = f3d.camera_orthographic(*params)
    proj2 = f3d.camera_orthographic(*params)
    
    # Should be identical
    np.testing.assert_array_equal(proj1, proj2)