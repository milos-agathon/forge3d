# D4: Model transforms & math helpers tests
import math
import numpy as np
import pytest

import forge3d as f3d


def test_basic_transforms():
    """Test basic T/R/S transform creation."""
    # Translation
    T = f3d.translate(1.0, 2.0, 3.0)
    assert T.shape == (4, 4)
    assert T.dtype == np.float32
    expected_translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(T[0:3, 3], expected_translation, rtol=1e-6)
    
    # Rotation around Z (90 degrees should swap X and Y)
    Rz = f3d.rotate_z(90.0)
    assert Rz.shape == (4, 4)
    assert Rz.dtype == np.float32
    
    # Scale
    S = f3d.scale(2.0, 1.0, 0.5)
    assert S.shape == (4, 4) 
    assert S.dtype == np.float32
    expected_diagonal = np.array([2.0, 1.0, 0.5, 1.0], dtype=np.float32)
    actual_diagonal = np.diag(S)
    np.testing.assert_allclose(actual_diagonal, expected_diagonal, rtol=1e-6)


def test_rotation_correctness():
    """Test that rotations work correctly."""
    # 90 degree rotation around Z should transform (1,0,0) -> (0,1,0)
    Rz90 = f3d.rotate_z(90.0)
    point = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Homogeneous
    rotated = Rz90 @ point
    
    expected = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(rotated, expected, atol=1e-6)
    
    # 90 degree rotation around X should transform (0,1,0) -> (0,0,1)
    Rx90 = f3d.rotate_x(90.0)
    point_y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    rotated_y = Rx90 @ point_y
    
    expected_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(rotated_y, expected_y, atol=1e-6)
    
    # 90 degree rotation around Y should transform (1,0,0) -> (0,0,-1)
    Ry90 = f3d.rotate_y(90.0)
    point_x = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotated_x = Ry90 @ point_x
    
    expected_x = np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(rotated_x, expected_x, atol=1e-6)


def test_compose_trs_acceptance_criterion():
    """Test the specific acceptance criterion: compose_model(T(1,2,3), Rz(90deg), S(2,1,1))."""
    # From task.xml: compose_model(T(1,2,3), Rz(90deg), S(2,1,1)) applied to unit x-axis 
    # yields world ≈ (1,2,3)+Rz(90)*(2,0,0) within 1e-5.
    
    translation = (1.0, 2.0, 3.0)
    rotation = (0.0, 0.0, 90.0)  # 90 degrees around Z
    scale = (2.0, 1.0, 1.0)
    
    model_matrix = f3d.compose_trs(translation, rotation, scale)
    
    # Apply to unit X-axis
    unit_x = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    result = model_matrix @ unit_x
    
    # Expected: (1,2,3) + Rz(90) * (2,0,0) = (1,2,3) + (0,2,0) = (1,4,3)
    # Because Rz(90) transforms (2,0,0) -> (0,2,0)
    expected = np.array([1.0, 4.0, 3.0, 1.0], dtype=np.float32)
    
    np.testing.assert_allclose(result, expected, atol=1e-5,
                              err_msg="TRS composition doesn't match acceptance criterion")


def test_trs_order():
    """Test that T*R*S order is applied correctly (scale first, then rotate, then translate)."""
    # Create a point that will show the transformation order clearly
    # Start with point (1, 0, 0)
    # 1. Scale by (2, 1, 1) -> (2, 0, 0) 
    # 2. Rotate 90° around Z -> (0, 2, 0)
    # 3. Translate by (1, 1, 0) -> (1, 3, 0)
    
    T = f3d.translate(1.0, 1.0, 0.0)
    R = f3d.rotate_z(90.0)
    S = f3d.scale(2.0, 1.0, 1.0)
    
    # Manual composition: T * R * S
    manual_trs = T @ R @ S
    
    # Using compose_trs
    composed_trs = f3d.compose_trs((1.0, 1.0, 0.0), (0.0, 0.0, 90.0), (2.0, 1.0, 1.0))
    
    # Both should be the same (adjust tolerance for floating-point precision)
    np.testing.assert_allclose(manual_trs, composed_trs, rtol=1e-6, atol=1e-6)
    
    # Test with unit X vector
    point = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    result = composed_trs @ point
    expected = np.array([1.0, 3.0, 0.0, 1.0], dtype=np.float32)  # As calculated above
    
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_matrix_operations():
    """Test matrix multiplication and inversion."""
    # Create two matrices
    T = f3d.translate(1.0, 0.0, 0.0)
    R = f3d.rotate_z(45.0)
    
    # Multiply them
    TR = f3d.multiply_matrices(T, R)
    
    # Should be same as T @ R
    expected = T @ R
    np.testing.assert_allclose(TR, expected, rtol=1e-6)
    
    # Test inversion
    T_inv = f3d.invert_matrix(T)
    identity = f3d.multiply_matrices(T, T_inv)
    
    expected_identity = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(identity, expected_identity, atol=1e-6)


def test_uniform_scale():
    """Test uniform scaling utility."""
    S_uniform = f3d.scale_uniform(2.5)
    S_manual = f3d.scale(2.5, 2.5, 2.5)
    
    np.testing.assert_allclose(S_uniform, S_manual, rtol=1e-6)


def test_look_at_transform():
    """Test object look-at transformation."""
    # Object at origin looking towards (1, 0, 0)
    transform = f3d.look_at_transform((0, 0, 0), (1, 0, 0), (0, 1, 0))
    
    assert transform.shape == (4, 4)
    assert transform.dtype == np.float32
    
    # The transform should be finite
    assert np.all(np.isfinite(transform))
    
    # Position should be at origin
    position = transform[0:3, 3]
    expected_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(position, expected_pos, atol=1e-6)


def test_normal_matrix():
    """Test normal matrix computation."""
    # Create a model matrix with non-uniform scale
    model = f3d.compose_trs((0, 0, 0), (0, 0, 0), (2.0, 1.0, 3.0))
    
    # Compute normal matrix
    normal_mat = f3d.compute_normal_matrix(model)
    
    assert normal_mat.shape == (4, 4)
    assert normal_mat.dtype == np.float32
    assert np.all(np.isfinite(normal_mat))
    
    # For a scale matrix, the normal matrix should have reciprocal scales
    # (since it's the inverse transpose of the upper-left 3x3)
    # The exact values depend on the normalization, but it should be invertible
    det = np.linalg.det(normal_mat[0:3, 0:3])  # Upper-left 3x3
    assert abs(det) > 1e-6, "Normal matrix should be invertible"


def test_transform_validation():
    """Test input validation for transform functions."""
    # Test that invalid matrices are rejected
    invalid_matrix = np.ones((3, 3), dtype=np.float32)  # Wrong shape
    
    with pytest.raises(RuntimeError, match="Expected \\(4,4\\) matrix"):
        f3d.multiply_matrices(invalid_matrix, np.eye(4, dtype=np.float32))
    
    # Test that non-contiguous arrays are handled
    non_contiguous = np.eye(4, dtype=np.float32)[::2, ::2]  # This creates a non-contiguous view
    
    # This might pass or fail depending on implementation - the key is it should not crash
    try:
        result = f3d.invert_matrix(non_contiguous)
        assert result.shape == (4, 4)
    except RuntimeError:
        # This is acceptable - we can reject non-contiguous arrays
        pass


def test_transform_deterministic():
    """Test that transform functions are deterministic."""
    # Same inputs should give same outputs
    T1 = f3d.translate(1.5, 2.5, 3.5)
    T2 = f3d.translate(1.5, 2.5, 3.5)
    np.testing.assert_array_equal(T1, T2)
    
    R1 = f3d.rotate_y(37.5)
    R2 = f3d.rotate_y(37.5)
    np.testing.assert_array_equal(R1, R2)
    
    S1 = f3d.scale(1.1, 2.2, 3.3)
    S2 = f3d.scale(1.1, 2.2, 3.3)
    np.testing.assert_array_equal(S1, S2)


def test_identity_transforms():
    """Test that identity transforms work correctly."""
    # Zero translation should be identity
    T_zero = f3d.translate(0.0, 0.0, 0.0)
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(T_zero, expected, atol=1e-6)
    
    # Zero rotation should be identity
    R_zero = f3d.rotate_z(0.0)
    np.testing.assert_allclose(R_zero, expected, atol=1e-6)
    
    # Unit scale should be identity  
    S_unit = f3d.scale(1.0, 1.0, 1.0)
    np.testing.assert_allclose(S_unit, expected, atol=1e-6)
    
    # Unit uniform scale should be identity
    S_uniform = f3d.scale_uniform(1.0)
    np.testing.assert_allclose(S_uniform, expected, atol=1e-6)


def test_compose_trs_edge_cases():
    """Test TRS composition with edge cases.""" 
    # Identity composition
    identity_trs = f3d.compose_trs((0, 0, 0), (0, 0, 0), (1, 1, 1))
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(identity_trs, expected, atol=1e-6)
    
    # Large values
    large_trs = f3d.compose_trs((1000, 2000, 3000), (180, 180, 180), (10, 20, 30))
    assert large_trs.shape == (4, 4)
    assert np.all(np.isfinite(large_trs))
    
    # Small values
    small_trs = f3d.compose_trs((1e-3, 2e-3, 3e-3), (0.1, 0.2, 0.3), (1.001, 0.999, 1.002))
    assert small_trs.shape == (4, 4)
    assert np.all(np.isfinite(small_trs))