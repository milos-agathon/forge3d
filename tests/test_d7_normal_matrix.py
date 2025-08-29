# D7: Normal matrix computation tests
import math
import numpy as np
import pytest

import forge3d as f3d


def test_normal_matrix_identity():
    """Test normal matrix computation for identity model matrix."""
    identity = np.eye(4, dtype=np.float32)
    normal_matrix = f3d.compute_normal_matrix(identity)
    
    assert normal_matrix.shape == (4, 4)
    assert normal_matrix.dtype == np.float32
    
    # For identity matrix, normal matrix should also be identity (upper-left 3x3)
    expected_3x3 = np.eye(3, dtype=np.float32)
    actual_3x3 = normal_matrix[0:3, 0:3]
    
    np.testing.assert_allclose(actual_3x3, expected_3x3, atol=1e-6)


def test_normal_matrix_translation_only():
    """Test that translation doesn't affect normal matrix."""
    # Translation matrix should not affect normals
    T = f3d.translate(5.0, 10.0, -3.0)
    normal_matrix = f3d.compute_normal_matrix(T)
    
    # Normal matrix should be identity for pure translation
    expected_3x3 = np.eye(3, dtype=np.float32)
    actual_3x3 = normal_matrix[0:3, 0:3]
    
    np.testing.assert_allclose(actual_3x3, expected_3x3, atol=1e-6,
                              err_msg="Translation should not affect normal matrix")


def test_normal_matrix_uniform_scale():
    """Test normal matrix for uniform scaling."""
    # Uniform scale by 2.0
    S = f3d.scale_uniform(2.0)
    normal_matrix = f3d.compute_normal_matrix(S)
    
    # For uniform scale, normal matrix should be inverse of scale = 1/scale
    # Since we're dealing with the inverse transpose, uniform scale becomes 1/scale
    expected_scale = 1.0 / 2.0
    expected_3x3 = np.eye(3, dtype=np.float32) * expected_scale
    actual_3x3 = normal_matrix[0:3, 0:3]
    
    np.testing.assert_allclose(actual_3x3, expected_3x3, rtol=1e-6,
                              err_msg="Uniform scale normal matrix should be 1/scale * I")


def test_normal_matrix_non_uniform_scale():
    """Test normal matrix for non-uniform scaling (the key use case)."""
    # Non-uniform scale - this is where normal matrix is crucial
    S = f3d.scale(2.0, 3.0, 4.0)
    normal_matrix = f3d.compute_normal_matrix(S)
    
    # For diagonal scale matrix S = diag(sx, sy, sz),
    # normal matrix = (S^-1)^T = diag(1/sx, 1/sy, 1/sz)
    expected_diag = np.array([1.0/2.0, 1.0/3.0, 1.0/4.0], dtype=np.float32)
    actual_diag = np.diag(normal_matrix[0:3, 0:3])
    
    np.testing.assert_allclose(actual_diag, expected_diag, rtol=1e-6,
                              err_msg="Non-uniform scale normal matrix should be reciprocal scales")


def test_normal_matrix_rotation():
    """Test that rotations preserve normal matrix orthogonality."""
    # 45-degree rotation around Y axis
    R = f3d.rotate_y(45.0)
    normal_matrix = f3d.compute_normal_matrix(R)
    
    # For pure rotation, normal matrix should equal the rotation matrix
    # (since rotation matrices are orthogonal: R^-1 = R^T, so (R^-1)^T = R)
    R_3x3 = R[0:3, 0:3]
    normal_3x3 = normal_matrix[0:3, 0:3]
    
    np.testing.assert_allclose(normal_3x3, R_3x3, rtol=1e-6,
                              err_msg="Normal matrix for rotation should equal rotation matrix")
    
    # Check that it's still orthogonal (R * R^T = I)
    should_be_identity = normal_3x3 @ normal_3x3.T
    np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-6,
                              err_msg="Normal matrix should preserve orthogonality")


def test_normal_matrix_complex_transform():
    """Test normal matrix for combined T*R*S transformation."""
    # Create a complex transformation
    translation = (1.0, 2.0, 3.0)
    rotation = (30.0, 45.0, 60.0)  # Euler angles
    scale = (2.0, 0.5, 3.0)  # Non-uniform scale
    
    model_matrix = f3d.compose_trs(translation, rotation, scale)
    normal_matrix = f3d.compute_normal_matrix(model_matrix)
    
    assert normal_matrix.shape == (4, 4)
    assert normal_matrix.dtype == np.float32
    assert np.all(np.isfinite(normal_matrix))
    
    # For TRS, the normal matrix should be computed from the 3x3 upper-left part
    # The exact values are complex, but we can verify basic properties
    normal_3x3 = normal_matrix[0:3, 0:3]
    
    # Check that it's invertible (non-zero determinant)
    det = np.linalg.det(normal_3x3)
    assert abs(det) > 1e-6, "Normal matrix should be invertible"


def test_normal_matrix_deterministic():
    """Test that normal matrix computation is deterministic."""
    # Same model matrix should give same normal matrix
    model1 = f3d.compose_trs((1.0, 2.0, 3.0), (45.0, 30.0, 60.0), (2.0, 1.5, 0.8))
    model2 = f3d.compose_trs((1.0, 2.0, 3.0), (45.0, 30.0, 60.0), (2.0, 1.5, 0.8))
    
    normal1 = f3d.compute_normal_matrix(model1)
    normal2 = f3d.compute_normal_matrix(model2)
    
    np.testing.assert_array_equal(normal1, normal2)


def test_normal_vector_transformation():
    """Test actual normal vector transformation using the normal matrix."""
    # Create a model matrix with non-uniform scaling
    model_matrix = f3d.scale(2.0, 1.0, 0.5)  # Squash in Z, stretch in X
    normal_matrix = f3d.compute_normal_matrix(model_matrix)
    
    # Test normal vector pointing up (Y direction)
    normal_vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # w=0 for vector
    
    # Transform the normal
    transformed = normal_matrix @ normal_vector
    transformed_3d = transformed[0:3]
    
    # Normalize the result (as would happen in the shader)
    normalized = transformed_3d / np.linalg.norm(transformed_3d)
    
    # For this specific case (scale Y by 1.0), Y component should remain dominant
    # but the exact values depend on the inverse transpose calculation
    assert normalized[1] > 0.5, "Y component should remain significant"
    assert np.abs(np.linalg.norm(normalized) - 1.0) < 1e-6, "Result should be normalized"


def test_normal_matrix_edge_cases():
    """Test normal matrix computation with edge cases."""
    # Very small scale (near zero)
    small_scale = f3d.scale_uniform(1e-6)
    try:
        normal_matrix = f3d.compute_normal_matrix(small_scale)
        assert np.all(np.isfinite(normal_matrix)), "Normal matrix should handle small scales"
    except Exception:
        # It's acceptable to fail for degenerate cases
        pass
    
    # Large scale
    large_scale = f3d.scale_uniform(1e6)
    normal_matrix = f3d.compute_normal_matrix(large_scale)
    assert np.all(np.isfinite(normal_matrix)), "Normal matrix should handle large scales"


def test_normal_matrix_validation():
    """Test input validation for normal matrix computation."""
    # Test with invalid matrix shape
    invalid_matrix = np.ones((3, 3), dtype=np.float32)
    
    with pytest.raises(RuntimeError, match="Expected \\(4,4\\) matrix"):
        f3d.compute_normal_matrix(invalid_matrix)


def test_normal_matrix_mathematical_property():
    """Test the mathematical property: (M^-1)^T transforms normals correctly."""
    # Create a simple shear transformation (not just T/R/S)
    # This tests the general mathematical correctness
    shear_matrix = np.array([
        [1.0, 0.5, 0.0, 0.0],  # Shear in X based on Y
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    normal_matrix = f3d.compute_normal_matrix(shear_matrix)
    
    # Verify the mathematical relationship: normal_matrix = (M^-1)^T for the 3x3 part
    M_3x3 = shear_matrix[0:3, 0:3]
    M_inv = np.linalg.inv(M_3x3)
    expected_normal = M_inv.T
    
    actual_normal = normal_matrix[0:3, 0:3]
    
    np.testing.assert_allclose(actual_normal, expected_normal, rtol=1e-6,
                              err_msg="Normal matrix should equal (M^-1)^T")


def test_normal_matrix_in_terrain_context():
    """Test normal matrix in the context of terrain rendering (integration test)."""
    # This would test the full pipeline: model matrix -> normal matrix -> shader
    # For now, just verify that we can create terrain uniforms with normal matrices
    
    # Create a simple model transformation for terrain
    terrain_transform = f3d.compose_trs((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))  # Stretch Y
    normal_matrix = f3d.compute_normal_matrix(terrain_transform)
    
    # Verify that normal matrix correctly handles the Y-stretch
    # When model stretches Y by 2, normals should be compressed by 1/2 in Y
    expected_y_scale = 1.0 / 2.0  # Inverse of the Y scale
    actual_y_scale = normal_matrix[1, 1]  # Y component of normal matrix
    
    np.testing.assert_allclose(actual_y_scale, expected_y_scale, rtol=1e-6,
                              err_msg="Normal matrix should handle terrain Y-exaggeration correctly")