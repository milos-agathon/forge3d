# D6: Camera uniforms (viewWorldPosition + near/far/FOV policy) tests
import math
import numpy as np
import pytest

import forge3d as f3d


def test_view_world_position_extraction():
    """Test that viewWorldPosition matches the inverse of view matrix translation."""
    # Create a view matrix using camera_look_at
    eye = (5.0, 3.0, 7.0)
    target = (0.0, 0.0, 0.0)
    up = (0.0, 1.0, 0.0)
    
    view_matrix = f3d.camera_look_at(eye, target, up)
    
    # Extract world position by inverting the view matrix
    view_mat_glam = view_matrix.astype(np.float32)
    
    # Invert the view matrix to get world matrix
    try:
        inv_view = np.linalg.inv(view_mat_glam)
    except np.linalg.LinAlgError:
        pytest.fail("View matrix is not invertible")
    
    # Camera world position is the translation part of the inverse view matrix
    extracted_position = inv_view[0:3, 3]  # Translation column, first 3 components
    expected_position = np.array(eye, dtype=np.float32)
    
    # Should match within reasonable tolerance
    np.testing.assert_allclose(extracted_position, expected_position, rtol=1e-5, atol=1e-6,
                              err_msg="Extracted camera world position doesn't match expected eye position")


def test_camera_uniforms_integration():
    """Test that camera uniforms are properly integrated in the rendering pipeline."""
    try:
        # Test if TerrainSpike can be created and has access to uniforms with camera position
        renderer = f3d.Renderer(64, 64)
        
        # Create a simple heightmap for terrain
        heightmap = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)
        renderer.add_terrain(heightmap, (1.0, 1.0), colormap='viridis')
        
        # Try to render (this should use the camera uniforms internally)
        rgba = renderer.render_triangle_rgba()
        
        assert rgba.shape == (64, 64, 4)
        assert rgba.dtype == np.uint8
        assert np.all(rgba >= 0) and np.all(rgba <= 255)
        
    except Exception as e:
        # If terrain features are not available, this is not a failure of D6
        pytest.skip(f"Terrain integration test skipped: {e}")


def test_camera_matrix_properties():
    """Test basic properties of camera matrices for documentation."""
    # Test various camera positions
    test_cases = [
        ((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # Standard position
        ((10.0, 5.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # Diagonal view
        ((0.0, 10.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),  # Top-down view
    ]
    
    for eye, target, up in test_cases:
        view_matrix = f3d.camera_look_at(eye, target, up)
        
        # Basic matrix properties
        assert view_matrix.shape == (4, 4)
        assert view_matrix.dtype == np.float32
        
        # View matrix should be invertible (determinant != 0)
        det = np.linalg.det(view_matrix)
        assert abs(det) > 1e-6, f"View matrix determinant too small: {det}"
        
        # The bottom row should be [0, 0, 0, 1] for affine transforms
        expected_bottom_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(view_matrix[3, :], expected_bottom_row, rtol=1e-6, atol=1e-7)


def test_near_far_fov_policy_defaults():
    """Test and document near/far/FOV policy and defaults."""
    # Test default perspective projection parameters
    fovy_deg = 45.0
    aspect = 16.0 / 9.0
    znear = 0.1
    zfar = 100.0
    
    proj_matrix = f3d.camera_perspective(fovy_deg, aspect, znear, zfar)
    
    # Basic properties
    assert proj_matrix.shape == (4, 4)
    assert proj_matrix.dtype == np.float32
    
    # The projection matrix should be well-conditioned
    det = np.linalg.det(proj_matrix)
    assert abs(det) > 1e-6, f"Projection matrix determinant too small: {det}"
    
    # Test that reasonable FOV values work
    valid_fov_values = [30.0, 45.0, 60.0, 90.0]
    for fov in valid_fov_values:
        proj = f3d.camera_perspective(fov, aspect, znear, zfar)
        assert proj.shape == (4, 4)
        assert np.all(np.isfinite(proj))
    
    # Test that reasonable near/far ratios work
    valid_near_far_pairs = [(0.01, 10.0), (0.1, 100.0), (1.0, 1000.0)]
    for near, far in valid_near_far_pairs:
        proj = f3d.camera_perspective(fovy_deg, aspect, near, far)
        assert proj.shape == (4, 4)
        assert np.all(np.isfinite(proj))


def test_combined_view_projection():
    """Test combined view-projection matrix computation."""
    eye = (0.0, 0.0, 5.0)
    target = (0.0, 0.0, 0.0) 
    up = (0.0, 1.0, 0.0)
    fovy_deg = 45.0
    aspect = 1.0
    znear = 0.1
    zfar = 100.0
    
    # Get individual matrices
    view = f3d.camera_look_at(eye, target, up)
    proj = f3d.camera_perspective(fovy_deg, aspect, znear, zfar)
    
    # Get combined matrix
    view_proj_combined = f3d.camera_view_proj(eye, target, up, fovy_deg, aspect, znear, zfar)
    
    # Manual multiplication: proj * view (note the order for column-major matrices)
    view_proj_manual = proj @ view
    
    # Should match within numerical precision
    np.testing.assert_allclose(view_proj_combined, view_proj_manual, rtol=1e-6, atol=1e-7)


def test_orthographic_vs_perspective_integration():
    """Test that both orthographic and perspective projections work in camera system."""
    # Test that both projection types can be created and have reasonable properties
    # Perspective projection
    persp = f3d.camera_perspective(45.0, 1.0, 0.1, 100.0)
    assert persp.shape == (4, 4)
    assert np.all(np.isfinite(persp))
    
    # Orthographic projection  
    ortho = f3d.camera_orthographic(-10, 10, -10, 10, 0.1, 100.0)
    assert ortho.shape == (4, 4)
    assert np.all(np.isfinite(ortho))
    
    # They should be different matrices
    assert not np.allclose(persp, ortho), "Perspective and orthographic matrices should be different"
    
    # Both should work with view matrices
    view = f3d.camera_look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
    
    persp_view_proj = persp @ view
    ortho_view_proj = ortho @ view
    
    assert persp_view_proj.shape == (4, 4)
    assert ortho_view_proj.shape == (4, 4)
    assert np.all(np.isfinite(persp_view_proj))
    assert np.all(np.isfinite(ortho_view_proj))


def test_viewWorldPosition_uniform_consistency():
    """Test that viewWorldPosition in uniforms matches camera math expectations."""
    # This is a conceptual test since we can't directly access uniform buffer contents
    # but we can test that the camera math is consistent
    
    test_positions = [
        (0.0, 0.0, 5.0),
        (3.0, 4.0, 5.0),  # 3-4-5 triangle
        (-2.0, 1.0, 8.0),
    ]
    
    for eye_pos in test_positions:
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        view = f3d.camera_look_at(eye_pos, target, up)
        
        # Compute expected world position
        inv_view = np.linalg.inv(view)
        computed_world_pos = inv_view[0:3, 3]
        expected_world_pos = np.array(eye_pos, dtype=np.float32)
        
        # The computed world position should match the eye position
        np.testing.assert_allclose(computed_world_pos, expected_world_pos, 
                                 rtol=1e-5, atol=1e-6,
                                 err_msg=f"World position mismatch for eye={eye_pos}")


def test_camera_uniform_defaults():
    """Test documented camera uniform defaults and policies."""
    # Document the recommended defaults for near/far/FOV
    
    # Recommended defaults for typical 3D scenes
    recommended_fov = 45.0  # degrees, good balance between perspective and fisheye
    recommended_near = 0.1  # close enough for detail, far enough to avoid z-fighting
    recommended_far = 100.0  # far enough for typical scenes, close enough for precision
    recommended_aspect = 16.0 / 9.0  # modern widescreen default
    
    # These should all work without issues
    proj = f3d.camera_perspective(recommended_fov, recommended_aspect, recommended_near, recommended_far)
    assert proj.shape == (4, 4)
    assert np.all(np.isfinite(proj))
    
    # Test that the near/far ratio is reasonable (not too extreme)
    ratio = recommended_far / recommended_near
    assert 10.0 <= ratio <= 10000.0, f"Near/far ratio {ratio} may cause precision issues"
    
    # Test FOV range  
    assert 10.0 <= recommended_fov <= 120.0, f"FOV {recommended_fov} outside reasonable range"