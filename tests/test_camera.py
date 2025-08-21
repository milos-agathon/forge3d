"""
Tests for T2.1 Camera & Uniform buffer functionality.

Tests camera math functions, validation, TerrainSpike integration, and 
adherence to exact error message requirements.
"""

import numpy as np
import pytest
import math

try:
    import forge3d as f3d
    from forge3d import camera_look_at, camera_perspective, camera_view_proj
    HAS_MAIN_MODULE = True
except ImportError:
    try:
        import _forge3d as f3d
        camera_look_at = f3d.camera_look_at
        camera_perspective = f3d.camera_perspective
        camera_view_proj = f3d.camera_view_proj
        HAS_MAIN_MODULE = False
    except ImportError:
        pytest.skip("forge3d module not available", allow_module_level=True)

# Error messages that must match exactly (escaped for regex)
import re
ERROR_FOVY = re.escape("fovy_deg must be finite and in (0, 180)")
ERROR_NEAR = re.escape("znear must be finite and > 0")
ERROR_FAR = re.escape("zfar must be finite and > znear")
ERROR_ASPECT = re.escape("aspect must be finite and > 0")
ERROR_VECFINITE = re.escape("eye/target/up components must be finite")
ERROR_UPCOLINEAR = re.escape("up vector must not be colinear with view direction")
ERROR_CLIP = re.escape("clip_space must be 'wgpu' or 'gl'")

# Numerical tolerances
RTOL = 1e-5
ATOL = 1e-6


class TestCameraLookAt:
    """Test camera_look_at function."""
    
    def test_shape_and_dtype(self):
        """Test that camera_look_at returns correct shape and dtype."""
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        result = camera_look_at(eye, target, up)
        
        assert result.shape == (4, 4), f"Expected shape (4, 4), got {result.shape}"
        assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"
        assert result.flags.c_contiguous, "Result must be C-contiguous"
    
    def test_numerical_correctness(self):
        """Test numerical correctness of view matrix computation."""
        # Camera looking from (0,0,3) to (0,0,0) with up (0,1,0)
        # Expected: view[2,3] ≈ -3 (Z translation in view space)
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        view = camera_look_at(eye, target, up)
        
        # In RH coordinate system looking down -Z, the camera at (0,0,3) 
        # should have Z translation of -3 in view matrix
        assert abs(view[2, 3] - (-3.0)) < ATOL, f"Expected view[2,3] ≈ -3, got {view[2, 3]}"
    
    def test_validation_infinite_components(self):
        """Test validation of infinite components."""
        # Test infinite eye
        with pytest.raises(RuntimeError, match=ERROR_VECFINITE):
            camera_look_at((float('inf'), 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        
        # Test infinite target
        with pytest.raises(RuntimeError, match=ERROR_VECFINITE):
            camera_look_at((0.0, 0.0, 3.0), (float('nan'), 0.0, 0.0), (0.0, 1.0, 0.0))
        
        # Test infinite up
        with pytest.raises(RuntimeError, match=ERROR_VECFINITE):
            camera_look_at((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, float('inf'), 0.0))
    
    def test_validation_colinear_up(self):
        """Test validation of colinear up vector."""
        # Up vector parallel to view direction should fail
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, -1.0)  # Parallel to view direction
        
        with pytest.raises(RuntimeError, match=ERROR_UPCOLINEAR):
            camera_look_at(eye, target, up)


class TestCameraPerspective:
    """Test camera_perspective function."""
    
    def test_shape_and_dtype(self):
        """Test that camera_perspective returns correct shape and dtype."""
        result = camera_perspective(45.0, 16.0/9.0, 0.1, 100.0)
        
        assert result.shape == (4, 4), f"Expected shape (4, 4), got {result.shape}"
        assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"
        assert result.flags.c_contiguous, "Result must be C-contiguous"
    
    def test_default_clip_space(self):
        """Test that default clip space is 'wgpu'."""
        proj_default = camera_perspective(45.0, 1.0, 0.1, 100.0)
        proj_wgpu = camera_perspective(45.0, 1.0, 0.1, 100.0, "wgpu")
        
        np.testing.assert_array_equal(proj_default, proj_wgpu)
    
    def test_gl_vs_wgpu_clip_space(self):
        """Test difference between GL and WGPU clip spaces."""
        fovy, aspect, znear, zfar = 45.0, 1.0, 0.1, 100.0
        
        proj_gl = camera_perspective(fovy, aspect, znear, zfar, "gl")
        proj_wgpu = camera_perspective(fovy, aspect, znear, zfar, "wgpu")
        
        # Verify they're different (GL uses [-1,1] Z, WGPU uses [0,1] Z)
        assert not np.allclose(proj_gl, proj_wgpu), "GL and WGPU projections should differ"
        
        # The main difference should be in the Z mapping - GL uses [-1,1], WGPU uses [0,1]
        # Verify that the X,Y components are the same
        np.testing.assert_allclose(proj_gl[:2, :], proj_wgpu[:2, :], rtol=RTOL, atol=ATOL,
                                   err_msg="X,Y components should be identical between GL and WGPU")
    
    def test_validation_fovy(self):
        """Test validation of field of view angle."""
        # Test invalid fovy values
        with pytest.raises(RuntimeError, match=ERROR_FOVY):
            camera_perspective(0.0, 1.0, 0.1, 100.0)  # fovy <= 0
        
        with pytest.raises(RuntimeError, match=ERROR_FOVY):
            camera_perspective(180.0, 1.0, 0.1, 100.0)  # fovy >= 180
        
        with pytest.raises(RuntimeError, match=ERROR_FOVY):
            camera_perspective(float('inf'), 1.0, 0.1, 100.0)  # infinite fovy
    
    def test_validation_aspect(self):
        """Test validation of aspect ratio."""
        with pytest.raises(RuntimeError, match=ERROR_ASPECT):
            camera_perspective(45.0, 0.0, 0.1, 100.0)  # aspect <= 0
        
        with pytest.raises(RuntimeError, match=ERROR_ASPECT):
            camera_perspective(45.0, float('inf'), 0.1, 100.0)  # infinite aspect
    
    def test_validation_near(self):
        """Test validation of near plane."""
        with pytest.raises(RuntimeError, match=ERROR_NEAR):
            camera_perspective(45.0, 1.0, 0.0, 100.0)  # znear <= 0
        
        with pytest.raises(RuntimeError, match=ERROR_NEAR):
            camera_perspective(45.0, 1.0, float('nan'), 100.0)  # infinite znear
    
    def test_validation_far(self):
        """Test validation of far plane."""
        with pytest.raises(RuntimeError, match=ERROR_FAR):
            camera_perspective(45.0, 1.0, 0.1, 0.05)  # zfar <= znear
        
        with pytest.raises(RuntimeError, match=ERROR_FAR):
            camera_perspective(45.0, 1.0, 0.1, float('inf'))  # infinite zfar
    
    def test_validation_clip_space(self):
        """Test validation of clip space parameter."""
        with pytest.raises(RuntimeError, match=ERROR_CLIP):
            camera_perspective(45.0, 1.0, 0.1, 100.0, "invalid")


class TestCameraViewProj:
    """Test camera_view_proj function."""
    
    def test_shape_and_dtype(self):
        """Test that camera_view_proj returns correct shape and dtype."""
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        result = camera_view_proj(eye, target, up, 45.0, 1.0, 0.1, 100.0)
        
        assert result.shape == (4, 4), f"Expected shape (4, 4), got {result.shape}"
        assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"
        assert result.flags.c_contiguous, "Result must be C-contiguous"
    
    def test_composition_equivalence(self):
        """Test that view_proj equals proj * view."""
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        fovy, aspect, znear, zfar = 45.0, 16.0/9.0, 0.1, 100.0
        
        # Get combined matrix
        view_proj = camera_view_proj(eye, target, up, fovy, aspect, znear, zfar, "wgpu")
        
        # Get individual matrices
        view = camera_look_at(eye, target, up)
        proj = camera_perspective(fovy, aspect, znear, zfar, "wgpu")
        
        # Verify view_proj = proj * view
        expected = proj @ view
        np.testing.assert_allclose(view_proj, expected, rtol=RTOL, atol=ATOL)
    
    def test_validation_all_parameters(self):
        """Test that all parameter validation works in view_proj."""
        # Test each type of validation error
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        # Test fovy validation
        with pytest.raises(RuntimeError, match=ERROR_FOVY):
            camera_view_proj(eye, target, up, 0.0, 1.0, 0.1, 100.0)
        
        # Test colinear up validation
        with pytest.raises(RuntimeError, match=ERROR_UPCOLINEAR):
            camera_view_proj(eye, target, (0.0, 0.0, -1.0), 45.0, 1.0, 0.1, 100.0)


@pytest.mark.skipif(not hasattr(f3d, 'TerrainSpike'), reason="TerrainSpike not available (terrain_spike feature disabled)")
class TestTerrainSpikeIntegration:
    """Test TerrainSpike camera integration."""
    
    def test_set_camera_look_at_exists(self):
        """Test that set_camera_look_at method exists."""
        spike = f3d.TerrainSpike(512, 512)
        assert hasattr(spike, 'set_camera_look_at'), "TerrainSpike should have set_camera_look_at method"
        assert hasattr(spike, 'debug_uniforms_f32'), "TerrainSpike should have debug_uniforms_f32 method"
    
    def test_set_camera_look_at_updates_uniforms(self):
        """Test that set_camera_look_at updates UBO and debug uniforms."""
        spike = f3d.TerrainSpike(512, 512)
        
        # Get initial uniforms
        initial_uniforms = spike.debug_uniforms_f32()
        assert len(initial_uniforms) == 44, f"Expected 44 floats (176 bytes / 4), got {len(initial_uniforms)}"
        
        # Set new camera
        eye = (1.0, 2.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        fovy_deg = 60.0
        znear = 0.1
        zfar = 100.0
        
        spike.set_camera_look_at(eye, target, up, fovy_deg, znear, zfar)
        
        # Get updated uniforms
        updated_uniforms = spike.debug_uniforms_f32()
        
        # Verify uniforms changed
        assert not np.allclose(initial_uniforms, updated_uniforms), "Uniforms should change after set_camera_look_at"
    
    def test_set_camera_look_at_validates_parameters(self):
        """Test that set_camera_look_at validates parameters correctly."""
        spike = f3d.TerrainSpike(512, 512)
        
        # Test invalid fovy
        with pytest.raises(RuntimeError, match=ERROR_FOVY):
            spike.set_camera_look_at((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0, 0.1, 100.0)
        
        # Test colinear up
        with pytest.raises(RuntimeError, match=ERROR_UPCOLINEAR):
            spike.set_camera_look_at((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 45.0, 0.1, 100.0)
    
    def test_debug_uniforms_match_expected_layout(self):
        """Test that debug uniforms match expected matrix layout."""
        spike = f3d.TerrainSpike(512, 512)
        
        # Set known camera parameters
        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        fovy_deg = 45.0
        znear = 0.1
        zfar = 100.0
        
        spike.set_camera_look_at(eye, target, up, fovy_deg, znear, zfar)
        uniforms = spike.debug_uniforms_f32()
        
        # Compute expected view and projection matrices
        aspect = 512.0 / 512.0  # width / height
        expected_view = camera_look_at(eye, target, up)
        expected_proj = camera_perspective(fovy_deg, aspect, znear, zfar, "wgpu")
        
        # Extract view matrix from uniforms (first 16 floats, stored column-major)
        view_from_uniforms = uniforms[:16].reshape(4, 4, order='F')  # Fortran order for column-major
        
        # Extract projection matrix from uniforms (next 16 floats, stored column-major)  
        proj_from_uniforms = uniforms[16:32].reshape(4, 4, order='F')
        
        # Verify matrices match (within tolerance)
        np.testing.assert_allclose(view_from_uniforms, expected_view, rtol=RTOL, atol=ATOL,
                                   err_msg="View matrix in uniforms doesn't match expected")
        np.testing.assert_allclose(proj_from_uniforms, expected_proj, rtol=RTOL, atol=ATOL,
                                   err_msg="Projection matrix in uniforms doesn't match expected")


def test_terrainspike_default_proj_is_wgpu():
    """Test that TerrainSpike defaults to WGPU clip space projection."""
    import numpy as np
    try:
        import forge3d as f3d
    except ImportError:
        import _forge3d as f3d
        
    if not hasattr(f3d, "TerrainSpike"):
        import pytest
        pytest.skip("TerrainSpike not built")
        
    W, H = 128, 96
    t = f3d.TerrainSpike(W, H, grid=32)
    u = t.debug_uniforms_f32()  # 44 floats
    
    # View (0:16), Proj (16:32) in column-major
    proj = np.array(u[16:32], dtype=np.float32).reshape(4, 4, order="F")
    fovy, znear, zfar = 45.0, 0.1, 100.0
    aspect = float(W) / float(H)
    
    # Import the camera_perspective function
    if hasattr(f3d, 'camera_perspective'):
        expected = f3d.camera_perspective(fovy, aspect, znear, zfar, clip_space="wgpu")
    else:
        # Fallback to module-level import
        from forge3d import camera_perspective
        expected = camera_perspective(fovy, aspect, znear, zfar, clip_space="wgpu")
    
    assert proj.shape == (4, 4) and expected.shape == (4, 4)
    assert np.allclose(proj, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])