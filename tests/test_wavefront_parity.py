# tests/test_wavefront_parity.py
# Tests for wavefront vs megakernel parity
# This file exists to ensure both path tracing engines produce identical deterministic results.
# RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_kernel.wgsl,tests/conftest.py

"""Test parity between wavefront and megakernel path tracing engines."""

import pytest
import numpy as np
from forge3d.path_tracing import render_rgba, render_aovs, TracerEngine
import os as _os
try:
    from forge3d import enumerate_adapters
    _HAVE_GPU = bool(enumerate_adapters())
except Exception:
    _HAVE_GPU = False

_WF_ENV = bool(_os.environ.get("FORGE3D_ENABLE_WAVEFRONT_TESTS"))
pytestmark = pytest.mark.skipif(
    (not _HAVE_GPU) or (not _WF_ENV),
    reason="Wavefront tests disabled (set FORGE3D_ENABLE_WAVEFRONT_TESTS=1 and require GPU)",
)


@pytest.mark.gpu  # Requires GPU for both engines
class TestWavefrontParity:
    """Test that wavefront and megakernel engines produce identical results."""
    
    def test_simple_sphere_parity(self):
        """Test that both engines render identical results for a simple sphere."""
        # Simple scene with one sphere
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.7, 0.3, 0.3]
            }
        ]
        camera = {}  # Default camera
        seed = 42
        frames = 4  # Low sample count for speed
        width, height = 64, 64  # Small resolution for speed
        
        # Render with megakernel
        result_mega = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.MEGAKERNEL
        )
        
        # Render with wavefront
        result_wave = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Results should be identical for deterministic rendering
        np.testing.assert_array_equal(result_mega, result_wave,
            "Megakernel and wavefront engines should produce identical results")
    
    def test_multiple_spheres_parity(self):
        """Test parity with multiple spheres and bounces."""
        # Scene with multiple spheres
        scene = [
            {
                'center': [-0.5, 0.0, -1.0],
                'radius': 0.3,
                'albedo': [0.8, 0.2, 0.2]
            },
            {
                'center': [0.5, 0.0, -1.0],
                'radius': 0.3,
                'albedo': [0.2, 0.8, 0.2]
            },
            {
                'center': [0.0, -0.5, -1.5],
                'radius': 0.2,
                'albedo': [0.2, 0.2, 0.8]
            }
        ]
        camera = {}
        seed = 123
        frames = 8
        width, height = 32, 32
        
        # Render with both engines
        result_mega = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.MEGAKERNEL
        )
        
        result_wave = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result_mega, result_wave,
            "Multiple sphere rendering should be identical between engines")
    
    def test_aov_parity(self):
        """Test that AOV outputs are identical between engines."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.4,
                'albedo': [0.6, 0.6, 0.6]
            }
        ]
        camera = {}
        seed = 999
        frames = 2
        width, height = 32, 32
        aovs = ("albedo", "normal", "depth", "visibility")
        
        # Render AOVs with megakernel
        aovs_mega = render_aovs(
            width, height, scene, camera,
            aovs=aovs, seed=seed, frames=frames,
            use_gpu=True, engine=TracerEngine.MEGAKERNEL
        )
        
        # Render AOVs with wavefront
        aovs_wave = render_aovs(
            width, height, scene, camera,
            aovs=aovs, seed=seed, frames=frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Compare each AOV
        for aov_name in aovs:
            if aov_name in aovs_mega and aov_name in aovs_wave:
                np.testing.assert_allclose(
                    aovs_mega[aov_name], aovs_wave[aov_name],
                    rtol=1e-6, atol=1e-6,
                    err_msg=f"AOV '{aov_name}' should be identical between engines"
                )
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results (sanity check)."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.5, 0.5, 0.5]
            }
        ]
        camera = {}
        frames = 4
        width, height = 32, 32
        
        # Render same scene with different seeds using wavefront
        result1 = render_rgba(
            width, height, scene, camera, 12345, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        result2 = render_rgba(
            width, height, scene, camera, 54321, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Results should be different (anti-aliasing jitter)
        assert not np.array_equal(result1, result2), \
            "Different seeds should produce different results due to jitter"
    
    def test_same_seed_same_results(self):
        """Test that same seed produces identical results (determinism check)."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.7, 0.7, 0.7]
            }
        ]
        camera = {}
        seed = 777
        frames = 4
        width, height = 32, 32
        
        # Render same scene twice with same seed using wavefront
        result1 = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        result2 = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2,
            "Same seed should produce identical results")
    
    @pytest.mark.parametrize("engine", [TracerEngine.MEGAKERNEL, TracerEngine.WAVEFRONT])
    def test_empty_scene(self, engine):
        """Test that empty scene produces consistent background."""
        scene = []  # Empty scene
        camera = {}
        seed = 1
        frames = 1
        width, height = 16, 16
        
        result = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=engine
        )
        
        # Should produce gradient background
        assert result.shape == (height, width, 4)
        assert result.dtype == np.uint8
        
        # Alpha should be 255 everywhere
        assert np.all(result[:, :, 3] == 255)
        
        # RGB should vary (gradient)
        rgb = result[:, :, :3]
        assert np.std(rgb) > 0, "Background should have variation"
    
    def test_high_sample_count_convergence(self):
        """Test that high sample counts converge to similar results."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.5, 0.5, 0.5]
            }
        ]
        camera = {}
        seed = 2023
        width, height = 24, 24
        
        # Low sample count
        result_low = render_rgba(
            width, height, scene, camera, seed, frames=4,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # High sample count
        result_high = render_rgba(
            width, height, scene, camera, seed, frames=64,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Results should be similar but not identical due to monte carlo
        # Convert to float for comparison
        low_f = result_low[:, :, :3].astype(np.float32) / 255.0
        high_f = result_high[:, :, :3].astype(np.float32) / 255.0
        
        # Should be reasonably close (within noise)
        mse = np.mean((low_f - high_f) ** 2)
        assert mse < 0.1, f"High and low sample results should converge (MSE: {mse})"
    
    def test_resolution_independence(self):
        """Test that different resolutions produce consistent per-pixel results."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.4, 0.6, 0.8]
            }
        ]
        camera = {}
        seed = 404
        frames = 8
        
        # Render at different resolutions
        result_16 = render_rgba(
            16, 16, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        result_32 = render_rgba(
            32, 32, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Both should have proper shape and range
        assert result_16.shape == (16, 16, 4)
        assert result_32.shape == (32, 32, 4)
        assert np.all((result_16 >= 0) & (result_16 <= 255))
        assert np.all((result_32 >= 0) & (result_32 <= 255))
        
        # Center pixels should be similar (sphere center)
        center_16 = result_16[8, 8, :3]
        center_32 = result_32[16, 16, :3]
        
        # Allow some tolerance for different sampling patterns
        diff = np.abs(center_16.astype(np.float32) - center_32.astype(np.float32))
        assert np.max(diff) < 50, "Center pixels should be similar across resolutions"
