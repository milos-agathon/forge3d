# tests/test_wavefront_compaction.py
# Tests for wavefront stream compaction
# This file exists to verify that stream compaction correctly removes terminated rays and maintains efficiency.
# RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_compact.wgsl,tests/conftest.py

"""Test stream compaction in wavefront path tracing."""

import pytest
import numpy as np
from forge3d.path_tracing import render_rgba, render_aovs, TracerEngine


@pytest.mark.gpu  # Requires GPU for wavefront engine
class TestWavefrontCompaction:
    """Test that stream compaction works correctly in wavefront path tracing."""
    
    def test_compaction_with_early_termination(self):
        """Test that compaction handles early path termination correctly."""
        # Scene designed to cause early termination (high absorption)
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.1, 0.1, 0.1]  # Dark material -> early termination
            }
        ]
        camera = {}
        seed = 12345
        frames = 16  # Enough samples to test compaction
        width, height = 32, 32
        
        # Render with wavefront (which uses compaction)
        result = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Should produce valid result despite early termination
        assert result.shape == (height, width, 4)
        assert result.dtype == np.uint8
        assert np.all((result >= 0) & (result <= 255))
        
        # Should have hit the sphere (non-background pixels)
        center_pixel = result[height//2, width//2, :3]
        background_estimate = np.array([153, 178, 229], dtype=np.uint8)  # Sky gradient
        center_diff = np.abs(center_pixel.astype(np.float32) - background_estimate.astype(np.float32))
        assert np.sum(center_diff) > 50, "Should hit sphere, not just background"
    
    def test_compaction_efficiency_test(self):
        """Test compaction by comparing scenes with different termination rates."""
        camera = {}
        seed = 999
        frames = 32
        width, height = 24, 24
        
        # Scene with high termination (dark materials)
        scene_dark = [
            {
                'center': [-0.3, 0.0, -1.0],
                'radius': 0.3,
                'albedo': [0.05, 0.05, 0.05]  # Very dark -> early termination
            },
            {
                'center': [0.3, 0.0, -1.0],
                'radius': 0.3,
                'albedo': [0.05, 0.05, 0.05]   # Very dark -> early termination
            }
        ]
        
        # Scene with low termination (bright materials)
        scene_bright = [
            {
                'center': [-0.3, 0.0, -1.0],
                'radius': 0.3,
                'albedo': [0.9, 0.9, 0.9]  # Bright -> more bounces
            },
            {
                'center': [0.3, 0.0, -1.0],
                'radius': 0.3,
                'albedo': [0.9, 0.9, 0.9]   # Bright -> more bounces
            }
        ]
        
        # Both should render successfully despite different compaction loads
        result_dark = render_rgba(
            width, height, scene_dark, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        result_bright = render_rgba(
            width, height, scene_bright, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Both should be valid
        assert result_dark.shape == (height, width, 4)
        assert result_bright.shape == (height, width, 4)
        
        # Bright scene should be... brighter
        dark_mean = np.mean(result_dark[:, :, :3].astype(np.float32))
        bright_mean = np.mean(result_bright[:, :, :3].astype(np.float32))
        assert bright_mean > dark_mean, "Bright materials should produce brighter image"
    
    def test_compaction_with_complex_scene(self):
        """Test compaction with complex scene that creates varied ray paths."""
        # Complex scene with mixed materials
        scene = [
            # Floor
            {
                'center': [0.0, -1.5, -2.0],
                'radius': 1.0,
                'albedo': [0.8, 0.8, 0.8]
            },
            # Left sphere (absorbing)
            {
                'center': [-0.6, 0.0, -1.5],
                'radius': 0.3,
                'albedo': [0.2, 0.1, 0.1]
            },
            # Center sphere (scattering)
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.4,
                'albedo': [0.7, 0.7, 0.7]
            },
            # Right sphere (absorbing)
            {
                'center': [0.6, 0.0, -1.5],
                'radius': 0.3,
                'albedo': [0.1, 0.1, 0.2]
            }
        ]
        camera = {}
        seed = 2024
        frames = 16
        width, height = 48, 48
        
        # Should render successfully with complex ray termination patterns
        result = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Verify result validity
        assert result.shape == (height, width, 4)
        assert np.all((result >= 0) & (result <= 255))
        
        # Should have some color variation due to different materials
        rgb_variance = np.var(result[:, :, :3].astype(np.float32))
        assert rgb_variance > 100, f"Should have color variation (var: {rgb_variance})"
    
    def test_compaction_max_depth_termination(self):
        """Test compaction behavior with maximum depth termination."""
        # Scene that could cause deep bounces (but will hit max depth)
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.95, 0.95, 0.95]  # High albedo -> many bounces
            }
        ]
        camera = {}
        seed = 777
        frames = 8
        width, height = 16, 16
        
        # Render with wavefront
        result = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Should terminate gracefully at max depth
        assert result.shape == (height, width, 4)
        assert np.all(np.isfinite(result))
        
        # Should be bright due to high albedo
        center_brightness = np.sum(result[height//2, width//2, :3])
        assert center_brightness > 300, "High albedo should produce bright result"
    
    def test_compaction_empty_queue_handling(self):
        """Test that compaction handles empty queues gracefully."""
        # Simple scene that should terminate quickly
        scene = [
            {
                'center': [0.0, 0.0, -5.0],  # Far away
                'radius': 0.1,                # Small
                'albedo': [0.1, 0.1, 0.1]     # Dark
            }
        ]
        camera = {}
        seed = 1
        frames = 4  # Low sample count
        width, height = 8, 8  # Very small
        
        # Most rays should miss and terminate quickly
        result = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Should still produce valid result
        assert result.shape == (height, width, 4)
        
        # Should be mostly background
        background_pixels = np.sum(result[:, :, :3], axis=2) > 300
        background_ratio = np.sum(background_pixels) / (width * height)
        assert background_ratio > 0.8, "Most pixels should be background"
    
    def test_compaction_determinism(self):
        """Test that compaction doesn't break determinism."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.4,
                'albedo': [0.6, 0.4, 0.2]
            }
        ]
        camera = {}
        seed = 42
        frames = 16
        width, height = 32, 32
        
        # Render same scene twice
        result1 = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        result2 = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Should be identical despite compaction
        np.testing.assert_array_equal(result1, result2,
            "Compaction should not break determinism")
    
    def test_compaction_with_aovs(self):
        """Test that compaction works correctly with AOV rendering."""
        scene = [
            {
                'center': [0.0, 0.0, -1.0],
                'radius': 0.5,
                'albedo': [0.3, 0.7, 0.5]
            }
        ]
        camera = {}
        seed = 555
        frames = 8
        width, height = 24, 24
        aovs = ("albedo", "normal", "depth", "visibility")
        
        # Render AOVs with wavefront (uses compaction)
        result_aovs = render_aovs(
            width, height, scene, camera,
            aovs=aovs, seed=seed, frames=frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # All AOVs should be present and valid
        for aov_name in aovs:
            assert aov_name in result_aovs, f"AOV '{aov_name}' should be present"
            aov_data = result_aovs[aov_name]
            
            if aov_name == "depth":
                # Depth should have valid values where sphere is hit
                assert aov_data.shape == (height, width)
                finite_depths = aov_data[np.isfinite(aov_data)]
                if len(finite_depths) > 0:
                    assert np.all(finite_depths > 0), "Depth values should be positive"
            elif aov_name == "visibility":
                # Visibility should be 0 or 255
                assert aov_data.shape == (height, width)
                assert np.all((aov_data == 0) | (aov_data == 255)), "Visibility should be binary"
            else:
                # Color AOVs should be finite
                assert aov_data.shape == (height, width, 3)
                assert np.all(np.isfinite(aov_data)), f"AOV '{aov_name}' should have finite values"
    
    def test_queue_capacity_stress(self):
        """Test compaction under queue capacity stress."""
        # Large scene to stress queue capacity
        scene = []
        for i in range(6):  # Multiple spheres
            for j in range(6):
                x = (i - 2.5) * 0.3
                z = -1.0 - j * 0.2
                scene.append({
                    'center': [x, 0.0, z],
                    'radius': 0.1,
                    'albedo': [0.5 + 0.1*i, 0.5 + 0.1*j, 0.5]
                })
        
        camera = {}
        seed = 2023
        frames = 4  # Keep low to avoid timeout
        width, height = 32, 32
        
        # Should handle large scene without crashing
        result = render_rgba(
            width, height, scene, camera, seed, frames,
            use_gpu=True, engine=TracerEngine.WAVEFRONT
        )
        
        # Should produce valid result
        assert result.shape == (height, width, 4)
        assert np.all(np.isfinite(result))
        
        # Should have variety due to multiple objects
        unique_colors = len(np.unique(result.reshape(-1, 4), axis=0))
        assert unique_colors > 10, f"Should have color variety (unique: {unique_colors})"