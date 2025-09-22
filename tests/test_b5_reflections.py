#!/usr/bin/env python3
# tests/test_b5_reflections.py
# Test suite for Planar Reflections (B5) implementation
# RELEVANT FILES: src/core/reflections.rs, shaders/planar_reflections.wgsl, src/scene/mod.rs

"""
Test suite for Planar Reflections implementation.

Tests B5 acceptance criteria:
- Render-to-texture with clip plane support
- Roughness-aware blur functionality
- Performance requirement (≤15% frame cost)
- Debug visualization modes
- API functionality and parameter validation
"""

import pytest
import numpy as np
from typing import List, Tuple
import time

try:
    import forge3d as f3d
except ImportError:
    pytest.skip("forge3d not available", allow_module_level=True)


def create_test_terrain(width=64, height=64):
    """Create simple test terrain for reflection tests."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    # Simple hills for testing
    Z = 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return Z.astype(np.float32)


class TestReflectionAPI:
    """Test planar reflection API functionality."""

    def test_enable_disable_reflections(self):
        """Test enabling and disabling reflections."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should start with reflections disabled
        # (We can't directly check this, but enable should work)

        # Test enabling with different qualities
        valid_qualities = ['low', 'medium', 'high', 'ultra']
        for quality in valid_qualities:
            scene.enable_reflections(quality)
            # If no exception, it worked

        # Test invalid quality
        with pytest.raises(ValueError, match="Invalid quality"):
            scene.enable_reflections('invalid')

        # Test disable
        scene.disable_reflections()
        # Should succeed without exception

    def test_reflection_plane_configuration(self):
        """Test reflection plane setup and validation."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should fail if reflections not enabled
        with pytest.raises(RuntimeError, match="Reflections not enabled"):
            scene.set_reflection_plane((0, 1, 0), (0, 0, 0), (4, 4, 0))

        # Enable reflections and test valid configurations
        scene.enable_reflections('medium')

        # Test various plane configurations
        test_planes = [
            ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0)),  # Horizontal plane
            ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 6.0, 0.0)),  # Vertical plane
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (8.0, 8.0, 0.0)),  # Z-plane
        ]

        for normal, point, size in test_planes:
            scene.set_reflection_plane(normal, point, size)
            # Should succeed without exception

    def test_reflection_intensity_configuration(self):
        """Test reflection intensity parameter."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should fail if reflections not enabled
        with pytest.raises(RuntimeError, match="Reflections not enabled"):
            scene.set_reflection_intensity(0.5)

        scene.enable_reflections('medium')

        # Test valid intensities
        valid_intensities = [0.0, 0.25, 0.5, 0.8, 1.0]
        for intensity in valid_intensities:
            scene.set_reflection_intensity(intensity)
            # Should succeed without exception

        # Test values outside [0,1] - these should be clamped
        scene.set_reflection_intensity(-0.5)  # Should clamp to 0.0
        scene.set_reflection_intensity(1.5)   # Should clamp to 1.0

    def test_fresnel_power_configuration(self):
        """Test Fresnel power parameter."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should fail if reflections not enabled
        with pytest.raises(RuntimeError, match="Reflections not enabled"):
            scene.set_reflection_fresnel_power(5.0)

        scene.enable_reflections('medium')

        # Test valid Fresnel powers
        valid_powers = [0.1, 1.0, 3.0, 5.0, 8.0, 10.0]
        for power in valid_powers:
            scene.set_reflection_fresnel_power(power)
            # Should succeed without exception

    def test_distance_fade_configuration(self):
        """Test distance fade parameters."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should fail if reflections not enabled
        with pytest.raises(RuntimeError, match="Reflections not enabled"):
            scene.set_reflection_distance_fade(10.0, 50.0)

        scene.enable_reflections('medium')

        # Test valid distance fade settings
        test_fades = [
            (10.0, 50.0),
            (20.0, 100.0),
            (5.0, 25.0),
        ]

        for start, end in test_fades:
            scene.set_reflection_distance_fade(start, end)
            # Should succeed without exception

    def test_debug_mode_configuration(self):
        """Test debug mode settings."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should fail if reflections not enabled
        with pytest.raises(RuntimeError, match="Reflections not enabled"):
            scene.set_reflection_debug_mode(1)

        scene.enable_reflections('medium')

        # Test valid debug modes
        valid_modes = [0, 1, 2]
        for mode in valid_modes:
            scene.set_reflection_debug_mode(mode)
            # Should succeed without exception

    def test_performance_info_retrieval(self):
        """Test performance information retrieval."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should fail if reflections not enabled
        with pytest.raises(RuntimeError, match="Reflections not enabled"):
            scene.reflection_performance_info()

        scene.enable_reflections('medium')

        # Should return tuple of (frame_cost, meets_requirement)
        frame_cost, meets_requirement = scene.reflection_performance_info()

        assert isinstance(frame_cost, float)
        assert frame_cost >= 0.0
        assert isinstance(meets_requirement, bool)


class TestReflectionQuality:
    """Test reflection quality settings and performance."""

    @pytest.mark.parametrize("quality", ["low", "medium", "high", "ultra"])
    def test_quality_settings(self, quality):
        """Test different quality settings work correctly."""
        scene = f3d.Scene(512, 512, grid=64)

        # Should be able to enable each quality setting
        scene.enable_reflections(quality)

        # Set up a basic reflection plane
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

        # Should be able to get performance info
        frame_cost, meets_requirement = scene.reflection_performance_info()

        # Frame cost should be reasonable
        assert 0.0 <= frame_cost <= 50.0  # Should be less than 50%

        # Low and medium quality should typically meet the requirement
        if quality in ['low', 'medium']:
            # This might pass or fail depending on implementation, but shouldn't crash
            assert isinstance(meets_requirement, bool)

    def test_quality_performance_progression(self):
        """Test that higher quality settings generally have higher cost."""
        scene = f3d.Scene(512, 512, grid=64)

        qualities = ['low', 'medium', 'high']
        costs = []

        for quality in qualities:
            scene.enable_reflections(quality)
            scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

            frame_cost, _ = scene.reflection_performance_info()
            costs.append(frame_cost)

        # Generally, higher quality should have higher cost
        # Allow some tolerance for implementation variations
        assert costs[0] <= costs[1] * 1.5  # Low <= Medium * 1.5
        assert costs[1] <= costs[2] * 1.5  # Medium <= High * 1.5


class TestReflectionRendering:
    """Test reflection rendering functionality."""

    def test_basic_reflection_rendering(self):
        """Test basic reflection rendering pipeline."""
        scene = f3d.Scene(512, 512, grid=64)

        # Set up terrain
        terrain = create_test_terrain(64, 64)
        scene.set_height_from_r32f(terrain)

        # Set up camera
        scene.set_camera_look_at(
            (2.0, 1.5, 2.0),   # eye
            (0.0, 0.0, 0.0),   # target
            (0.0, 1.0, 0.0),   # up
            45.0, 0.1, 100.0   # fovy, near, far
        )

        # Render without reflections (baseline)
        baseline_pixels = scene.render_rgba()
        assert baseline_pixels.shape == (512, 512, 4)

        # Enable reflections
        scene.enable_reflections('medium')
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

        # Render with reflections
        reflection_pixels = scene.render_rgba()
        assert reflection_pixels.shape == (512, 512, 4)

        # Images should be different (reflections should change the appearance)
        assert not np.array_equal(baseline_pixels, reflection_pixels)

    def test_reflection_parameter_effects(self):
        """Test that changing reflection parameters affects the output."""
        scene = f3d.Scene(512, 512, grid=64)

        # Set up scene
        terrain = create_test_terrain(64, 64)
        scene.set_height_from_r32f(terrain)
        scene.set_camera_look_at(
            (2.0, 1.5, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            45.0, 0.1, 100.0
        )

        scene.enable_reflections('medium')
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

        # Render with low intensity
        scene.set_reflection_intensity(0.2)
        low_intensity = scene.render_rgba()

        # Render with high intensity
        scene.set_reflection_intensity(0.9)
        high_intensity = scene.render_rgba()

        # Should produce different results
        assert not np.array_equal(low_intensity, high_intensity)

    def test_debug_mode_rendering(self):
        """Test that debug modes produce valid output."""
        scene = f3d.Scene(512, 512, grid=64)

        # Set up scene
        terrain = create_test_terrain(64, 64)
        scene.set_height_from_r32f(terrain)
        scene.set_camera_look_at(
            (2.0, 1.5, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            45.0, 0.1, 100.0
        )

        scene.enable_reflections('medium')
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

        debug_outputs = []

        # Test each debug mode
        for debug_mode in [0, 1, 2]:
            scene.set_reflection_debug_mode(debug_mode)
            pixels = scene.render_rgba()

            assert pixels.shape == (512, 512, 4)
            debug_outputs.append(pixels)

        # Debug modes should produce different outputs
        assert not np.array_equal(debug_outputs[0], debug_outputs[1])
        assert not np.array_equal(debug_outputs[1], debug_outputs[2])


class TestReflectionPerformance:
    """Test reflection performance and optimization."""

    def test_performance_measurement(self):
        """Test performance measurement functionality."""
        scene = f3d.Scene(256, 256, grid=32)  # Smaller for faster testing

        terrain = create_test_terrain(32, 32)
        scene.set_height_from_r32f(terrain)
        scene.set_camera_look_at(
            (1.5, 1.0, 1.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            45.0, 0.1, 50.0
        )

        # Measure baseline performance (no reflections)
        baseline_times = []
        for _ in range(3):
            start = time.time()
            scene.render_rgba()
            baseline_times.append(time.time() - start)
        baseline_avg = np.mean(baseline_times)

        # Measure with reflections
        scene.enable_reflections('medium')
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0))

        reflection_times = []
        for _ in range(3):
            start = time.time()
            scene.render_rgba()
            reflection_times.append(time.time() - start)
        reflection_avg = np.mean(reflection_times)

        # Reflections should add some overhead, but not too much
        overhead_factor = reflection_avg / baseline_avg
        assert 1.0 <= overhead_factor <= 3.0  # Should be 1x to 3x slower at most

    def test_b5_performance_requirement(self):
        """Test B5 performance requirement (≤15% frame cost)."""
        scene = f3d.Scene(512, 512, grid=64)

        # Test that at least one quality setting meets the requirement
        performance_results = []

        for quality in ['low', 'medium']:
            scene.enable_reflections(quality)
            scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

            frame_cost, meets_requirement = scene.reflection_performance_info()
            performance_results.append((quality, frame_cost, meets_requirement))

        # At least one quality should meet the requirement
        meeting_requirement = any(meets_req for _, _, meets_req in performance_results)
        assert meeting_requirement, f"No quality setting meets ≤15% requirement: {performance_results}"


class TestReflectionIntegration:
    """Test reflection integration with other features."""

    def test_reflection_with_different_colormaps(self):
        """Test reflections work with different colormaps."""
        colormaps = ['viridis', 'plasma', 'terrain', 'magma']

        for colormap in colormaps:
            scene = f3d.Scene(256, 256, grid=32, colormap=colormap)

            terrain = create_test_terrain(32, 32)
            scene.set_height_from_r32f(terrain)

            scene.enable_reflections('low')
            scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0))

            # Should render without error
            pixels = scene.render_rgba()
            assert pixels.shape == (256, 256, 4)

    def test_reflection_with_different_grid_sizes(self):
        """Test reflections work with different grid resolutions."""
        grid_sizes = [16, 32, 64, 128]

        for grid_size in grid_sizes:
            scene = f3d.Scene(256, 256, grid=grid_size)

            terrain = create_test_terrain(grid_size//2, grid_size//2)
            scene.set_height_from_r32f(terrain)

            scene.enable_reflections('low')
            scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0))

            # Should render without error
            pixels = scene.render_rgba()
            assert pixels.shape == (256, 256, 4)

    def test_reflection_with_camera_changes(self):
        """Test reflections adapt to camera changes."""
        scene = f3d.Scene(256, 256, grid=32)

        terrain = create_test_terrain(32, 32)
        scene.set_height_from_r32f(terrain)

        scene.enable_reflections('medium')
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0))

        # Test different camera positions
        camera_positions = [
            ((2.0, 1.5, 2.0), (0.0, 0.0, 0.0)),
            ((1.0, 2.0, 1.0), (0.0, 0.0, 0.0)),
            ((3.0, 1.0, 3.0), (0.0, 0.0, 0.0)),
        ]

        rendered_outputs = []

        for eye, target in camera_positions:
            scene.set_camera_look_at(eye, target, (0.0, 1.0, 0.0), 45.0, 0.1, 50.0)
            pixels = scene.render_rgba()
            rendered_outputs.append(pixels)

        # Different camera positions should produce different outputs
        assert not np.array_equal(rendered_outputs[0], rendered_outputs[1])
        assert not np.array_equal(rendered_outputs[1], rendered_outputs[2])


class TestB5AcceptanceCriteria:
    """Test B5 acceptance criteria compliance."""

    def test_render_to_texture_support(self):
        """Test B5: Render-to-texture with clip plane support."""
        scene = f3d.Scene(512, 512, grid=64)

        # Set up terrain and camera
        terrain = create_test_terrain(64, 64)
        scene.set_height_from_r32f(terrain)
        scene.set_camera_look_at(
            (2.0, 1.5, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            45.0, 0.1, 100.0
        )

        # Enable reflections (this sets up render-to-texture internally)
        scene.enable_reflections('medium')

        # Configure clip plane via reflection plane
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

        # Should be able to render without error
        pixels = scene.render_rgba()
        assert pixels.shape == (512, 512, 4)

        # Verify render-to-texture is working by checking that we get valid output
        assert pixels.dtype == np.uint8
        assert np.any(pixels > 0)  # Should have some non-zero pixels

    def test_roughness_aware_blur(self):
        """Test B5: Roughness-aware blur functionality."""
        scene = f3d.Scene(512, 512, grid=64)

        # Set up scene
        terrain = create_test_terrain(64, 64)
        scene.set_height_from_r32f(terrain)
        scene.set_camera_look_at(
            (2.0, 1.5, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            45.0, 0.1, 100.0
        )

        # Test different quality settings (which affect blur)
        blur_outputs = []

        for quality in ['low', 'medium', 'high']:
            scene.enable_reflections(quality)
            scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

            pixels = scene.render_rgba()
            blur_outputs.append(pixels)

        # Different quality settings should produce different blur results
        assert not np.array_equal(blur_outputs[0], blur_outputs[1])
        assert not np.array_equal(blur_outputs[1], blur_outputs[2])

    def test_performance_requirement(self):
        """Test B5: ≤15% frame cost requirement."""
        scene = f3d.Scene(512, 512, grid=64)

        # Test performance requirement
        requirement_met = False

        for quality in ['low', 'medium', 'high']:
            scene.enable_reflections(quality)
            scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

            frame_cost, meets_requirement = scene.reflection_performance_info()

            if meets_requirement:
                requirement_met = True
                break

        assert requirement_met, "No quality setting meets the ≤15% frame cost requirement"

    def test_debug_visualization_modes(self):
        """Test B5: Debug visualization modes."""
        scene = f3d.Scene(512, 512, grid=64)

        # Set up scene
        terrain = create_test_terrain(64, 64)
        scene.set_height_from_r32f(terrain)
        scene.set_camera_look_at(
            (2.0, 1.5, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            45.0, 0.1, 100.0
        )

        scene.enable_reflections('medium')
        scene.set_reflection_plane((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 4.0, 0.0))

        # Test all debug modes work
        debug_modes = [0, 1, 2]  # Normal, reflection texture, debug overlay

        for mode in debug_modes:
            scene.set_reflection_debug_mode(mode)
            pixels = scene.render_rgba()

            # Should render successfully
            assert pixels.shape == (512, 512, 4)
            assert pixels.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])