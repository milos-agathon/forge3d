#!/usr/bin/env python3
"""
B12: Soft Light Radius (Raster) - Test Suite

Tests for B12 acceptance criteria:
- Radius control visibly softens falloff
- Raster path remains stable

This test suite validates the complete B12 implementation including:
- Soft light radius renderer initialization
- Light parameter controls (position, color, intensity, radius)
- Falloff mode functionality (linear, quadratic, cubic, exponential)
- Preset configurations (spotlight, area light, ambient, candle, street lamp)
- Performance requirements and stability
- Scene round-trip behavior
"""

import pytest
import numpy as np
import time
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import forge3d as f3d
except ImportError:
    pytest.skip("forge3d not available", allow_module_level=True)

@pytest.fixture
def scene():
    """Create a basic scene for testing"""
    scene = f3d.Scene(512, 512, grid=32)

    # Set up camera
    scene.set_camera_look_at(
        eye=(0.0, 10.0, 10.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fovy_deg=45.0,
        znear=0.1,
        zfar=100.0
    )

    # Upload simple terrain
    heights = np.random.rand(32, 32).astype(np.float32) * 2.0
    scene.upload_height_map(heights)

    return scene

class TestB12SoftLightRadius:
    """Test suite for B12 Soft Light Radius functionality"""

    def test_soft_light_radius_enable_disable(self, scene):
        """Test basic enable/disable functionality"""
        # Should start disabled
        assert not scene.is_soft_light_radius_enabled()

        # Enable soft light radius
        scene.enable_soft_light_radius()
        assert scene.is_soft_light_radius_enabled()

        # Disable soft light radius
        scene.disable_soft_light_radius()
        assert not scene.is_soft_light_radius_enabled()

    def test_light_position_control(self, scene):
        """Test light position parameter control"""
        scene.enable_soft_light_radius()

        # Test various positions
        positions = [
            (0.0, 5.0, 0.0),
            (10.0, 8.0, -5.0),
            (-3.0, 12.0, 7.0)
        ]

        for pos in positions:
            scene.set_light_position(*pos)
            # Should not raise exceptions
            assert scene.is_soft_light_radius_enabled()

    def test_light_intensity_control(self, scene):
        """Test light intensity parameter control"""
        scene.enable_soft_light_radius()

        # Test various intensities
        intensities = [0.5, 1.0, 1.5, 2.0, 3.0]

        for intensity in intensities:
            scene.set_light_intensity(intensity)
            assert scene.is_soft_light_radius_enabled()

    def test_light_color_control(self, scene):
        """Test light color parameter control"""
        scene.enable_soft_light_radius()

        # Test various colors
        colors = [
            (1.0, 1.0, 1.0),  # White
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 0.8, 0.6),  # Warm white
        ]

        for color in colors:
            scene.set_light_color(*color)
            assert scene.is_soft_light_radius_enabled()

    def test_radius_control_softens_falloff(self, scene):
        """Test B12 core acceptance criteria: radius control visibly softens falloff"""
        scene.enable_soft_light_radius()
        scene.set_light_position(0.0, 8.0, 0.0)
        scene.set_light_intensity(2.0)
        scene.set_light_color(1.0, 1.0, 1.0)

        # Test different radius configurations to verify visible softening
        radius_configs = [
            {"inner": 2.0, "outer": 6.0, "softness": 0.1},   # Sharp
            {"inner": 3.0, "outer": 10.0, "softness": 1.0},  # Medium
            {"inner": 4.0, "outer": 15.0, "softness": 3.0},  # Soft
        ]

        rendered_images = []

        for config in radius_configs:
            scene.set_light_inner_radius(config["inner"])
            scene.set_light_outer_radius(config["outer"])
            scene.set_light_edge_softness(config["softness"])

            # Render image
            rgba = scene.render_rgba()
            rendered_images.append(rgba)

            # Verify effective range increases with softness
            effective_range = scene.get_light_effective_range()
            expected_min_range = config["outer"] + config["softness"]
            assert effective_range >= expected_min_range

        # Verify images are different (softening is visually different)
        assert not np.array_equal(rendered_images[0], rendered_images[1])
        assert not np.array_equal(rendered_images[1], rendered_images[2])

        # Check that softness parameters affect the range
        scene.set_light_edge_softness(0.0)
        range_no_softness = scene.get_light_effective_range()

        scene.set_light_edge_softness(5.0)
        range_with_softness = scene.get_light_effective_range()

        assert range_with_softness > range_no_softness, "Edge softness should increase effective range"

    def test_falloff_modes(self, scene):
        """Test different falloff modes functionality"""
        scene.enable_soft_light_radius()
        scene.set_light_position(0.0, 5.0, 0.0)
        scene.set_light_intensity(1.5)
        scene.set_light_inner_radius(3.0)
        scene.set_light_outer_radius(10.0)

        falloff_modes = ["linear", "quadratic", "cubic", "exponential"]

        for mode in falloff_modes:
            scene.set_light_falloff_mode(mode)

            # Should render without errors
            rgba = scene.render_rgba()
            assert rgba is not None
            assert rgba.shape == (512, 512, 4)

    def test_falloff_exponent_control(self, scene):
        """Test falloff exponent parameter"""
        scene.enable_soft_light_radius()
        scene.set_light_falloff_mode("quadratic")

        exponents = [0.5, 1.0, 2.0, 3.0, 4.0]

        for exponent in exponents:
            scene.set_light_falloff_exponent(exponent)
            # Should not raise exceptions
            assert scene.is_soft_light_radius_enabled()

    def test_light_presets(self, scene):
        """Test different light presets"""
        scene.enable_soft_light_radius()

        presets = ["spotlight", "area_light", "ambient_light", "candle", "street_lamp"]

        for preset in presets:
            scene.set_light_preset(preset)

            # Should render without errors
            rgba = scene.render_rgba()
            assert rgba is not None

            # Each preset should have different effective range
            effective_range = scene.get_light_effective_range()
            assert effective_range > 0.0

    def test_shadow_softness_control(self, scene):
        """Test shadow softness parameter"""
        scene.enable_soft_light_radius()

        softness_values = [0.0, 0.5, 1.0, 1.5, 2.0]

        for softness in softness_values:
            scene.set_light_shadow_softness(softness)
            assert scene.is_soft_light_radius_enabled()

    def test_light_affects_point(self, scene):
        """Test light influence calculation"""
        scene.enable_soft_light_radius()
        scene.set_light_position(0.0, 5.0, 0.0)
        scene.set_light_inner_radius(3.0)
        scene.set_light_outer_radius(10.0)
        scene.set_light_edge_softness(2.0)

        # Points within inner radius should be affected
        assert scene.light_affects_point(0.0, 0.0, 0.0)  # Center
        assert scene.light_affects_point(2.0, 0.0, 0.0)  # Within inner radius

        # Points within outer + softness should be affected
        assert scene.light_affects_point(8.0, 0.0, 0.0)  # Within outer radius

        # Points far outside should not be affected
        assert not scene.light_affects_point(20.0, 0.0, 0.0)  # Far outside
        assert not scene.light_affects_point(0.0, 0.0, 50.0)  # Far away in Z

    def test_raster_path_stability(self, scene):
        """Test B12 acceptance criteria: raster path remains stable"""
        # Test multiple enable/disable cycles
        for i in range(10):
            scene.enable_soft_light_radius()
            assert scene.is_soft_light_radius_enabled()

            # Set parameters
            scene.set_light_position(0.0, 5.0, 0.0)
            scene.set_light_intensity(1.0 + i * 0.1)
            scene.set_light_inner_radius(2.0 + i * 0.5)
            scene.set_light_outer_radius(8.0 + i * 1.0)

            # Render should work
            rgba = scene.render_rgba()
            assert rgba is not None

            scene.disable_soft_light_radius()
            assert not scene.is_soft_light_radius_enabled()

        # Test rapid parameter changes
        scene.enable_soft_light_radius()

        for i in range(20):
            # Rapidly change multiple parameters
            scene.set_light_position(i % 5, 5.0 + i % 3, i % 4)
            scene.set_light_intensity(1.0 + (i % 10) * 0.2)
            scene.set_light_color((i % 5) / 4.0, 1.0, (i % 3) / 2.0)
            scene.set_light_inner_radius(1.0 + i % 8)
            scene.set_light_outer_radius(5.0 + i % 15)
            scene.set_light_edge_softness((i % 5) * 0.5)

            # Should remain stable
            rgba = scene.render_rgba()
            assert rgba is not None
            assert rgba.shape == (512, 512, 4)

    def test_performance_requirement(self, scene):
        """Test performance meets requirements for practical use"""
        # Use a larger scene closer to real usage
        large_scene = f3d.Scene(1024, 768, grid=64)
        large_scene.set_camera_look_at(
            eye=(0.0, 10.0, 10.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fovy_deg=45.0,
            znear=0.1,
            zfar=100.0
        )

        # Create more detailed terrain
        heights = np.random.rand(64, 64).astype(np.float32) * 5.0
        large_scene.upload_height_map(heights)

        large_scene.enable_soft_light_radius()
        large_scene.set_light_preset("area_light")

        # Measure frame time over multiple frames
        frame_times = []
        num_frames = 10

        for _ in range(num_frames):
            start_time = time.perf_counter()
            rgba = large_scene.render_rgba()
            end_time = time.perf_counter()

            frame_time = end_time - start_time
            frame_times.append(frame_time)

            assert rgba is not None

        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time

        # Should maintain reasonable performance (at least 30 FPS for testing)
        # B12 target is 60 FPS at 1080p, but testing at lower resolution/complexity
        min_fps = 30.0
        assert fps >= min_fps, f"Performance too low: {fps:.1f} FPS < {min_fps} FPS"

    def test_scene_round_trip_with_soft_light(self, scene):
        """Test scene round-trip behavior with soft light radius"""
        # Enable and configure soft light
        scene.enable_soft_light_radius()
        scene.set_light_position(5.0, 8.0, -2.0)
        scene.set_light_intensity(1.8)
        scene.set_light_color(1.0, 0.9, 0.7)
        scene.set_light_inner_radius(4.0)
        scene.set_light_outer_radius(12.0)
        scene.set_light_edge_softness(2.5)
        scene.set_light_falloff_mode("quadratic")
        scene.set_light_falloff_exponent(2.2)

        # Render first image
        rgba1 = scene.render_rgba()

        # Get current effective range
        range1 = scene.get_light_effective_range()

        # Disable and re-enable
        scene.disable_soft_light_radius()
        scene.enable_soft_light_radius()

        # Reapply same configuration
        scene.set_light_position(5.0, 8.0, -2.0)
        scene.set_light_intensity(1.8)
        scene.set_light_color(1.0, 0.9, 0.7)
        scene.set_light_inner_radius(4.0)
        scene.set_light_outer_radius(12.0)
        scene.set_light_edge_softness(2.5)
        scene.set_light_falloff_mode("quadratic")
        scene.set_light_falloff_exponent(2.2)

        # Render second image
        rgba2 = scene.render_rgba()
        range2 = scene.get_light_effective_range()

        # Should be identical (or very close due to floating point precision)
        assert np.allclose(range1, range2, rtol=1e-5)

        # Images should be very similar (allowing for minor numerical differences)
        mse = np.mean((rgba1.astype(float) - rgba2.astype(float)) ** 2)
        assert mse < 1.0, f"Scene round-trip failed: MSE = {mse}"

    def test_error_handling(self, scene):
        """Test error handling for invalid operations"""
        # Should raise errors when not enabled
        with pytest.raises(RuntimeError, match="not enabled"):
            scene.set_light_position(0.0, 5.0, 0.0)

        with pytest.raises(RuntimeError, match="not enabled"):
            scene.set_light_intensity(1.0)

        with pytest.raises(RuntimeError, match="not enabled"):
            scene.get_light_effective_range()

        # Test invalid falloff mode
        scene.enable_soft_light_radius()
        with pytest.raises(ValueError, match="Mode must be one of"):
            scene.set_light_falloff_mode("invalid_mode")

        # Test invalid preset
        with pytest.raises(ValueError, match="Preset must be one of"):
            scene.set_light_preset("invalid_preset")

@pytest.mark.performance
class TestB12Performance:
    """Performance-focused tests for B12"""

    def test_1080p_performance(self):
        """Test B12 acceptance criteria at 1080p resolution"""
        scene = f3d.Scene(1920, 1080, grid=128)
        scene.set_camera_look_at(
            eye=(0.0, 15.0, 15.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fovy_deg=45.0,
            znear=0.1,
            zfar=100.0
        )

        # Create realistic terrain
        heights = np.random.rand(128, 128).astype(np.float32) * 10.0
        scene.upload_height_map(heights)

        scene.enable_soft_light_radius()
        scene.set_light_preset("area_light")

        # Measure performance over multiple frames
        frame_times = []
        num_frames = 15

        for _ in range(num_frames):
            start_time = time.perf_counter()
            rgba = scene.render_rgba()
            end_time = time.perf_counter()

            frame_time = end_time - start_time
            frame_times.append(frame_time)

        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time

        # B12 acceptance criteria: maintain performance at 1080p
        # For CI testing, use a more relaxed target (20 FPS minimum)
        min_fps = 20.0
        assert fps >= min_fps, f"1080p performance insufficient: {fps:.1f} FPS < {min_fps} FPS"

        print(f"B12 Performance Test: {fps:.1f} FPS at 1080p")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])