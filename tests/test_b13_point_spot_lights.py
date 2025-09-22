#!/usr/bin/env python3
"""
B13: Point & Spot Lights (Realtime) - Test Suite

Tests for B13 acceptance criteria:
- Point/spot lights illuminate correctly
- Shadow toggles verified

This test suite validates the complete B13 implementation including:
- Point and spot light creation and management
- Light parameter controls (position, color, intensity, range, direction)
- Spot light cone angles and penumbra shaping
- Light presets functionality
- Shadow toggles for individual lights
- Performance requirements
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
        eye=(8.0, 10.0, 12.0),
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

class TestB13PointSpotLights:
    """Test suite for B13 Point & Spot Lights functionality"""

    def test_point_spot_lights_enable_disable(self, scene):
        """Test basic enable/disable functionality"""
        # Should start disabled
        assert not scene.is_point_spot_lights_enabled()

        # Enable point/spot lights
        scene.enable_point_spot_lights()
        assert scene.is_point_spot_lights_enabled()

        # Disable point/spot lights
        scene.disable_point_spot_lights()
        assert not scene.is_point_spot_lights_enabled()

    def test_point_light_creation(self, scene):
        """Test point light creation and basic properties"""
        scene.enable_point_spot_lights()

        # Add point light
        light_id = scene.add_point_light(
            0.0, 5.0, 0.0,      # position
            1.0, 0.8, 0.6,      # color (warm white)
            2.0,                # intensity
            15.0                # range
        )

        assert isinstance(light_id, int)
        assert light_id != 4294967295  # u32::MAX indicates failure

        # Check light count
        assert scene.get_light_count() == 1

        # Test light affects point calculation
        assert scene.check_light_affects_point(light_id, 0.0, 0.0, 0.0)  # Should be within range
        assert not scene.check_light_affects_point(light_id, 50.0, 0.0, 0.0)  # Should be outside range

    def test_spot_light_creation(self, scene):
        """Test spot light creation with cone angles and penumbra"""
        scene.enable_point_spot_lights()

        # Add spot light
        light_id = scene.add_spot_light(
            0.0, 8.0, 0.0,          # position
            0.0, -1.0, 0.0,         # direction (pointing down)
            1.0, 1.0, 0.9,          # color
            2.5,                    # intensity
            20.0,                   # range
            20.0,                   # inner cone angle (degrees)
            35.0,                   # outer cone angle (degrees)
            1.2                     # penumbra softness
        )

        assert isinstance(light_id, int)
        assert light_id != 4294967295

        # Check light count
        assert scene.get_light_count() == 1

        # Test spot light affects point (should be more selective than point light)
        assert scene.check_light_affects_point(light_id, 0.0, 0.0, 0.0)  # Directly below
        # Point to the side may or may not be affected depending on cone
        side_affected = scene.check_light_affects_point(light_id, 10.0, 0.0, 0.0)
        # Should definitely not affect points far outside range
        assert not scene.check_light_affects_point(light_id, 50.0, 0.0, 0.0)

    def test_light_presets(self, scene):
        """Test predefined light presets"""
        scene.enable_point_spot_lights()

        presets = [
            "room_light", "desk_lamp", "street_light", "spotlight",
            "headlight", "flashlight", "candle", "warm_lamp"
        ]

        light_ids = []
        for preset in presets:
            light_id = scene.add_light_preset(preset, 0.0, 5.0, 0.0)
            assert isinstance(light_id, int)
            assert light_id != 4294967295
            light_ids.append(light_id)

            # Each preset should create a valid light
            assert scene.get_light_count() == len(light_ids)

        # Test that presets create different lighting (render should work)
        for light_id in light_ids[:-1]:  # Remove all but last
            scene.remove_light(light_id)

        rgba = scene.render_rgba()
        assert rgba is not None
        assert rgba.shape == (512, 512, 4)

    def test_light_parameter_control(self, scene):
        """Test individual light parameter controls"""
        scene.enable_point_spot_lights()

        # Create a point light to modify
        light_id = scene.add_point_light(0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0)

        # Test position control
        scene.set_light_position(light_id, 5.0, 8.0, -3.0)

        # Test color control
        scene.set_light_color(light_id, 1.0, 0.5, 0.2)

        # Test intensity control
        scene.set_light_intensity(light_id, 2.5)

        # Test range control
        scene.set_light_range(light_id, 18.0)

        # Should render without errors
        rgba = scene.render_rgba()
        assert rgba is not None

    def test_spot_light_cone_and_penumbra(self, scene):
        """Test spot light cone angles and penumbra shaping"""
        scene.enable_point_spot_lights()

        # Create a spot light
        light_id = scene.add_spot_light(
            0.0, 8.0, 0.0, 0.0, -1.0, 0.0,
            1.0, 1.0, 1.0, 2.0, 15.0,
            25.0, 40.0, 1.0
        )

        # Test direction control
        scene.set_light_direction(light_id, 0.5, -1.0, 0.0)

        # Test cone angle control
        scene.set_spot_light_cone(light_id, 15.0, 30.0)  # Tighter cone

        # Test penumbra control
        scene.set_spot_light_penumbra(light_id, 2.0)  # Softer penumbra

        # Should render without errors
        rgba = scene.render_rgba()
        assert rgba is not None

        # Test that penumbra affects the rendering
        scene.set_spot_light_penumbra(light_id, 0.1)  # Sharp penumbra
        rgba_sharp = scene.render_rgba()

        scene.set_spot_light_penumbra(light_id, 3.0)  # Very soft penumbra
        rgba_soft = scene.render_rgba()

        # Images should be different (penumbra affects appearance)
        assert not np.array_equal(rgba_sharp, rgba_soft)

    def test_shadow_toggles_verification(self, scene):
        """Test B13 acceptance criteria: shadow toggles verified"""
        scene.enable_point_spot_lights()

        # Add lights with shadows
        light1 = scene.add_point_light(0.0, 8.0, 0.0, 1.0, 1.0, 1.0, 2.0, 15.0)
        light2 = scene.add_spot_light(-5.0, 6.0, 0.0, 0.5, -1.0, 0.0, 1.0, 0.8, 0.6, 1.8, 12.0, 20.0, 35.0, 1.0)

        # Test global shadow quality settings
        shadow_qualities = ["off", "low", "medium", "high"]
        for quality in shadow_qualities:
            scene.set_shadow_quality(quality)
            # Should render without errors
            rgba = scene.render_rgba()
            assert rgba is not None

        # Test individual light shadow toggles
        scene.set_shadow_quality("medium")

        # Disable shadows for first light
        scene.set_light_shadows(light1, False)
        rgba_light1_no_shadow = scene.render_rgba()
        assert rgba_light1_no_shadow is not None

        # Re-enable shadows for first light, disable for second
        scene.set_light_shadows(light1, True)
        scene.set_light_shadows(light2, False)
        rgba_light2_no_shadow = scene.render_rgba()
        assert rgba_light2_no_shadow is not None

        # Images should be different (shadow toggles affect rendering)
        assert not np.array_equal(rgba_light1_no_shadow, rgba_light2_no_shadow)

        print("✓ Shadow toggles verified - individual lights can have shadows enabled/disabled")

    def test_multiple_lights_illumination(self, scene):
        """Test that multiple lights illuminate correctly together"""
        scene.enable_point_spot_lights(max_lights=8)
        scene.set_ambient_lighting(0.1, 0.1, 0.12, 0.2)

        # Add multiple lights of different types
        lights = []

        # Point lights
        lights.append(scene.add_point_light(0.0, 8.0, 0.0, 1.0, 1.0, 1.0, 2.0, 15.0))
        lights.append(scene.add_point_light(-8.0, 5.0, -8.0, 1.0, 0.3, 0.3, 1.5, 12.0))
        lights.append(scene.add_point_light(8.0, 5.0, 8.0, 0.3, 0.3, 1.0, 1.5, 12.0))

        # Spot lights
        lights.append(scene.add_spot_light(0.0, 12.0, -10.0, 0.0, -0.8, 0.6, 1.0, 1.0, 0.9, 2.5, 18.0, 18.0, 30.0, 0.8))

        assert scene.get_light_count() == 4

        # All lights should be valid
        for light_id in lights:
            assert light_id != 4294967295

        # Scene should render with multiple lights
        rgba = scene.render_rgba()
        assert rgba is not None
        assert rgba.shape == (512, 512, 4)

        # Test light interactions by modifying individual lights
        original_intensity = 2.0
        for light_id in lights[:2]:  # Modify first two lights
            scene.set_light_intensity(light_id, original_intensity * 0.5)

        rgba_dimmed = scene.render_rgba()
        assert rgba_dimmed is not None

        # Should be visually different
        mse = np.mean((rgba.astype(float) - rgba_dimmed.astype(float)) ** 2)
        assert mse > 1.0, "Light intensity changes should affect rendering"

    def test_light_management(self, scene):
        """Test light addition, removal, and management"""
        scene.enable_point_spot_lights(max_lights=5)

        # Add lights
        light_ids = []
        for i in range(3):
            light_id = scene.add_point_light(
                i * 2.0, 5.0, 0.0,
                1.0, 0.5, 0.2,
                1.5, 10.0
            )
            light_ids.append(light_id)

        assert scene.get_light_count() == 3

        # Remove one light
        removed = scene.remove_light(light_ids[1])
        assert removed is True
        assert scene.get_light_count() == 2

        # Try to remove non-existent light
        removed = scene.remove_light(99999)
        assert removed is False
        assert scene.get_light_count() == 2

        # Clear all lights
        scene.clear_all_lights()
        assert scene.get_light_count() == 0

        # Should still render (with no lights)
        rgba = scene.render_rgba()
        assert rgba is not None

    def test_ambient_lighting_control(self, scene):
        """Test ambient lighting control"""
        scene.enable_point_spot_lights()

        # Test different ambient lighting settings
        ambient_configs = [
            (0.0, 0.0, 0.0, 0.0),      # No ambient
            (0.2, 0.2, 0.3, 0.5),      # Cool ambient
            (0.3, 0.2, 0.1, 0.3),      # Warm ambient
            (0.1, 0.1, 0.1, 1.0),      # Bright ambient
        ]

        for r, g, b, intensity in ambient_configs:
            scene.set_ambient_lighting(r, g, b, intensity)

            # Should render without errors
            rgba = scene.render_rgba()
            assert rgba is not None

    def test_debug_visualization_modes(self, scene):
        """Test debug visualization modes"""
        scene.enable_point_spot_lights()

        # Add a light for debug visualization
        light_id = scene.add_point_light(0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 2.0, 12.0)

        debug_modes = ["normal", "show_light_bounds", "show_shadows"]

        for mode in debug_modes:
            scene.set_lighting_debug_mode(mode)

            # Should render without errors
            rgba = scene.render_rgba()
            assert rgba is not None

        # Return to normal mode
        scene.set_lighting_debug_mode("normal")

    def test_light_range_affects_illumination(self, scene):
        """Test that light range correctly affects illumination"""
        scene.enable_point_spot_lights()

        # Create light with small range
        light_id = scene.add_point_light(0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 2.0, 5.0)

        # Point within range should be affected
        assert scene.check_light_affects_point(light_id, 0.0, 0.0, 0.0)
        assert scene.check_light_affects_point(light_id, 3.0, 0.0, 0.0)

        # Point outside range should not be affected
        assert not scene.check_light_affects_point(light_id, 10.0, 0.0, 0.0)

        # Increase range and test again
        scene.set_light_range(light_id, 15.0)
        assert scene.check_light_affects_point(light_id, 10.0, 0.0, 0.0)  # Now should be affected

    def test_max_lights_limitation(self, scene):
        """Test maximum lights limitation"""
        max_lights = 3
        scene.enable_point_spot_lights(max_lights=max_lights)

        # Add lights up to the maximum
        light_ids = []
        for i in range(max_lights):
            light_id = scene.add_point_light(i * 2.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0)
            assert light_id != 4294967295  # Should succeed
            light_ids.append(light_id)

        assert scene.get_light_count() == max_lights

        # Try to add one more light (should fail)
        with pytest.raises(RuntimeError, match="Maximum number of lights exceeded"):
            scene.add_point_light(10.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0)

        # Count should remain the same
        assert scene.get_light_count() == max_lights

    def test_scene_round_trip_with_lights(self, scene):
        """Test scene round-trip behavior with lights"""
        scene.enable_point_spot_lights()

        # Configure complex lighting setup
        light1 = scene.add_point_light(5.0, 8.0, -2.0, 1.0, 0.8, 0.6, 2.2, 16.0)
        light2 = scene.add_spot_light(-3.0, 6.0, 4.0, 0.3, -1.0, -0.2, 0.7, 0.9, 1.0, 1.8, 14.0, 22.0, 38.0, 1.5)

        scene.set_ambient_lighting(0.15, 0.12, 0.18, 0.25)
        scene.set_shadow_quality("medium")
        scene.set_light_shadows(light1, True)
        scene.set_light_shadows(light2, False)

        # Render first image
        rgba1 = scene.render_rgba()
        count1 = scene.get_light_count()

        # Disable and re-enable lights
        scene.disable_point_spot_lights()
        scene.enable_point_spot_lights()

        # Recreate same lighting setup
        new_light1 = scene.add_point_light(5.0, 8.0, -2.0, 1.0, 0.8, 0.6, 2.2, 16.0)
        new_light2 = scene.add_spot_light(-3.0, 6.0, 4.0, 0.3, -1.0, -0.2, 0.7, 0.9, 1.0, 1.8, 14.0, 22.0, 38.0, 1.5)

        scene.set_ambient_lighting(0.15, 0.12, 0.18, 0.25)
        scene.set_shadow_quality("medium")
        scene.set_light_shadows(new_light1, True)
        scene.set_light_shadows(new_light2, False)

        # Render second image
        rgba2 = scene.render_rgba()
        count2 = scene.get_light_count()

        # Should have same light count
        assert count1 == count2

        # Images should be very similar (allowing for minor numerical differences)
        mse = np.mean((rgba1.astype(float) - rgba2.astype(float)) ** 2)
        assert mse < 2.0, f"Scene round-trip failed: MSE = {mse}"

    def test_error_handling(self, scene):
        """Test error handling for invalid operations"""
        # Should raise errors when not enabled
        with pytest.raises(RuntimeError, match="not enabled"):
            scene.add_point_light(0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0)

        with pytest.raises(RuntimeError, match="not enabled"):
            scene.get_light_count()

        scene.enable_point_spot_lights()

        # Test invalid light IDs
        with pytest.raises(ValueError, match="not found"):
            scene.set_light_position(99999, 0.0, 0.0, 0.0)

        with pytest.raises(ValueError, match="not found"):
            scene.check_light_affects_point(99999, 0.0, 0.0, 0.0)

        # Test invalid preset
        with pytest.raises(ValueError, match="Preset must be one of"):
            scene.add_light_preset("invalid_preset", 0.0, 5.0, 0.0)

        # Test invalid shadow quality
        with pytest.raises(ValueError, match="Quality must be one of"):
            scene.set_shadow_quality("invalid_quality")

        # Test invalid debug mode
        with pytest.raises(ValueError, match="Mode must be one of"):
            scene.set_lighting_debug_mode("invalid_mode")

@pytest.mark.performance
class TestB13Performance:
    """Performance-focused tests for B13"""

    def test_multiple_lights_performance(self):
        """Test performance with multiple lights"""
        scene = f3d.Scene(1024, 768, grid=64)
        scene.set_camera_look_at(
            eye=(10.0, 12.0, 15.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fovy_deg=45.0,
            znear=0.1,
            zfar=100.0
        )

        # Create terrain
        heights = np.random.rand(64, 64).astype(np.float32) * 5.0
        scene.upload_height_map(heights)

        scene.enable_point_spot_lights(max_lights=16)
        scene.set_ambient_lighting(0.1, 0.1, 0.12, 0.2)

        # Add multiple lights
        light_count = 12
        for i in range(light_count):
            angle = i * 2 * np.pi / light_count
            x = 8.0 * np.cos(angle)
            z = 8.0 * np.sin(angle)
            y = 5.0 + 2.0 * np.sin(i * 0.5)

            if i % 2 == 0:
                # Point light
                scene.add_point_light(x, y, z, 1.0, 0.8, 0.6, 1.5, 12.0)
            else:
                # Spot light
                scene.add_spot_light(x, y, z, -x/8, -1.0, -z/8, 0.9, 0.7, 1.0, 1.8, 15.0, 25.0, 40.0, 1.0)

        # Measure performance
        frame_times = []
        num_frames = 10

        for _ in range(num_frames):
            start_time = time.perf_counter()
            rgba = scene.render_rgba()
            end_time = time.perf_counter()

            frame_time = end_time - start_time
            frame_times.append(frame_time)

        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time

        # Should maintain reasonable performance with multiple lights
        min_fps = 20.0  # Reasonable minimum for testing
        assert fps >= min_fps, f"Performance too low: {fps:.1f} FPS < {min_fps} FPS with {light_count} lights"

        print(f"B13 Performance Test: {fps:.1f} FPS with {light_count} lights")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])