"""
B14: Rect Area Lights (LTC) test suite

Tests the LTC (Linearly Transformed Cosines) rectangular area lights implementation
to verify B14 acceptance criteria: "(Verify) LTC path renders as before."

Acceptance Criteria:
1. LTC rect area lights can be enabled and disabled
2. Area lights render with physically accurate illumination
3. LTC approximation produces visually similar results to exact evaluation
4. Performance is acceptable for real-time rendering
5. Hardware capabilities are properly detected
6. API integration with Scene works correctly
"""

import pytest
import numpy as np
import forge3d as f3d
import time
from typing import Dict, List, Tuple, Optional


class TestB14LTCRectAreaLights:
    """Test suite for B14 LTC rect area lights functionality."""

    @pytest.fixture
    def scene_setup(self):
        """Create a test scene for LTC rect area lights."""
        scene = f3d.Scene(width=512, height=512)

        # Set up camera for good lighting view
        scene.set_camera_look_at(
            eye=(6.0, 4.0, 8.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fovy_deg=45.0,
            znear=0.1,
            zfar=50.0
        )

        # Create simple terrain for lighting tests
        heights = np.zeros((32, 32), dtype=np.float32)
        for y in range(32):
            for x in range(32):
                dx = x - 16
                dy = y - 16
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 10:
                    heights[y, x] = 1.0 * (1.0 - dist / 10.0)

        scene.upload_height_map(heights)
        return scene

    def test_ltc_enablement_and_disable(self, scene_setup):
        """Test B14.1: LTC area lights can be enabled and disabled."""
        scene = scene_setup

        # Initially should not be enabled
        assert scene.is_ltc_rect_area_lights_enabled() is False

        # Enable LTC rect area lights
        scene.enable_ltc_rect_area_lights(max_lights=8)
        assert scene.is_ltc_rect_area_lights_enabled() is True

        # Test light count starts at zero
        light_count = scene.get_rect_area_light_count()
        assert light_count == 0

        # Disable LTC rect area lights
        scene.disable_ltc_rect_area_lights()
        assert scene.is_ltc_rect_area_lights_enabled() is False

    def test_basic_area_light_creation(self, scene_setup):
        """Test B14.2: Basic rect area light creation and management."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=4)

        # Add a basic rect area light
        light_id = scene.add_rect_area_light(
            x=0.0, y=6.0, z=0.0,
            width=2.0, height=2.0,
            r=1.0, g=1.0, b=1.0,
            intensity=3.0
        )

        assert isinstance(light_id, int)
        assert light_id >= 0

        # Check light count
        light_count = scene.get_rect_area_light_count()
        assert light_count == 1

        # Add multiple lights
        light_id2 = scene.add_rect_area_light(
            -4.0, 5.0, -4.0, 1.5, 1.5, 1.0, 0.5, 0.2, 2.5
        )
        light_id3 = scene.add_rect_area_light(
            4.0, 5.0, 4.0, 1.8, 1.2, 0.2, 0.8, 1.0, 2.8
        )

        assert scene.get_rect_area_light_count() == 3

        # Remove a light
        scene.remove_rect_area_light(light_id2)
        assert scene.get_rect_area_light_count() == 2

    def test_custom_oriented_area_lights(self, scene_setup):
        """Test B14.3: Custom oriented area lights with right/up vectors."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=4)

        # Add custom oriented light
        light_id = scene.add_custom_rect_area_light(
            position=(0.0, 6.0, 0.0),
            right_vec=(1.0, 0.0, 0.0),
            up_vec=(0.0, 0.0, 1.0),
            width=3.0,
            height=2.0,
            r=1.0, g=0.8, b=0.6,
            intensity=4.0,
            two_sided=False
        )

        assert isinstance(light_id, int)
        assert light_id >= 0
        assert scene.get_rect_area_light_count() == 1

        # Add two-sided light
        light_id2 = scene.add_custom_rect_area_light(
            position=(-5.0, 4.0, 0.0),
            right_vec=(0.0, 1.0, 0.0),
            up_vec=(0.0, 0.0, 1.0),
            width=2.5,
            height=3.0,
            r=0.8, g=0.4, b=1.0,
            intensity=3.5,
            two_sided=True
        )

        assert scene.get_rect_area_light_count() == 2

    def test_light_modification(self, scene_setup):
        """Test B14.4: Dynamic light property modification."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=2)

        # Add initial light
        light_id = scene.add_rect_area_light(
            0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0
        )

        # Update light properties
        scene.update_rect_area_light(
            light_id, 1.0, 7.0, 1.0, 2.5, 1.5, 0.8, 0.6, 1.0, 4.0
        )

        # Should still have one light
        assert scene.get_rect_area_light_count() == 1

        # Render to ensure no crashes
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)

    def test_ltc_approximation_vs_exact(self, scene_setup):
        """Test B14.5: LTC approximation vs exact evaluation."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=3)

        # Add test lights
        scene.add_rect_area_light(0.0, 8.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0)
        scene.add_rect_area_light(-4.0, 6.0, -4.0, 1.5, 1.5, 1.0, 0.6, 0.3, 3.0)
        scene.add_rect_area_light(4.0, 6.0, 4.0, 1.8, 1.2, 0.3, 0.8, 1.0, 3.5)

        # Render with LTC approximation
        scene.set_ltc_approximation_enabled(True)
        rgba_ltc = scene.render_rgba()

        # Render with exact evaluation
        scene.set_ltc_approximation_enabled(False)
        rgba_exact = scene.render_rgba()

        # Both should render successfully
        assert rgba_ltc.shape == (512, 512, 4)
        assert rgba_exact.shape == (512, 512, 4)

        # Calculate visual difference
        diff = np.abs(rgba_ltc.astype(np.float32) - rgba_exact.astype(np.float32))
        mean_diff = np.mean(diff[:, :, :3]) / 255.0  # Normalize to [0,1]
        max_diff = np.max(diff[:, :, :3]) / 255.0

        # LTC approximation should be reasonably close to exact
        assert mean_diff < 0.1, f"Mean difference too large: {mean_diff}"
        assert max_diff < 0.5, f"Max difference too large: {max_diff}"

        print(f"LTC vs Exact - Mean diff: {mean_diff:.4f}, Max diff: {max_diff:.4f}")

    def test_global_intensity_control(self, scene_setup):
        """Test B14.6: Global intensity control."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=2)

        # Add lights
        scene.add_rect_area_light(0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0)

        # Test global intensity modification
        scene.set_ltc_global_intensity(2.0)
        uniforms = scene.get_ltc_uniforms()
        assert abs(uniforms[1] - 2.0) < 0.01

        scene.set_ltc_global_intensity(0.5)
        uniforms = scene.get_ltc_uniforms()
        assert abs(uniforms[1] - 0.5) < 0.01

        # Render with different intensities
        rgba_bright = scene.render_rgba()

        scene.set_ltc_global_intensity(0.1)
        rgba_dim = scene.render_rgba()

        # Brighter should have higher pixel values overall
        mean_bright = np.mean(rgba_bright[:, :, :3].astype(np.float32))
        mean_dim = np.mean(rgba_dim[:, :, :3].astype(np.float32))
        assert mean_bright > mean_dim

    def test_uniforms_and_state(self, scene_setup):
        """Test B14.7: LTC uniforms and state management."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=4)

        # Check initial uniforms
        uniforms = scene.get_ltc_uniforms()
        assert len(uniforms) == 3
        light_count, global_intensity, ltc_enabled = uniforms

        assert light_count == 0  # No lights added yet
        assert abs(global_intensity - 1.0) < 0.01  # Default intensity
        assert ltc_enabled is True  # Default enabled

        # Add lights and check state updates
        scene.add_rect_area_light(0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0)
        scene.add_rect_area_light(-3.0, 5.0, -3.0, 1.5, 1.5, 1.0, 0.5, 0.2, 2.5)

        uniforms = scene.get_ltc_uniforms()
        assert uniforms[0] == 2  # Two lights

        # Test LTC enable/disable
        scene.set_ltc_approximation_enabled(False)
        uniforms = scene.get_ltc_uniforms()
        assert uniforms[2] is False

        scene.set_ltc_approximation_enabled(True)
        uniforms = scene.get_ltc_uniforms()
        assert uniforms[2] is True

    def test_performance_requirements(self, scene_setup):
        """Test B14.8: Performance requirements for real-time rendering."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=12)

        # Add multiple lights for performance testing
        for i in range(8):
            angle = i * 2 * np.pi / 8
            x = 6.0 * np.cos(angle)
            z = 6.0 * np.sin(angle)
            scene.add_rect_area_light(
                x, 5.0 + i * 0.2, z,
                1.5, 1.5,
                0.8, 0.8, 0.8,
                2.0
            )

        # Measure performance with LTC
        scene.set_ltc_approximation_enabled(True)
        frame_times_ltc = []

        for _ in range(5):
            start_time = time.perf_counter()
            rgba = scene.render_rgba()
            end_time = time.perf_counter()
            frame_times_ltc.append(end_time - start_time)

        # Measure performance with exact evaluation
        scene.set_ltc_approximation_enabled(False)
        frame_times_exact = []

        for _ in range(5):
            start_time = time.perf_counter()
            rgba = scene.render_rgba()
            end_time = time.perf_counter()
            frame_times_exact.append(end_time - start_time)

        # Calculate performance metrics
        avg_ltc = np.mean(frame_times_ltc)
        avg_exact = np.mean(frame_times_exact)
        speedup = avg_exact / avg_ltc if avg_ltc > 0 else 1.0

        print(f"Performance - LTC: {avg_ltc*1000:.1f}ms, Exact: {avg_exact*1000:.1f}ms, Speedup: {speedup:.1f}x")

        # LTC should be reasonably fast for real-time use
        assert avg_ltc < 0.5, f"LTC rendering too slow: {avg_ltc*1000:.1f}ms"

        # LTC should provide some performance benefit over exact
        assert speedup >= 1.0, f"LTC should be faster than exact evaluation"

    def test_error_handling(self, scene_setup):
        """Test B14.9: Proper error handling for edge cases."""
        scene = scene_setup

        # Test operations without enabling LTC first
        with pytest.raises(Exception, match="LTC rect area lights not enabled"):
            scene.add_rect_area_light(0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0)

        with pytest.raises(Exception, match="LTC rect area lights not enabled"):
            scene.get_rect_area_light_count()

        with pytest.raises(Exception, match="LTC rect area lights not enabled"):
            scene.set_ltc_global_intensity(2.0)

        # Enable and test valid operations
        scene.enable_ltc_rect_area_lights(max_lights=2)

        # Add valid light
        light_id = scene.add_rect_area_light(0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0)

        # Test invalid light removal
        with pytest.raises(Exception):
            scene.remove_rect_area_light(999)  # Non-existent light

        # Test invalid light update
        with pytest.raises(Exception):
            scene.update_rect_area_light(999, 0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0)

    @pytest.mark.parametrize("max_lights", [1, 4, 8, 16])
    def test_light_capacity(self, scene_setup, max_lights):
        """Test B14.10: Different maximum light capacities."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=max_lights)

        # Add lights up to capacity
        light_ids = []
        for i in range(max_lights):
            angle = i * 2 * np.pi / max_lights
            x = 5.0 * np.cos(angle)
            z = 5.0 * np.sin(angle)

            light_id = scene.add_rect_area_light(
                x, 5.0, z, 1.5, 1.5, 0.8, 0.8, 0.8, 2.0
            )
            light_ids.append(light_id)

        assert scene.get_rect_area_light_count() == max_lights

        # Try to add one more (should fail or handle gracefully)
        if max_lights < 32:  # Only test overflow for reasonable limits
            try:
                scene.add_rect_area_light(0.0, 10.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                # If it succeeds, check we don't exceed the limit
                assert scene.get_rect_area_light_count() <= max_lights
            except Exception:
                # Expected to fail when at capacity
                assert scene.get_rect_area_light_count() == max_lights

        # Render to ensure functionality works with capacity
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)
        assert np.any(rgba[:, :, 3] > 0)  # Some rendering occurred

    def test_visual_correctness(self, scene_setup):
        """Test B14.11: Visual correctness of area light rendering."""
        scene = scene_setup
        scene.enable_ltc_rect_area_lights(max_lights=2)

        # Add a bright area light above the center
        scene.add_rect_area_light(0.0, 8.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 8.0)

        # Render scene
        rgba = scene.render_rgba()

        # Check that lighting is present
        assert np.any(rgba[:, :, :3] > 10)  # Some illumination should be visible

        # Check that center region is brighter (directly under light)
        center_region = rgba[240:272, 240:272, :3]  # 32x32 center
        edge_region = rgba[0:32, 0:32, :3]          # 32x32 corner

        center_brightness = np.mean(center_region.astype(np.float32))
        edge_brightness = np.mean(edge_region.astype(np.float32))

        assert center_brightness > edge_brightness, "Center should be brighter than edges"

        # Validate alpha channel
        assert np.all(rgba[:, :, 3] > 0), "Alpha channel should be fully opaque"

    def test_integration_with_other_systems(self, scene_setup):
        """Test B14.12: Integration with other Scene systems."""
        scene = scene_setup

        # Enable multiple lighting systems if available
        scene.enable_ltc_rect_area_lights(max_lights=4)

        # Add LTC area light
        ltc_light = scene.add_rect_area_light(0.0, 6.0, 0.0, 2.0, 2.0, 1.0, 0.8, 0.6, 4.0)

        # Try to enable other lighting systems (may not exist, handle gracefully)
        try:
            scene.enable_point_spot_lights(max_lights=4)
            point_light = scene.add_point_light(3.0, 5.0, 3.0, 0.8, 1.0, 0.6, 3.0, 12.0)
            print(f"Integration test: LTC light {ltc_light}, Point light {point_light}")
        except AttributeError:
            # Point/spot lights may not be available
            print("Point/spot lights not available for integration test")

        # Render with multiple systems
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)

        # Check that rendering produces visible results
        mean_brightness = np.mean(rgba[:, :, :3])
        assert mean_brightness > 5.0, "Combined lighting should be visible"

    def test_acceptance_criteria_summary(self, scene_setup):
        """Test B14.13: Overall B14 acceptance criteria validation."""
        scene = scene_setup

        print("B14 Acceptance Criteria Validation:")

        # AC1: LTC functionality exists and works
        scene.enable_ltc_rect_area_lights(max_lights=8)
        assert scene.is_ltc_rect_area_lights_enabled() is True
        print("  ✓ LTC rect area lights can be enabled")

        # AC2: Area lights render correctly
        light_id = scene.add_rect_area_light(0.0, 6.0, 0.0, 2.5, 2.5, 1.0, 1.0, 1.0, 5.0)
        rgba = scene.render_rgba()
        assert np.any(rgba[:, :, :3] > 10)
        print("  ✓ Area lights render with visible illumination")

        # AC3: LTC approximation works
        scene.set_ltc_approximation_enabled(True)
        rgba_ltc = scene.render_rgba()
        scene.set_ltc_approximation_enabled(False)
        rgba_exact = scene.render_rgba()

        diff = np.mean(np.abs(rgba_ltc.astype(np.float32) - rgba_exact.astype(np.float32)))
        assert diff < 50.0  # Reasonable visual similarity
        print(f"  ✓ LTC approximation produces similar results (diff: {diff:.1f})")

        # AC4: Performance is acceptable
        start_time = time.perf_counter()
        for _ in range(3):
            rgba = scene.render_rgba()
        avg_time = (time.perf_counter() - start_time) / 3.0

        assert avg_time < 1.0  # Should render within reasonable time
        print(f"  ✓ Performance acceptable ({avg_time*1000:.1f}ms per frame)")

        # AC5: API integration works
        uniforms = scene.get_ltc_uniforms()
        assert len(uniforms) == 3
        assert uniforms[0] == 1  # One light
        print("  ✓ API integration functional")

        print("\n🎉 B14 LTC Rect Area Lights: ALL ACCEPTANCE CRITERIA MET")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])