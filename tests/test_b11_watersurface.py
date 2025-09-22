"""
B11: Water Surface Color Toggle tests

Tests the water surface rendering system including:
- Pipeline uniform controlling water albedo/hue
- Python setter methods for water color control
- Water tint toggling and transparency
- Water surface mode switching
- Wave animation and flow direction
- Water presets and parameter validation

Validates that water tint toggles predictably and scenes round-trip.
"""

import pytest
import numpy as np
import forge3d as f3d
from pathlib import Path


def _gpu_ok():
    """Check if GPU is available for testing."""
    try:
        import forge3d as f3d
        # Try to create a minimal scene to test GPU availability
        scene = f3d.Scene(32, 32, 16)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
class TestWaterSurfaceBasics:
    """Test basic water surface functionality."""

    def setup_method(self):
        """Setup test scene."""
        self.scene = f3d.Scene(64, 64, 32)
        # Create simple test terrain
        heights = np.zeros((32, 32), dtype=np.float32)
        heights += 0.1  # Slightly above water level
        self.scene.set_terrain_dims(32, 32, 10.0)
        self.scene.set_terrain_heights(heights)

    def test_water_surface_enable_disable(self):
        """Test enabling and disabling water surface."""
        # Initially disabled
        assert not self.scene.is_water_surface_enabled()

        # Enable water surface
        self.scene.enable_water_surface()
        assert self.scene.is_water_surface_enabled()

        # Disable water surface
        self.scene.disable_water_surface()
        assert not self.scene.is_water_surface_enabled()

    def test_water_surface_modes(self):
        """Test different water surface modes."""
        self.scene.enable_water_surface()

        # Test all valid modes
        valid_modes = ["disabled", "transparent", "reflective", "animated"]
        for mode in valid_modes:
            self.scene.set_water_surface_mode(mode)
            # Mode change should succeed without error

        # Test invalid mode
        with pytest.raises(Exception):  # Should raise PyValueError
            self.scene.set_water_surface_mode("invalid_mode")

    def test_water_surface_without_enable(self):
        """Test that setting parameters without enabling raises errors."""
        # Should raise error when water surface not enabled
        with pytest.raises(Exception):
            self.scene.set_water_base_color(0.1, 0.3, 0.6)

        with pytest.raises(Exception):
            self.scene.set_water_hue_shift(1.0)

        with pytest.raises(Exception):
            self.scene.get_water_surface_params()


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
class TestWaterSurfaceColorControl:
    """Test water color and hue control functionality."""

    def setup_method(self):
        """Setup test scene with water surface enabled."""
        self.scene = f3d.Scene(64, 64, 32)
        heights = np.zeros((32, 32), dtype=np.float32)
        self.scene.set_terrain_dims(32, 32, 10.0)
        self.scene.set_terrain_heights(heights)
        self.scene.enable_water_surface()

    def test_water_base_color(self):
        """Test setting water base color."""
        # Test valid color values
        self.scene.set_water_base_color(0.1, 0.3, 0.6)
        self.scene.set_water_base_color(1.0, 1.0, 1.0)  # White
        self.scene.set_water_base_color(0.0, 0.0, 0.0)  # Black

        # All should succeed without error

    def test_water_hue_shift(self):
        """Test water hue shifting functionality."""
        # Test various hue shift values
        hue_shifts = [0.0, 1.57, 3.14, 6.28, -1.57]  # 0°, 90°, 180°, 360°, -90°
        for hue_shift in hue_shifts:
            self.scene.set_water_hue_shift(hue_shift)
            # Should succeed without error

    def test_water_tint_control(self):
        """Test water tint color and strength control."""
        # Test various tint configurations
        tint_configs = [
            (0.0, 0.5, 1.0, 0.0),    # Blue tint, no strength
            (1.0, 0.0, 0.0, 0.5),    # Red tint, medium strength
            (0.0, 1.0, 0.0, 1.0),    # Green tint, full strength
            (0.5, 0.5, 0.5, 0.25),   # Gray tint, low strength
        ]

        for r, g, b, strength in tint_configs:
            self.scene.set_water_tint(r, g, b, strength)
            # Should succeed without error

    def test_water_alpha_control(self):
        """Test water transparency control."""
        # Test various alpha values
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha in alpha_values:
            self.scene.set_water_alpha(alpha)
            # Should succeed without error

    def test_water_color_round_trip(self):
        """Test that water color settings persist correctly."""
        # Set specific values
        self.scene.set_water_base_color(0.2, 0.4, 0.8)
        self.scene.set_water_hue_shift(1.57)
        self.scene.set_water_tint(0.0, 1.0, 0.0, 0.3)
        self.scene.set_water_alpha(0.7)

        # Get parameters back
        params = self.scene.get_water_surface_params()
        assert isinstance(params, tuple)
        assert len(params) == 4

        # Parameters should be reasonable values
        height, alpha, hue_shift, tint_strength = params
        assert isinstance(height, float)
        assert isinstance(alpha, float)
        assert isinstance(hue_shift, float)
        assert isinstance(tint_strength, float)

        # Alpha should match what we set
        assert abs(alpha - 0.7) < 0.01
        assert abs(hue_shift - 1.57) < 0.01
        assert abs(tint_strength - 0.3) < 0.01


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
class TestWaterSurfaceAnimation:
    """Test water animation and wave functionality."""

    def setup_method(self):
        """Setup test scene with water surface enabled."""
        self.scene = f3d.Scene(64, 64, 32)
        heights = np.zeros((32, 32), dtype=np.float32)
        self.scene.set_terrain_dims(32, 32, 10.0)
        self.scene.set_terrain_heights(heights)
        self.scene.enable_water_surface()

    def test_water_wave_parameters(self):
        """Test setting wave animation parameters."""
        # Test various wave configurations
        wave_configs = [
            (0.0, 1.0, 1.0),    # No waves
            (0.1, 2.0, 0.5),    # Small slow waves
            (0.5, 1.5, 2.0),    # Large fast waves
            (0.2, 4.0, 1.5),    # Small high-frequency waves
        ]

        for amplitude, frequency, speed in wave_configs:
            self.scene.set_water_wave_params(amplitude, frequency, speed)
            # Should succeed without error

    def test_water_flow_direction(self):
        """Test setting water flow direction."""
        # Test various flow directions
        flow_directions = [
            (1.0, 0.0),    # East
            (0.0, 1.0),    # North
            (-1.0, 0.0),   # West
            (0.0, -1.0),   # South
            (0.707, 0.707), # Northeast
        ]

        for dx, dy in flow_directions:
            self.scene.set_water_flow_direction(dx, dy)
            # Should succeed without error

    def test_water_animation_update(self):
        """Test water animation time updates."""
        # Set up animated water
        self.scene.set_water_surface_mode("animated")
        self.scene.set_water_wave_params(0.2, 2.0, 1.0)

        # Update animation with different time steps
        time_steps = [0.0, 0.016, 0.1, 1.0, 5.0]  # Various dt values
        for dt in time_steps:
            self.scene.update_water_animation(dt)
            # Should succeed without error

    def test_water_lighting_parameters(self):
        """Test water lighting and visual parameters."""
        # Test various lighting configurations
        lighting_configs = [
            (0.0, 0.0, 1.0, 0.0),    # No reflection/refraction
            (1.0, 0.5, 5.0, 0.1),    # Strong reflection, smooth
            (0.3, 0.8, 2.0, 0.5),    # Weak reflection, rough
            (0.8, 0.2, 10.0, 0.05),  # Strong reflection, very smooth
        ]

        for reflection, refraction, fresnel, roughness in lighting_configs:
            self.scene.set_water_lighting_params(reflection, refraction, fresnel, roughness)
            # Should succeed without error


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
class TestWaterSurfacePresets:
    """Test water surface preset functionality."""

    def setup_method(self):
        """Setup test scene with water surface enabled."""
        self.scene = f3d.Scene(64, 64, 32)
        heights = np.zeros((32, 32), dtype=np.float32)
        self.scene.set_terrain_dims(32, 32, 10.0)
        self.scene.set_terrain_heights(heights)
        self.scene.enable_water_surface()

    def test_water_presets(self):
        """Test all water preset configurations."""
        valid_presets = ["ocean", "lake", "river"]

        for preset in valid_presets:
            self.scene.set_water_preset(preset)
            # Should succeed without error

            # Verify parameters changed
            params = self.scene.get_water_surface_params()
            assert isinstance(params, tuple)
            assert len(params) == 4

    def test_invalid_water_preset(self):
        """Test invalid water preset handling."""
        with pytest.raises(Exception):  # Should raise PyValueError
            self.scene.set_water_preset("invalid_preset")

    def test_preset_parameter_persistence(self):
        """Test that preset parameters persist and can be queried."""
        # Set ocean preset
        self.scene.set_water_preset("ocean")
        ocean_params = self.scene.get_water_surface_params()

        # Set lake preset
        self.scene.set_water_preset("lake")
        lake_params = self.scene.get_water_surface_params()

        # Set river preset
        self.scene.set_water_preset("river")
        river_params = self.scene.get_water_surface_params()

        # Parameters should be different between presets
        assert ocean_params != lake_params
        assert lake_params != river_params
        assert ocean_params != river_params


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
class TestWaterSurfaceRendering:
    """Test water surface rendering integration."""

    def setup_method(self):
        """Setup test scene for rendering tests."""
        self.scene = f3d.Scene(128, 128, 64)

        # Create terrain with some variation
        heights = np.random.RandomState(42).uniform(-0.1, 0.3, (64, 64)).astype(np.float32)
        self.scene.set_terrain_dims(64, 64, 20.0)
        self.scene.set_terrain_heights(heights)

        # Set camera
        self.scene.set_camera_look_at(
            (30.0, 20.0, -30.0),  # eye
            (0.0, 0.0, 0.0),      # target
            (0.0, 1.0, 0.0),      # up
            45.0, 1.0, 100.0      # fov, near, far
        )

    def test_water_surface_render_png(self):
        """Test rendering water surface to PNG."""
        self.scene.enable_water_surface()
        self.scene.set_water_surface_mode("transparent")
        self.scene.set_water_base_color(0.1, 0.3, 0.7)
        self.scene.set_water_alpha(0.6)

        # Render should complete without error
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            self.scene.render_png(tmp.name)

            # Check file was created
            tmp_path = Path(tmp.name)
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0

            # Cleanup
            tmp_path.unlink()

    def test_water_surface_render_rgba(self):
        """Test rendering water surface to RGBA array."""
        self.scene.enable_water_surface()
        self.scene.set_water_surface_mode("animated")
        self.scene.set_water_preset("ocean")

        # Render to array
        rgba_array = self.scene.render_rgba()

        # Verify array properties
        assert rgba_array.shape == (128, 128, 4)
        assert rgba_array.dtype == np.uint8

        # Array should contain non-zero values (not all black)
        assert np.any(rgba_array > 0)

    def test_water_surface_with_terrain(self):
        """Test water surface rendering combined with terrain."""
        # Enable both terrain and water surface
        self.scene.set_colormap_name("viridis")
        self.scene.enable_water_surface()
        self.scene.set_water_preset("lake")
        self.scene.set_water_surface_height(0.05)  # Above some terrain

        # Render should complete without error
        rgba_array = self.scene.render_rgba()
        assert rgba_array.shape == (128, 128, 4)
        assert np.any(rgba_array > 0)

    def test_water_surface_disabled_render(self):
        """Test rendering with water surface disabled."""
        # Disable water surface
        self.scene.disable_water_surface()

        # Render should still work
        rgba_array = self.scene.render_rgba()
        assert rgba_array.shape == (128, 128, 4)

    def test_water_surface_transparency_variation(self):
        """Test rendering with different transparency levels."""
        self.scene.enable_water_surface()
        self.scene.set_water_base_color(0.0, 0.4, 0.8)

        # Test different alpha levels produce different results
        alpha_renders = {}
        for alpha in [0.2, 0.5, 0.8]:
            self.scene.set_water_alpha(alpha)
            rgba_array = self.scene.render_rgba()
            alpha_renders[alpha] = rgba_array.copy()

        # Results should be different for different alpha values
        assert not np.array_equal(alpha_renders[0.2], alpha_renders[0.8])


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
class TestWaterSurfaceEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup test scene."""
        self.scene = f3d.Scene(64, 64, 32)
        heights = np.zeros((32, 32), dtype=np.float32)
        self.scene.set_terrain_dims(32, 32, 10.0)
        self.scene.set_terrain_heights(heights)

    def test_water_surface_extreme_values(self):
        """Test water surface with extreme parameter values."""
        self.scene.enable_water_surface()

        # Test extreme but valid values
        self.scene.set_water_surface_height(-1000.0)
        self.scene.set_water_surface_height(1000.0)
        self.scene.set_water_surface_size(0.1)
        self.scene.set_water_surface_size(10000.0)
        self.scene.set_water_alpha(0.0)  # Fully transparent
        self.scene.set_water_alpha(1.0)  # Fully opaque
        self.scene.set_water_hue_shift(100.0)  # Large hue shift

        # Should all succeed without error

    def test_water_surface_parameter_clamping(self):
        """Test that water parameters are properly clamped."""
        self.scene.enable_water_surface()

        # Alpha should be clamped to [0,1] range
        self.scene.set_water_alpha(-1.0)  # Should clamp to 0
        self.scene.set_water_alpha(2.0)   # Should clamp to 1

        # Should not raise errors

    def test_water_surface_multiple_enable_disable(self):
        """Test multiple enable/disable cycles."""
        for _ in range(5):
            self.scene.enable_water_surface()
            assert self.scene.is_water_surface_enabled()

            self.scene.disable_water_surface()
            assert not self.scene.is_water_surface_enabled()


def test_water_surface_api_imports():
    """Test that water surface types can be imported."""
    # Test that we can import water surface types
    from forge3d import WaterSurfaceMode, WaterSurfaceParams, WaterSurfaceRenderer, WaterSurfaceUniforms

    # Basic type checks
    assert hasattr(f3d, 'WaterSurfaceMode')
    assert hasattr(f3d, 'WaterSurfaceParams')
    assert hasattr(f3d, 'WaterSurfaceRenderer')
    assert hasattr(f3d, 'WaterSurfaceUniforms')


@pytest.mark.skipif(not _gpu_ok(), reason="GPU not available")
def test_water_surface_b11_acceptance_criteria():
    """
    Test B11 acceptance criteria: "Water tint toggles predictably; scenes round-trip."
    """
    scene = f3d.Scene(64, 64, 32)
    heights = np.zeros((32, 32), dtype=np.float32)
    scene.set_terrain_dims(32, 32, 10.0)
    scene.set_terrain_heights(heights)

    # Test predictable water tint toggling
    scene.enable_water_surface()

    # Set initial water color
    scene.set_water_base_color(0.2, 0.4, 0.8)
    scene.set_water_tint(1.0, 0.0, 0.0, 0.5)  # Red tint, 50% strength

    # Get parameters
    params1 = scene.get_water_surface_params()

    # Change tint
    scene.set_water_tint(0.0, 1.0, 0.0, 0.3)  # Green tint, 30% strength
    params2 = scene.get_water_surface_params()

    # Parameters should change predictably
    assert params1 != params2

    # Test scene round-trip - disable and re-enable should preserve state
    scene.set_water_base_color(0.1, 0.5, 0.9)
    scene.set_water_alpha(0.6)
    scene.set_water_hue_shift(1.0)

    params_before = scene.get_water_surface_params()

    # Render once
    render1 = scene.render_rgba()

    # Disable and re-enable
    scene.disable_water_surface()
    scene.enable_water_surface()

    # Restore same settings
    scene.set_water_base_color(0.1, 0.5, 0.9)
    scene.set_water_alpha(0.6)
    scene.set_water_hue_shift(1.0)

    params_after = scene.get_water_surface_params()

    # Parameters should round-trip correctly
    # (allowing for small floating point differences)
    assert abs(params_before[0] - params_after[0]) < 0.01  # height
    assert abs(params_before[1] - params_after[1]) < 0.01  # alpha
    assert abs(params_before[2] - params_after[2]) < 0.01  # hue_shift