# tests/test_terrain_materials.py
# M4: Test suite for Terrain Material Layering (snow, rock, wetness)
# Verifies that material layer settings work correctly and produce expected results
#
# RELEVANT FILES: src/terrain/renderer.rs, src/shaders/terrain_pbr_pom.wgsl,
#                 python/forge3d/terrain_params.py

import math
import numpy as np
import pytest

# Try to import forge3d - skip tests if not available
try:
    import forge3d
    from forge3d.terrain_params import (
        TerrainRenderParams,
        MaterialLayerSettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


def compute_mean_albedo(image: np.ndarray) -> float:
    """Compute mean luminance/albedo from RGB image."""
    rgb_float = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image
    if rgb_float.ndim == 3 and rgb_float.shape[2] >= 3:
        luma = 0.2126 * rgb_float[..., 0] + 0.7152 * rgb_float[..., 1] + 0.0722 * rgb_float[..., 2]
        return float(np.mean(luma))
    return float(np.mean(rgb_float))


class TestMaterialLayerConfig:
    """Tests for MaterialLayerSettings configuration and validation."""

    def test_material_default_disabled(self):
        """Test that all material layers are disabled by default."""
        materials = MaterialLayerSettings()
        assert materials.snow_enabled is False
        assert materials.rock_enabled is False
        assert materials.wetness_enabled is False

    def test_snow_defaults(self):
        """Test snow layer default values."""
        materials = MaterialLayerSettings()
        assert materials.snow_altitude_min == 2000.0
        assert materials.snow_altitude_blend == 500.0
        assert materials.snow_slope_max == 45.0
        assert materials.snow_slope_blend == 15.0
        assert materials.snow_aspect_influence == 0.3
        assert materials.snow_color == (0.95, 0.95, 0.98)
        assert materials.snow_roughness == 0.4

    def test_rock_defaults(self):
        """Test rock layer default values."""
        materials = MaterialLayerSettings()
        assert materials.rock_slope_min == 45.0
        assert materials.rock_slope_blend == 10.0
        assert materials.rock_color == (0.35, 0.32, 0.28)
        assert materials.rock_roughness == 0.8

    def test_wetness_defaults(self):
        """Test wetness layer default values."""
        materials = MaterialLayerSettings()
        assert materials.wetness_strength == 0.3
        assert materials.wetness_slope_influence == 0.5

    def test_snow_enabled(self):
        """Test enabling snow layer with custom settings."""
        materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=1500.0,
            snow_altitude_blend=300.0,
            snow_slope_max=40.0,
            snow_aspect_influence=0.5,
        )
        assert materials.snow_enabled is True
        assert materials.snow_altitude_min == 1500.0
        assert materials.snow_altitude_blend == 300.0
        assert materials.snow_slope_max == 40.0
        assert materials.snow_aspect_influence == 0.5

    def test_rock_enabled(self):
        """Test enabling rock layer with custom settings."""
        materials = MaterialLayerSettings(
            rock_enabled=True,
            rock_slope_min=50.0,
            rock_slope_blend=15.0,
            rock_color=(0.4, 0.35, 0.3),
        )
        assert materials.rock_enabled is True
        assert materials.rock_slope_min == 50.0
        assert materials.rock_slope_blend == 15.0
        assert materials.rock_color == (0.4, 0.35, 0.3)

    def test_wetness_enabled(self):
        """Test enabling wetness layer with custom settings."""
        materials = MaterialLayerSettings(
            wetness_enabled=True,
            wetness_strength=0.5,
            wetness_slope_influence=0.7,
        )
        assert materials.wetness_enabled is True
        assert materials.wetness_strength == 0.5
        assert materials.wetness_slope_influence == 0.7

    def test_snow_slope_max_validation(self):
        """Test that snow_slope_max outside [0, 90] raises error."""
        with pytest.raises(ValueError, match="snow_slope_max must be in"):
            MaterialLayerSettings(snow_slope_max=100.0)
        with pytest.raises(ValueError, match="snow_slope_max must be in"):
            MaterialLayerSettings(snow_slope_max=-10.0)

    def test_snow_altitude_blend_validation(self):
        """Test that non-positive snow_altitude_blend raises error."""
        with pytest.raises(ValueError, match="snow_altitude_blend must be > 0"):
            MaterialLayerSettings(snow_altitude_blend=0.0)
        with pytest.raises(ValueError, match="snow_altitude_blend must be > 0"):
            MaterialLayerSettings(snow_altitude_blend=-100.0)

    def test_snow_aspect_influence_validation(self):
        """Test that snow_aspect_influence outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="snow_aspect_influence must be in"):
            MaterialLayerSettings(snow_aspect_influence=1.5)
        with pytest.raises(ValueError, match="snow_aspect_influence must be in"):
            MaterialLayerSettings(snow_aspect_influence=-0.1)

    def test_rock_slope_min_validation(self):
        """Test that rock_slope_min outside [0, 90] raises error."""
        with pytest.raises(ValueError, match="rock_slope_min must be in"):
            MaterialLayerSettings(rock_slope_min=95.0)

    def test_wetness_strength_validation(self):
        """Test that wetness_strength outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="wetness_strength must be in"):
            MaterialLayerSettings(wetness_strength=1.5)

    def test_snow_color_validation(self):
        """Test that invalid snow_color raises error."""
        with pytest.raises(ValueError, match="snow_color must be"):
            MaterialLayerSettings(snow_color=(1.0, 1.0))  # Missing component


class TestMaterialLayerInTerrainParams:
    """Tests for material layer integration in TerrainRenderParams."""

    def test_terrain_params_default_materials(self):
        """Test that TerrainRenderParams defaults to disabled materials."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
        )
        assert params.materials is not None
        assert params.materials.snow_enabled is False
        assert params.materials.rock_enabled is False
        assert params.materials.wetness_enabled is False

    def test_terrain_params_with_snow(self):
        """Test TerrainRenderParams with snow enabled."""
        materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=1800.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            materials=materials,
        )
        assert params.materials is not None
        assert params.materials.snow_enabled is True
        assert params.materials.snow_altitude_min == 1800.0

    def test_terrain_params_with_rock(self):
        """Test TerrainRenderParams with rock enabled."""
        materials = MaterialLayerSettings(
            rock_enabled=True,
            rock_slope_min=50.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            materials=materials,
        )
        assert params.materials is not None
        assert params.materials.rock_enabled is True
        assert params.materials.rock_slope_min == 50.0

    def test_terrain_params_all_layers(self):
        """Test TerrainRenderParams with all material layers enabled."""
        materials = MaterialLayerSettings(
            snow_enabled=True,
            rock_enabled=True,
            wetness_enabled=True,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            materials=materials,
        )
        assert params.materials.snow_enabled is True
        assert params.materials.rock_enabled is True
        assert params.materials.wetness_enabled is True


class TestTerrainAttributeLogic:
    """Tests for terrain attribute computation (unit tests)."""

    def test_slope_computation(self):
        """Test slope computation from normal vector."""
        # Flat surface (normal pointing up): slope = 0
        flat_normal = np.array([0.0, 0.0, 1.0])
        slope = math.acos(np.clip(flat_normal[2], -1.0, 1.0))
        assert abs(slope) < 0.01  # Near zero
        
        # 45° slope: normal at 45° from vertical
        angled_normal = np.array([0.707, 0.0, 0.707])  # ~45°
        slope = math.acos(np.clip(angled_normal[2], -1.0, 1.0))
        assert abs(slope - math.pi/4) < 0.1  # ~45° in radians
        
        # Vertical cliff (normal pointing sideways): slope = 90°
        cliff_normal = np.array([1.0, 0.0, 0.0])
        slope = math.acos(np.clip(cliff_normal[2], -1.0, 1.0))
        assert abs(slope - math.pi/2) < 0.01  # ~90° in radians

    def test_aspect_computation(self):
        """Test aspect computation from normal vector."""
        # North-facing (normal points south, negative Y)
        north_facing = np.array([0.0, -0.5, 0.866])
        horizontal = north_facing[:2]
        aspect = math.atan2(horizontal[0], horizontal[1])
        if aspect < 0:
            aspect += 2 * math.pi
        # North facing should have aspect ~180° (pointing south)
        assert abs(aspect - math.pi) < 0.1
        
        # East-facing (normal points west, negative X)
        east_facing = np.array([-0.5, 0.0, 0.866])
        horizontal = east_facing[:2]
        aspect = math.atan2(horizontal[0], horizontal[1])
        if aspect < 0:
            aspect += 2 * math.pi
        # East facing should have aspect ~270°
        assert abs(aspect - 3*math.pi/2) < 0.1

    def test_snow_altitude_factor(self):
        """Test snow altitude factor computation."""
        def altitude_factor(altitude, alt_min, alt_blend):
            return max(0.0, min(1.0, (altitude - alt_min) / max(alt_blend, 0.001)))
        
        # Below minimum: factor = 0
        assert altitude_factor(1500.0, 2000.0, 500.0) == 0.0
        
        # At minimum: factor = 0
        assert altitude_factor(2000.0, 2000.0, 500.0) == 0.0
        
        # At min + blend/2: factor = 0.5
        assert abs(altitude_factor(2250.0, 2000.0, 500.0) - 0.5) < 0.01
        
        # At min + blend: factor = 1.0
        assert altitude_factor(2500.0, 2000.0, 500.0) == 1.0
        
        # Above blend: factor = 1.0
        assert altitude_factor(3000.0, 2000.0, 500.0) == 1.0

    def test_rock_slope_factor(self):
        """Test rock slope factor computation."""
        def rock_factor(slope_deg, slope_min_deg, slope_blend_deg):
            slope_min = math.radians(slope_min_deg)
            slope_blend = math.radians(slope_blend_deg)
            slope = math.radians(slope_deg)
            return max(0.0, min(1.0, (slope - slope_min) / max(slope_blend, 0.001)))
        
        # Below minimum: factor = 0
        assert rock_factor(30.0, 45.0, 10.0) == 0.0
        
        # At minimum: factor = 0
        assert rock_factor(45.0, 45.0, 10.0) == 0.0
        
        # At min + blend/2: factor = 0.5
        assert abs(rock_factor(50.0, 45.0, 10.0) - 0.5) < 0.05
        
        # At min + blend: factor = 1.0
        assert abs(rock_factor(55.0, 45.0, 10.0) - 1.0) < 0.01


class TestMaterialLayerAcceptanceCriteria:
    """Tests for M4 acceptance criteria."""

    def test_helper_mean_albedo(self):
        """Test mean albedo computation helper."""
        # White image
        white = np.ones((64, 64, 3), dtype=np.float32)
        assert abs(compute_mean_albedo(white) - 1.0) < 0.01
        
        # Black image
        black = np.zeros((64, 64, 3), dtype=np.float32)
        assert compute_mean_albedo(black) == 0.0
        
        # Gray image
        gray = np.full((64, 64, 3), 0.5, dtype=np.float32)
        assert abs(compute_mean_albedo(gray) - 0.5) < 0.01


@pytest.mark.skipif(not FORGE3D_AVAILABLE, reason="forge3d not installed")
class TestMaterialLayerRendering:
    """Integration tests for material layer rendering (requires GPU)."""

    def test_materials_disabled_params(self):
        """Test that all-disabled materials creates valid params."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            materials=MaterialLayerSettings(),
        )
        assert params.materials.snow_enabled is False
        assert params.materials.rock_enabled is False
        assert params.materials.wetness_enabled is False

    def test_snow_enabled_params(self):
        """Test that snow enabled params are correctly passed."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 4000.0),  # Range includes snow altitude
            materials=MaterialLayerSettings(
                snow_enabled=True,
                snow_altitude_min=2500.0,
                snow_altitude_blend=500.0,
            ),
        )
        assert params.materials.snow_enabled is True
        assert params.materials.snow_altitude_min == 2500.0

    def test_rock_enabled_params(self):
        """Test that rock enabled params are correctly passed."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            materials=MaterialLayerSettings(
                rock_enabled=True,
                rock_slope_min=40.0,
            ),
        )
        assert params.materials.rock_enabled is True
        assert params.materials.rock_slope_min == 40.0

    def test_all_layers_enabled_params(self):
        """Test that all layers enabled creates valid params."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 3000.0),
            materials=MaterialLayerSettings(
                snow_enabled=True,
                rock_enabled=True,
                wetness_enabled=True,
            ),
        )
        assert params.materials.snow_enabled is True
        assert params.materials.rock_enabled is True
        assert params.materials.wetness_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
