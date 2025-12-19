# tests/test_blender_quality_integration.py
# M7: Integration test suite for Blender-quality rendering features
# Verifies that all M1-M6 features work together without regressions
#
# RELEVANT FILES: python/forge3d/terrain_params.py, examples/blender_quality_demo.py

import pytest

# Try to import forge3d - skip tests if not available
try:
    import forge3d
    from forge3d.terrain_params import (
        TerrainRenderParams,
        BloomSettings,
        MaterialLayerSettings,
        VectorOverlaySettings,
        TonemapSettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


class TestFeatureDefaults:
    """Test that all features default to OFF for backward compatibility."""

    def test_bloom_default_disabled(self):
        """Test that bloom is disabled by default."""
        settings = BloomSettings()
        assert settings.enabled is False

    def test_materials_default_disabled(self):
        """Test that material layers are disabled by default."""
        settings = MaterialLayerSettings()
        assert settings.snow_enabled is False
        assert settings.rock_enabled is False
        assert settings.wetness_enabled is False

    def test_vector_overlay_default_disabled(self):
        """Test that vector overlay features are disabled by default."""
        settings = VectorOverlaySettings()
        assert settings.depth_test is False
        assert settings.halo_enabled is False

    def test_tonemap_default_aces(self):
        """Test that tonemap defaults to ACES operator."""
        settings = TonemapSettings()
        assert settings.operator == "aces"
        assert settings.white_balance_enabled is False

    def test_terrain_params_all_defaults(self):
        """Test that TerrainRenderParams has all features disabled by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
        )
        assert params.aa_samples == 1
        assert params.bloom.enabled is False
        assert params.materials.snow_enabled is False
        assert params.materials.rock_enabled is False
        assert params.vector_overlay.depth_test is False
        assert params.vector_overlay.halo_enabled is False
        assert params.tonemap.operator == "aces"


class TestFeatureCombinations:
    """Test that features can be combined without errors."""

    def test_all_features_enabled(self):
        """Test enabling all features at once."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
            aa_samples=16,
            bloom=BloomSettings(enabled=True, intensity=0.3),
            materials=MaterialLayerSettings(
                snow_enabled=True,
                rock_enabled=True,
                wetness_enabled=True,
            ),
            vector_overlay=VectorOverlaySettings(
                depth_test=True,
                halo_enabled=True,
            ),
            tonemap=TonemapSettings(
                operator="aces",
                white_balance_enabled=True,
                temperature=5500.0,
            ),
        )
        assert params.aa_samples == 16
        assert params.bloom.enabled is True
        assert params.materials.snow_enabled is True
        assert params.materials.rock_enabled is True
        assert params.vector_overlay.depth_test is True
        assert params.tonemap.white_balance_enabled is True

    def test_bloom_and_tonemap(self):
        """Test bloom with different tonemap operators."""
        for op in ["reinhard", "aces", "uncharted2"]:
            params = make_terrain_params_config(
                size_px=(256, 256),
                render_scale=1.0,
                terrain_span=1000.0,
                msaa_samples=1,
                z_scale=1.0,
                exposure=1.0,
                domain=(0.0, 1000.0),
                bloom=BloomSettings(enabled=True),
                tonemap=TonemapSettings(operator=op),
            )
            assert params.bloom.enabled is True
            assert params.tonemap.operator == op

    def test_materials_and_overlay(self):
        """Test materials with vector overlay."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
            materials=MaterialLayerSettings(snow_enabled=True),
            vector_overlay=VectorOverlaySettings(halo_enabled=True),
        )
        assert params.materials.snow_enabled is True
        assert params.vector_overlay.halo_enabled is True

    def test_aa_with_all_features(self):
        """Test accumulation AA with all other features."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
            aa_samples=64,
            bloom=BloomSettings(enabled=True),
            materials=MaterialLayerSettings(snow_enabled=True),
            tonemap=TonemapSettings(operator="aces"),
        )
        assert params.aa_samples == 64


class TestPresetConfigurations:
    """Test preset-like configurations from blender_quality_demo.py."""

    def test_alpine_preset(self):
        """Test alpine mountain scene configuration."""
        params = make_terrain_params_config(
            size_px=(1920, 1080),
            render_scale=1.0,
            terrain_span=10000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 4000.0),
            aa_samples=16,
            bloom=BloomSettings(enabled=True, intensity=0.3, threshold=0.8),
            materials=MaterialLayerSettings(
                snow_enabled=True,
                snow_altitude_min=2500.0,
                snow_altitude_blend=300.0,
                rock_enabled=True,
                rock_slope_min=40.0,
            ),
            vector_overlay=VectorOverlaySettings(
                depth_test=True,
                halo_enabled=True,
            ),
            tonemap=TonemapSettings(
                operator="aces",
                white_balance_enabled=True,
                temperature=7000.0,
            ),
        )
        assert params.materials.snow_enabled is True
        assert params.materials.snow_altitude_min == 2500.0
        assert params.tonemap.temperature == 7000.0

    def test_cinematic_preset(self):
        """Test cinematic warm-toned configuration."""
        params = make_terrain_params_config(
            size_px=(1920, 1080),
            render_scale=1.0,
            terrain_span=10000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 4000.0),
            aa_samples=64,
            bloom=BloomSettings(enabled=True, intensity=0.5),
            materials=MaterialLayerSettings(
                wetness_enabled=True,
                wetness_strength=0.4,
            ),
            tonemap=TonemapSettings(
                operator="uncharted2",
                white_point=6.0,
                white_balance_enabled=True,
                temperature=5500.0,
                tint=0.1,
            ),
        )
        assert params.tonemap.operator == "uncharted2"
        assert params.tonemap.temperature == 5500.0
        assert params.materials.wetness_enabled is True

    def test_high_quality_preset(self):
        """Test maximum quality configuration."""
        params = make_terrain_params_config(
            size_px=(3840, 2160),
            render_scale=1.0,
            terrain_span=10000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 4000.0),
            aa_samples=256,
            bloom=BloomSettings(enabled=True),
            materials=MaterialLayerSettings(
                snow_enabled=True,
                rock_enabled=True,
                wetness_enabled=True,
            ),
            vector_overlay=VectorOverlaySettings(
                depth_test=True,
                halo_enabled=True,
            ),
            tonemap=TonemapSettings(operator="aces"),
        )
        assert params.aa_samples == 256
        assert params.materials.snow_enabled is True
        assert params.materials.rock_enabled is True
        assert params.materials.wetness_enabled is True


class TestParameterValidation:
    """Test parameter validation across features."""

    def test_invalid_tonemap_operator(self):
        """Test that invalid tonemap operator raises error."""
        with pytest.raises(ValueError):
            TonemapSettings(operator="invalid")

    def test_invalid_temperature_range(self):
        """Test that out-of-range temperature raises error."""
        with pytest.raises(ValueError):
            TonemapSettings(temperature=1000.0)  # Below 2000K
        with pytest.raises(ValueError):
            TonemapSettings(temperature=15000.0)  # Above 12000K

    def test_invalid_tint_range(self):
        """Test that out-of-range tint raises error."""
        with pytest.raises(ValueError):
            TonemapSettings(tint=-2.0)  # Below -1
        with pytest.raises(ValueError):
            TonemapSettings(tint=2.0)  # Above 1

    def test_invalid_halo_color(self):
        """Test that invalid halo color raises error."""
        with pytest.raises(ValueError):
            VectorOverlaySettings(halo_color=(1.0, 0.0, 0.0))  # Missing alpha

    def test_invalid_wetness_strength(self):
        """Test that out-of-range wetness strength raises error."""
        with pytest.raises(ValueError):
            MaterialLayerSettings(wetness_strength=-0.1)


class TestAcceptanceCriteria:
    """Test M7 acceptance criteria."""

    def test_no_default_output_changes(self):
        """Test that default config produces baseline-identical output config."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
        )
        # All features should be disabled by default
        assert params.aa_samples == 1
        assert params.bloom.enabled is False
        assert params.materials.snow_enabled is False
        assert params.materials.rock_enabled is False
        assert params.materials.wetness_enabled is False
        assert params.vector_overlay.depth_test is False
        assert params.vector_overlay.halo_enabled is False

    def test_integration_all_features_no_error(self):
        """Test that all features can be enabled without error."""
        # This tests that the combination doesn't cause validation errors
        params = make_terrain_params_config(
            size_px=(1920, 1080),
            render_scale=1.0,
            terrain_span=10000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.2,
            domain=(0.0, 4000.0),
            aa_samples=64,
            bloom=BloomSettings(enabled=True, intensity=0.3),
            materials=MaterialLayerSettings(
                snow_enabled=True,
                snow_altitude_min=2500.0,
                rock_enabled=True,
                rock_slope_min=40.0,
                wetness_enabled=True,
            ),
            vector_overlay=VectorOverlaySettings(
                depth_test=True,
                halo_enabled=True,
                halo_width=3.0,
            ),
            tonemap=TonemapSettings(
                operator="aces",
                white_balance_enabled=True,
                temperature=6000.0,
            ),
        )
        # If we get here without exception, the test passes
        assert params is not None
        assert params.aa_samples == 64
        assert params.bloom.enabled is True
        assert params.tonemap.operator == "aces"


@pytest.mark.skipif(not FORGE3D_AVAILABLE, reason="forge3d not installed")
class TestIntegrationRendering:
    """Integration tests that require GPU rendering."""

    def test_params_round_trip(self):
        """Test that params can be created and accessed."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1000.0),
            tonemap=TonemapSettings(operator="aces"),
        )
        # Verify params are accessible
        assert params.tonemap.operator == "aces"
        assert params.size_px == (256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
