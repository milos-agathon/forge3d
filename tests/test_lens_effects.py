"""M5: Tests for Lens Effects post-processing.

Tests the lens effects settings configuration and validates effect properties.
"""
import pytest

from forge3d.terrain_params import (
    LensEffectsSettings,
    TerrainRenderParams,
    make_terrain_params_config,
)


class TestLensEffectsSettings:
    """Tests for LensEffectsSettings dataclass."""

    def test_lens_effects_settings_default(self):
        """LensEffectsSettings should be disabled by default."""
        settings = LensEffectsSettings()
        assert settings.enabled is False
        assert settings.distortion == 0.0
        assert settings.chromatic_aberration == 0.0
        assert settings.vignette_strength == 0.0
        assert settings.vignette_radius == 0.7
        assert settings.vignette_softness == 0.3

    def test_lens_effects_settings_enabled(self):
        """LensEffectsSettings can be enabled with custom parameters."""
        settings = LensEffectsSettings(
            enabled=True,
            distortion=0.1,
            chromatic_aberration=0.02,
            vignette_strength=0.5,
        )
        assert settings.enabled is True
        assert settings.distortion == 0.1
        assert settings.chromatic_aberration == 0.02
        assert settings.vignette_strength == 0.5

    def test_lens_effects_has_distortion_property(self):
        """has_distortion property returns correct values."""
        # No distortion
        settings = LensEffectsSettings(distortion=0.0)
        assert settings.has_distortion is False

        # Small distortion (below threshold)
        settings = LensEffectsSettings(distortion=0.0005)
        assert settings.has_distortion is False

        # Barrel distortion
        settings = LensEffectsSettings(distortion=0.1)
        assert settings.has_distortion is True

        # Pincushion distortion (negative)
        settings = LensEffectsSettings(distortion=-0.05)
        assert settings.has_distortion is True

    def test_lens_effects_has_chromatic_aberration_property(self):
        """has_chromatic_aberration property returns correct values."""
        # No CA
        settings = LensEffectsSettings(chromatic_aberration=0.0)
        assert settings.has_chromatic_aberration is False

        # Small CA (below threshold)
        settings = LensEffectsSettings(chromatic_aberration=0.0005)
        assert settings.has_chromatic_aberration is False

        # Visible CA
        settings = LensEffectsSettings(chromatic_aberration=0.02)
        assert settings.has_chromatic_aberration is True

    def test_lens_effects_has_vignette_property(self):
        """has_vignette property returns correct values."""
        # No vignette
        settings = LensEffectsSettings(vignette_strength=0.0)
        assert settings.has_vignette is False

        # Small vignette (below threshold)
        settings = LensEffectsSettings(vignette_strength=0.0005)
        assert settings.has_vignette is False

        # Visible vignette
        settings = LensEffectsSettings(vignette_strength=0.5)
        assert settings.has_vignette is True

    def test_lens_effects_has_any_effect_property(self):
        """has_any_effect property returns correct values."""
        # No effects
        settings = LensEffectsSettings()
        assert settings.has_any_effect is False

        # Only distortion
        settings = LensEffectsSettings(distortion=0.1)
        assert settings.has_any_effect is True

        # Only CA
        settings = LensEffectsSettings(chromatic_aberration=0.02)
        assert settings.has_any_effect is True

        # Only vignette
        settings = LensEffectsSettings(vignette_strength=0.5)
        assert settings.has_any_effect is True

        # All effects
        settings = LensEffectsSettings(
            distortion=0.1,
            chromatic_aberration=0.02,
            vignette_strength=0.5,
        )
        assert settings.has_any_effect is True

    def test_lens_effects_vignette_validation(self):
        """LensEffectsSettings validates vignette parameters."""
        # Valid parameters
        LensEffectsSettings(vignette_strength=0.0)
        LensEffectsSettings(vignette_strength=1.0)
        LensEffectsSettings(vignette_radius=0.0)
        LensEffectsSettings(vignette_radius=1.0)
        LensEffectsSettings(vignette_softness=0.0)
        LensEffectsSettings(vignette_softness=1.0)

        # Invalid vignette_strength
        with pytest.raises(ValueError, match="vignette_strength must be >= 0"):
            LensEffectsSettings(vignette_strength=-0.1)

        # Invalid vignette_radius
        with pytest.raises(ValueError, match="vignette_radius must be in"):
            LensEffectsSettings(vignette_radius=-0.1)
        with pytest.raises(ValueError, match="vignette_radius must be in"):
            LensEffectsSettings(vignette_radius=1.1)

        # Invalid vignette_softness
        with pytest.raises(ValueError, match="vignette_softness must be >= 0"):
            LensEffectsSettings(vignette_softness=-0.1)


class TestTerrainRenderParamsWithLensEffects:
    """Tests for TerrainRenderParams with lens effects settings."""

    def test_terrain_params_default_lens_effects(self):
        """TerrainRenderParams should have disabled lens effects by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.lens_effects is not None
        assert params.lens_effects.enabled is False

    def test_terrain_params_with_lens_effects_enabled(self):
        """TerrainRenderParams can have lens effects enabled."""
        le = LensEffectsSettings(
            enabled=True,
            distortion=0.1,
            chromatic_aberration=0.02,
            vignette_strength=0.5,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            lens_effects=le,
        )
        assert params.lens_effects is not None
        assert params.lens_effects.enabled is True
        assert params.lens_effects.distortion == 0.1


class TestLensEffectsPhysics:
    """Tests for lens effects physics and optics."""

    def test_distortion_types(self):
        """Test barrel and pincushion distortion types."""
        # Barrel distortion (positive - edges curve inward)
        barrel = LensEffectsSettings(distortion=0.15)
        assert barrel.distortion > 0
        assert barrel.has_distortion is True

        # Pincushion distortion (negative - edges curve outward)
        pincushion = LensEffectsSettings(distortion=-0.1)
        assert pincushion.distortion < 0
        assert pincushion.has_distortion is True

    def test_chromatic_aberration_range(self):
        """Test typical chromatic aberration values."""
        # Subtle CA (typical for good lenses)
        subtle = LensEffectsSettings(chromatic_aberration=0.005)
        assert subtle.has_chromatic_aberration is True

        # Strong CA (cheap or artistic)
        strong = LensEffectsSettings(chromatic_aberration=0.05)
        assert strong.has_chromatic_aberration is True

    def test_vignette_configurations(self):
        """Test typical vignette configurations."""
        # Subtle vignette
        subtle = LensEffectsSettings(
            vignette_strength=0.2,
            vignette_radius=0.8,
            vignette_softness=0.4,
        )
        assert subtle.has_vignette is True

        # Strong vignette (artistic)
        strong = LensEffectsSettings(
            vignette_strength=0.8,
            vignette_radius=0.5,
            vignette_softness=0.2,
        )
        assert strong.has_vignette is True

        # Hard edge vignette
        hard = LensEffectsSettings(
            vignette_strength=1.0,
            vignette_radius=0.6,
            vignette_softness=0.05,
        )
        assert hard.has_vignette is True
