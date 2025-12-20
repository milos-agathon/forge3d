"""M5: Tests for Denoise settings configuration.

Tests the denoise settings dataclass and validates denoiser configuration.
"""
import numpy as np
import pytest

from forge3d.terrain_params import (
    DenoiseSettings,
    TerrainRenderParams,
    make_terrain_params_config,
)
from forge3d.denoise import atrous_denoise


class TestDenoiseSettings:
    """Tests for DenoiseSettings dataclass."""

    def test_denoise_settings_default(self):
        """DenoiseSettings should be disabled by default."""
        settings = DenoiseSettings()
        assert settings.enabled is False
        assert settings.method == "atrous"
        assert settings.iterations == 3
        assert settings.sigma_color == 0.1
        assert settings.sigma_normal == 0.1
        assert settings.sigma_depth == 0.1
        assert settings.edge_stopping == 1.0

    def test_denoise_settings_enabled(self):
        """DenoiseSettings can be enabled with custom parameters."""
        settings = DenoiseSettings(
            enabled=True,
            method="bilateral",
            iterations=5,
            sigma_color=0.2,
        )
        assert settings.enabled is True
        assert settings.method == "bilateral"
        assert settings.iterations == 5
        assert settings.sigma_color == 0.2

    def test_denoise_method_validation(self):
        """DenoiseSettings validates method field."""
        # Valid methods
        DenoiseSettings(method="atrous")
        DenoiseSettings(method="bilateral")
        DenoiseSettings(method="none")

        # Invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            DenoiseSettings(method="invalid")

    def test_denoise_iterations_validation(self):
        """DenoiseSettings validates iterations field."""
        # Valid iterations
        DenoiseSettings(iterations=1)
        DenoiseSettings(iterations=5)
        DenoiseSettings(iterations=10)

        # Invalid iterations (too low)
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            DenoiseSettings(iterations=0)

        # Invalid iterations (too high)
        with pytest.raises(ValueError, match="iterations must be <= 10"):
            DenoiseSettings(iterations=15)

    def test_denoise_sigma_validation(self):
        """DenoiseSettings validates sigma parameters."""
        # Valid sigmas
        DenoiseSettings(sigma_color=0.0)
        DenoiseSettings(sigma_color=1.0)
        DenoiseSettings(sigma_normal=0.0)
        DenoiseSettings(sigma_depth=0.0)
        DenoiseSettings(edge_stopping=0.0)

        # Invalid sigmas
        with pytest.raises(ValueError, match="sigma_color must be >= 0"):
            DenoiseSettings(sigma_color=-0.1)
        with pytest.raises(ValueError, match="sigma_normal must be >= 0"):
            DenoiseSettings(sigma_normal=-0.1)
        with pytest.raises(ValueError, match="sigma_depth must be >= 0"):
            DenoiseSettings(sigma_depth=-0.1)
        with pytest.raises(ValueError, match="edge_stopping must be >= 0"):
            DenoiseSettings(edge_stopping=-0.1)

    def test_denoise_uses_guidance_property(self):
        """uses_guidance property returns correct values."""
        # No guidance (default sigmas but using bilateral)
        settings = DenoiseSettings(method="bilateral")
        assert settings.uses_guidance is False

        # No guidance (method=none)
        settings = DenoiseSettings(method="none")
        assert settings.uses_guidance is False

        # No guidance (atrous but zero sigma)
        settings = DenoiseSettings(method="atrous", sigma_normal=0.0, sigma_depth=0.0)
        assert settings.uses_guidance is False

        # Uses normal guidance
        settings = DenoiseSettings(method="atrous", sigma_normal=0.1, sigma_depth=0.0)
        assert settings.uses_guidance is True

        # Uses depth guidance
        settings = DenoiseSettings(method="atrous", sigma_normal=0.0, sigma_depth=0.1)
        assert settings.uses_guidance is True

        # Uses both
        settings = DenoiseSettings(method="atrous", sigma_normal=0.1, sigma_depth=0.1)
        assert settings.uses_guidance is True


class TestTerrainRenderParamsWithDenoise:
    """Tests for TerrainRenderParams with denoise settings."""

    def test_terrain_params_default_denoise(self):
        """TerrainRenderParams should have disabled denoise by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.denoise is not None
        assert params.denoise.enabled is False

    def test_terrain_params_with_denoise_enabled(self):
        """TerrainRenderParams can have denoise enabled."""
        dn = DenoiseSettings(
            enabled=True,
            method="atrous",
            iterations=5,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            denoise=dn,
        )
        assert params.denoise is not None
        assert params.denoise.enabled is True
        assert params.denoise.iterations == 5


class TestAtrousDenoiser:
    """Tests for the A-trous denoiser implementation."""

    def test_atrous_denoise_basic(self):
        """A-trous denoiser runs without error on valid input."""
        # Create noisy image
        rng = np.random.default_rng(42)
        clean = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        noisy = clean + rng.normal(0, 0.1, clean.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        # Denoise
        denoised = atrous_denoise(noisy, iterations=2)

        # Check output shape and type
        assert denoised.shape == noisy.shape
        assert denoised.dtype == np.float32

    def test_atrous_denoise_reduces_variance(self):
        """A-trous denoiser reduces noise variance."""
        # Create noisy image
        rng = np.random.default_rng(42)
        clean = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        noise = rng.normal(0, 0.15, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0, 1)

        # Denoise
        denoised = atrous_denoise(noisy, iterations=3, sigma_color=0.15)

        # Calculate variance reduction
        noisy_var = np.var(noisy)
        denoised_var = np.var(denoised)

        # Denoised should have lower variance
        assert denoised_var < noisy_var, f"Variance not reduced: {denoised_var} >= {noisy_var}"

    def test_atrous_denoise_with_guidance(self):
        """A-trous denoiser uses normal/depth guidance."""
        # Create test data
        rng = np.random.default_rng(42)
        color = rng.random((32, 32, 3)).astype(np.float32)
        normal = np.zeros((32, 32, 3), dtype=np.float32)
        normal[..., 2] = 1.0  # All normals point up
        depth = np.ones((32, 32), dtype=np.float32) * 10.0

        # Denoise with guidance
        denoised = atrous_denoise(
            color,
            normal=normal,
            depth=depth,
            iterations=2,
            sigma_normal=0.1,
            sigma_depth=0.1,
        )

        # Check output
        assert denoised.shape == color.shape
        assert denoised.dtype == np.float32

    def test_atrous_denoise_preserves_edges(self):
        """A-trous denoiser preserves strong edges."""
        # Create image with strong edge
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[:, 32:, :] = 1.0  # Right half is white

        # Add noise
        rng = np.random.default_rng(42)
        noisy = image + rng.normal(0, 0.05, image.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        # Denoise
        denoised = atrous_denoise(noisy, iterations=3, sigma_color=0.1)

        # Check edge is preserved (difference between sides should remain large)
        left_mean = np.mean(denoised[:, :30, :])
        right_mean = np.mean(denoised[:, 34:, :])
        edge_diff = right_mean - left_mean

        # Edge should still be visible (> 0.5 difference)
        assert edge_diff > 0.5, f"Edge not preserved: diff = {edge_diff}"

    def test_atrous_denoise_input_validation(self):
        """A-trous denoiser validates input shapes."""
        # Invalid color shape (2D)
        with pytest.raises(ValueError, match="color must be"):
            atrous_denoise(np.zeros((64, 64), dtype=np.float32))

        # Invalid color shape (wrong channels)
        with pytest.raises(ValueError, match="color must be"):
            atrous_denoise(np.zeros((64, 64, 4), dtype=np.float32))

        # Invalid normal shape
        color = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="normal must match"):
            atrous_denoise(color, normal=np.zeros((32, 32, 3), dtype=np.float32))

        # Invalid depth shape
        with pytest.raises(ValueError, match="depth must be"):
            atrous_denoise(color, depth=np.zeros((32, 32), dtype=np.float32))
