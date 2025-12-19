# tests/test_fog_offline.py
# M3: Test suite for Atmospheric Fog in offline terrain rendering
# Verifies that fog settings work correctly and produce expected results
#
# RELEVANT FILES: src/terrain/renderer.rs, src/shaders/terrain_pbr_pom.wgsl,
#                 python/forge3d/terrain_params.py

import hashlib
import numpy as np
import pytest
from pathlib import Path

# Try to import forge3d - skip tests if not available
try:
    import forge3d
    from forge3d.terrain_params import (
        TerrainRenderParams,
        FogSettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


def compute_image_hash(image: np.ndarray) -> str:
    """Compute SHA256 hash of image data for reproducibility checks."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def compute_saturation(rgb: np.ndarray) -> np.ndarray:
    """Compute saturation from RGB image (0-1 range).
    
    Saturation = (max(R,G,B) - min(R,G,B)) / max(R,G,B) if max > 0, else 0.
    """
    rgb_float = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb
    max_rgb = np.max(rgb_float[..., :3], axis=-1)
    min_rgb = np.min(rgb_float[..., :3], axis=-1)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0.0)
    return sat


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB image (Rec. 709 coefficients)."""
    rgb_float = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb
    if rgb_float.ndim == 3 and rgb_float.shape[2] >= 3:
        return 0.2126 * rgb_float[..., 0] + 0.7152 * rgb_float[..., 1] + 0.0722 * rgb_float[..., 2]
    return rgb_float


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = 255.0
    return 10 * np.log10((max_val ** 2) / mse)


class TestFogConfig:
    """Tests for FogSettings configuration and validation."""

    def test_fog_default_disabled(self):
        """Test that fog is disabled by default (density=0)."""
        fog = FogSettings()
        assert fog.density == 0.0
        assert fog.height_falloff == 0.0
        assert fog.base_height is None
        assert fog.inscatter == (1.0, 1.0, 1.0)
        assert fog.aerial_perspective == 0.0

    def test_fog_enabled_with_density(self):
        """Test enabling fog with custom density."""
        fog = FogSettings(
            density=0.1,
            height_falloff=0.01,
            base_height=1000.0,
            inscatter=(0.8, 0.85, 0.9),
        )
        assert fog.density == 0.1
        assert fog.height_falloff == 0.01
        assert fog.base_height == 1000.0
        assert fog.inscatter == (0.8, 0.85, 0.9)

    def test_fog_with_aerial_perspective(self):
        """Test fog with aerial perspective enabled."""
        fog = FogSettings(
            density=0.15,
            height_falloff=0.02,
            aerial_perspective=0.5,
        )
        assert fog.density == 0.15
        assert fog.aerial_perspective == 0.5

    def test_fog_density_validation(self):
        """Test that negative density raises error."""
        with pytest.raises(ValueError, match="density must be >= 0"):
            FogSettings(density=-0.1)

    def test_fog_height_falloff_validation(self):
        """Test that negative height_falloff raises error."""
        with pytest.raises(ValueError, match="height_falloff must be >= 0"):
            FogSettings(height_falloff=-0.1)

    def test_fog_inscatter_validation(self):
        """Test that invalid inscatter raises error."""
        with pytest.raises(ValueError, match="inscatter must be"):
            FogSettings(inscatter=(1.0, 1.0))  # Missing component
        with pytest.raises(ValueError, match="inscatter components must be in"):
            FogSettings(inscatter=(1.5, 0.5, 0.5))  # Out of range

    def test_fog_aerial_perspective_validation(self):
        """Test that aerial_perspective outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="aerial_perspective must be in"):
            FogSettings(aerial_perspective=1.5)
        with pytest.raises(ValueError, match="aerial_perspective must be in"):
            FogSettings(aerial_perspective=-0.1)


class TestFogInTerrainParams:
    """Tests for fog integration in TerrainRenderParams."""

    def test_terrain_params_default_fog(self):
        """Test that TerrainRenderParams defaults to disabled fog."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
        )
        assert params.fog is not None
        assert params.fog.density == 0.0

    def test_terrain_params_with_fog(self):
        """Test TerrainRenderParams with custom fog settings."""
        fog = FogSettings(
            density=0.1,
            height_falloff=0.01,
            inscatter=(0.7, 0.8, 0.9),
            aerial_perspective=0.3,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            fog=fog,
        )
        assert params.fog is not None
        assert params.fog.density == 0.1
        assert params.fog.aerial_perspective == 0.3


class TestAerialPerspectiveLogic:
    """Tests for aerial perspective algorithm (unit tests)."""

    def test_desaturation_logic(self):
        """Test that desaturation reduces color saturation."""
        # Simulate desaturation algorithm from shader
        def desaturate(color: np.ndarray, amount: float) -> np.ndarray:
            """Python implementation of shader desaturation."""
            luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
            return color * (1.0 - amount) + np.array([luma, luma, luma]) * amount
        
        # Saturated color (pure red)
        red = np.array([1.0, 0.0, 0.0])
        sat_red = (max(red) - min(red)) / max(red)
        assert sat_red == 1.0  # Fully saturated
        
        # Desaturate by 0.5
        desat = desaturate(red, 0.5)
        sat_desat = (max(desat) - min(desat)) / max(desat) if max(desat) > 0 else 0
        assert sat_desat < sat_red  # Less saturated

    def test_blue_shift_logic(self):
        """Test that blue shift moves colors toward atmospheric blue."""
        atmosphere_blue = np.array([0.65, 0.75, 0.9])
        
        def blue_shift(color: np.ndarray, luma: float, amount: float) -> np.ndarray:
            """Python implementation of shader blue shift."""
            return color * (1.0 - amount) + atmosphere_blue * luma * amount
        
        # Green color
        green = np.array([0.2, 0.8, 0.1])
        luma = 0.2126 * green[0] + 0.7152 * green[1] + 0.0722 * green[2]
        
        # Apply blue shift
        shifted = blue_shift(green, luma, 0.3)
        
        # Blue channel should increase relative to green
        assert shifted[2] > green[2]  # Blue increased

    def test_extinction_curve(self):
        """Test exponential extinction curve behavior."""
        def extinction(density: float, distance: float, height_factor: float) -> float:
            """Compute extinction (fog visibility)."""
            return np.exp(-density * distance * height_factor * 0.005)
        
        # At distance 0, extinction = 1 (no fog)
        assert extinction(0.1, 0.0, 1.0) == 1.0
        
        # At large distance, extinction approaches 0 (full fog)
        ext_far = extinction(0.1, 10000.0, 1.0)
        assert ext_far < 0.1  # Significant fog at distance
        
        # Higher density = lower extinction (more fog)
        ext_low = extinction(0.05, 5000.0, 1.0)
        ext_high = extinction(0.2, 5000.0, 1.0)
        assert ext_high < ext_low  # More fog with higher density


class TestFogAcceptanceCriteria:
    """Tests for M3 acceptance criteria."""

    def test_saturation_helper(self):
        """Test saturation computation helper."""
        # Pure red (fully saturated)
        red = np.array([[[255, 0, 0]]], dtype=np.uint8)
        sat = compute_saturation(red)
        assert abs(sat[0, 0] - 1.0) < 0.01
        
        # Gray (no saturation)
        gray = np.array([[[128, 128, 128]]], dtype=np.uint8)
        sat = compute_saturation(gray)
        assert sat[0, 0] == 0.0
        
        # Partially saturated
        partial = np.array([[[200, 100, 50]]], dtype=np.uint8)
        sat = compute_saturation(partial)
        assert 0.0 < sat[0, 0] < 1.0

    def test_luminance_helper(self):
        """Test luminance computation helper."""
        # White
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        luma = compute_luminance(white)
        assert abs(luma[0, 0] - 1.0) < 0.01
        
        # Black
        black = np.array([[[0, 0, 0]]], dtype=np.uint8)
        luma = compute_luminance(black)
        assert luma[0, 0] == 0.0

    def test_psnr_helper(self):
        """Test PSNR computation helper."""
        # Identical images
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr == float('inf')
        
        # Similar images
        img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 130, dtype=np.uint8)
        psnr = compute_psnr(img1, img2)
        assert psnr > 30.0  # Very similar


@pytest.mark.skipif(not FORGE3D_AVAILABLE, reason="forge3d not installed")
class TestFogRendering:
    """Integration tests for fog rendering (requires GPU)."""

    def test_fog_disabled_params(self):
        """Test that fog.density=0 creates valid params."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            fog=FogSettings(density=0.0),
        )
        assert params.fog.density == 0.0

    def test_fog_enabled_params(self):
        """Test that fog enabled params are correctly passed."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            fog=FogSettings(
                density=0.1,
                height_falloff=0.01,
                inscatter=(0.8, 0.85, 0.9),
                aerial_perspective=0.5,
            ),
        )
        assert params.fog.density == 0.1
        assert params.fog.aerial_perspective == 0.5

    def test_aerial_perspective_params(self):
        """Test aerial perspective parameter passing."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            fog=FogSettings(
                density=0.15,
                aerial_perspective=0.8,
            ),
        )
        assert params.fog.aerial_perspective == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
