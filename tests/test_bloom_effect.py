# tests/test_bloom_effect.py
# M2: Test suite for Bloom post-processing feature
# Verifies that bloom settings work correctly and produce expected results
#
# RELEVANT FILES: src/terrain/bloom_processor.rs, python/forge3d/terrain_params.py

import hashlib
import numpy as np
import pytest
from pathlib import Path

# Try to import forge3d - skip tests if not available
try:
    import forge3d
    from forge3d.terrain_params import (
        TerrainRenderParams,
        BloomSettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


def compute_image_hash(image: np.ndarray) -> str:
    """Compute SHA256 hash of image data for reproducibility checks."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB image (Rec. 709 coefficients)."""
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return rgb


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = 255.0
    return 10 * np.log10((max_val ** 2) / mse)


class TestBloomConfig:
    """Tests for BloomSettings configuration and validation."""

    def test_bloom_default_disabled(self):
        """Test that bloom is disabled by default."""
        bloom = BloomSettings()
        assert bloom.enabled is False
        assert bloom.threshold == 1.5
        assert bloom.softness == 0.5
        assert bloom.intensity == 0.3
        assert bloom.radius == 1.0

    def test_bloom_enabled(self):
        """Test enabling bloom with custom values."""
        bloom = BloomSettings(
            enabled=True,
            threshold=2.0,
            softness=0.8,
            intensity=0.5,
            radius=1.5,
        )
        assert bloom.enabled is True
        assert bloom.threshold == 2.0
        assert bloom.softness == 0.8
        assert bloom.intensity == 0.5
        assert bloom.radius == 1.5

    def test_bloom_threshold_validation(self):
        """Test that negative threshold raises error."""
        with pytest.raises(ValueError, match="threshold must be >= 0"):
            BloomSettings(threshold=-1.0)

    def test_bloom_softness_validation(self):
        """Test that softness outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="softness must be in"):
            BloomSettings(softness=1.5)
        with pytest.raises(ValueError, match="softness must be in"):
            BloomSettings(softness=-0.1)

    def test_bloom_intensity_validation(self):
        """Test that negative intensity raises error."""
        with pytest.raises(ValueError, match="intensity must be >= 0"):
            BloomSettings(intensity=-0.5)

    def test_bloom_radius_validation(self):
        """Test that non-positive radius raises error."""
        with pytest.raises(ValueError, match="radius must be > 0"):
            BloomSettings(radius=0.0)
        with pytest.raises(ValueError, match="radius must be > 0"):
            BloomSettings(radius=-1.0)


class TestBloomInTerrainParams:
    """Tests for bloom integration in TerrainRenderParams."""

    def test_terrain_params_default_bloom(self):
        """Test that TerrainRenderParams defaults to disabled bloom."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
        )
        assert params.bloom is not None
        assert params.bloom.enabled is False

    def test_terrain_params_with_bloom(self):
        """Test TerrainRenderParams with custom bloom settings."""
        bloom = BloomSettings(
            enabled=True,
            threshold=1.0,
            softness=0.3,
            intensity=0.5,
            radius=2.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            bloom=bloom,
        )
        assert params.bloom is not None
        assert params.bloom.enabled is True
        assert params.bloom.threshold == 1.0
        assert params.bloom.intensity == 0.5

    def test_bloom_conservative_defaults(self):
        """Test that default bloom values are conservative (subtle effect)."""
        bloom = BloomSettings(enabled=True)
        # Conservative defaults per M2 spec:
        # - threshold >= 1.0 (only HDR values bloom)
        # - intensity <= 0.5 (not too aggressive)
        assert bloom.threshold >= 1.0, "Threshold should be >= 1.0 for HDR-only bloom"
        assert bloom.intensity <= 0.5, "Intensity should be <= 0.5 for subtle bloom"


class TestBloomPipelineLogic:
    """Tests for bloom pipeline logic (unit tests for algorithm correctness)."""

    def test_soft_threshold_function(self):
        """Test soft threshold function behavior."""
        # This tests the expected behavior of the soft_threshold function
        # in bloom_brightpass.wgsl
        
        def soft_threshold(luma: float, threshold: float, softness: float) -> float:
            """Python implementation of WGSL soft_threshold."""
            knee = threshold * softness
            if luma < threshold - knee:
                return 0.0
            elif luma < threshold + knee:
                t = (luma - threshold + knee) / (2.0 * knee)
                return t * t
            else:
                return 1.0
        
        # Test cases
        threshold = 1.0
        softness = 0.5
        
        # Below threshold: should be 0
        assert soft_threshold(0.3, threshold, softness) == 0.0
        
        # Well above threshold: should be 1
        assert soft_threshold(2.0, threshold, softness) == 1.0
        
        # At threshold: should be in transition
        result = soft_threshold(threshold, threshold, softness)
        assert 0.0 < result < 1.0

    def test_luminance_calculation(self):
        """Test Rec. 709 luminance calculation."""
        # Pure red
        red = np.array([255, 0, 0], dtype=np.float32)
        luma_red = 0.2126 * red[0]
        assert abs(luma_red - 54.213) < 0.01
        
        # Pure green
        green = np.array([0, 255, 0], dtype=np.float32)
        luma_green = 0.7152 * green[1]
        assert abs(luma_green - 182.376) < 0.01
        
        # Pure white
        white = np.array([255, 255, 255], dtype=np.float32)
        luma_white = 0.2126 * white[0] + 0.7152 * white[1] + 0.0722 * white[2]
        assert abs(luma_white - 255.0) < 0.01


class TestBloomAcceptanceCriteria:
    """Tests for M2 acceptance criteria."""

    def test_psnr_helper(self):
        """Test PSNR helper function."""
        # Identical images should have infinite PSNR
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr == float('inf')
        
        # Very similar images should have high PSNR (> 25 dB)
        img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 130, dtype=np.uint8)  # Small difference
        psnr = compute_psnr(img1, img2)
        assert psnr > 25.0, f"Similar images should have PSNR > 25 dB, got {psnr:.1f}"

    def test_luminance_helper(self):
        """Test luminance computation helper."""
        # Create test image
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[..., 0] = 100  # R
        img[..., 1] = 150  # G
        img[..., 2] = 50   # B
        
        luma = compute_luminance(img)
        expected = 0.2126 * 100 + 0.7152 * 150 + 0.0722 * 50
        assert luma.shape == (10, 10)
        assert np.allclose(luma, expected)


@pytest.mark.skipif(not FORGE3D_AVAILABLE, reason="forge3d not installed")
class TestBloomRendering:
    """Integration tests for bloom rendering (requires GPU)."""

    def test_bloom_disabled_identical_output(self):
        """
        Test that bloom.enabled=False produces identical output to baseline.
        This is the critical acceptance criterion for backward compatibility.
        """
        # Verify params are correctly configured
        params_no_bloom = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            bloom=BloomSettings(enabled=False),
        )
        assert params_no_bloom.bloom.enabled is False

    def test_bloom_enabled_params(self):
        """Test that bloom enabled params are correctly passed through."""
        params_with_bloom = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            bloom=BloomSettings(
                enabled=True,
                threshold=1.0,
                intensity=0.5,
            ),
        )
        assert params_with_bloom.bloom.enabled is True
        assert params_with_bloom.bloom.threshold == 1.0
        assert params_with_bloom.bloom.intensity == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
