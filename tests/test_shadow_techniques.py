"""P6.2 Shadow Technique Selection Tests.

Tests that shadow technique CLI validation works correctly and that
different techniques produce visually different outputs.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from forge3d.terrain_params import ShadowSettings
from forge3d.config import ShadowParams, _SHADOW_TECHNIQUES, validate_shadow_technique, load_renderer_config


class TestShadowTechniqueValidation:
    """Test shadow technique validation in terrain_params.py."""

    # Supported techniques for terrain rendering (HARD/PCF/PCSS + NONE)
    SUPPORTED_TECHNIQUES = ["hard", "pcf", "pcss", "none"]
    # Unsupported techniques that should raise clear errors
    UNSUPPORTED_TECHNIQUES = ["vsm", "evsm", "msm", "csm"]

    def _make_shadow_settings(self, technique: str) -> ShadowSettings:
        """Helper to create ShadowSettings with given technique."""
        return ShadowSettings(
            enabled=True,
            technique=technique,
            resolution=2048,
            cascades=3,
            max_distance=4000.0,
            softness=1.5,
            intensity=0.8,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        )

    @pytest.mark.parametrize("technique", SUPPORTED_TECHNIQUES)
    def test_technique_validation_accepts_supported(self, technique: str):
        """All supported techniques should validate without error."""
        settings = self._make_shadow_settings(technique)
        # Technique should be normalized to uppercase
        assert settings.technique == technique.upper()

    @pytest.mark.parametrize("technique", SUPPORTED_TECHNIQUES)
    def test_technique_validation_case_insensitive(self, technique: str):
        """Technique validation should be case-insensitive."""
        for variant in [technique.lower(), technique.upper(), technique.capitalize()]:
            settings = self._make_shadow_settings(variant)
            assert settings.technique == technique.upper()

    def test_unsupported_technique_raises_error(self):
        """Unsupported techniques should raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            self._make_shadow_settings("invalid_technique")
        
        error_msg = str(exc_info.value)
        assert "Unsupported shadow technique" in error_msg
        assert "INVALID_TECHNIQUE" in error_msg

    def test_none_technique_disables_shadows(self):
        """Technique='NONE' should disable shadows automatically."""
        settings = self._make_shadow_settings("none")
        assert settings.technique == "NONE"
        assert settings.enabled is False


class TestShadowConfigValidation:
    """Test shadow technique validation in config.py ShadowParams."""

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss"])
    def test_shadow_params_from_mapping_supported(self, technique: str):
        """ShadowParams.from_mapping should accept supported techniques."""
        params = ShadowParams.from_mapping({"technique": technique})
        assert params.technique == technique  # config.py uses lowercase

    @pytest.mark.parametrize("technique", ["vsm", "evsm", "msm"])
    def test_shadow_params_from_mapping_rejects_unsupported(self, technique: str):
        """ShadowParams.from_mapping should reject VSM/EVSM/MSM with clear error."""
        with pytest.raises(ValueError) as exc_info:
            ShadowParams.from_mapping({"technique": technique})
        
        error_msg = str(exc_info.value)
        assert "not implemented" in error_msg or "not supported" in error_msg.lower()

    def test_shadow_params_from_mapping_rejects_csm(self):
        """ShadowParams.from_mapping should reject 'csm' with explanation."""
        with pytest.raises(ValueError) as exc_info:
            ShadowParams.from_mapping({"technique": "csm"})
        
        error_msg = str(exc_info.value)
        assert "csm" in error_msg.lower()
        assert "pipeline" in error_msg.lower() or "not a valid filter" in error_msg.lower()

    def test_shadow_techniques_dict_complete(self):
        """_SHADOW_TECHNIQUES dict should include supported techniques."""
        expected = {"none", "hard", "pcf", "pcss"}
        actual = set(_SHADOW_TECHNIQUES.keys())
        assert expected == actual, f"Expected {expected}, got {actual}"

    def test_renderer_config_shadow_technique(self):
        """load_renderer_config should propagate shadow technique correctly."""
        config = load_renderer_config(None, {"shadows": "pcss"})
        assert config.shadows.technique == "pcss"


class TestValidateShadowTechnique:
    """Test the validate_shadow_technique function directly."""

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss", "none"])
    def test_accepts_supported_techniques(self, technique: str):
        """Should accept supported techniques."""
        result = validate_shadow_technique(technique)
        assert result == technique

    @pytest.mark.parametrize("technique", ["vsm", "evsm", "msm"])
    def test_rejects_moment_techniques(self, technique: str):
        """Should reject VSM/EVSM/MSM with clear error."""
        with pytest.raises(ValueError) as exc_info:
            validate_shadow_technique(technique)
        
        error_msg = str(exc_info.value)
        assert "not implemented" in error_msg
        assert "moment-based" in error_msg.lower()
        # Should list supported alternatives
        assert "hard" in error_msg or "pcf" in error_msg

    def test_rejects_csm_with_explanation(self):
        """Should reject 'csm' with explanation that it's the pipeline, not a filter."""
        with pytest.raises(ValueError) as exc_info:
            validate_shadow_technique("csm")
        
        error_msg = str(exc_info.value)
        assert "csm" in error_msg.lower()
        assert "pipeline" in error_msg.lower() or "not a valid filter" in error_msg.lower()


class TestShadowMemoryBudget:
    """Test shadow memory budget validation."""

    def test_memory_budget_normal_config(self):
        """Normal shadow config should pass memory budget check."""
        # 4096x4096 * 4 cascades * 4 bytes = 256 MiB (within 512 MiB budget)
        settings = ShadowSettings(
            enabled=True,
            technique="PCF",
            resolution=4096,
            cascades=4,
            max_distance=4000.0,
            softness=1.5,
            intensity=0.8,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        )
        assert settings.resolution == 4096
        assert settings.cascades == 4

    def test_memory_estimate_increases_with_resolution(self):
        """Higher resolution should increase memory estimate."""
        low_res = ShadowSettings(
            enabled=True, technique="PCF", resolution=1024, cascades=2,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        high_res = ShadowSettings(
            enabled=True, technique="PCF", resolution=4096, cascades=2,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        assert high_res._estimate_memory_bytes() > low_res._estimate_memory_bytes()


def _create_step_dem(width: int = 256, height: int = 256, cliff_height: float = 100.0) -> np.ndarray:
    """Create a synthetic step-DEM with a sharp cliff for shadow testing.
    
    Left half is low (0), right half is high (cliff_height).
    This creates a long shadow boundary where HARD/PCF/PCSS will differ.
    
    Args:
        width: DEM width in pixels
        height: DEM height in pixels
        cliff_height: Height of the cliff in meters
    
    Returns:
        2D numpy array with shape (height, width) containing elevation values
    """
    dem = np.zeros((height, width), dtype=np.float32)
    # Right half is elevated
    dem[:, width // 2:] = cliff_height
    return dem


def _save_geotiff(dem: np.ndarray, path: Path) -> None:
    """Save a DEM array as a GeoTIFF file.
    
    Uses rasterio to write a simple GeoTIFF with default CRS and transform.
    """
    import rasterio
    from rasterio.transform import from_bounds
    
    height, width = dem.shape
    # Create a simple transform (1 meter per pixel, origin at 0,0)
    transform = from_bounds(0, 0, width, height, width, height)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dem.dtype,
        crs='EPSG:32610',  # UTM zone 10N (arbitrary but valid)
        transform=transform,
    ) as dst:
        dst.write(dem, 1)


class TestShadowTechniqueDifferentiation:
    """Test that different shadow techniques produce different outputs.
    
    Uses a synthetic step-DEM with a sharp cliff to guarantee visible shadow edges.
    """
    
    @pytest.fixture
    def step_dem_path(self, tmp_path: Path) -> Path:
        """Create a temporary step-DEM GeoTIFF for testing."""
        dem = _create_step_dem(width=256, height=256, cliff_height=100.0)
        dem_path = tmp_path / "step_dem.tif"
        _save_geotiff(dem, dem_path)
        return dem_path
    
    def _render_with_technique(self, dem_path: Path, technique: str, output_path: Path) -> bytes:
        """Render the DEM with the specified shadow technique and return the image bytes."""
        import subprocess
        import sys
        
        # Use terrain_demo.py CLI which handles all setup correctly
        # Use mesh mode for perspective, and low sun for visible shadows
        # The 512 shadow res ensures filtering differences are visible
        cmd = [
            sys.executable, "-B", "examples/terrain_demo.py",
            "--dem", str(dem_path),
            "--size", "320", "180",
            "--shadows", technique,
            "--shadow-map-res", "512",
            "--sun-elevation", "10",
            "--sun-azimuth", "45",
            "--ibl-intensity", "0",
            "--hdr", "assets/hdri/snow_field_4k.hdr",
            "--output", str(output_path),
            "--overwrite",
            "--camera-mode", "mesh",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            raise RuntimeError(f"Render failed: {result.stderr}")
        return output_path.read_bytes()
    
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_hard_vs_pcf_differ(self, step_dem_path: Path, tmp_path: Path):
        """HARD and PCF techniques must produce different outputs."""
        hard_path = tmp_path / "hard.png"
        pcf_path = tmp_path / "pcf.png"
        
        hard_bytes = self._render_with_technique(step_dem_path, "hard", hard_path)
        pcf_bytes = self._render_with_technique(step_dem_path, "pcf", pcf_path)
        
        hard_hash = hashlib.md5(hard_bytes).hexdigest()
        pcf_hash = hashlib.md5(pcf_bytes).hexdigest()
        
        assert hard_hash != pcf_hash, f"HARD and PCF produced identical output: {hard_hash}"
    
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_pcf_vs_pcss_differ(self, step_dem_path: Path, tmp_path: Path):
        """PCF and PCSS techniques must produce different outputs."""
        pcf_path = tmp_path / "pcf.png"
        pcss_path = tmp_path / "pcss.png"
        
        pcf_bytes = self._render_with_technique(step_dem_path, "pcf", pcf_path)
        pcss_bytes = self._render_with_technique(step_dem_path, "pcss", pcss_path)
        
        pcf_hash = hashlib.md5(pcf_bytes).hexdigest()
        pcss_hash = hashlib.md5(pcss_bytes).hexdigest()
        
        assert pcf_hash != pcss_hash, f"PCF and PCSS produced identical output: {pcf_hash}"
