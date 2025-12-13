"""P6.2 Shadow Technique Selection Tests.

Tests that shadow technique CLI validation works correctly and that
different techniques can be selected without errors.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from forge3d.terrain_params import ShadowSettings
from forge3d.config import ShadowParams, _SHADOW_TECHNIQUES, load_renderer_config


class TestShadowTechniqueValidation:
    """Test shadow technique validation in terrain_params.py."""

    # All techniques that should be supported
    SUPPORTED_TECHNIQUES = ["hard", "pcf", "pcss", "vsm", "evsm", "msm", "csm", "none"]

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
        # Should list supported techniques
        assert "CSM" in error_msg or "PCF" in error_msg

    def test_none_technique_disables_shadows(self):
        """Technique='NONE' should disable shadows automatically."""
        settings = self._make_shadow_settings("none")
        assert settings.technique == "NONE"
        assert settings.enabled is False


class TestShadowConfigValidation:
    """Test shadow technique validation in config.py ShadowParams."""

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss", "vsm", "evsm", "msm", "csm"])
    def test_shadow_params_from_mapping(self, technique: str):
        """ShadowParams.from_mapping should accept all supported techniques."""
        params = ShadowParams.from_mapping({"technique": technique})
        assert params.technique == technique  # config.py uses lowercase

    def test_shadow_techniques_dict_complete(self):
        """_SHADOW_TECHNIQUES dict should include all expected techniques."""
        expected = {"none", "hard", "pcf", "pcss", "vsm", "evsm", "msm", "csm"}
        actual = set(_SHADOW_TECHNIQUES.keys())
        assert expected.issubset(actual), f"Missing techniques: {expected - actual}"

    def test_renderer_config_shadow_technique(self):
        """load_renderer_config should propagate shadow technique correctly."""
        config = load_renderer_config(None, {"shadows": "pcss"})
        assert config.shadows.technique == "pcss"


class TestTerrainShadowTechniqueValidation:
    """Test that VSM/EVSM/MSM are correctly rejected for terrain rendering."""

    UNSUPPORTED_TERRAIN_TECHNIQUES = ["vsm", "evsm", "msm"]
    SUPPORTED_TERRAIN_TECHNIQUES = ["none", "hard", "pcf", "pcss", "csm"]

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

    @pytest.mark.parametrize("technique", UNSUPPORTED_TERRAIN_TECHNIQUES)
    def test_unsupported_techniques_raise_for_terrain(self, technique: str):
        """VSM/EVSM/MSM should raise ValueError when validated for terrain."""
        settings = self._make_shadow_settings(technique)
        # Settings creation succeeds (config layer accepts them)
        assert settings.technique == technique.upper()
        
        # But terrain validation should fail with clear error
        with pytest.raises(ValueError) as exc_info:
            settings.validate_for_terrain()
        
        error_msg = str(exc_info.value)
        assert "not implemented for terrain" in error_msg
        assert technique.upper() in error_msg
        # Should list supported alternatives
        assert "CSM" in error_msg or "PCSS" in error_msg

    @pytest.mark.parametrize("technique", SUPPORTED_TERRAIN_TECHNIQUES)
    def test_supported_techniques_pass_terrain_validation(self, technique: str):
        """HARD/PCF/PCSS/CSM/NONE should pass terrain validation."""
        settings = self._make_shadow_settings(technique)
        # Should not raise
        settings.validate_for_terrain()
        assert settings.technique == technique.upper()

    def test_terrain_supported_techniques_constant(self):
        """TERRAIN_SUPPORTED_TECHNIQUES should match expected set."""
        expected = {"NONE", "HARD", "PCF", "PCSS", "CSM"}
        assert ShadowSettings.TERRAIN_SUPPORTED_TECHNIQUES == expected

    def test_all_techniques_superset_of_terrain(self):
        """ALL_TECHNIQUES should be superset of TERRAIN_SUPPORTED_TECHNIQUES."""
        assert ShadowSettings.TERRAIN_SUPPORTED_TECHNIQUES.issubset(
            ShadowSettings.ALL_TECHNIQUES
        )


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

    def test_memory_budget_vsm_within_budget(self):
        """VSM at reasonable resolution should pass budget check."""
        # VSM uses extra memory for moment maps (2 channels)
        settings = ShadowSettings(
            enabled=True,
            technique="VSM",
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
        mem = settings._estimate_memory_bytes()
        assert mem < ShadowSettings.MAX_SHADOW_MEMORY_BYTES

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

    def test_memory_estimate_moment_techniques(self):
        """VSM/EVSM/MSM should use more memory than PCF due to moment maps."""
        pcf = ShadowSettings(
            enabled=True, technique="PCF", resolution=2048, cascades=3,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        vsm = ShadowSettings(
            enabled=True, technique="VSM", resolution=2048, cascades=3,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        evsm = ShadowSettings(
            enabled=True, technique="EVSM", resolution=2048, cascades=3,
            max_distance=4000.0, softness=1.5, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        # VSM uses 2-channel moments, EVSM uses 4-channel
        assert vsm._estimate_memory_bytes() > pcf._estimate_memory_bytes()
        assert evsm._estimate_memory_bytes() > vsm._estimate_memory_bytes()


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
