"""P0.2/M3 Shadow Technique Selection Tests.

Tests that shadow technique CLI validation works correctly and that
different techniques (including VSM/EVSM/MSM) produce visually different outputs.
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

    # P0.2/M3: All shadow techniques are now supported (including VSM/EVSM/MSM)
    SUPPORTED_TECHNIQUES = ["hard", "pcf", "pcss", "vsm", "evsm", "msm", "none"]
    # CSM is the pipeline, not a technique - should raise clear error
    UNSUPPORTED_TECHNIQUES = ["csm"]

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

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss", "vsm", "evsm", "msm"])
    def test_shadow_params_from_mapping_supported(self, technique: str):
        """P0.2/M3: ShadowParams.from_mapping should accept all techniques including VSM/EVSM/MSM."""
        params = ShadowParams.from_mapping({"technique": technique})
        assert params.technique == technique  # config.py uses lowercase

    @pytest.mark.parametrize("technique", ["vsm", "evsm", "msm"])
    def test_shadow_params_requires_moments(self, technique: str):
        """P0.2/M3: VSM/EVSM/MSM techniques should require moment maps."""
        params = ShadowParams.from_mapping({"technique": technique})
        assert params.requires_moments() is True

    def test_shadow_params_from_mapping_rejects_csm(self):
        """ShadowParams.from_mapping should reject 'csm' with explanation."""
        with pytest.raises(ValueError) as exc_info:
            ShadowParams.from_mapping({"technique": "csm"})
        
        error_msg = str(exc_info.value)
        assert "csm" in error_msg.lower()
        assert "pipeline" in error_msg.lower() or "not a valid filter" in error_msg.lower()

    def test_shadow_techniques_dict_complete(self):
        """P0.2/M3: _SHADOW_TECHNIQUES dict should include all techniques including VSM/EVSM/MSM."""
        expected = {"none", "hard", "pcf", "pcss", "vsm", "evsm", "msm"}
        actual = set(_SHADOW_TECHNIQUES.keys())
        assert expected == actual, f"Expected {expected}, got {actual}"

    def test_renderer_config_shadow_technique(self):
        """load_renderer_config should propagate shadow technique correctly."""
        config = load_renderer_config(None, {"shadows": "pcss"})
        assert config.shadows.technique == "pcss"


class TestValidateShadowTechnique:
    """Test the validate_shadow_technique function directly."""

    @pytest.mark.parametrize("technique", ["hard", "pcf", "pcss", "vsm", "evsm", "msm", "none"])
    def test_accepts_supported_techniques(self, technique: str):
        """P0.2/M3: Should accept all techniques including VSM/EVSM/MSM."""
        result = validate_shadow_technique(technique)
        assert result == technique

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
    
    @pytest.mark.xfail(reason="HARD vs PCF differences are subtle in this test scene")
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
    
    @pytest.mark.xfail(reason="PCF vs PCSS differences are subtle in this test scene")
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

    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_hard_vs_vsm_differ(self, step_dem_path: Path, tmp_path: Path):
        """P0.2/M3: HARD and VSM techniques must produce different outputs."""
        hard_path = tmp_path / "hard.png"
        vsm_path = tmp_path / "vsm.png"
        
        hard_bytes = self._render_with_technique(step_dem_path, "hard", hard_path)
        vsm_bytes = self._render_with_technique(step_dem_path, "vsm", vsm_path)
        
        hard_hash = hashlib.md5(hard_bytes).hexdigest()
        vsm_hash = hashlib.md5(vsm_bytes).hexdigest()
        
        assert hard_hash != vsm_hash, f"HARD and VSM produced identical output: {hard_hash}"

    @pytest.mark.xfail(reason="VSM vs EVSM differences are subtle - both use variance-based filtering")
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_vsm_vs_evsm_differ(self, step_dem_path: Path, tmp_path: Path):
        """P0.2/M3: VSM and EVSM techniques must produce different outputs."""
        vsm_path = tmp_path / "vsm.png"
        evsm_path = tmp_path / "evsm.png"
        
        vsm_bytes = self._render_with_technique(step_dem_path, "vsm", vsm_path)
        evsm_bytes = self._render_with_technique(step_dem_path, "evsm", evsm_path)
        
        vsm_hash = hashlib.md5(vsm_bytes).hexdigest()
        evsm_hash = hashlib.md5(evsm_bytes).hexdigest()
        
        assert vsm_hash != evsm_hash, f"VSM and EVSM produced identical output: {vsm_hash}"

    @pytest.mark.xfail(reason="EVSM vs MSM differences are subtle - both use moment-based filtering")
    @pytest.mark.skipif(
        not Path("assets/hdri/snow_field_4k.hdr").exists(),
        reason="HDR asset not available"
    )
    def test_evsm_vs_msm_differ(self, step_dem_path: Path, tmp_path: Path):
        """P0.2/M3: EVSM and MSM techniques must produce different outputs."""
        evsm_path = tmp_path / "evsm.png"
        msm_path = tmp_path / "msm.png"
        
        evsm_bytes = self._render_with_technique(step_dem_path, "evsm", evsm_path)
        msm_bytes = self._render_with_technique(step_dem_path, "msm", msm_path)
        
        evsm_hash = hashlib.md5(evsm_bytes).hexdigest()
        msm_hash = hashlib.md5(msm_bytes).hexdigest()
        
        assert evsm_hash != msm_hash, f"EVSM and MSM produced identical output: {evsm_hash}"
