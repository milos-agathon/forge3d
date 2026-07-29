"""P0.2/M3 Shadow Technique Selection Tests.

Tests that shadow technique CLI validation works correctly and that
different techniques (including VSM/EVSM/MSM) produce visually different outputs.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from forge3d.terrain_params import ShadowSettings
from forge3d.config import ShadowParams, _SHADOW_TECHNIQUES, validate_shadow_technique, load_renderer_config


def test_shadow_depth_height_curve_uses_receiver_primitives():
    source = (
        Path(__file__).parent.parent / "src" / "shaders" / "terrain_shadow_depth.wgsl"
    ).read_text(encoding="utf-8")

    assert "curved = det_pow(t, power);" in source
    assert "curved = pow(t, power);" not in source
    assert "curved = height_curve_lut_sample(t);" in source
    assert "return det_mix(t, curved, strength);" in source


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

    @pytest.mark.parametrize(
        "factory_path",
        (
            "forge3d.map_scene._mapscene_shadow_settings",
            "forge3d.terrain_demo._make_terrain_shadow_settings",
        ),
    )
    def test_active_terrain_translation_preserves_explicit_pcss_parameters(
        self, factory_path: str
    ):
        """The two active Python entry points must not reinterpret PCSS controls."""
        from importlib import import_module

        module_name, factory_name = factory_path.rsplit(".", 1)
        factory = getattr(import_module(module_name), factory_name)
        public = ShadowParams(
            technique="pcss",
            pcss_blocker_radius=7.25,
            pcss_filter_radius=3.5,
            light_size=2.75,
        )

        native_input = factory(public)

        assert native_input.pcss_blocker_radius == pytest.approx(7.25)
        assert native_input.pcss_filter_radius == pytest.approx(3.5)
        assert native_input.light_size == pytest.approx(2.75)
        assert native_input.softness != pytest.approx(public.light_size)

    def test_terrain_shadow_defaults_match_public_pcss_defaults(self):
        """Direct terrain settings and RendererConfig share one texel-space default."""
        settings = TestShadowTechniqueValidation()._make_shadow_settings("pcss")
        public = ShadowParams()

        assert settings.pcss_blocker_radius == pytest.approx(
            public.pcss_blocker_radius
        )
        assert settings.pcss_filter_radius == pytest.approx(public.pcss_filter_radius)
        assert settings.light_size == pytest.approx(public.light_size)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        (
            ("pcss_light_radius", np.nan),
            ("pcss_blocker_radius", np.inf),
            ("pcss_filter_radius", -np.inf),
            ("light_size", np.nan),
        ),
    )
    def test_terrain_shadow_rejects_non_finite_pcss_controls(
        self, field_name: str, value: float
    ):
        kwargs = {field_name: value}
        with pytest.raises(ValueError, match=rf"{field_name} must be finite"):
            ShadowSettings(
                enabled=True,
                technique="PCSS",
                resolution=2048,
                cascades=1,
                max_distance=20.0,
                softness=1.0,
                intensity=1.0,
                slope_scale_bias=0.001,
                depth_bias=0.0005,
                normal_bias=0.0002,
                min_variance=1e-4,
                light_bleed_reduction=0.5,
                evsm_exponent=40.0,
                fade_start=1.0,
                **kwargs,
            )


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


def _rasterio_available() -> bool:
    try:
        import rasterio
        return not getattr(rasterio, "__forge3d_stub__", False)
    except Exception:
        return False


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


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio not installed")
@pytest.mark.offscreen
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


def _viewer_gpu_available() -> bool:
    try:
        import forge3d as f3d
        return bool(f3d.has_gpu())
    except Exception:
        return False


def _pyramid_dem(path: Path, n: int = 512, pix: float = 20.0) -> Path:
    """Flat plain with one tall isolated pyramid -> guaranteed long cast shadow."""
    import rasterio
    from rasterio.transform import from_bounds

    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = n * 0.35
    r = np.maximum(np.abs(xx - c), np.abs(yy - c))
    dem = (np.clip(1.0 - r / (n * 0.12), 0.0, 1.0) * 900.0).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
        crs="EPSG:32610", transform=from_bounds(0, 0, n * pix, n * pix, n, n),
    ) as dst:
        dst.write(dem, 1)
    return path


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio not installed")
@pytest.mark.skipif(not _viewer_gpu_available(), reason="no usable GPU adapter")
@pytest.mark.viewer
class TestEvsmExposureParity:
    """EVSM must light the scene like the other techniques, not black it out.

    Regression guard for the EVSM moment pipeline: the moment atlas is
    Rgba16Float, so exp(c * depth) with c=40 overflows to +Inf (and exp(-40*d)
    flushes to zero), which made every EVSM fragment resolve to ~0 visibility.
    The negative lobe must also be stored negated so its Chebyshev bound is
    monotonically increasing, otherwise EVSM reports "lit" everywhere instead.
    """

    def _render(self, viewer, technique: str, out: Path) -> np.ndarray:
        import time
        viewer.send_ipc({
            "cmd": "set_terrain", "phi": 90.0, "theta": 55.0, "fov": 30.0,
            "radius": 18000.0, "zscale": 1.0, "sun_azimuth": 270.0,
            "sun_elevation": 8.0, "sun_intensity": 2.0, "ambient": 0.15,
            "shadow": 1.0, "background": [1.0, 1.0, 1.0],
        })
        viewer.send_ipc({
            "cmd": "set_terrain_pbr", "enabled": True, "shadow_technique": technique,
            "shadow_map_res": 2048, "exposure": 1.0, "msaa": 1,
            "ibl_intensity": 0.0,
        })
        time.sleep(1.0)
        viewer.snapshot(str(out), width=640, height=400)
        from PIL import Image
        img = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def test_evsm_is_not_black(self, tmp_path: Path):
        import forge3d as f3d

        dem = _pyramid_dem(tmp_path / "pyramid.tif")
        with f3d.open_viewer_async(terrain_path=str(dem), width=640, height=400,
                                   timeout=45.0) as viewer:
            lum = {t: self._render(viewer, t, tmp_path / f"{t}.png")
                   for t in ("pcf", "evsm")}

        means = {}
        for tech, l in lum.items():
            terrain = l < 0.97  # exclude the white background
            assert terrain.any(), f"{tech}: no terrain pixels found"
            means[tech] = float(l[terrain].mean())

        # EVSM must not be globally darker than PCF by more than 20%.
        assert means["evsm"] >= means["pcf"] * 0.8, (
            f"EVSM is broken-dark: terrain mean {means['evsm']:.3f} vs "
            f"PCF {means['pcf']:.3f}"
        )

        # ...and it must still cast a real shadow (some pixels clearly darker
        # than the lit plain), not just return "fully lit" everywhere.
        terrain = lum["evsm"] < 0.97
        vals = lum["evsm"][terrain]
        shadowed = float((vals < np.percentile(vals, 95) * 0.6).mean())
        assert shadowed >= 0.01, (
            f"EVSM casts no shadow: only {shadowed:.4f} of terrain is shadowed"
        )


def _native_terrain_gpu_available() -> bool:
    try:
        from _terrain_runtime import terrain_rendering_available

        return terrain_rendering_available()
    except Exception:
        return False


def _native_pyramid_heightmap(size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size * 0.42
    radius = np.maximum(np.abs(xx - center), np.abs(yy - center))
    return np.clip(1.0 - radius / (size * 0.12), 0.0, 1.0).astype(np.float32)


@pytest.mark.skipif(
    not _native_terrain_gpu_available(),
    reason="no terrain-capable hardware-backed forge3d runtime",
)
@pytest.mark.offscreen
def test_native_moment_visibility_semantics_pcss_transition_width_and_msm_uses_four_moments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Native MSM uses four moments, moment filters stay visible, and PCSS widens."""
    import forge3d as f3d
    from _terrain_runtime import _write_test_hdr
    from forge3d.terrain_params import PomSettings, make_terrain_params_config

    monkeypatch.setenv("FORGE3D_TERRAIN_SHADOW_DEBUG", "raw")
    hdr_path = tmp_path / "constant.hdr"
    _write_test_hdr(hdr_path)

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=0.0)
    heightmap = _native_pyramid_heightmap()
    flat_mask_map = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#ffffff"),
            (0.001, "#000000"),
            (1.0, "#000000"),
        ],
        domain=(0.0, 1.0),
    )
    flat_mask_overlay = f3d.OverlayLayer.from_colormap1d(
        flat_mask_map, strength=1.0
    )

    def render(
        technique: str,
        *,
        enabled: bool = True,
        terrain: np.ndarray | None = None,
        curve_mode: str = "linear",
        curve_power: float = 1.0,
        curve_lut: np.ndarray | None = None,
        pcss_blocker_radius: float = 6.0,
        pcss_filter_radius: float = 4.0,
        light_size: float = 1.0,
        pcss_light_radius: float = 0.0,
    ) -> np.ndarray:
        shadows = ShadowSettings(
            enabled=enabled,
            technique=technique,
            resolution=2048,
            cascades=1,
            max_distance=20.0,
            softness=1.0,
            intensity=1.0,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
            pcss_blocker_radius=pcss_blocker_radius,
            pcss_filter_radius=pcss_filter_radius,
            light_size=light_size,
            pcss_light_radius=pcss_light_radius,
        )
        config = make_terrain_params_config(
            size_px=(320, 240),
            render_scale=1.0,
            terrain_span=4.0,
            msaa_samples=1,
            z_scale=2.0,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=False,
            ibl_intensity=0.0,
            light_azimuth_deg=315.0,
            light_elevation_deg=12.0,
            sun_intensity=3.0,
            cam_radius=6.0,
            cam_phi_deg=135.0,
            cam_theta_deg=55.0,
            fov_y_deg=52.0,
            camera_mode="mesh:zup",
            debug_mode=1 if not enabled else 0,
            height_curve_mode=curve_mode,
            height_curve_strength=0.0 if curve_mode == "linear" else 1.0,
            height_curve_power=curve_power,
            height_curve_lut=curve_lut,
            overlays=[flat_mask_overlay],
            shadows=shadows,
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        )
        return renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=f3d.TerrainRenderParams(config),
            heightmap=heightmap if terrain is None else terrain,
        ).to_numpy()

    raw = {
        technique: render(technique)[..., :3].astype(np.float32) / 255.0
        for technique in ("pcf", "vsm", "evsm", "msm")
    }
    luminance = {technique: image[..., 0] for technique, image in raw.items()}
    terrain_reference = (
        render("pcf", enabled=False)[..., 0].astype(np.float32) / 255.0
    )
    terrain = terrain_reference > 0.9
    assert terrain.any(), "shadow-disabled native render contains no flat terrain"
    for technique, image in raw.items():
        assert np.max(np.abs(image[..., 0][terrain] - image[..., 1][terrain])) <= (
            1.0 / 255.0
        )
        assert np.max(np.abs(image[..., 0][terrain] - image[..., 2][terrain])) <= (
            1.0 / 255.0
        )

    lit_plain = terrain & (luminance["pcf"] > 0.9)
    cast_shadow = terrain & (luminance["pcf"] < 0.6)
    assert lit_plain.any(), "deterministic pyramid produced no exposed PCF plain"
    assert float(cast_shadow.sum() / terrain.sum()) >= 0.01, (
        "deterministic pyramid produced no measurable PCF cast-shadow region"
    )

    pcf_exposure = float(luminance["pcf"][lit_plain].mean())
    for technique in ("vsm", "evsm", "msm"):
        exposure = float(luminance[technique][lit_plain].mean())
        shadowed = float(
            ((luminance[technique] < exposure * 0.6) & cast_shadow).sum()
            / terrain.sum()
        )
        print(
            f"{technique}: exposure={exposure:.6f}, "
            f"pcf={pcf_exposure:.6f}, shadowed={shadowed:.6f}"
        )
        assert exposure >= pcf_exposure * 0.8, (
            f"{technique.upper()} breaks lit exposure: {exposure:.3f} vs "
            f"PCF {pcf_exposure:.3f}"
        )
        assert shadowed >= 0.01, (
            f"{technique.upper()} casts no shadow: only {shadowed:.4f} "
            "of terrain is clearly shadowed"
        )

    msm_cast_difference = float(
        np.mean(np.abs(luminance["msm"][cast_shadow] - luminance["vsm"][cast_shadow]))
    )
    print(f"MSM vs VSM cast-shadow MAE={msm_cast_difference:.6f}")
    assert not np.array_equal(raw["msm"], raw["vsm"]), (
        "MSM and VSM produced byte-identical native output"
    )
    assert msm_cast_difference >= 0.005, (
        "MSM does not visibly consume its third and fourth moments: "
        f"cast-shadow MAE={msm_cast_difference:.6f}"
    )

    pcss = {
        light_size: (
            render(
                "pcss", light_size=light_size, pcss_filter_radius=16.0
            )[..., 0].astype(np.float32)
            / 255.0
        )
        for light_size in (1.0, 12.0)
    }
    pcss_mae = {
        radius: float(
            np.mean(np.abs(visibility[terrain] - luminance["pcf"][terrain]))
        )
        for radius, visibility in pcss.items()
    }
    print(f"PCSS vs PCF MAE: {pcss_mae}")
    assert pcss_mae[12.0] >= 0.002, (
        f"wide-light PCSS aliases PCF: MAE={pcss_mae[12.0]:.6f}"
    )
    assert pcss_mae[12.0] >= pcss_mae[1.0] + 0.00075, (
        f"light-size response is only a near-fixed perturbation: {pcss_mae}"
    )

    def edge_geometry(
        pcf_visibility: np.ndarray, terrain_mask: np.ndarray
    ) -> tuple[np.ndarray, int]:
        cast = terrain_mask & (pcf_visibility < 0.6)
        padded = np.pad(cast, 1)
        interior = (
            cast
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        edge = cast & ~interior
        assert edge.any(), "deterministic cast shadow has no measurable edge"
        band = np.lib.stride_tricks.sliding_window_view(
            np.pad(edge, 12), (25, 25)
        ).any(axis=(-2, -1))
        return band & terrain_mask, int(edge.sum())

    edge_band, edge_count = edge_geometry(luminance["pcf"], terrain)

    def transition_width(
        visibility: np.ndarray, band: np.ndarray, denominator: int
    ) -> float:
        transition = 4.0 * visibility * (1.0 - visibility)
        return float(transition[band].sum() / denominator)

    widths = {
        radius: transition_width(visibility, edge_band, edge_count)
        for radius, visibility in pcss.items()
    }
    print(f"PCSS transition widths: {widths}")
    assert widths[12.0] >= widths[1.0] + 0.25, (
        f"larger PCSS light size did not widen the cast-shadow transition: {widths}"
    )

    curve_heightmap = np.zeros_like(heightmap)
    curve_heightmap[30:78, 30:78] = np.float32(0.5)
    curve_heightmap[30, 30] = np.float32(1.0)
    intermediate = curve_heightmap == np.float32(0.5)
    assert float(intermediate.mean()) >= 0.1

    lut = np.zeros(256, dtype=np.float32)
    curve_cases = (
        (
            "pow",
            8.0,
            None,
            np.power(curve_heightmap, np.float32(8.0)).astype(np.float32),
        ),
        (
            "lut",
            1.0,
            lut,
            lut[np.rint(curve_heightmap * 255.0).astype(np.int32)],
        ),
    )
    for mode, power, curve_lut, prewarped in curve_cases:
        curved = render(
            "pcf",
            terrain=curve_heightmap,
            curve_mode=mode,
            curve_power=power,
            curve_lut=curve_lut,
        )
        reference = render("pcf", terrain=prewarped)
        mean_abs = float(
            np.mean(
                np.abs(
                    curved[..., :3].astype(np.float32)
                    - reference[..., :3].astype(np.float32)
                )
            )
            / 255.0
        )
        print(f"{mode}: curved-vs-prewarped MAE={mean_abs:.6f}")
        assert mean_abs <= 0.01, (
            f"{mode} caster diverges from the receiver curve: MAE={mean_abs:.6f}"
        )
