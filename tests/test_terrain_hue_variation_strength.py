from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE
from forge3d.terrain_params import make_terrain_params_config


ROOT = Path(__file__).resolve().parents[1]


def _make_params(*, hue_variation_strength: float = 0.08, material_slope_bias: float = 1.0):
    return make_terrain_params_config(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=1000.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1000.0),
        hue_variation_strength=hue_variation_strength,
        material_slope_bias=material_slope_bias,
    )


def test_terrain_material_variation_defaults_preserve_historical_values() -> None:
    params = _make_params()
    assert params.hue_variation_strength == pytest.approx(0.08)
    assert params.material_slope_bias == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.125, 0.125), (0.2, 0.2), (1.0, 0.2)],
)
def test_hue_variation_strength_is_clamped(requested: float, expected: float) -> None:
    assert _make_params(hue_variation_strength=requested).hue_variation_strength == pytest.approx(
        expected
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_hue_variation_strength_rejects_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="hue_variation_strength must be finite"):
        _make_params(hue_variation_strength=invalid)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
)
def test_material_slope_bias_is_clamped(requested: float, expected: float) -> None:
    assert _make_params(material_slope_bias=requested).material_slope_bias == pytest.approx(expected)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_material_slope_bias_rejects_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="material_slope_bias must be finite"):
        _make_params(material_slope_bias=invalid)


@pytest.mark.parametrize(
    ("settings", "expected_hue", "expected_slope_bias"),
    [
        ({}, 0.08, 1.0),
        ({"hue_variation_strength": 0.0, "material_slope_bias": 0.0}, 0.0, 0.0),
    ],
)
def test_mapscene_lighting_settings_reach_terrain_material_variation_params(
    monkeypatch,
    settings: dict[str, float],
    expected_hue: float,
    expected_slope_bias: float,
) -> None:
    import forge3d.map_scene as map_scene

    class FakeColormap:
        @staticmethod
        def from_stops(**_kwargs):
            return object()

    class FakeOverlay:
        @staticmethod
        def from_colormap1d(*_args, **_kwargs):
            return object()

    class FakeParams:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(f3d, "Colormap1D", FakeColormap)
    monkeypatch.setattr(f3d, "OverlayLayer", FakeOverlay)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)

    heightmap = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(data=heightmap, crs="EPSG:32610"),
        camera=f3d.OrbitCamera(distance=100.0),
        lighting=f3d.LightingPreset(settings=settings),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    captured = map_scene._build_mapscene_terrain_params(scene.recipe, heightmap, (64, 64))

    assert captured is not None
    assert captured.config.hue_variation_strength == pytest.approx(expected_hue)
    assert captured.config.material_slope_bias == pytest.approx(expected_slope_bias)


def test_terrain_material_variation_is_uploaded_and_consumed_as_uniforms() -> None:
    upload = (ROOT / "src/terrain/renderer/upload.rs").read_text(encoding="utf-8")
    shader = (ROOT / "src/shaders/terrain_pbr_pom.wgsl").read_text(encoding="utf-8")

    assert "params.hue_variation_strength.clamp(0.0, 0.2)" in upload
    assert "params.material_slope_bias.clamp(0.0, 1.0)" in upload
    assert "clamp(u_overlay.params3.z, 0.0, 0.2)" in shader
    assert "clamp(u_overlay.params3.w, 0.0, 1.0)" in shader
    assert "let hue_variation_strength = 0.08" not in shader


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Requires compiled _forge3d extension")
def test_native_terrain_render_params_exposes_material_variation_controls() -> None:
    native = f3d.TerrainRenderParams(
        _make_params(hue_variation_strength=0.0, material_slope_bias=0.0)
    )
    assert native.hue_variation_strength == pytest.approx(0.0)
    assert native.material_slope_bias == pytest.approx(0.0)
