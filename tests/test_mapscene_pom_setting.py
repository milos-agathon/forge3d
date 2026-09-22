# tests/test_mapscene_pom_setting.py
# MapScene exposes parallax occlusion mapping through lighting.settings["pom"].
# POM ray-marches height in discrete steps; on a displaced mesh (camera_mode
# "mesh") the stepping shows as contour-like bands, so callers must be able to
# turn it off. Absent keeps the renderer default so existing goldens are unchanged.
# RELEVANT FILES: python/forge3d/map_scene.py, python/forge3d/terrain_params.py

from __future__ import annotations

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain_params import make_terrain_params_config


def _captured_pom(monkeypatch, settings):
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
    return captured.config.pom


def test_absent_pom_setting_keeps_renderer_default(monkeypatch) -> None:
    default = make_terrain_params_config(size_px=(64, 64), render_scale=1.0, terrain_span=8.0,
                                         msaa_samples=1, z_scale=1.0, exposure=1.0,
                                         domain=(0.0, 1.0)).pom
    assert _captured_pom(monkeypatch, {}) == default
    assert default.enabled is True


@pytest.mark.parametrize("value", [False, {"enabled": False}])
def test_pom_can_be_disabled(monkeypatch, value) -> None:
    pom = _captured_pom(monkeypatch, {"pom": value})
    assert pom.enabled is False
    assert pom.scale == 0.0
    assert pom.shadow is False and pom.occlusion is False


def test_pom_mapping_overrides_fields(monkeypatch) -> None:
    pom = _captured_pom(monkeypatch, {"pom": {"scale": 0.02, "max_steps": 64}})
    assert pom.enabled is True
    assert pom.scale == pytest.approx(0.02)
    assert pom.max_steps == 64
    assert pom.min_steps == 12


def test_pom_rejects_unknown_keys_and_types(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unknown pom settings"):
        _captured_pom(monkeypatch, {"pom": {"strength": 1.0}})
    with pytest.raises(TypeError):
        _captured_pom(monkeypatch, {"pom": "off"})
