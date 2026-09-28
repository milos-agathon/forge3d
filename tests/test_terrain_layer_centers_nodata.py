"""Configurable material layer bands and no-data holes on the flat terrain path."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import (
    _terrain_rendering_available_inprocess,
    terrain_rendering_available,
)
from forge3d._native import NATIVE_AVAILABLE
from forge3d.terrain_params import make_terrain_params_config

ROOT = Path(__file__).resolve().parents[1]
GPU_AVAILABLE = (
    terrain_rendering_available()
    if os.environ.get("GITHUB_ACTIONS") == "true"
    else _terrain_rendering_available_inprocess()
)
TEST_PRIVATE_KEY = bytes(range(32))
VIRTUAL_SIZE = 512


def _make_params(**overrides):
    return make_terrain_params_config(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=1000.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1000.0),
        **overrides,
    )


def test_layer_centers_and_nodata_default_to_disabled() -> None:
    params = _make_params()
    assert params.material_layer_centers is None
    assert params.nodata_height_below is None


def test_layer_centers_are_normalized_to_a_float_tuple() -> None:
    params = _make_params(material_layer_centers=[0, 0.25, 0.5, 1])
    assert params.material_layer_centers == (0.0, 0.25, 0.5, 1.0)


@pytest.mark.parametrize(
    ("centers", "message"),
    [
        ((), "1 to 4 values"),
        ((0.0, 0.2, 0.4, 0.6, 0.8), "1 to 4 values"),
        ((0.0, 1.5), "within \\[0, 1\\]"),
        ((0.0, float("nan")), "within \\[0, 1\\]"),
        ((0.0, 0.5, 0.5), "strictly increasing"),
        ((0.6, 0.2), "strictly increasing"),
    ],
)
def test_layer_centers_reject_invalid_bands(centers, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _make_params(material_layer_centers=centers)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nodata_height_below_rejects_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="nodata_height_below must be finite"):
        _make_params(nodata_height_below=invalid)


def test_nodata_fill_leaves_finite_heightmaps_untouched() -> None:
    from forge3d.map_scene import _mapscene_nodata_heightmap

    heightmap = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    filled, threshold = _mapscene_nodata_heightmap(heightmap)
    assert filled is heightmap
    assert threshold is None


def test_nodata_fill_places_holes_below_every_valid_height() -> None:
    from forge3d.map_scene import _mapscene_nodata_heightmap

    heightmap = np.linspace(100.0, 200.0, 16, dtype=np.float32).reshape(4, 4)
    heightmap[0, :] = np.nan
    filled, threshold = _mapscene_nodata_heightmap(heightmap)

    valid = np.isfinite(heightmap)
    assert np.isfinite(filled).all()
    assert np.array_equal(filled[valid], heightmap[valid])
    assert threshold is not None
    assert (filled[~valid] < threshold).all()
    assert threshold < heightmap[valid].min()


def test_nodata_fill_rejects_heightmaps_without_data() -> None:
    from forge3d.map_scene import _mapscene_nodata_heightmap

    with pytest.raises(ValueError, match="no finite heights"):
        _mapscene_nodata_heightmap(np.full((4, 4), np.nan, dtype=np.float32))


def test_mapscene_forwards_layer_centers_and_nodata(monkeypatch) -> None:
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

    heightmap = np.linspace(10.0, 20.0, 64, dtype=np.float32).reshape(8, 8)
    heightmap[:2] = np.nan
    filled, threshold = map_scene._mapscene_nodata_heightmap(heightmap)
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(data=heightmap, crs="EPSG:32610"),
        camera=f3d.OrbitCamera(distance=100.0),
        lighting=f3d.LightingPreset(
            settings={"material_layer_centers": [0.0, 0.2, 0.6, 1.0]}
        ),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    captured = map_scene._build_mapscene_terrain_params(
        scene.recipe, filled, (64, 64), nodata_height_below=threshold
    )

    assert captured is not None
    assert captured.config.material_layer_centers == (0.0, 0.2, 0.6, 1.0)
    assert captured.config.nodata_height_below == pytest.approx(threshold)
    # The no-data fill stays out of the colour/material domain.
    assert tuple(captured.config.clamp.height_range) == pytest.approx(
        (float(np.nanmin(heightmap)), float(np.nanmax(heightmap)))
    )


def test_layer_centers_and_nodata_reach_the_uniforms_and_shader() -> None:
    upload = (ROOT / "src/terrain/renderer/upload.rs").read_text(encoding="utf-8")
    shader = (ROOT / "src/shaders/terrain_pbr_pom.wgsl").read_text(encoding="utf-8")

    assert "params.material_layer_centers.as_deref()" in upload
    assert "params.nodata_height_below.unwrap_or(0.0)" in upload
    assert (
        "u_overlay.params6.z > 0.5 && sample_height(uv) < u_overlay.params6.y"
        in shader
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Requires compiled _forge3d extension")
def test_native_params_expose_layer_centers_and_nodata() -> None:
    native = f3d.TerrainRenderParams(
        _make_params(material_layer_centers=(0.0, 0.5, 1.0), nodata_height_below=-2.5)
    )
    assert native.material_layer_centers == pytest.approx([0.0, 0.5, 1.0])
    assert native.nodata_height_below == pytest.approx(-2.5)

    default = f3d.TerrainRenderParams(_make_params())
    assert default.material_layer_centers is None
    assert default.nodata_height_below is None


def _ramp_heightmap(size: int) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.ascontiguousarray(np.tile(ramp, (size, 1)))


def _render_source_map(tmp_path: Path, name: str, heightmap: np.ndarray, settings: dict) -> np.ndarray:
    image_path = tmp_path / f"{name}.png"
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heightmap,
            crs="EPSG:32610",
            metadata={
                "asset_status": "fixture",
                "virtual_texture": {
                    "enabled": True,
                    "families": [
                        {"family": "albedo", "virtual_size_px": [VIRTUAL_SIZE, VIRTUAL_SIZE]}
                    ],
                    "procedural_sources": True,
                    "source_count": 4,
                    "source_size": VIRTUAL_SIZE,
                },
            },
        ),
        camera=f3d.OrbitCamera(
            target=(0.0, 0.0, 0.0),
            distance=4.0,
            azimuth_deg=90.0,
            elevation_deg=89.0,
            fov_deg=50.0,
        ),
        lighting=f3d.LightingPreset(
            settings={
                "albedo_mode": "material",
                "colormap_strength": 0.0,
                "hue_variation_strength": 0.0,
                "material_slope_bias": 0.0,
                **settings,
            }
        ),
        output=f3d.OutputSpec(path=str(image_path), width=256, height=256, samples=1),
    )
    scene.render(emit_provenance=True, provenance_signing_key=TEST_PRIVATE_KEY)
    return np.load(image_path.with_name(f"{name}.source_map.npy"))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires a terrain-capable GPU runtime")
def test_nodata_cells_render_empty_and_unattributed(tmp_path) -> None:
    full = _ramp_heightmap(160)
    holed = full.copy()
    # A no-data strip inside the framed part of the ramp (this camera frames
    # the first half of the columns).
    holed[:, 20:40] = np.nan

    full_map = _render_source_map(tmp_path, "full", full, {})
    holed_map = _render_source_map(tmp_path, "holed", holed, {})

    # Holes only remove attribution: every attributed pixel of the holed render
    # is also attributed in the complete render, and some pixels are now empty.
    assert np.count_nonzero(holed_map) > 0
    assert np.all(full_map[holed_map != 0] != 0)
    assert np.count_nonzero(holed_map) < np.count_nonzero(full_map)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires a terrain-capable GPU runtime")
def test_layer_centers_move_the_attributed_bands(tmp_path) -> None:
    heightmap = _ramp_heightmap(160)
    even = _render_source_map(tmp_path, "even", heightmap, {})
    low = _render_source_map(
        tmp_path, "low", heightmap, {"material_layer_centers": [0.0, 0.1, 0.2, 0.3]}
    )

    top = 4  # the highest layer's albedo source id
    assert np.count_nonzero(low == top) > np.count_nonzero(even == top)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires a terrain-capable GPU runtime")
def test_layer_center_count_must_match_the_material_layers(tmp_path) -> None:
    with pytest.raises(Exception, match="material_layer_centers has 3 values"):
        _render_source_map(
            tmp_path, "mismatch", _ramp_heightmap(64), {"material_layer_centers": [0.0, 0.5, 1.0]}
        )
