from __future__ import annotations

import json

import numpy as np
import pytest

import forge3d as f3d
from forge3d import map_scene


def _building_geojson() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "b1",
                "properties": {
                    "height": 28,
                    "building:material": "brick",
                    "roof:shape": "gabled",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.25, 0.25],
                            [0.75, 0.25],
                            [0.75, 0.75],
                            [0.25, 0.75],
                            [0.25, 0.25],
                        ]
                    ],
                },
            }
        ],
    }


def _scene(layers):
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "flat-dem", "width": 8, "height": 8},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=layers,
    )


def _mixed_roof_layer() -> f3d.MapSceneBuildingLayer:
    features = []
    roof_types = ("flat", "gabled", "hipped", "pyramidal")
    for idx, roof_type in enumerate(roof_types):
        x0 = 0.08 + idx * 0.22
        x1 = x0 + 0.15
        y0 = 0.22
        y1 = 0.54
        features.append(
            {
                "id": f"b-{roof_type}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
                },
                "properties": {
                    "height": 24.0 + idx * 6.0,
                    "roof:shape": roof_type,
                    "building:material": "brick" if idx % 2 else "concrete",
                },
            }
        )
    return f3d.MapSceneBuildingLayer(
        layer_id="buildings",
        source={"source_id": "mixed-roofs", "asset_status": "fixture"},
        support_level="supported",
        geometry_count=len(features),
        material_status="scalar_pbr_underdeveloped",
        features=features,
        metadata={"source_id": "mixed-roofs", "asset_status": "fixture"},
    )


def test_geojson_building_layer_is_scalar_render_supported(tmp_path) -> None:
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(_building_geojson()), encoding="utf-8")

    layer = f3d.MapSceneBuildingLayer.from_geojson(path, metadata={"asset_status": "fixture"})
    report = _scene([layer]).validate()

    assert layer.support_level == "supported"
    assert layer.geometry_count == 1
    assert report.status == "ok"
    assert report.supported_features["buildings.scalar_materials"] == "supported"
    assert report.supported_features["buildings.mapscene_render"] == "supported"
    assert "buildings.mapscene_render" not in report.unsupported_features
    assert not any(d.code == "placeholder_fallback" for d in report.diagnostics)


def test_mapscene_render_composites_building_layer(tmp_path) -> None:
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(_building_geojson()), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer.from_geojson(path, metadata={"asset_status": "fixture"})

    with_building = tmp_path / "with-building.png"
    without_building = tmp_path / "without-building.png"
    report = _scene([layer]).render(str(with_building), allow_placeholder=True)
    _scene([]).render(str(without_building), allow_placeholder=True)

    from PIL import Image

    building_pixels = np.asarray(Image.open(with_building).convert("RGBA"))
    base_pixels = np.asarray(Image.open(without_building).convert("RGBA"))

    assert report.supported_features["mapscene.building_composite"] == "supported"
    assert np.count_nonzero(np.any(building_pixels != base_pixels, axis=2)) > 100


def test_native_building_mesh_batches_preserve_mixed_roof_geometry() -> None:
    batches = map_scene._native_building_mesh_batches_for_layers([_mixed_roof_layer()])

    assert batches is not None
    assert {batch["roof_type"] for batch in batches} == {"flat", "gabled", "hipped", "pyramidal"}
    assert {batch["feature_id"] for batch in batches} == {
        "b-flat",
        "b-gabled",
        "b-hipped",
        "b-pyramidal",
    }
    for batch in batches:
        positions = np.asarray(batch["positions"], dtype=np.float32)
        assert positions.shape[1] == 3
        assert np.asarray(batch["indices"]).shape[1] == 3
        assert np.asarray(batch["normals"]).shape == positions.shape
        max_y = float(np.max(positions[:, 1]))
        if batch["roof_type"] == "flat":
            assert max_y == pytest.approx(float(batch["wall_height"]), abs=1.0e-5)
        else:
            assert max_y > float(batch["wall_height"])


def test_native_building_shadow_mesh_projects_from_batch_geometry() -> None:
    batches = map_scene._native_building_mesh_batches_for_layers([_mixed_roof_layer()])
    assert batches is not None

    shadow = map_scene._native_building_projected_shadow_mesh(batches, (0.4, 0.6, 0.2))

    assert shadow is not None
    positions, indices, normals = shadow
    assert np.asarray(positions).shape == (len(batches) * 4, 3)
    assert np.asarray(indices).shape == (len(batches) * 2, 3)
    assert np.asarray(normals).shape == np.asarray(positions).shape
    assert np.allclose(np.asarray(normals)[:, 1], 1.0)
    assert float(np.max(np.asarray(positions)[:, 1])) == pytest.approx(0.012)


def test_building_batches_serialize_to_terrain_scatter_contract() -> None:
    scene = _scene([_mixed_roof_layer()])

    result = map_scene._terrain_scatter_building_batches_for_recipe(
        scene.recipe,
        np.zeros((8, 8), dtype=np.float32),
    )

    assert result is not None
    scatter_batches, metadata = result
    assert metadata["building_backend"] == "terrain_scatter_instanced_mesh"
    assert metadata["building_shadow_model"] == "terrain_csm_mesh_cast_receive"
    assert metadata["building_batch_count"] == 4
    assert set(metadata["building_roof_types"].values()) == {"flat", "gabled", "hipped", "pyramidal"}
    assert set(metadata["building_batch_ids"]) == {
        "b-flat",
        "b-gabled",
        "b-hipped",
        "b-pyramidal",
    }
    for batch in scatter_batches:
        assert str(batch["name"]).startswith("building:")
        assert batch["terrain_contact"]["enabled"] is True
        assert batch["terrain_blend"]["enabled"] is False
        assert np.asarray(batch["transforms"], dtype=np.float32).shape == (1, 16)
        mesh = batch["levels"][0]["mesh"]
        positions = np.asarray(mesh["positions"], dtype=np.float32)
        assert positions.ndim == 2 and positions.shape[1] == 3
        assert np.asarray(mesh["indices"], dtype=np.uint32).ndim == 2
        assert float(np.max(positions[:, 1])) > 0.0


def test_native_offscreen_uses_terrain_scatter_buildings_without_projected_compositor(monkeypatch) -> None:
    scene = _scene([_mixed_roof_layer()])
    base = np.zeros((32, 32, 4), dtype=np.uint8)
    base[..., 3] = 255
    metadata = {
        "building_backend": "terrain_scatter_instanced_mesh",
        "building_batch_count": 4,
        "building_batch_ids": {"b-flat": 0},
        "building_roof_types": {"b-flat": "flat"},
        "building_shadow_model": "terrain_csm_mesh_cast_receive",
    }

    monkeypatch.setattr(
        map_scene,
        "_render_terrain_renderer_result",
        lambda *_args, **_kwargs: map_scene._MapSceneNativeRenderResult(
            rgba=base.copy(),
            metadata=metadata,
        ),
    )

    def fail_projected_compositor(*_args, **_kwargs):
        raise AssertionError("projected building compositor should not run for terrain scatter buildings")

    monkeypatch.setattr(map_scene, "_composite_native_building_layers", fail_projected_compositor)

    result = map_scene._render_native_offscreen_result(scene.recipe, {}, allow_placeholder=True)

    assert result is not None
    assert result.metadata["building_backend"] == "terrain_scatter_instanced_mesh"
    assert result.metadata["building_shadow_model"] == "terrain_csm_mesh_cast_receive"


def test_native_building_compositor_sends_roof_mesh_to_scene(tmp_path, monkeypatch) -> None:
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(_building_geojson()), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer.from_geojson(path, metadata={"asset_status": "fixture"})
    recipe = _scene([layer]).recipe
    base = np.zeros((48, 48, 4), dtype=np.uint8)
    base[..., 3] = 255
    calls: dict[str, object] = {"batches": []}

    class FakeScene:
        def __init__(self, width, height):
            calls["size"] = (width, height)
            self._base = np.zeros((height, width, 4), dtype=np.uint8)
            self._base[..., 3] = 255

        def disable_terrain(self):
            calls["disable_terrain"] = True

        def set_raster_overlay(self, image, *_args):
            self._base = np.asarray(image, dtype=np.uint8).copy()

        def set_camera_look_at(self, *_args):
            calls["camera"] = True

        def add_instanced_mesh(self, positions, indices, transforms, *, normals=None, color=None, light_dir=None, light_intensity=None):
            batch = {
                "positions": np.asarray(positions, dtype=np.float32),
                "indices": np.asarray(indices, dtype=np.uint32),
                "transforms": np.asarray(transforms, dtype=np.float32),
                "normals": np.asarray(normals, dtype=np.float32),
                "color": color,
                "light_dir": light_dir,
                "light_intensity": light_intensity,
            }
            calls["batches"].append(batch)
            return len(calls["batches"]) - 1

        def render_rgba(self):
            out = self._base.copy()
            out[20:28, 20:28, :3] = (180, 110, 80)
            return out

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeScene)

    composited, used_native, metadata = map_scene._composite_native_building_layers(base, recipe)

    assert used_native is True
    assert calls["size"] == (48, 48)
    assert calls["disable_terrain"] is True
    assert calls["camera"] is True
    assert metadata["building_backend"] == "native_instanced_mesh"
    assert metadata["building_batch_count"] == 1
    assert metadata["building_batch_ids"] == {"b1": 0}
    assert metadata["building_shadow_model"] == "projected_native_shadow_mesh"
    positions = calls["batches"][0]["positions"]
    indices = calls["batches"][0]["indices"]
    normals = calls["batches"][0]["normals"]
    assert positions.shape[1] == 3
    assert indices.shape[1] == 3
    assert normals.shape == positions.shape
    assert float(np.max(positions[:, 1])) > (28.0 / 45.0)
    shadow_batch = calls["batches"][1]
    assert shadow_batch["color"] == (1.0, 1.0, 1.0, 1.0)
    assert shadow_batch["light_intensity"] == 1.0
    assert np.asarray(shadow_batch["positions"]).shape[1] == 3
    assert np.count_nonzero(np.any(composited != base, axis=2)) > 0
