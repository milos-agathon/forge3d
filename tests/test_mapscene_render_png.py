import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
import numpy as np

from forge3d.helpers.offscreen import save_png_deterministic


def _supported_scene(seed: int = 19) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "asset_status": "fixture"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=750.0, azimuth_deg=30.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.25),
        output=f3d.OutputSpec(width=48, height=32, format="png"),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=seed),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                opacity=0.8,
                metadata={"width": 8, "height": 8, "asset_status": "fixture"},
            ),
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                        "properties": {"class": "primary"},
                    }
                ],
                style={"version": 8, "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#ffffff"}}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "harbor",
                        "kind": "point",
                        "text": "Harbor",
                        "geometry": {"type": "Point", "coordinates": (24.0, 16.0, 0.0)},
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Harbor"))},
            ),
        ],
    )


def test_render_writes_source_derived_png_for_supported_scene(tmp_path):
    first_path = tmp_path / "first.png"

    scene = _supported_scene()
    report = scene.render(str(first_path))

    assert first_path.exists()
    assert first_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report.status == "ok"
    assert report.supported_features["mapscene.render_png"] == "supported"
    assert "mapscene.render_backend" not in report.unsupported_features
    assert not any(
        diagnostic.code == "placeholder_fallback" and diagnostic.layer_id == "mapscene.render_png"
        for diagnostic in report.diagnostics
    )
    assert scene.last_validation_report is not None
    assert scene.last_render_path == str(first_path)


def test_render_uses_output_spec_path_for_supported_png(tmp_path):
    output_path = tmp_path / "from-output-spec.png"
    scene = _supported_scene()
    scene.recipe.output.path = str(output_path)

    report = scene.render()

    assert output_path.exists()
    assert report.status == "ok"
    assert scene.last_validation_report is not None
    assert scene.last_render_path == str(output_path)


def test_render_output_changes_when_source_data_changes(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_scene = _supported_scene(seed=19)
    second_scene = _supported_scene(seed=19)
    second_scene.recipe.layers = (
        f3d.RasterOverlay(
            layer_id="ortho-alt",
            path="fixtures/alternate-ortho.tif",
            crs="EPSG:32610",
            opacity=0.35,
            metadata={"width": 16, "height": 16, "source_id": "alternate-raster", "asset_status": "fixture"},
        ),
        f3d.VectorOverlay(
            layer_id="roads-alt",
            crs="EPSG:32610",
            features=[
                {
                    "id": "road-alt",
                    "geometry": {"type": "LineString", "coordinates": [(1.0, 0.0), (0.0, 1.0)]},
                    "properties": {"class": "secondary"},
                }
            ],
            style={"version": 8, "layers": [{"id": "roads-alt", "type": "line", "paint": {"line-color": "#00ff00"}}]},
        ),
    )

    first_report = first_scene.render(str(first_path))
    second_report = second_scene.render(str(second_path))

    assert first_report.status == "ok"
    assert second_report.status == "ok"
    assert first_path.exists()
    assert second_path.exists()
    assert first_path.read_bytes() != second_path.read_bytes()


def _native_asset_scene(tmp_path):
    terrain_path = tmp_path / "terrain.npy"
    heightmap = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    np.save(terrain_path, heightmap)

    raster_path = tmp_path / "ortho.png"
    raster = np.zeros((32, 48, 4), dtype=np.uint8)
    raster[..., 0] = 32
    raster[..., 1] = 196
    raster[..., 2] = 120
    raster[..., 3] = 255
    save_png_deterministic(raster_path, raster)

    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=str(terrain_path),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=750.0, azimuth_deg=30.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.25),
        output=f3d.OutputSpec(width=48, height=32, format="png"),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=23),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path=str(raster_path),
                crs="EPSG:32610",
                opacity=0.8,
                metadata={"width": 48, "height": 32},
            ),
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                        "properties": {"class": "primary"},
                    }
                ],
                style={"version": 8, "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#ffffff"}}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "harbor",
                        "kind": "point",
                        "text": "Harbor",
                        "geometry": {"type": "Point", "coordinates": (24.0, 16.0, 0.0)},
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Harbor"))},
            ),
        ],
    )


def test_render_uses_native_offscreen_for_real_terrain_and_raster_assets(tmp_path, monkeypatch):
    scene = _native_asset_scene(tmp_path)

    class FakeNativeScene:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.heightmap = None
            self.overlay = None

        def set_height_from_r32f(self, heightmap):
            self.heightmap = heightmap

        def set_camera_look_at(self, *_args):
            return None

        def set_raster_overlay(self, overlay, *_args):
            self.overlay = overlay

        def render_rgba(self):
            assert self.heightmap is not None
            assert self.overlay is not None
            rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            rgba[..., 0] = 16
            rgba[..., 1] = 48
            rgba[..., 2] = 96
            rgba[..., 3] = 255
            return rgba

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeScene)

    def fail_source_derived_render(*_args, **_kwargs):
        raise AssertionError("source-derived fallback was used for fixture-backed native render")

    monkeypatch.setattr(map_scene, "_render_source_derived_rgba", fail_source_derived_render)
    output_path = tmp_path / "native-offscreen.png"

    report = scene.render(str(output_path))

    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report.status == "ok"
    assert report.supported_features["mapscene.render_png"] == "supported"
    assert report.supported_features["mapscene.render_backend"] == "supported"
    assert scene.last_render_backend == "native/offscreen"
    assert scene.compiled_label_plans["labels"].accepted


def test_render_falls_back_when_native_adapter_is_unavailable(tmp_path, monkeypatch):
    scene = _native_asset_scene(tmp_path)

    class PanicException(BaseException):
        pass

    PanicException.__module__ = "pyo3_runtime"

    class UnavailableNativeScene:
        def __init__(self, *_args):
            raise PanicException("No suitable GPU adapter")

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: UnavailableNativeScene)
    output_path = tmp_path / "fallback.png"

    report = scene.render(str(output_path))

    assert output_path.exists()
    assert report.status == "ok"
    assert report.supported_features["mapscene.render_png"] == "supported"
    assert scene.last_render_backend == "source-derived"
