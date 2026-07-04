import pytest
from pathlib import Path
import struct
import zlib

import forge3d as f3d
import forge3d.map_scene as map_scene
import numpy as np

from forge3d.helpers.offscreen import save_png_deterministic
from _terrain_runtime import terrain_rendering_available


def _png_ihdr(path: Path) -> tuple[int, int, int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert data[12:16] == b"IHDR"
    width, height, bit_depth, color_type, _compression, _filter, _interlace = struct.unpack(">IIBBBBB", data[16:29])
    return width, height, bit_depth, color_type


def _png_rgba16_samples(path: Path) -> np.ndarray:
    data = path.read_bytes()
    pos = 8
    width = height = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", chunk_data)
            assert bit_depth == 16
            assert color_type == 6
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    assert width is not None and height is not None
    raw = zlib.decompress(bytes(idat))
    rows = []
    stride = int(width) * 4 * 2
    offset = 0
    for _ in range(int(height)):
        assert raw[offset] == 0
        offset += 1
        rows.append(np.frombuffer(raw[offset : offset + stride], dtype=">u2").astype(np.uint16).reshape(int(width), 4))
        offset += stride
    return np.stack(rows, axis=0)


def _public_label_vector_scene(path: Path) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=120.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=96, height=64, format="png", path=str(path)),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {"id": "a", "geometry": {"type": "LineString", "coordinates": [(0.08, 0.20), (0.92, 0.72)]}},
                    {"id": "b", "geometry": {"type": "LineString", "coordinates": [(0.12, 0.78), (0.88, 0.28)]}},
                ],
                width_px=4,
                line_cap="round",
                line_join="round",
                dash_array=[8, 4],
                style={"version": 8, "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#f9fafb"}}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "summit",
                        "kind": "point",
                        "text": "Summit",
                        "geometry": {"type": "Point", "coordinates": (30.0, 20.0, 0.0)},
                    },
                    {
                        "id": "trail",
                        "kind": "point",
                        "text": "Trail",
                        "geometry": {"type": "Point", "coordinates": (66.0, 42.0, 0.0)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("SummitTrail"))},
            ),
        ],
        reproducibility_profile=f3d.ReproducibilityProfile(seed=42),
    )


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


def test_render_requires_explicit_placeholder_for_symbolic_scene(tmp_path):
    output_path = tmp_path / "blocked-placeholder.png"
    scene = _supported_scene()

    with pytest.raises(RuntimeError, match="allow_placeholder=True"):
        scene.render(str(output_path))

    assert not output_path.exists()


def test_render_writes_placeholder_png_when_explicitly_allowed(tmp_path):
    first_path = tmp_path / "first.png"

    scene = _supported_scene()
    report = scene.render(str(first_path), allow_placeholder=True)

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
    assert scene.last_render_backend == "placeholder"


def test_save_png_deterministic_writes_bit_exact_rgba16(tmp_path):
    output_path = tmp_path / "rgba16.png"
    rgba16 = np.array(
        [
            [[0, 32768, 65535, 65535], [17, 1024, 4096, 8192]],
            [[12345, 23456, 34567, 45678], [65535, 0, 1, 2]],
        ],
        dtype=np.uint16,
    )

    save_png_deterministic(output_path, rgba16, bit_depth=16)

    assert _png_ihdr(output_path) == (2, 2, 16, 6)
    assert np.array_equal(_png_rgba16_samples(output_path), rgba16)


def test_render_writes_16bit_png_when_requested(tmp_path):
    output_path = tmp_path / "map-16bit.png"
    scene = _supported_scene()
    scene.recipe.output.path = str(output_path)
    scene.recipe.output.bit_depth = 16

    report = scene.render(allow_placeholder=True)

    assert _png_ihdr(output_path) == (48, 32, 16, 6)
    assert report.supported_features["output.bit_depth.16"] == "supported"
    assert report.supported_features["mapscene.render_png_16bit"] == "supported"
    assert scene.last_render_metadata["bit_depth"] == 16


def test_render_uses_output_spec_path_for_supported_png(tmp_path):
    output_path = tmp_path / "from-output-spec.png"
    scene = _supported_scene()
    scene.recipe.output.path = str(output_path)

    report = scene.render(allow_placeholder=True)

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

    first_report = first_scene.render(str(first_path), allow_placeholder=True)
    second_report = second_scene.render(str(second_path), allow_placeholder=True)

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

        def disable_terrain(self):
            return None

        def set_native_text_atlas(self, *_args, **_kwargs):
            return None

        def enable_native_text(self):
            return None

        def add_native_text_rect_uv_halo(self, *_args):
            return None

        def render_rgba(self):
            if self.heightmap is None and self.overlay is not None:
                return self.overlay
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
    assert report.supported_features["mapscene.vector_composite"] == "supported"
    assert report.supported_features["mapscene.label_composite"] == "supported"
    assert scene.last_render_backend == "gpu_terrain"
    assert scene.compiled_label_plans["labels"].accepted


def test_render_routes_terrain_through_terrain_renderer(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )
    calls: dict[str, object] = {}

    def fail_legacy_scene():
        raise AssertionError("MapScene native render must use TerrainRenderer, not Scene.render_rgba")

    class FakeSession:
        def __init__(self, *, window=False):
            calls["session_window"] = window

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            calls["material_set"] = "default"
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            calls["ibl_path"] = str(path)
            return "ibl"

    class FakeParams:
        def __init__(self, config):
            calls["params_size"] = tuple(config.size_px)
            self.config = config

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 0] = 11
            rgba[..., 1] = 22
            rgba[..., 2] = 33
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, session):
            calls["renderer_session"] = session

        def render_terrain_pbr_pom(self, *, material_set, env_maps, params, heightmap, target=None, water_mask=None):
            calls["render_call"] = {
                "material_set": material_set,
                "env_maps": env_maps,
                "heightmap_shape": tuple(heightmap.shape),
                "target": target,
                "water_mask": water_mask,
                "params_type": type(params).__name__,
            }
            return FakeFrame()

    monkeypatch.setattr(map_scene, "_native_scene_class", fail_legacy_scene)
    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    output_path = tmp_path / "terrain-renderer.png"
    scene.render(str(output_path))

    assert scene.last_render_backend == "gpu_terrain"
    assert calls["session_window"] is False
    assert calls["material_set"] == "default"
    assert calls["params_size"] == (64, 64)
    assert calls["render_call"] == {
        "material_set": "material-set",
        "env_maps": "ibl",
        "heightmap_shape": (8, 8),
        "target": None,
        "water_mask": None,
        "params_type": "FakeParams",
    }


def test_gpu_backend_blocks_label_vector_cpu_fallback_without_placeholder(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=80, height=40, format="png"),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                    }
                ],
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "roads",
                            "type": "line",
                            "paint": {"line-color": "#00ff00", "line-width": 3},
                        }
                    ],
                },
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "ab",
                        "kind": "point",
                        "text": "AB",
                        "geometry": {"type": "Point", "coordinates": (20.0, 12.0, 0.0)},
                        "typography": {"color": "#ff0000", "halo_color": "#0000ff"},
                    }
                ],
                glyph_atlas={"glyphs": ["A", "B"]},
            ),
        ],
    )

    class FakeNativeScene:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def set_height_from_r32f(self, _heightmap):
            return None

        def set_camera_look_at(self, *_args):
            return None

        def render_rgba(self):
            rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeScene)
    monkeypatch.setattr(map_scene, "_composite_native_vector_layers", lambda base, _recipe: (base, False))
    output_path = tmp_path / "styled-composite.png"

    with pytest.raises(RuntimeError, match="native label/vector compositing"):
        scene.render(str(output_path))

    report = scene.render(str(output_path), allow_placeholder=True)

    from forge3d._png import load_png_rgba

    image = load_png_rgba(output_path)
    green_pixels = np.count_nonzero((image[..., 0] == 0) & (image[..., 1] == 255) & (image[..., 2] == 0))
    red_pixels = np.count_nonzero((image[..., 0] > 160) & (image[..., 1] < 80) & (image[..., 2] < 80))
    blue_pixels = np.count_nonzero((image[..., 2] > 120) & (image[..., 0] < 120) & (image[..., 1] < 120))

    assert report.status == "ok"
    assert report.supported_features["mapscene.vector_composite"] == "supported"
    assert report.supported_features["mapscene.label_composite"] == "supported"
    assert green_pixels > 50
    assert red_pixels > 5
    assert blue_pixels > 5


def test_public_label_vector_render_is_deterministic_and_not_placeholder(tmp_path):
    if not terrain_rendering_available():
        pytest.skip("real MapScene label/vector render requires a terrain-capable GPU runtime")

    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = _public_label_vector_scene(first_path)
    second = _public_label_vector_scene(second_path)

    first_report = first.render()
    second_report = second.render()

    assert first_report.status == "ok"
    assert second_report.status == "ok"
    assert first.last_render_backend == "gpu_terrain"
    assert second.last_render_backend == "gpu_terrain"
    assert first.compiled_label_plans["labels"].to_dict()["accepted"] == second.compiled_label_plans["labels"].to_dict()["accepted"]
    assert first_path.read_bytes() == second_path.read_bytes()

    placeholder = map_scene._render_source_derived_rgba(first.recipe, first.compiled_label_plans)
    placeholder_path = tmp_path / "placeholder.png"
    save_png_deterministic(placeholder_path, placeholder)

    assert first_path.read_bytes() != placeholder_path.read_bytes()


def test_gpu_backend_uses_native_sdf_text_for_labels(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=80, height=64, format="png"),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "ab",
                        "kind": "point",
                        "text": "AB",
                        "geometry": {"type": "Point", "coordinates": (20.0, 12.0, 0.0)},
                        "typography": {"color": "#ff0000", "halo_color": "#0000ff", "halo_width_px": 2},
                    }
                ],
                glyph_atlas={"glyphs": ["A", "B"]},
            )
        ],
    )
    calls: dict[str, object] = {"glyphs": 0}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return rgba

    class FakeNativeTextScene:
        def __init__(self, width, height):
            calls["scene_size"] = (width, height)
            self.base = None

        def disable_terrain(self):
            calls["terrain_disabled"] = True

        def set_raster_overlay(self, image, alpha, offset_xy, scale):
            calls["base_shape"] = tuple(image.shape)
            calls["base_alpha"] = alpha
            self.base = np.asarray(image).copy()

        def set_native_text_atlas(self, atlas, channels=None, smoothing=None):
            calls["atlas_shape"] = tuple(atlas.shape)
            calls["atlas_channels"] = channels
            calls["atlas_smoothing"] = smoothing

        def enable_native_text(self):
            calls["enabled"] = True

        def add_native_text_rect_uv_halo(self, *args):
            calls["glyphs"] = int(calls["glyphs"]) + 1
            calls["last_glyph_arg_count"] = len(args)

        def render_rgba(self):
            assert self.base is not None
            return self.base

    import forge3d._map_scene_render as render_helpers

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_rgba", fake_terrain)
    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeTextScene)
    monkeypatch.setattr(
        render_helpers,
        "_draw_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CPU text fallback used")),
    )

    output_path = tmp_path / "native-text.png"
    scene.render(str(output_path))

    assert output_path.exists()
    assert calls["scene_size"] == (80, 64)
    assert calls["terrain_disabled"] is True
    assert calls["base_shape"] == (64, 80, 4)
    assert calls["atlas_channels"] == 1
    assert calls["glyphs"] == 2
    assert calls["last_glyph_arg_count"] == 17


def test_native_scene_render_rgba_draws_raster_overlay_and_sdf_text():
    try:
        scene = f3d.Scene(80, 64)
    except Exception as exc:
        pytest.skip(f"native Scene unavailable: {exc}")

    scene.disable_terrain()
    base = np.zeros((64, 80, 4), dtype=np.uint8)
    base[..., 0] = 48
    base[..., 3] = 255
    scene.set_raster_overlay(base, 1.0, None, None)

    atlas = np.full((8, 8, 4), 255, dtype=np.uint8)
    scene.set_native_text_atlas(atlas, 4, 1.0)
    scene.enable_native_text()
    scene.add_native_text_rect_uv(10.0, 12.0, 30.0, 20.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    rgba = np.asarray(scene.render_rgba())
    red_base = np.count_nonzero((rgba[..., 0] >= 6) & (rgba[..., 1] < 20) & (rgba[..., 2] < 20))
    green_text = np.count_nonzero((rgba[..., 1] > 160) & (rgba[..., 0] < 80) & (rgba[..., 2] < 80))

    assert red_base > 1000
    assert green_text > 100


def test_gpu_backend_uses_native_vector_oit_for_line_layers(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=80, height=64, format="png"),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                    }
                ],
                width_px=3,
                style={
                    "version": 8,
                    "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#00ff00"}}],
                },
            )
        ],
    )
    calls: dict[str, object] = {}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_vector_render(width, height, *, points_xy=None, point_rgba=None, point_size=None, polylines=None, polyline_rgba=None, stroke_width=None):
        calls["vector"] = {
            "size": (width, height),
            "points_xy": points_xy,
            "point_rgba": point_rgba,
            "point_size": point_size,
            "polylines": polylines,
            "polyline_rgba": polyline_rgba,
            "stroke_width": stroke_width,
        }
        top = np.zeros((height, width, 4), dtype=np.uint8)
        top[8:12, 8:12] = (0, 255, 0, 255)
        return top

    def fake_composite(bottom, top, *, premultiplied=False):
        calls["composite"] = (tuple(bottom.shape), tuple(top.shape), premultiplied)
        out = np.asarray(bottom).copy()
        mask = np.asarray(top)[..., 3] > 0
        out[mask] = top[mask]
        return out

    import forge3d._map_scene_render as render_helpers

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)
    monkeypatch.setattr(map_scene, "_alpha_composite_rgba", fake_composite)
    monkeypatch.setattr(
        render_helpers,
        "_draw_polyline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CPU vector fallback used")),
    )

    output_path = tmp_path / "native-vector.png"
    scene.render(str(output_path))

    assert output_path.exists()
    assert calls["vector"]["size"] == (80, 64)
    assert calls["vector"]["points_xy"] is None
    assert calls["vector"]["point_rgba"] is None
    assert calls["vector"]["point_size"] is None
    assert calls["vector"]["stroke_width"] == [3.0]
    assert calls["vector"]["polyline_rgba"] == [(0.0, 1.0, 0.0, 1.0)]
    polyline = calls["vector"]["polylines"][0]
    assert polyline[0] == (-1.0, 1.0)
    assert polyline[-1] == (1.0, -1.0)
    assert calls["composite"] == ((64, 80, 4), (64, 80, 4), False)


def test_outputspec_offline_fields_roundtrip_through_recipe():
    output = f3d.OutputSpec(
        width=64,
        height=32,
        format="exr",
        path="render.exr",
        samples=8,
        denoiser="atrous",
        aovs=("albedo", "normal", "depth"),
        hdr=True,
    )

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=output,
    )
    loaded = f3d.MapScene._recipe_from_dict(scene.recipe.to_dict())

    assert loaded.output is not None
    assert loaded.output.format == "exr"
    assert loaded.output.samples == 8
    assert loaded.output.denoiser == "atrous"
    assert loaded.output.aovs == ("albedo", "normal", "depth")
    assert loaded.output.hdr is True


def test_outputspec_rejects_unsupported_uv_aov():
    with pytest.raises(ValueError, match="Unsupported OutputSpec AOV"):
        f3d.OutputSpec(width=64, height=64, aovs=("uv",))


def test_render_records_offline_samples_and_writes_aovs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls: dict[str, object] = {"exr_writes": []}

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(
            width=16,
            height=12,
            format="png",
            path=str(tmp_path / "offline.png"),
            samples=3,
            denoiser="atrous",
            aovs=("albedo", "normal", "depth"),
            hdr=True,
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=77),
    )

    class FakeSession:
        def __init__(self, *, window=False):
            calls["session_window"] = window

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(*_args, **_kwargs):
            return "ibl"

    class FakeParams:
        def __init__(self, config):
            calls["aa_samples"] = config.aa_samples
            calls["aa_seed"] = config.aa_seed
            calls["aov"] = (config.aov.enabled, config.aov.albedo, config.aov.normal, config.aov.depth)
            calls["denoise"] = (config.denoise.enabled, config.denoise.method)
            self.config = config

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((12, 16, 4), dtype=np.uint8)
            rgba[..., 0] = 32
            rgba[..., 1] = 64
            rgba[..., 2] = 128
            rgba[..., 3] = 255
            return rgba

    class FakeHdrFrame:
        size = (16, 12)

        def save(self, path):
            calls["hdr_save"] = str(path)
            Path(path).write_bytes(b"HDR")

    class FakeAovFrame:
        def albedo(self):
            return np.full((12, 16, 3), 0.25, dtype=np.float32)

        def normal(self):
            return np.full((12, 16, 3), 0.5, dtype=np.float32)

        def depth(self):
            return np.full((12, 16), 0.75, dtype=np.float32)

    class FakeTerrainRenderer:
        def __init__(self, session):
            calls["renderer_session"] = session

        def render_terrain_pbr_pom(self, **_kwargs):
            raise AssertionError("multisample MapScene render used the one-shot path")

        def render_with_aov(self, **_kwargs):
            raise AssertionError("multisample MapScene render used render_with_aov instead of render_offline")

    def fake_render_offline(renderer, material_set, env_maps, params, heightmap, *, settings, progress_callback=None, water_mask=None):
        calls["offline"] = {
            "renderer_type": type(renderer).__name__,
            "material_set": material_set,
            "env_maps": env_maps,
            "heightmap_shape": tuple(heightmap.shape),
            "settings": (settings.enabled, settings.max_samples, settings.min_samples, settings.batch_size),
            "water_mask": water_mask,
            "progress_callback": progress_callback,
            "params_type": type(params).__name__,
        }
        return SimpleNamespace(
            frame=FakeFrame(),
            hdr_frame=FakeHdrFrame(),
            aov_frame=FakeAovFrame(),
            metadata={
                "samples_used": 3,
                "target_samples": 3,
                "denoiser_used": "atrous",
                "final_p95_delta": 0.002,
                "converged_ratio": 0.75,
                "adaptive": False,
            },
        )

    import forge3d.offline as offline_mod

    def fake_exr_writer(path, array, channel_prefix="beauty"):
        calls["exr_writes"].append((str(path), tuple(array.shape), str(channel_prefix), float(np.mean(array))))
        Path(path).write_bytes(b"EXR")

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)
    monkeypatch.setattr(offline_mod, "render_offline", fake_render_offline)
    monkeypatch.setattr(
        map_scene,
        "_aov_arrays_for_rgba",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthetic AOV arrays used")),
    )
    monkeypatch.setattr(map_scene, "_write_exr_array", fake_exr_writer)

    report = scene.render()

    assert Path(scene.last_render_path).exists()
    assert calls["session_window"] is False
    assert calls["aa_samples"] == 3
    assert calls["aa_seed"] == 77
    assert calls["aov"] == (True, True, True, True)
    assert calls["denoise"] == (True, "atrous")
    assert calls["offline"] == {
        "renderer_type": "FakeTerrainRenderer",
        "material_set": "material-set",
        "env_maps": "ibl",
        "heightmap_shape": (4, 4),
        "settings": (True, 3, 3, 3),
        "water_mask": None,
        "progress_callback": None,
        "params_type": "FakeParams",
    }
    assert calls["hdr_save"] == str(tmp_path / "offline.exr")
    assert sorted(calls["exr_writes"]) == [
        (str(tmp_path / "offline_aov-albedo.exr"), (12, 16, 3), "albedo", 0.25),
        (str(tmp_path / "offline_aov-depth.exr"), (12, 16), "depth", 0.75),
        (str(tmp_path / "offline_aov-normal.exr"), (12, 16, 3), "normal", 0.5),
    ]
    assert scene.last_render_metadata == {
        "samples_used": 3,
        "target_samples": 3,
        "denoiser_used": "atrous",
        "aov_paths": {
            "albedo": str(tmp_path / "offline_aov-albedo.exr"),
            "normal": str(tmp_path / "offline_aov-normal.exr"),
            "depth": str(tmp_path / "offline_aov-depth.exr"),
        },
        "hdr": True,
        "format": "png",
        "bit_depth": 8,
        "aa_seed": 77,
        "final_p95_delta": 0.002,
        "converged_ratio": 0.75,
        "adaptive": False,
    }
    assert report.supported_features["mapscene.offline_accumulation"] == "supported"
    assert report.supported_features["mapscene.aov_export"] == "supported"
    assert report.supported_features["mapscene.hdr_output"] == "supported"


def test_render_exr_uses_exr_report_feature(tmp_path, monkeypatch):
    def fake_exr_writer(path, array, channel_prefix="beauty"):
        Path(path).write_bytes(b"EXR")

    class FakeNativeScene:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def set_height_from_r32f(self, _heightmap):
            return None

        def set_camera_look_at(self, *_args):
            return None

        def render_rgba(self):
            rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=8, height=8, format="exr", path=str(tmp_path / "beauty.exr")),
    )

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeScene)
    monkeypatch.setattr(map_scene, "_numpy_to_exr_writer", lambda: fake_exr_writer)

    report = scene.render()

    assert Path(scene.last_render_path).suffix == ".exr"
    assert report.supported_features["mapscene.render_exr"] == "supported"
    assert "mapscene.render_png" not in report.supported_features


def test_render_blocks_placeholder_when_native_adapter_is_unavailable(tmp_path, monkeypatch):
    scene = _native_asset_scene(tmp_path)

    class PanicException(BaseException):
        pass

    PanicException.__module__ = "pyo3_runtime"

    class UnavailableTerrainRenderer:
        def __init__(self, *_args):
            raise PanicException("No suitable GPU adapter")

    monkeypatch.setattr(f3d, "TerrainRenderer", UnavailableTerrainRenderer)
    output_path = tmp_path / "fallback.png"

    with pytest.raises(RuntimeError, match="allow_placeholder=True"):
        scene.render(str(output_path))

    assert not output_path.exists()


def test_render_falls_back_when_explicitly_allowed(tmp_path, monkeypatch):
    scene = _native_asset_scene(tmp_path)

    class PanicException(BaseException):
        pass

    PanicException.__module__ = "pyo3_runtime"

    class UnavailableTerrainRenderer:
        def __init__(self, *_args):
            raise PanicException("No suitable GPU adapter")

    monkeypatch.setattr(f3d, "TerrainRenderer", UnavailableTerrainRenderer)
    output_path = tmp_path / "fallback.png"

    report = scene.render(str(output_path), allow_placeholder=True)

    assert output_path.exists()
    assert report.status == "ok"
    assert report.supported_features["mapscene.render_png"] == "supported"
    assert scene.last_render_backend == "placeholder"


def test_render_geotiff_terrain_reaches_native_offscreen_adapter(tmp_path, monkeypatch):
    terrain_path = tmp_path / "terrain.tif"
    terrain_path.write_bytes(b"fixture")

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=str(terrain_path),
            crs="EPSG:32610",
            metadata={"width": 2, "height": 2},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=500.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=8, height=6, format="png"),
    )

    class FakeDem:
        data = np.arange(4, dtype=np.float32).reshape(2, 2)

    import forge3d.io as io

    monkeypatch.setattr(io, "load_dem", lambda *args, **kwargs: FakeDem())

    class FakeSession:
        def __init__(self, *, window=False):
            pass

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(*_args, **_kwargs):
            return "ibl"

    class FakeParams:
        def __init__(self, config):
            self.config = config

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 1] = 127
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        seen_heightmap = None

        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, *, heightmap, **_kwargs):
            FakeTerrainRenderer.seen_heightmap = heightmap
            return FakeFrame()

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    output_path = tmp_path / "geotiff.png"
    scene.render(str(output_path))

    assert output_path.exists()
    assert scene.last_render_backend == "gpu_terrain"
    assert FakeTerrainRenderer.seen_heightmap is not None
    assert FakeTerrainRenderer.seen_heightmap.dtype == np.float32
