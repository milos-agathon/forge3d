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


def test_render_blocks_symbolic_scene_without_native_terrain(tmp_path):
    output_path = tmp_path / "blocked-placeholder.png"
    scene = _supported_scene()

    with pytest.raises(f3d.MapSceneNativeUnavailable) as excinfo:
        scene.render(str(output_path))

    assert not output_path.exists(), "a blocked render must not write placeholder pixels"
    block = excinfo.value.diagnostic
    assert block["status"] == "diagnostic_block"
    assert block["layer"] == "terrain"
    assert block["required_native"]
    assert scene.last_validation_report is not None
    assert scene.last_render_backend is None


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
    if not terrain_rendering_available():
        pytest.skip("16-bit MapScene render requires a terrain-capable GPU runtime")
    output_path = tmp_path / "map-16bit.png"
    scene = _native_asset_scene(tmp_path)
    scene.recipe.output.path = str(output_path)
    scene.recipe.output.bit_depth = 16

    report = scene.render()

    assert _png_ihdr(output_path) == (48, 32, 16, 6)
    assert report.supported_features["output.bit_depth.16"] == "supported"
    assert report.supported_features["mapscene.render_png_16bit"] == "supported"
    assert scene.last_render_metadata["bit_depth"] == 16


def test_render_uses_output_spec_path_for_supported_png(tmp_path):
    if not terrain_rendering_available():
        pytest.skip("MapScene render requires a terrain-capable GPU runtime")
    output_path = tmp_path / "from-output-spec.png"
    scene = _native_asset_scene(tmp_path)
    scene.recipe.output.path = str(output_path)

    report = scene.render()

    assert output_path.exists()
    assert report.status == "ok"
    assert scene.last_validation_report is not None
    assert scene.last_render_path == str(output_path)


def test_render_output_changes_when_source_data_changes(tmp_path):
    if not terrain_rendering_available():
        pytest.skip("MapScene render requires a terrain-capable GPU runtime")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "render.png"
    second_path = second_dir / "render.png"
    first_scene = _native_asset_scene(first_dir)
    second_scene = _native_asset_scene(second_dir)

    # Change the second scene's raster overlay source pixels on disk.
    raster = np.zeros((32, 48, 4), dtype=np.uint8)
    raster[..., 0] = 220
    raster[..., 3] = 255
    save_png_deterministic(second_dir / "ortho.png", raster)

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
    # SUTURA: the source-derived CPU fallback renderer no longer exists.
    assert not hasattr(map_scene, "_render_source_derived_rgba")
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

    with pytest.raises(f3d.MapSceneNativeUnavailable) as excinfo:
        scene.render(str(output_path))

    assert not output_path.exists(), "a blocked render must not write placeholder pixels"
    blocks = excinfo.value.diagnostics
    assert {block["layer"] for block in blocks} == {"roads", "labels"}
    for block in blocks:
        assert block["status"] == "diagnostic_block"
        assert block["required_native"]


def test_public_label_vector_render_is_deterministic_and_not_placeholder(tmp_path):
    if not terrain_rendering_available():
        pytest.skip("real MapScene label/vector render requires a terrain-capable GPU runtime")

    paths = [tmp_path / f"frame-{index}.png" for index in range(4)]
    scenes = [_public_label_vector_scene(path) for path in paths]
    reports = [scene.render() for scene in scenes]

    assert all(report.status == "ok" for report in reports)
    assert all(scene.last_render_backend == "gpu_terrain" for scene in scenes)
    accepted = [
        scene.compiled_label_plans["labels"].to_dict()["accepted"]
        for scene in scenes
    ]
    assert accepted == [accepted[0]] * len(accepted)
    image_bytes = [path.read_bytes() for path in paths]
    assert image_bytes == [image_bytes[0]] * len(image_bytes)

    rgba = f3d.png_to_numpy(paths[-1])
    opaque_magenta = np.all(rgba == np.array([255, 0, 255, 255], dtype=np.uint8), axis=2)
    assert not np.any(opaque_magenta), (
        "repeated MapScene renders must not expose Metal's timestamp-corruption marker"
    )

    # SUTURA: the CPU placeholder renderer no longer exists at all.
    assert not hasattr(map_scene, "_render_source_derived_rgba")


def test_gpu_backend_uses_native_msdf_text_for_labels(tmp_path, monkeypatch):
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
    calls: dict[str, object] = {"glyphs": 0, "rects": []}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

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
            calls["rects"].append(tuple(args))

        def render_rgba(self):
            assert self.base is not None
            return self.base

    import forge3d._map_scene_render as render_helpers

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
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
    assert calls["atlas_channels"] == 3
    assert calls["glyphs"] == 2
    assert calls["last_glyph_arg_count"] == 17

    from forge3d.text_atlas import default_latin_atlas_paths, load_atlas_metrics

    metrics = load_atlas_metrics(default_latin_atlas_paths()[1])
    accepted = scene.compiled_label_plans["labels"].accepted[0]
    positioned = [glyph for glyph in accepted.positioned_glyphs if glyph["has_outline"]]
    anchor_x, anchor_y = map_scene._render_label_anchor(accepted, 80, 64)
    render_size = 12.0
    atlas_scale = render_size / metrics["font_size"]
    expected = []
    for item in positioned:
        glyph = metrics["glyphs_by_id"][f'{item["font_index"]}:{item["glyph_id"]}']
        expected.append((
            anchor_x + item["origin"][0] * render_size + glyph["ox"] * atlas_scale,
            anchor_y + item["origin"][1] * render_size + glyph["oy"] * atlas_scale,
            glyph["w"] * atlas_scale,
            glyph["h"] * atlas_scale,
            glyph["x"] / metrics["width"],
            glyph["y"] / metrics["height"],
            (glyph["x"] + glyph["w"]) / metrics["width"],
            (glyph["y"] + glyph["h"]) / metrics["height"],
        ))
    rects = calls["rects"]
    assert len(rects) == len(expected)
    for actual, expected_rect in zip(rects, expected):
        assert actual[:8] == pytest.approx(expected_rect)


def test_native_label_compositor_uses_explicit_positioned_line_ranges(monkeypatch):
    layer = f3d.LabelLayer(
        layer_id="labels",
        labels=[{
            "id": "wrapped",
            "kind": "point",
            "text": "ABCD",
            "line_ranges": [[0, 2], [2, 4]],
            "geometry": {"type": "Point", "coordinates": (20.0, 12.0, 0.0)},
        }],
        glyph_atlas={"glyphs": list("ABCD")},
    )
    plan = f3d.LabelPlan.compile(
        labels=layer.labels,
        camera={},
        viewport=(80, 64),
        glyph_atlas=layer.glyph_atlas,
    )
    assert plan.accepted
    accepted = plan.accepted[0]
    assert list(accepted.line_ranges) == [(0, 2), (2, 4)]
    assert sorted({glyph["line_index"] for glyph in accepted.positioned_glyphs}) == [0, 1]

    calls = {"rects": []}

    class FakeNativeTextScene:
        def __init__(self, *_args):
            pass

        def disable_terrain(self):
            pass

        def set_raster_overlay(self, *_args):
            pass

        def set_native_text_atlas(self, *_args):
            pass

        def enable_native_text(self):
            pass

        def add_native_text_rect_uv_halo(self, *args):
            calls["rects"].append(args)

        def render_rgba(self):
            return np.zeros((64, 80, 4), dtype=np.uint8)

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeTextScene)
    recipe = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(data=np.zeros((2, 2), dtype=np.float32)),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=80, height=64),
        layers=[layer],
    )

    map_scene._composite_native_label_layers(
        np.zeros((64, 80, 4), dtype=np.uint8), recipe, {"labels": plan}
    )

    assert len(calls["rects"]) == len(
        [glyph for glyph in accepted.positioned_glyphs if glyph["has_outline"]]
    )
    line_y = {
        glyph["line_index"]: glyph["origin"][1]
        for glyph in accepted.positioned_glyphs
        if glyph["has_outline"]
    }
    assert line_y[1] > line_y[0]
    rendered_y = {round(rect[1], 6) for rect in calls["rects"]}
    assert len(rendered_y) >= 2


def test_native_label_compositor_rejects_missing_shaped_identity(monkeypatch):
    layer = f3d.LabelLayer(
        layer_id="labels",
        labels=[{
            "id": "ligature",
            "kind": "point",
            "text": "office",
            "geometry": {"type": "Point", "coordinates": (20.0, 12.0, 0.0)},
        }],
        glyph_atlas={"glyphs": list("office")},
    )
    plan = f3d.LabelPlan.compile(
        labels=layer.labels,
        camera={},
        viewport=(80, 64),
        glyph_atlas=layer.glyph_atlas,
    )
    assert plan.accepted
    original = plan.accepted[0].positioned_glyphs[0]
    plan.accepted[0].positioned_glyphs = ({**original, "glyph_id": 65535},)

    class FakeNativeTextScene:
        def __init__(self, *_args):
            pass

        def disable_terrain(self):
            pass

        def set_raster_overlay(self, *_args):
            pass

        def set_native_text_atlas(self, *_args):
            pass

        def enable_native_text(self):
            pass

        def add_native_text_rect_uv_halo(self, *_args):
            raise AssertionError("missing shaped identity must fail before drawing")

        def render_rgba(self):
            return np.zeros((64, 80, 4), dtype=np.uint8)

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeTextScene)
    recipe = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(data=np.zeros((2, 2), dtype=np.float32)),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=80, height=64),
        layers=[layer],
    )

    with pytest.raises(f3d.MapSceneTextLayoutError) as caught:
        map_scene._composite_native_label_layers(
            np.zeros((64, 80, 4), dtype=np.uint8), recipe, {"labels": plan}
        )

    assert caught.value.diagnostic["reason"] == "missing_shaped_glyph"
    assert caught.value.diagnostic["identity"] == "0:65535"


def test_native_label_compositor_rejects_late_missing_identity_before_any_append(monkeypatch):
    from forge3d.text_atlas import default_latin_atlas_paths, load_atlas_metrics

    metrics = load_atlas_metrics(default_latin_atlas_paths()[1])
    glyph_ids = [key.split(":") for key in sorted(metrics["glyphs_by_id"])[:2]]
    assert len(glyph_ids) == 2
    positioned = [
        {
            "font_index": int(font_index),
            "glyph_id": int(glyph_id),
            "line_index": 0,
            "cluster": index,
            "origin": [float(index), 0.0],
            "advance": [1.0, 0.0],
            "has_outline": True,
        }
        for index, (font_index, glyph_id) in enumerate(glyph_ids)
    ]
    positioned[-1] = {**positioned[-1], "glyph_id": 65535}
    candidate = {
        "candidate_id": "word:center",
        "candidate_type": "center",
        "anchor": [20.0, 12.0, 0.0],
        "score": 0.0,
        "bounds": [20.0, 12.0, 44.0, 24.0],
        "terrain_sample": {},
        "details": {},
        "ordering_key": "word:center",
    }
    plan = f3d.LabelPlan.from_dict({
        "payload_version": 2,
        "accepted": [{
            "label_id": "word",
            "source_id": "word",
            "text": "Map",
            "geometry_type": "Point",
            "candidate": candidate,
            "candidates": [candidate],
            "priority_class": "default",
            "screen_bounds": [20.0, 12.0, 44.0, 24.0],
            "world_bounds": [20.0, 12.0, 0.0, 44.0, 24.0, 0.0],
            "typography": {
                "render_mapping": "positioned_glyphs_by_id",
                "font_sha256": metrics["font_sha256"],
            },
            "glyphs": ["M", "a"],
            "positioned_glyphs": positioned,
            "ordering_key": "word",
        }],
        "rejected": [],
        "diagnostics": [],
        "rationale": [],
    })
    layer = f3d.LabelLayer(layer_id="labels", labels=[])
    calls = {"appends": 0}

    class FakeNativeTextScene:
        def __init__(self, *_args):
            pass

        def disable_terrain(self):
            pass

        def set_raster_overlay(self, *_args):
            pass

        def set_native_text_atlas(self, *_args):
            pass

        def enable_native_text(self):
            pass

        def add_native_text_rect_uv_halo(self, *_args):
            calls["appends"] += 1

        def render_rgba(self):
            raise AssertionError("late missing glyph must fail before rendering")

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeTextScene)
    recipe = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(data=np.zeros((2, 2), dtype=np.float32)),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=80, height=64),
        layers=[layer],
    )

    with pytest.raises(f3d.MapSceneTextLayoutError) as caught:
        map_scene._composite_native_label_layers(
            np.zeros((64, 80, 4), dtype=np.uint8), recipe, {"labels": plan}
        )

    assert caught.value.diagnostic["reason"] == "missing_shaped_glyph"
    assert calls["appends"] == 0


@pytest.mark.parametrize("bad_origin", ([float("nan"), 0.0], ["bad", 0.0]))
def test_native_label_compositor_rejects_late_invalid_rect_before_any_append(
    monkeypatch, bad_origin
):
    layer = f3d.LabelLayer(
        layer_id="labels",
        labels=[{
            "id": "word",
            "kind": "point",
            "text": "AB",
            "geometry": {"type": "Point", "coordinates": (20.0, 12.0, 0.0)},
        }],
        glyph_atlas={"glyphs": list("AB")},
    )
    plan = f3d.LabelPlan.compile(
        labels=layer.labels,
        camera={},
        viewport=(80, 64),
        glyph_atlas=layer.glyph_atlas,
    )
    positioned = [dict(glyph) for glyph in plan.accepted[0].positioned_glyphs]
    positioned[-1]["origin"] = bad_origin
    plan.accepted[0].positioned_glyphs = tuple(positioned)
    calls = {"appends": 0}

    class FakeNativeTextScene:
        def __init__(self, *_args):
            pass

        def disable_terrain(self):
            pass

        def set_raster_overlay(self, *_args):
            pass

        def set_native_text_atlas(self, *_args):
            pass

        def enable_native_text(self):
            pass

        def add_native_text_rect_uv_halo(self, *_args):
            calls["appends"] += 1

        def render_rgba(self):
            raise AssertionError("invalid positioned glyph must fail before rendering")

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeTextScene)
    recipe = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(data=np.zeros((2, 2), dtype=np.float32)),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=80, height=64),
        layers=[layer],
    )

    with pytest.raises(f3d.MapSceneTextLayoutError) as caught:
        map_scene._composite_native_label_layers(
            np.zeros((64, 80, 4), dtype=np.uint8), recipe, {"labels": plan}
        )

    assert caught.value.diagnostic["reason"] == "invalid_text_rect"
    assert calls["appends"] == 0


@pytest.mark.parametrize("glyph_patch", ({"w": 0}, {"x": 10_000}))
def test_native_label_compositor_rejects_invalid_atlas_uv_before_any_append(
    monkeypatch, glyph_patch
):
    from forge3d import text_atlas

    real_load_atlas_metrics = text_atlas.load_atlas_metrics
    atlas_metrics = real_load_atlas_metrics(text_atlas.default_latin_atlas_paths()[1])

    def load_bad_metrics(path):
        metrics = real_load_atlas_metrics(path)
        identity = sorted(metrics["glyphs_by_id"])[0]
        metrics["glyphs_by_id"] = {
            key: ({**value, **glyph_patch} if key == identity else value)
            for key, value in metrics["glyphs_by_id"].items()
        }
        metrics["glyphs"] = {
            key: ({**value, **glyph_patch} if key == identity else value)
            for key, value in metrics["glyphs"].items()
        }
        return metrics

    monkeypatch.setattr(text_atlas, "load_atlas_metrics", load_bad_metrics)
    font_index, glyph_id = (int(part) for part in sorted(atlas_metrics["glyphs_by_id"])[0].split(":"))
    candidate = {
        "candidate_id": "word:center",
        "candidate_type": "center",
        "anchor": [20.0, 12.0, 0.0],
        "score": 0.0,
        "bounds": [20.0, 12.0, 44.0, 24.0],
        "terrain_sample": {},
        "details": {},
        "ordering_key": "word:center",
    }
    plan = f3d.LabelPlan.from_dict({
        "payload_version": 2,
        "accepted": [{
            "label_id": "word",
            "source_id": "word",
            "text": "A",
            "geometry_type": "Point",
            "candidate": candidate,
            "candidates": [candidate],
            "priority_class": "default",
            "screen_bounds": [20.0, 12.0, 44.0, 24.0],
            "world_bounds": [20.0, 12.0, 0.0, 44.0, 24.0, 0.0],
            "typography": {
                "render_mapping": "positioned_glyphs_by_id",
                "font_sha256": atlas_metrics["font_sha256"],
            },
            "glyphs": ["A"],
            "positioned_glyphs": [{
                "font_index": font_index,
                "glyph_id": glyph_id,
                "line_index": 0,
                "cluster": 0,
                "origin": [0.0, 0.0],
                "advance": [1.0, 0.0],
                "has_outline": True,
            }],
            "ordering_key": "word",
        }],
        "rejected": [],
        "diagnostics": [],
        "rationale": [],
    })
    layer = f3d.LabelLayer(layer_id="labels", labels=[])
    calls = {"appends": 0}

    class FakeNativeTextScene:
        def __init__(self, *_args):
            pass

        def disable_terrain(self):
            pass

        def set_raster_overlay(self, *_args):
            pass

        def set_native_text_atlas(self, *_args):
            pass

        def enable_native_text(self):
            pass

        def add_native_text_rect_uv_halo(self, *_args):
            calls["appends"] += 1

        def render_rgba(self):
            raise AssertionError("invalid atlas UV must fail before rendering")

    monkeypatch.setattr(map_scene, "_native_scene_class", lambda: FakeNativeTextScene)
    recipe = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(data=np.zeros((2, 2), dtype=np.float32)),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=80, height=64),
        layers=[layer],
    )

    with pytest.raises(f3d.MapSceneTextLayoutError) as caught:
        map_scene._composite_native_label_layers(
            np.zeros((64, 80, 4), dtype=np.uint8), recipe, {"labels": plan}
        )

    assert caught.value.diagnostic["reason"] == "invalid_text_rect"
    assert calls["appends"] == 0


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
                        "properties": {"class": "primary", "width": 3},
                    },
                    {
                        "id": "road-2",
                        "geometry": {"type": "LineString", "coordinates": [(1.0, 0.0), (0.0, 1.0)]},
                        "properties": {"class": "secondary", "width": 5},
                    }
                ],
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "roads",
                            "type": "line",
                            "paint": {
                                "line-color": ["match", ["get", "class"], "primary", "#00ff00", "#ff0000"],
                                "line-width": ["get", "width"],
                            },
                        }
                    ],
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
    assert calls["vector"]["stroke_width"] == [3.0, 5.0]
    assert calls["vector"]["polyline_rgba"] == [(0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
    polyline = calls["vector"]["polylines"][0]
    assert polyline[0] == (-1.0, 1.0)
    assert polyline[-1] == (1.0, -1.0)
    assert calls["composite"] == ((64, 80, 4), (64, 80, 4), False)


def test_gpu_backend_uses_native_oit_for_inline_point_cloud_layer(tmp_path, monkeypatch):
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
            f3d.PointCloudLayer(
                layer_id="sample-points",
                crs="EPSG:32610",
                point_count=2,
                metadata={
                    "positions": [[0.0, 0.0, 0.0], [10.0, 5.0, 1.0]],
                    "bounds": [0.0, 0.0, 10.0, 5.0],
                    "color": "#ff8000",
                    "point_size": 6.0,
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
        return np.zeros((height, width, 4), dtype=np.uint8)

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    report = scene.render(str(tmp_path / "point-cloud.png"))

    assert calls["vector"]["size"] == (80, 64)
    assert calls["vector"]["points_xy"] == [(-1.0, 1.0), (1.0, -1.0)]
    assert calls["vector"]["point_rgba"] == [(1.0, pytest.approx(128 / 255), 0.0, pytest.approx(220 / 255))] * 2
    assert calls["vector"]["point_size"] == [6.0, 6.0]
    assert calls["vector"]["polylines"] is None
    assert report.supported_features["point_cloud.mapscene_render"] == "supported"
    assert scene.last_render_metadata["point_cloud_backend"] == "native_oit_points"


def test_gpu_backend_uses_edl_oit_for_point_cloud_layer(tmp_path, monkeypatch):
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
            f3d.PointCloudLayer(
                layer_id="edl-points",
                crs="EPSG:32610",
                point_count=2,
                metadata={
                    "positions": [[0.0, 0.0, 0.0], [10.0, 5.0, 1.0]],
                    "bounds": [0.0, 0.0, 10.0, 5.0],
                    "point_size": 6.0,
                    "shading": "edl",
                    "edl_strength": 2.25,
                    "edl_radius_px": 3.0,
                },
            )
        ],
    )
    calls: dict[str, object] = {}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fail_plain_oit(*_args, **_kwargs):
        raise AssertionError("plain OIT path used for EDL point cloud")

    def fake_edl_render(width, height, *, points_xy=None, point_size=None, edl_strength=None, edl_radius_px=None, **_kwargs):
        calls["edl"] = {
            "size": (width, height),
            "points_xy": points_xy,
            "point_size": point_size,
            "edl_strength": edl_strength,
            "edl_radius_px": edl_radius_px,
        }
        return np.zeros((height, width, 4), dtype=np.uint8)

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fail_plain_oit)
    monkeypatch.setattr(f3d, "vector_render_oit_edl_py", fake_edl_render, raising=False)

    report = scene.render(str(tmp_path / "point-cloud-edl.png"))

    assert calls["edl"]["size"] == (80, 64)
    assert calls["edl"]["points_xy"] == [(-1.0, 1.0), (1.0, -1.0)]
    assert calls["edl"]["point_size"] == [6.0, 6.0]
    assert calls["edl"]["edl_strength"] == 2.25
    assert calls["edl"]["edl_radius_px"] == 3.0
    assert report.supported_features["point_cloud.edl"] == "supported"
    assert scene.last_render_metadata["point_cloud_edl_backend"] == "weighted_oit_depth_edl"


def test_gpu_backend_uses_pointcloud_dataset_for_file_backed_layer(tmp_path, monkeypatch):
    point_path = tmp_path / "points.laz"
    point_path.write_bytes(b"LASF")
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
            f3d.PointCloudLayer(
                layer_id="sample-points",
                path=str(point_path),
                crs="EPSG:32610",
                point_count=2,
                metadata={"point_size": 5.0, "point_budget": 2},
            )
        ],
    )
    calls: dict[str, object] = {}

    class FakeBounds:
        min = (100.0, 200.0, 0.0)
        max = (110.0, 205.0, 1.0)

    class FakePointData:
        positions = np.asarray([[100.0, 200.0, 0.0], [110.0, 205.0, 1.0]], dtype=np.float32)
        colors = np.asarray([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)

    class FakeDataset:
        bounds = FakeBounds()

        def read_points(self, *, budget=None):
            calls["budget"] = budget
            return FakePointData()

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_vector_render(width, height, *, points_xy=None, point_rgba=None, point_size=None, **_kwargs):
        calls["vector"] = {
            "size": (width, height),
            "points_xy": points_xy,
            "point_rgba": point_rgba,
            "point_size": point_size,
        }
        return np.zeros((height, width, 4), dtype=np.uint8)

    import forge3d.pointcloud as pointcloud_mod

    def fake_open_pointcloud(path):
        calls["path"] = Path(path)
        return FakeDataset()

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(pointcloud_mod, "open_pointcloud", fake_open_pointcloud)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    report = scene.render(str(tmp_path / "point-cloud-file.png"))

    assert calls["path"] == point_path
    assert calls["budget"] == 2
    assert calls["vector"]["points_xy"] == [(-1.0, 1.0), (1.0, -1.0)]
    assert calls["vector"]["point_rgba"] == [
        (1.0, 0.0, 0.0, pytest.approx(220 / 255)),
        (0.0, pytest.approx(128 / 255), 1.0, pytest.approx(220 / 255)),
    ]
    assert calls["vector"]["point_size"] == [5.0, 5.0]
    assert report.supported_features["point_cloud.mapscene_render"] == "supported"


def test_gpu_backend_uses_pnts_tiles3d_layer_as_oit_points(tmp_path, monkeypatch):
    pnts_path = tmp_path / "tile.pnts"
    pnts_path.write_bytes(b"pnts")
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
            f3d.Tiles3DLayer(
                layer_id="points-tile",
                source={"path": str(pnts_path), "source_format": "pnts"},
                metadata={"point_size": 7.0},
            )
        ],
    )
    calls: dict[str, object] = {}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_decode_pnts(data):
        calls["pnts_bytes"] = data
        return {
            "point_count": 2,
            "positions": np.asarray([[0.0, 0.0, 0.0], [2.0, 4.0, 1.0]], dtype=np.float32),
            "colors": np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        }

    def fake_vector_render(width, height, *, points_xy=None, point_rgba=None, point_size=None, **_kwargs):
        calls["vector"] = {
            "size": (width, height),
            "points_xy": points_xy,
            "point_rgba": point_rgba,
            "point_size": point_size,
        }
        return np.zeros((height, width, 4), dtype=np.uint8)

    import forge3d.tiles3d as tiles3d_mod

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(tiles3d_mod, "decode_pnts", fake_decode_pnts)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    report = scene.render(str(tmp_path / "tiles-pnts.png"))

    assert calls["pnts_bytes"] == b"pnts"
    assert calls["vector"]["points_xy"] == [(-1.0, 1.0), (1.0, -1.0)]
    assert calls["vector"]["point_rgba"] == [
        (pytest.approx(10 / 255), pytest.approx(20 / 255), pytest.approx(30 / 255), pytest.approx(220 / 255)),
        (pytest.approx(40 / 255), pytest.approx(50 / 255), pytest.approx(60 / 255), pytest.approx(220 / 255)),
    ]
    assert calls["vector"]["point_size"] == [7.0, 7.0]
    assert report.supported_features["tiles3d.mapscene_render"] == "supported"
    assert scene.last_render_metadata["tiles3d_source_bytes"] == 2 * 3 * 8


def test_gpu_backend_uses_tileset_json_pnts_content_as_oit_points(tmp_path, monkeypatch):
    tileset_path = tmp_path / "tileset.json"
    pnts_path = tmp_path / "child.pnts"
    tileset_path.write_text('{"asset":{"version":"1.0"},"geometricError":0,"root":{"boundingVolume":{"sphere":[0,0,0,1]},"geometricError":0}}')
    pnts_path.write_bytes(b"pnts")
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
            f3d.Tiles3DLayer(
                layer_id="points-tileset",
                source={"path": str(tileset_path), "source_format": "tileset.json"},
                metadata={"point_size": 3.0, "camera_position": [1.0, 2.0, 3.0], "sse_threshold": 8.0},
            )
        ],
    )
    calls: dict[str, object] = {}

    class FakeDataset:
        @classmethod
        def from_tileset_json(cls, path):
            calls["tileset_path"] = Path(path)
            return cls()

        def traverse(self, camera_position, *, sse_threshold=16.0, max_depth=32):
            calls["traverse"] = (camera_position, sse_threshold, max_depth)
            return [{"resolved_path": str(pnts_path)}, {"resolved_path": str(tmp_path / "mesh.b3dm")}]

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_decode_pnts(data):
        calls["pnts_bytes"] = data
        return {
            "point_count": 2,
            "positions": np.asarray([[5.0, 5.0, 0.0], [7.0, 9.0, 1.0]], dtype=np.float32),
            "colors": np.asarray([[90, 80, 70], [60, 50, 40]], dtype=np.uint8),
        }

    def fake_vector_render(width, height, *, points_xy=None, point_rgba=None, point_size=None, **_kwargs):
        calls["vector"] = {
            "size": (width, height),
            "points_xy": points_xy,
            "point_rgba": point_rgba,
            "point_size": point_size,
        }
        return np.zeros((height, width, 4), dtype=np.uint8)

    import forge3d.tiles3d as tiles3d_mod

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(tiles3d_mod, "Tiles3dDataset", FakeDataset)
    monkeypatch.setattr(tiles3d_mod, "decode_pnts", fake_decode_pnts)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    report = scene.render(str(tmp_path / "tileset-pnts.png"))

    assert calls["tileset_path"] == tileset_path
    assert calls["traverse"] == ((1.0, 2.0, 3.0), 8.0, 32)
    assert calls["pnts_bytes"] == b"pnts"
    expected_points = map_scene._project_tiles3d_perspective(
        [[5.0, 5.0, 0.0], [7.0, 9.0, 1.0]],
        {"point_size": 3.0, "camera_position": [1.0, 2.0, 3.0], "sse_threshold": 8.0},
        80,
        64,
    )
    assert calls["vector"]["points_xy"] == expected_points
    assert calls["vector"]["point_size"] == [3.0, 3.0]
    assert report.supported_features["tiles3d.mapscene_render"] == "supported"
    assert scene.last_render_metadata["tiles3d_source_bytes"] == 2 * 3 * 8


def test_gpu_backend_uses_b3dm_tiles3d_layer_as_oit_wireframe(tmp_path, monkeypatch):
    b3dm_path = tmp_path / "tile.b3dm"
    b3dm_path.write_bytes(b"b3dm")
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
            f3d.Tiles3DLayer(
                layer_id="mesh-tile",
                source={"path": str(b3dm_path), "source_format": "b3dm"},
                metadata={"bounds": [0.0, 0.0, 2.0, 2.0], "mesh_width": 4.0, "mesh_color": "#3366ff"},
            )
        ],
    )
    calls: dict[str, object] = {}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_decode_b3dm(data):
        calls["b3dm_bytes"] = data
        return {
            "positions": np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
            "indices": np.asarray([0, 1, 2], dtype=np.uint32),
        }

    def fake_vector_render(width, height, *, points_xy=None, polylines=None, polyline_rgba=None, stroke_width=None, **_kwargs):
        calls["vector"] = {
            "size": (width, height),
            "points_xy": points_xy,
            "polylines": polylines,
            "polyline_rgba": polyline_rgba,
            "stroke_width": stroke_width,
        }
        return np.zeros((height, width, 4), dtype=np.uint8)

    import forge3d.tiles3d as tiles3d_mod

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(tiles3d_mod, "decode_b3dm", fake_decode_b3dm)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    report = scene.render(str(tmp_path / "tiles-b3dm.png"))

    assert calls["b3dm_bytes"] == b"b3dm"
    assert calls["vector"]["points_xy"] is None
    assert calls["vector"]["polylines"] == [[(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 1.0)]]
    assert calls["vector"]["polyline_rgba"] == [(pytest.approx(51 / 255), pytest.approx(102 / 255), 1.0, pytest.approx(230 / 255))]
    assert calls["vector"]["stroke_width"] == [4.0]
    assert report.supported_features["tiles3d.mapscene_render"] == "supported"
    assert scene.last_render_metadata["tiles3d_backend"] == "native_oit_geometry"
    assert scene.last_render_metadata["tiles3d_source_bytes"] == 3 * 3 * 8
    assert "point_cloud_backend" not in scene.last_render_metadata


def test_gpu_backend_uses_tileset_json_b3dm_content_as_oit_wireframe(tmp_path, monkeypatch):
    tileset_path = tmp_path / "tileset.json"
    b3dm_path = tmp_path / "child.b3dm"
    tileset_path.write_text('{"asset":{"version":"1.0"},"geometricError":0,"root":{"boundingVolume":{"sphere":[0,0,0,1]},"geometricError":0}}')
    b3dm_path.write_bytes(b"b3dm")
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
            f3d.Tiles3DLayer(
                layer_id="mesh-tileset",
                source={"path": str(tileset_path), "source_format": "tileset.json"},
                metadata={"bounds": [0.0, 0.0, 2.0, 2.0], "camera_position": [1.0, 1.0, 4.0]},
            )
        ],
    )
    calls: dict[str, object] = {}

    class FakeDataset:
        @classmethod
        def from_tileset_json(cls, path):
            calls["tileset_path"] = Path(path)
            return cls()

        def traverse(self, camera_position, *, sse_threshold=16.0, max_depth=32):
            calls["traverse"] = (camera_position, sse_threshold, max_depth)
            return [{"resolved_path": str(b3dm_path)}, {"resolved_path": str(tmp_path / "points.pnts.ignore")}]

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_decode_b3dm(data):
        calls["b3dm_bytes"] = data
        return {
            "positions": np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
            "indices": np.asarray([0, 1, 2], dtype=np.uint32),
        }

    def fake_vector_render(width, height, *, polylines=None, stroke_width=None, **_kwargs):
        calls["vector"] = {"size": (width, height), "polylines": polylines, "stroke_width": stroke_width}
        return np.zeros((height, width, 4), dtype=np.uint8)

    import forge3d.tiles3d as tiles3d_mod

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(tiles3d_mod, "Tiles3dDataset", FakeDataset)
    monkeypatch.setattr(tiles3d_mod, "decode_b3dm", fake_decode_b3dm)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    report = scene.render(str(tmp_path / "tileset-b3dm.png"))

    assert calls["tileset_path"] == tileset_path
    assert calls["traverse"] == ((1.0, 1.0, 4.0), 16.0, 32)
    assert calls["b3dm_bytes"] == b"b3dm"
    expected_vertices = map_scene._project_tiles3d_perspective(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        {"bounds": [0.0, 0.0, 2.0, 2.0], "camera_position": [1.0, 1.0, 4.0]},
        80,
        64,
    )
    assert calls["vector"]["polylines"] == [[
        expected_vertices[0],
        expected_vertices[1],
        expected_vertices[2],
        expected_vertices[0],
    ]]
    assert calls["vector"]["stroke_width"] == [2.0]
    assert report.supported_features["tiles3d.mapscene_render"] == "supported"
    assert scene.last_render_metadata["tiles3d_backend"] == "native_oit_geometry"
    assert scene.last_render_metadata["tiles3d_source_bytes"] == 3 * 3 * 8


def test_gpu_backend_precise_polygon_path_uses_data_driven_fill_and_outline(tmp_path, monkeypatch):
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
                layer_id="zones",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "zone-1",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4), (0.1, 0.1)]],
                        },
                        "properties": {"zone": "park", "opacity": 0.5, "stroke": 2},
                    },
                    {
                        "id": "zone-2",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9), (0.6, 0.6)]],
                        },
                        "properties": {"zone": "industrial", "opacity": 0.75, "stroke": 4},
                    },
                ],
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "zones-fill",
                            "type": "fill",
                            "paint": {
                                "fill-color": ["match", ["get", "zone"], "park", "#00ff00", "#ff0000"],
                                "fill-opacity": ["get", "opacity"],
                            },
                        },
                        {
                            "id": "zones-outline",
                            "type": "line",
                            "paint": {
                                "line-color": ["match", ["get", "zone"], "park", "#0000ff", "#ffff00"],
                                "line-width": ["get", "stroke"],
                            },
                        },
                    ],
                },
            )
        ],
    )
    calls: dict[str, object] = {"polygons": None, "lines": None}

    def fake_terrain(_recipe, _heightmap):
        rgba = np.zeros((64, 80, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    def fake_polygon_render(width, height, exteriors, holes=None, fill_rgba=None, stroke_rgba=None, stroke_width=None, fill_rgba_list=None, coordinates_are_ndc=None):
        calls["polygons"] = {
            "size": (width, height),
            "exterior_count": len(exteriors),
            "holes": holes,
            "fill_rgba": fill_rgba,
            "fill_rgba_list": fill_rgba_list,
            "coordinates_are_ndc": coordinates_are_ndc,
        }
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        return rgba

    def fake_vector_render(width, height, *, points_xy=None, point_rgba=None, point_size=None, polylines=None, polyline_rgba=None, stroke_width=None):
        calls["lines"] = {
            "size": (width, height),
            "polylines": polylines,
            "polyline_rgba": polyline_rgba,
            "stroke_width": stroke_width,
        }
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        return rgba

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    monkeypatch.setattr(f3d, "vector_render_polygons_fill_py", fake_polygon_render)
    monkeypatch.setattr(f3d, "vector_render_oit_py", fake_vector_render)

    output_path = tmp_path / "polygon-choropleth.png"
    scene.render(str(output_path))

    assert output_path.exists()
    assert calls["polygons"]["size"] == (80, 64)
    assert calls["polygons"]["exterior_count"] == 2
    assert calls["polygons"]["coordinates_are_ndc"] is True
    fills = calls["polygons"]["fill_rgba_list"]
    assert fills[0][:3] == (0.0, 1.0, 0.0)
    assert fills[0][3] == pytest.approx(128 / 255)
    assert fills[1][:3] == (1.0, 0.0, 0.0)
    assert fills[1][3] == pytest.approx(191 / 255)
    assert calls["lines"]["polyline_rgba"] == [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    assert calls["lines"]["stroke_width"] == [2.0, 4.0]


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
    render_metadata = dict(scene.last_render_metadata)
    assert render_metadata.pop("offline_accumulation_ms") >= 0.0
    assert render_metadata.pop("timing_source") == "python_perf_counter"
    assert render_metadata == {
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

    with pytest.raises(f3d.MapSceneNativeUnavailable) as excinfo:
        scene.render(str(output_path))

    assert not output_path.exists()
    block = excinfo.value.diagnostic
    assert block["status"] == "diagnostic_block"
    assert block["layer"] == "terrain"
    assert "TerrainRenderer" in block["required_native"]
    assert scene.last_render_backend is None


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
