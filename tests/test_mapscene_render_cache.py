"""GPU-free MapScene render-cache key and store contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d as f3d


def _scene(path: Path, *, azimuth_deg: float = 35.0) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"source_id": "mapscene-cache-test"},
        ),
        camera=f3d.OrbitCamera(
            target=(0.0, 0.0, 0.0), distance=75.0, azimuth_deg=azimuth_deg
        ),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=8, height=8, format="png", path=str(path)),
    )


@pytest.fixture
def fake_renderer(monkeypatch, tmp_path) -> list[float]:
    """Mock only the native boundary; run the real adapter and cache store."""
    import hashlib
    import forge3d.map_scene as ms
    import forge3d.anamnesis as anamnesis
    import forge3d._anamnesis_scene as adapter
    from forge3d._native import get_native_module

    plain_key = getattr(get_native_module(), "_anamnesis_plain_key", None)

    calls: list[float] = []
    environment = tmp_path / "unit-environment.hdr"
    environment.write_bytes(b"unit environment")

    class Resources:
        def anamnesis_resource_fingerprint(self, ibl):
            return hashlib.sha256(Path(ibl).read_bytes()).digest()

    def render_native(recipe, compiled, **_kwargs):
        calls.append(float(recipe.camera.azimuth_deg))
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., 0] = int(round(recipe.camera.azimuth_deg)) % 256
        rgba[..., 1] = hashlib.sha256(ms._load_native_heightmap(recipe.terrain).tobytes()).digest()[0]
        rgba[..., 3] = 255
        return ms._MapSceneNativeRenderResult(rgba=rgba)

    monkeypatch.setattr(ms, "_render_native_offscreen_result", render_native)
    monkeypatch.setattr(ms, "_native_ibl_path", lambda recipe: (environment, False))
    monkeypatch.setattr(ms, "_terrain_resources_for_recipe", lambda recipe: (
        Resources(), recipe.lighting.settings["ibl"]["path"]))
    monkeypatch.setattr(adapter, "_reject_gpu_errors", lambda: None)
    monkeypatch.setattr(anamnesis, "get_native_module", lambda: SimpleNamespace(
        _anamnesis_plain_key=plain_key,
        capabilities=lambda: {"backend": "unit-cpu", "granted": [], "limits": {},
                              "dx12_compiler": "unit", "naga_capabilities": []}))
    return calls


def test_output_destination_change_hits_and_writes_requested_target(
    fake_renderer, tmp_path
):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = _scene(first_path)
    second = _scene(second_path)
    cache = tmp_path / "cache"

    first.render(cache=cache)
    second.render(cache=cache)

    assert fake_renderer == [35.0]
    assert first.last_render_metadata["cache_hit"] is False
    assert second.last_render_metadata["cache_hit"] is True
    assert second.last_render_path == str(second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_prior_effects_off_label_visibility_cache_is_invalidated(
    monkeypatch, fake_renderer, tmp_path
):
    import forge3d.anamnesis as anamnesis

    model = _scene(tmp_path / "frame.png")
    cache = tmp_path / "cache"
    render_sequence = anamnesis.render_sequence

    def previous_visibility_namespace(recipe, **kwargs):
        from forge3d._canonical_json import canonical_json_bytes
        from test_chronos_flythrough import _path_free_scene
        legacy_recipe = _path_free_scene(recipe)
        def legacy_render(_recipe, _frame):
            recipe._render_impl(str(recipe.recipe.output.path))
            return Path(recipe.recipe.output.path).read_bytes()
        return render_sequence(
            legacy_recipe, **kwargs, render_frame=legacy_render,
            render_frame_fingerprint=b"forge3d.python.mapscene.render/v2",
            render_frame_context=canonical_json_bytes(legacy_recipe, error_context="legacy render inputs"),
        )

    with monkeypatch.context() as patch:
        patch.setattr(anamnesis, "render_sequence", previous_visibility_namespace)
        model.render(cache=cache)
    model.render(cache=cache)
    assert model.last_render_metadata["cache_hit"] is False
    assert fake_renderer == [35.0, 35.0]
    model.render(cache=cache)
    assert model.last_render_metadata["cache_hit"] is True
    assert fake_renderer == [35.0, 35.0]


def test_secondary_file_byte_change_misses_styled_raster_cache(fake_renderer, tmp_path):
    from test_raster_styles import bivariate
    scene = _scene(tmp_path / "frame.png")
    primary, secondary = tmp_path / "primary.npy", tmp_path / "secondary.npy"
    np.save(primary, np.zeros((4, 4), np.float32))
    np.save(secondary, np.zeros((4, 4), np.float32))
    scene.recipe.layers = [f3d.RasterOverlay("thematic", path=primary, secondary_path=secondary,
        style=bivariate(), crs="EPSG:32610")]
    cache = tmp_path / "cache"
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is False
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    np.save(secondary, np.full((4, 4), 30, np.float32))
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is False
    assert fake_renderer == [35.0, 35.0]


def test_different_scenes_share_cache_root_without_thrashing(fake_renderer, tmp_path):
    output = tmp_path / "shared.png"
    first = _scene(output, azimuth_deg=35.0)
    second = _scene(output, azimuth_deg=73.0)
    cache = tmp_path / "cache"
    hits = []
    blobs = []

    for scene in (first, second, first, second):
        scene.render(cache=cache)
        hits.append(scene.last_render_metadata["cache_hit"])
        blobs.append(output.read_bytes())

    assert fake_renderer == [35.0, 73.0]
    assert hits == [False, False, True, True]
    assert blobs[0] != blobs[1]
    assert blobs[0] == blobs[2]
    assert blobs[1] == blobs[3]
    assert len([path for path in cache.iterdir() if path.is_dir()]) == 2


def test_context_ignores_destination_but_changes_with_camera(
    monkeypatch, fake_renderer, tmp_path
):
    scenes = [
        _scene(tmp_path / "first.png"),
        _scene(tmp_path / "second.png"),
        _scene(tmp_path / "third.png", azimuth_deg=73.0),
    ]
    captured = [_capture_context(monkeypatch, scene, tmp_path / "out.png") for scene in scenes]
    assert captured[0] == captured[1]
    assert captured[0] != captured[2]
    for context in captured:
        assert all(name not in context["recipe"]["output"] for name in ("path", "directory", "filename"))
    assert fake_renderer == []


def test_camera_change_after_hit_misses_and_replaces_pixels(fake_renderer, tmp_path):
    output = tmp_path / "frame.png"
    scene = _scene(output)
    cache = tmp_path / "cache"

    scene.render(cache=cache)
    original_bytes = output.read_bytes()
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True

    scene.recipe = replace(
        scene.recipe, camera=replace(scene.recipe.camera, azimuth_deg=73.0)
    )
    scene.render(cache=cache)
    changed_bytes = output.read_bytes()
    assert scene.last_render_metadata["cache_hit"] is False
    assert changed_bytes != original_bytes
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    assert output.read_bytes() == changed_bytes
    assert fake_renderer == [35.0, 73.0]


def _capture_context(monkeypatch, scene: f3d.MapScene, output: Path) -> dict:
    import forge3d._anamnesis_scene as adapter

    class ContextCaptured(Exception):
        pass

    captured = []

    def capture(self, declare, frames):
        captured.append(self.base_states[0])
        raise ContextCaptured

    with monkeypatch.context() as patch:
        patch.setattr(adapter._SceneExecution, "declare", capture)
        with pytest.raises(ContextCaptured):
            scene.render(str(output), cache=output.parent / "cache")
    assert len(captured) == 1
    return captured[0]


def test_frame_free_context_keys_actual_owned_inputs(monkeypatch, fake_renderer, tmp_path):
    import hashlib

    scene = _scene(tmp_path / "frame.png")
    context = _capture_context(monkeypatch, scene, tmp_path / "frame.png")
    expected = scene.recipe.to_dict()
    expected["output"].pop("path", None)
    assert context["recipe"] == expected
    assert context["frame"] is None
    assert context["height"]["sha256"] == hashlib.sha256(scene.recipe.terrain.data.tobytes()).hexdigest()
    assert context["native_resources"] == hashlib.sha256(b"unit environment").hexdigest()


def test_compiled_chronos_frame_changes_context_for_same_recipe(monkeypatch, fake_renderer, tmp_path):
    import forge3d.map_scene as ms

    output = tmp_path / "frame.png"
    scene = _scene(output)
    plan = scene.compile_plan()
    monkeypatch.setattr(ms, "_chronos_label_plans", lambda frame: {})
    contexts = []
    for frame_json in ('{"frame":0}', '{"frame":1}'):
        scene.compiled_plan = replace(
            plan, frame=SimpleNamespace(to_json=lambda value=frame_json: value)
        )
        contexts.append(_capture_context(monkeypatch, scene, output))
    assert contexts[0] != contexts[1]
    assert contexts[0]["frame"] == {"frame": 0}
    assert contexts[1]["frame"] == {"frame": 1}


@pytest.mark.parametrize("source", ["array", "file"])
def test_same_shape_height_edit_misses_and_matches_fresh(fake_renderer, tmp_path, source):
    scene = _scene(tmp_path / "frame.png")
    height_path = tmp_path / "height.npy"
    if source == "file":
        np.save(height_path, scene.recipe.terrain.data)
        scene.recipe.terrain = replace(scene.recipe.terrain, path=height_path, data=None)
    cache = tmp_path / "cache"
    scene.render(cache=cache)
    first = Path(scene.last_render_path).read_bytes()
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    if source == "file":
        np.save(height_path, values)
    else:
        scene.recipe.terrain.data[:] = values
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is False
    changed = Path(scene.last_render_path).read_bytes()
    assert changed != first
    scene.render(cache=None)
    assert Path(scene.last_render_path).read_bytes() == changed
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    assert fake_renderer == [35.0, 35.0, 35.0]


def test_blank_batch_templates_preserve_camera_and_scalar_types(fake_renderer, tmp_path):
    from forge3d._anamnesis_scene import _OwnedScene, _SceneExecution, _plain_frame_key, _path_free
    scenes = {index: _scene(tmp_path / f"{index}.png", azimuth_deg=azimuth)
              for index, azimuth in enumerate((35.0, 45.0))}
    first, second = (scene.recipe for scene in scenes.values())
    policy = first.render_policy
    assert _plain_frame_key(first, policy) is not None
    assert _plain_frame_key(first, policy) == _plain_frame_key(second, policy)
    adapter = _SceneExecution(scenes, list(scenes), tmp_path, cache_requested=True)
    for index, scene in scenes.items():
        assert isinstance(adapter.scenes[index], _OwnedScene)
        assert adapter.scenes[index]._scene is None
        assert scene.recipe.terrain.data.flags.writeable
        assert adapter.base_states[index]["recipe"] == _path_free(scene.recipe)
    first.terrain.metadata["type_sensitive"] = True
    second.terrain.metadata["type_sensitive"] = 1
    assert _plain_frame_key(first, policy) != _plain_frame_key(second, policy)
    adapter = _SceneExecution(scenes, list(scenes), tmp_path, cache_requested=True)
    assert adapter.base_states[0]["recipe"]["terrain"]["metadata"]["type_sensitive"] is True
    assert type(adapter.base_states[1]["recipe"]["terrain"]["metadata"]["type_sensitive"]) is int


def test_blank_batch_rechecks_changed_validation_inputs(fake_renderer, tmp_path):
    from forge3d._anamnesis_scene import _SceneExecution
    scenes = {index: _scene(tmp_path / f"{index}.png") for index in (0, 1)}
    scenes[1].recipe.diagnostics_policy = {"gpu_memory_budget_bytes": 0}
    scenes[1].render_policy = "fail_on_warning"
    with pytest.raises(RuntimeError, match="blocked by diagnostics"):
        _SceneExecution(scenes, list(scenes), tmp_path, cache_requested=True)


def test_warm_key_material_does_not_bypass_blob_verification(fake_renderer, tmp_path):
    import json
    import forge3d.anamnesis as an
    scene = _scene(tmp_path / "frame.png")
    cache = tmp_path / "cache"
    scene.render(cache=cache)
    store = next(value for value in an._STORE_POOL.values() if value.root.parent == cache)
    first = dict(store._key_material)
    if isinstance(an.get_native_module()._anamnesis_plain_key, type(len)):
        assert first
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    if first:  # Native-only optimization; Python-only runtimes use the full path.
        assert store._key_material.keys() == first.keys()
        assert all(value is first[key] for key, value in store._key_material.items())
    key = json.loads((store.root / "last_manifest.json").read_text())["entries"]["0:frame.output"]
    from anamnesis_packed_store import write_blob
    write_blob(store.root, key, b"corrupt")
    with pytest.raises(RuntimeError, match="prediction mismatch"):
        scene.render(cache=cache)


def test_scene_constructor_hooks_keep_the_ordinary_path(fake_renderer, monkeypatch, tmp_path):
    from forge3d._anamnesis_scene import _OwnedScene, _SceneExecution
    import forge3d.map_scene as ms
    scenes = {index: _scene(tmp_path / f"{index}.png") for index in (0, 1)}
    constructor = ms.MapScene.__init__
    calls = []
    def observed(self, *args, **kwargs):
        calls.append(self)
        constructor(self, *args, **kwargs)
    monkeypatch.setattr(ms.MapScene, "__init__", observed)
    adapter = _SceneExecution(scenes, list(scenes), tmp_path, cache_requested=True)
    assert len(calls) == len(scenes)
    assert all(not isinstance(scene, _OwnedScene) for scene in adapter.scenes.values())


@pytest.mark.parametrize("source", ["array", "npy", "geotiff", "geotiff-worldfile", "geotiff-aux"])
@pytest.mark.parametrize("metadata", [{}, {"width": 4, "height": 4},
                                      {"resolution": [2.0, 3.0]}],
                         ids=["no-size", "size", "resolution"])
def test_cached_terrain_inputs_match_fresh(fake_renderer, monkeypatch, tmp_path, source, metadata):
    """Observe the production span/grid/sample boundary, without a GPU."""
    import hashlib
    import json
    import forge3d.map_scene as ms

    scene = _scene(tmp_path / "frame.png")
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = tmp_path / ("dem.tif" if source.startswith("geotiff") else "dem.npy")
    external_metadata = source in {"geotiff-worldfile", "geotiff-aux"}
    if source.startswith("geotiff"):
        import rasterio
        from rasterio.transform import from_origin
        assert not getattr(rasterio, "__forge3d_stub__", False), "real rasterio is required"
        values[0, 0] = -9999.0
        options = ({"PROFILE": "BASELINE", "TFW": "YES"} if source == "geotiff-worldfile" else
                   {"PROFILE": "BASELINE"} if source == "geotiff-aux" else {})
        with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1,
                           dtype="float32", crs="EPSG:32610", nodata=-9999.0,
                           transform=from_origin(500000.0, 4100064.0, 30.0, 30.0), **options) as target:
            target.write(values, 1)
        if external_metadata:
            assert path.with_suffix(".tif.aux.xml").exists()
            assert path.with_suffix(".tfw").exists() == (source == "geotiff-worldfile")
    elif source == "npy":
        np.save(path, values)
    scene.recipe.terrain = replace(scene.recipe.terrain, metadata=dict(metadata, source_id="terrain-input-contract"),
        data=values if source == "array" else None, path=None if source == "array" else path,
        nodata_policy="preserve")
    inputs = []

    def render_native(recipe, compiled, **kwargs):
        height = ms._load_native_heightmap(recipe.terrain)
        grid = ms._terrain_alignment_grid(recipe.terrain,
            target_crs=recipe.target_crs or recipe.terrain.crs, fallback_shape=height.shape)
        if grid is not None:
            grid = dict(grid, transform=list(grid["transform"]))
        observed = {"span": max(1.0, ms._terrain_scene_diagonal(recipe.terrain)),
                    "grid": grid, "shape": list(height.shape),
                    "samples": hashlib.sha256(height.tobytes()).hexdigest(),
                    "nodata_policy": recipe.terrain.nodata_policy}
        inputs.append(observed)
        digest = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).digest()
        rgba = np.tile(np.frombuffer(digest[:3] + b"\xff", np.uint8), (8, 8, 1))
        return ms._MapSceneNativeRenderResult(rgba=rgba)

    monkeypatch.setattr(ms, "_render_native_offscreen_result", render_native)
    scene.render(cache=None)
    fresh = Path(scene.last_render_path).read_bytes()
    scene.render(cache=tmp_path / "cache")
    assert scene.last_render_metadata["cache_hit"] is False
    if external_metadata:
        assert scene.last_render_metadata["bypass_reason"] == "scene_terrain_external_metadata"
    assert inputs[0] == inputs[1]
    assert Path(scene.last_render_path).read_bytes() == fresh
    scene.render(cache=tmp_path / "cache")
    assert scene.last_render_metadata["cache_hit"] is (not external_metadata)
    assert len(inputs) == (3 if external_metadata else 2)
    if external_metadata:
        assert inputs[0] == inputs[2]
        assert not (tmp_path / "cache").exists()
    assert Path(scene.last_render_path).read_bytes() == fresh


@pytest.mark.parametrize("change", ["size", "georeferencing"])
def test_equal_terrain_samples_with_different_extent_miss(fake_renderer, tmp_path, change):
    scene = _scene(tmp_path / "frame.png")
    values = scene.recipe.terrain.data.copy()
    path = tmp_path / "dem.tif"

    def write_grid(x):
        import rasterio
        from rasterio.transform import from_origin
        assert not getattr(rasterio, "__forge3d_stub__", False), "real rasterio is required"
        with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1,
                           dtype="float32", crs="EPSG:32610",
                           transform=from_origin(x, 4100064.0, 30.0, 30.0)) as target:
            target.write(values, 1)

    if change == "georeferencing":
        write_grid(500000.0)
        scene.recipe.terrain = replace(scene.recipe.terrain, data=None, path=path, metadata={})
    cache = tmp_path / "cache"
    scene.render(cache=cache)
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    if change == "georeferencing":
        write_grid(500030.0)
    else:
        scene.recipe.terrain.metadata = {"width": 8, "height": 4, "resolution": [2.0, 3.0]}
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is False
    assert len(fake_renderer) == 2


def test_elevation_mutations_invalidate_plan_and_callback_cache(fake_renderer,tmp_path):
    scene = _scene(tmp_path / "frame.png")
    cache = tmp_path / "cache"
    scene.render(cache=cache)
    original = scene.compiled_plan.recipe_hash
    scene.recipe.terrain.data[0,0] = 3
    scene.render(cache=cache)
    assert scene.compiled_plan.recipe_hash != original
    assert scene.last_render_metadata["cache_hit"] is False
    second = scene.compiled_plan.recipe_hash
    scene.recipe = replace(scene.recipe,terrain=replace(scene.recipe.terrain,data=np.ones((4,4),np.float32)))
    scene.render(cache=cache)
    assert scene.compiled_plan.recipe_hash != second
    assert scene.last_render_metadata["cache_hit"] is False
    scene.render(path=str(tmp_path / "other.png"),cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    assert len(fake_renderer)==3


def test_terrain_identity_canonical_zero_nan_payloads_and_location():
    identity = lambda a: f3d.TerrainSource(data=a).to_dict()
    assert identity(np.array([[0.]], np.float32)) == identity(np.array([[-0.]], np.float32))
    a = np.array([[0x7fc00001, 0x3f800000]], dtype=np.uint32).view(np.float32)
    b = np.array([[0x7fffffff, 0x3f800000]], dtype=np.uint32).view(np.float32)
    assert identity(a) == identity(b)
    assert identity(a) != identity(a[:, ::-1])
    assert identity(np.array([[np.inf]])) != identity(np.array([[-np.inf]]))


def test_nan_hole_terrain_compiles_and_renders(tmp_path):
    scene = _scene(tmp_path / "nan.png")
    scene.recipe.terrain.data[1, 1] = np.nan
    scene.compile_plan()
    scene.render()
    assert (tmp_path / "nan.png").is_file()
