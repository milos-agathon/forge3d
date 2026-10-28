"""Production scene dispatch and independent native pixel comparisons."""
from dataclasses import replace
import json
import os

import numpy as np
import pytest

import forge3d as f3d
from forge3d.anamnesis import render_sequence


def test_reference_dry_run_does_not_open_the_content_store(tmp_path):
    root = tmp_path / "unopened"
    result = render_sequence(
        {"terrain": {}, "output": {}}, frames=[0], cache=root, dry_run=True,
        capabilities={"backend": "unit-cpu", "granted": [], "limits": {}},
    )
    assert not root.exists()
    assert result.predicted_recompute and not result.observed_recompute


@pytest.mark.parametrize("change", ["directory", "contents", "working_directory", "file_appears", "file_disappears"])
def test_default_material_identity_rechecks_sources(tmp_path, monkeypatch, change):
    from forge3d._anamnesis_scene import _default_material_identity

    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        (directory / "assets/materials").mkdir(parents=True)
    monkeypatch.chdir(first)
    monkeypatch.delenv("FORGE3D_MATERIAL_DIR", raising=False)
    texture = first / "assets/materials/rock_albedo.png"
    if change != "file_appears":
        texture.write_bytes(b"first texture")
    before = _default_material_identity()
    assert _default_material_identity() == before
    if change == "directory":
        monkeypatch.setenv("FORGE3D_MATERIAL_DIR", str(second / "assets/materials"))
    elif change == "working_directory":
        monkeypatch.chdir(second)
    elif change == "file_disappears":
        texture.unlink()
    else:
        texture.write_bytes(b"changed texture")
    assert _default_material_identity() != before


def test_snapshot_status_and_plain_key_do_not_invoke_unknown_callbacks():
    from forge3d._anamnesis_scene import _snapshot_recipes
    from forge3d._native import get_native_module
    from test_chronos_flythrough import _synthetic_scene
    native = get_native_module()
    source = _synthetic_scene().recipe
    owned, callback_free = _snapshot_recipes({0: source}, with_status=True)
    assert callback_free is True
    assert owned[0] is not source
    class Hook(dict):
        def __deepcopy__(self, memo):
            return dict(self)
    source.terrain.metadata = Hook(source.terrain.metadata)
    _, callback_free = _snapshot_recipes({0: source}, with_status=True)
    assert callback_free is False
    class Trap:
        def __repr__(self):
            raise AssertionError("plain-key probe invoked user repr")
        def __eq__(self, other):
            raise AssertionError("plain-key probe invoked user comparison")
    assert native._anamnesis_plain_key({"unknown": Trap()}) is None
    assert native._anamnesis_plain_key({}, {"unknown": Trap()}) is None
    keys = [native._anamnesis_plain_key({"value": value}) for value in (True, 1, 1.0)]
    assert len(set(keys)) == 3
    assert native._anamnesis_plain_key({"b": 2, "a": 1}) == native._anamnesis_plain_key({"a": 1, "b": 2})
    cycle = []
    cycle.append(cycle)
    assert native._anamnesis_plain_key(cycle) is None


def test_scene_adapter_releases_snapshots_when_callbacks_are_released(tmp_path):
    import gc
    import weakref
    from forge3d._anamnesis_scene import _SceneExecution

    enabled = gc.isenabled()
    gc.disable()  # Prove ownership, independently of cyclic-GC scheduling.
    try:
        adapter = _SceneExecution({}, [], tmp_path, cache_requested=True)
        assert adapter._anamnesis_scene_adapter is adapter
        callbacks = adapter.executors
        reference = weakref.ref(adapter)
        del adapter
        assert reference() is not None, "callbacks must retain their execution state"
        del callbacks
        assert reference() is None, "completed adapters must not retain frame snapshots in a cycle"
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_scene_snapshots_preserve_aliases_cycles_arrays_and_copy_hooks(monkeypatch):
    import copy
    import forge3d.map_scene as ms
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    recipe = _synthetic_scene().recipe
    shared = [1.0]
    shared.append(shared)
    tuple_child = []
    tuple_cycle = (tuple_child,)
    tuple_child.append(tuple_cycle)
    recipe.terrain.metadata = {"left": shared, "right": shared, "camera": recipe.camera, "tuple_cycle": tuple_cycle}
    recipe.camera.target = [0.0, 0.0, 0.0]
    snapshots = _snapshot_recipes({0: recipe, 1: recipe})
    for snapshot in snapshots.values():
        metadata = snapshot.terrain.metadata
        assert metadata["left"] is metadata["right"]
        assert metadata["left"][1] is metadata["left"]
        assert metadata["camera"] is snapshot.camera
        assert metadata["tuple_cycle"][0][0] is metadata["tuple_cycle"]
        assert metadata["tuple_cycle"] is not tuple_cycle
        assert snapshot.camera.target is not recipe.camera.target
        assert snapshot.terrain.data is not recipe.terrain.data
        assert snapshot.terrain.data.tobytes() == recipe.terrain.data.tobytes()
    snapshots[0].terrain.metadata["left"][0] = 2.0
    snapshots[0].terrain.data[0, 0] += 1.0
    assert snapshots[1].terrain.metadata["left"][0] == shared[0] == 1.0
    assert snapshots[1].terrain.data.tobytes() == recipe.terrain.data.tobytes()

    called = []
    def custom_copy(value, memo):
        called.append(value.distance)
        return ms.OrbitCamera(distance=value.distance + 1.0)
    monkeypatch.setattr(ms.OrbitCamera, "__deepcopy__", custom_copy, raising=False)
    hooked = _snapshot_recipes({0: recipe})[0]
    assert called and hooked.camera.distance == recipe.camera.distance + 1.0
    assert hooked.camera.distance == copy.deepcopy(recipe).camera.distance


def test_snapshot_preparation_without_native_preserves_frame_isolation(monkeypatch):
    import forge3d._native as native
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    recipe = _synthetic_scene().recipe
    monkeypatch.setattr(native, "get_native_module", lambda: None)
    snapshots = _snapshot_recipes({0: recipe, 1: recipe})
    assert snapshots[0] is not snapshots[1] and snapshots[0] is not recipe
    assert snapshots[0].terrain.data is not snapshots[1].terrain.data
    assert np.array_equal(snapshots[0].terrain.data, recipe.terrain.data)


def test_native_snapshot_preserves_state_aliases_and_deepcopy_protocols():
    import copy
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    class HookedList(list):
        def __deepcopy__(self, memo):
            result = type(self)([copy.deepcopy(self[0], memo), "hooked"])
            memo[id(self)] = result
            return result

    class Reduced:
        def __init__(self, value):
            self.value = value

        def __reduce_ex__(self, protocol):
            return type(self), (self.value,)

    recipe = _synthetic_scene().recipe
    recipe.camera.target = [0.0, 0.0, 0.0]
    recipe.terrain.metadata = {
        "state": recipe.camera.__dict__,
        "list": HookedList([recipe.camera]),
        "reduced": Reduced(recipe.camera),
        "objects": np.array([recipe.camera], dtype=object),
    }
    expected = copy.deepcopy(recipe)
    actual = _snapshot_recipes({0: recipe, 1: recipe})
    for frame in actual.values():
        metadata = frame.terrain.metadata
        # Python reconstructs an object's __dict__ when a custom hook first
        # copies it; compare that behavior rather than requiring dict identity.
        assert (metadata["state"] is frame.camera.__dict__) == (
            expected.terrain.metadata["state"] is expected.camera.__dict__)
        assert metadata["state"]["target"] is frame.camera.target
        assert metadata["list"][0] is frame.camera
        assert metadata["list"][1] == expected.terrain.metadata["list"][1] == "hooked"
        assert metadata["reduced"].value is frame.camera
        assert metadata["objects"][0] is frame.camera
        assert frame.camera is not recipe.camera
    assert actual[0].camera is not actual[1].camera
    assert actual[0].terrain.metadata["objects"] is not actual[1].terrain.metadata["objects"]


def test_native_snapshot_honors_copyreg_and_dictionary_mutation_errors(monkeypatch):
    import copy
    import copyreg
    import forge3d.map_scene as ms
    from forge3d._anamnesis_scene import _snapshot_recipes
    from forge3d._native import get_native_module
    from test_chronos_flythrough import _synthetic_scene

    recipe = _synthetic_scene().recipe
    monkeypatch.setitem(copyreg.dispatch_table, ms.OrbitCamera,
                        lambda camera: (ms.OrbitCamera, (), {"distance": camera.distance + 2.0}))
    expected = copy.deepcopy(recipe)
    assert _snapshot_recipes({0: recipe})[0].camera.distance == expected.camera.distance

    class MutatesParent:
        def __init__(self, parent):
            self.parent = parent

        def __deepcopy__(self, memo):
            self.parent["added"] = None
            return None

    for copier in (lambda value: get_native_module()._anamnesis_snapshot_recipes({0: value}, ()), copy.deepcopy):
        value = {}
        value["hook"] = MutatesParent(value)
        with pytest.raises(RuntimeError, match="dictionary changed size during iteration"):
            copier(value)


def test_native_snapshot_propagates_copy_errors_and_recursion_limit():
    import copy
    import sys
    from forge3d._native import get_native_module

    snapshot = get_native_module()._anamnesis_snapshot_recipes

    class Broken:
        def __deepcopy__(self, memo):
            raise ValueError("copy hook failed")

    for copier in (lambda value: snapshot({0: value}, ()), copy.deepcopy):
        with pytest.raises(ValueError, match="copy hook failed"):
            copier([Broken()])
    root = node = []
    for _ in range(sys.getrecursionlimit()):
        child = []
        node.append(child)
        node = child
    with pytest.raises(RecursionError):
        snapshot({0: root}, ())
    # Unwinding the recursion guard must leave the next call usable.
    assert snapshot({0: [1.0]}, ()) == {0: [1.0]}


@pytest.mark.parametrize("limit", [200, 400])
def test_native_snapshot_respects_configured_python_recursion_limit(limit):
    import sys
    from forge3d._native import get_native_module

    snapshot = get_native_module()._anamnesis_snapshot_recipes
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        root = node = []
        for _ in range(limit):
            child = []
            node.append(child)
            node = child
        with pytest.raises(RecursionError):
            snapshot({0: root}, ())
        assert snapshot({0: [1.0]}, ()) == {0: [1.0]}
    finally:
        sys.setrecursionlimit(previous)


def test_scene_joins_context_before_synchronous_capability_query(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import time
    import forge3d.anamnesis as an
    import forge3d._anamnesis_scene as adapter_module
    import forge3d._native as native_boundary

    joined = []
    class PendingContext:
        def result(self):
            joined.append(True)
    pending = PendingContext()
    monkeypatch.setattr(native_boundary, "get_native_module", lambda: SimpleNamespace(
        _anamnesis_prepare_context=time.monotonic))
    monkeypatch.setattr(an, "_signature_executor", lambda: SimpleNamespace(submit=lambda fn: pending))
    monkeypatch.setattr(an, "_preload_packed_store", lambda root: nullcontext())
    monkeypatch.setattr(adapter_module, "_scenes", lambda recipe, frames: {0: object()})
    monkeypatch.setattr(adapter_module, "_SceneExecution", lambda *args, **kwargs: SimpleNamespace(
        cache_eligible=True, executors={}, fingerprints={}, contexts={}))
    expected = object()
    def synchronous_capabilities(*args, **kwargs):
        assert joined, "context worker must finish before a native query holds the GIL"
        return expected
    monkeypatch.setattr(an, "render_sequence", synchronous_capabilities)
    assert adapter_module.try_render_sequence(
        object(), [0], tmp_path, dry_run=False, max_bytes=1024,
        verify_reads=True, capabilities=None) is expected
    assert getattr(adapter_module._DEFAULT_ENVIRONMENT, "context_future", None) is None


@pytest.mark.parametrize("depth", [64, 128])
def test_native_snapshot_deep_subtrees_preserve_aliases_and_cycles(depth):
    from forge3d._native import get_native_module

    root = node = []
    for _ in range(depth):
        child = []
        node.append(child)
        node = child
    shared = {"value": [1.0]}
    node.extend([root, shared, shared])
    actual = get_native_module()._anamnesis_snapshot_recipes({0: root}, ())[0]
    node = actual
    for _ in range(depth):
        node = node[0]
    assert node[0] is actual
    assert node[1] is node[2] and node[1] is not shared
    assert node[1] == shared


@pytest.mark.parametrize("metadata", [
    {"value": 1.25, "nested": {1: "integer key", None: "optional key"}},
    {"negative_zero": -0.0, "negative_small": -1e-20, "text": '"-0.0"'},
    {"unicode": "é", "sequence": (None, False, 1, 1.0, {"value": 2.0})},
])
def test_scene_uniform_encoder_preserves_exact_canonical_bytes(metadata):
    from forge3d._anamnesis_scene import _SceneExecution, _path_free
    from forge3d._canonical_json import canonical_json_bytes
    from test_chronos_flythrough import _synthetic_scene

    recipe = _synthetic_scene().recipe
    recipe.terrain.metadata.update(metadata)
    state = {"frame": 250, "state": {"recipe": _path_free(recipe), "frame": None,
                                    "height": {"shape": [64, 64], "dtype": "float32", "sha256": "01"},
                                    "environment": {"VF_COLOR_DEBUG_MODE": None}}}
    adapter = object.__new__(_SceneExecution)
    adapter.fast_uniforms = True
    for label in ("terrain.render", "frame.output", "label.compile"):
        assert adapter.encode_uniforms(label, state, error_context="parity") == canonical_json_bytes(state, error_context="parity")
    adapter.fast_uniforms = False
    custom = {"frame": 250, "state": {True: {None: "custom serializer keys"}}}
    assert adapter.encode_uniforms("terrain.render", custom, error_context="parity") == canonical_json_bytes(custom, error_context="parity")
    adapter.fast_uniforms = True
    for invalid in (float("nan"), float("inf"), float("-inf")):
        state["state"]["recipe"]["terrain"]["metadata"]["invalid"] = invalid
        with pytest.raises(ValueError, match="parity does not permit NaN or infinite floats"):
            adapter.encode_uniforms("terrain.render", state, error_context="parity")


def _scenes(text):
    from test_chronos_flythrough import _synthetic_scene

    scenes = {frame: _synthetic_scene() for frame in (0, 1)}
    scenes[1].recipe = replace(scenes[1].recipe, camera=replace(scenes[1].recipe.camera, azimuth_deg=45.0), layers=(
        f3d.LabelLayer(
            layer_id="labels", labels=[{"id": "peak", "text": text, "geometry": {"type": "Point", "coordinates": (32.0, 32.0, 0.0)}}],
            glyph_atlas={"glyphs": sorted(set("PeakHill"))}, occlusion="none",
        ),
    ))
    return scenes


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="native scene acceptance requires a physical GPU")
@pytest.mark.parametrize("frozen", [False, True])
def test_public_scene_sequence_reuses_native_terrain_for_label_edit(tmp_path, monkeypatch, frozen):
    import forge3d.map_scene as ms
    from forge3d._png import decode_png

    original, modified = _scenes("Peak"), _scenes("Hill")
    if frozen:
        from forge3d.chronos import _prepare_frame_scene, _prepare_frozen_frame, _accepted_labels
        for scenes in (original, modified):
            for index, scene in list(scenes.items()):
                scene = _prepare_frame_scene(scene, index, 7, 1, tmp_path / f"unused-{index}.png")
                events = {key: (None, None) for key in _accepted_labels(scene)}
                _prepare_frozen_frame(scene, index, 7, 1, label_events=events)
                scenes[index] = scene
    phases, calls = ["original"], []
    native_render = ms._render_native_offscreen_result

    def observe(recipe, compiled, **kwargs):
        calls.append((phases[0], recipe.camera.azimuth_deg))
        return native_render(recipe, compiled, **kwargs)

    monkeypatch.setattr(ms, "_render_native_offscreen_result", observe)
    seed = render_sequence(original, frames=[0, 1], cache=tmp_path / "shared")
    phases[0] = "incremental"
    incremental = render_sequence(modified, frames=[0, 1], cache=tmp_path / "shared")
    phases[0] = "independent-cold"
    cold = render_sequence(modified, frames=[0, 1], cache=None)
    assert seed.frame_hashes[1] != incremental.frame_hashes[1]
    assert incremental.frame_blobs == cold.frame_blobs
    expected = {(-1, "label.compile"), (1, "label.composite"), (1, "frame.output")}
    assert set(incremental.predicted_recompute) == set(incremental.observed_recompute) == expected
    assert not [call for call in calls if call[0] == "incremental"]
    assert len([call for call in calls if call[0] == "independent-cold"]) == 2
    assert not cold.cache_report.hits and cold.cache_report.bytes_written == 0
    assert all(blob.startswith(b"\x89PNG\r\n\x1a\n") for blob in incremental.frame_blobs)
    assert np.count_nonzero(decode_png(seed.frame_blobs[0])[..., :3]) > 0
    # Independent public MapScene.render follows the complete production path,
    # including native label placement and deterministic PNG encoding.
    phases[0] = "public-cold"
    for frame in (0, 1):
        path = tmp_path / f"public-{frame}.png"
        modified[frame].render(str(path), cache=None)
        assert path.read_bytes() == incremental.frame_blobs[frame]

    if not frozen:
        # Reusing the text compositor must not retain old glyphs or output
        # dimensions. Independent uncached renders use fresh native scenes.
        for size, text in ((32, "P"), (80, "Peak"), (64, "Hill")):
            resized = _scenes(text)[1]
            resized.recipe = replace(resized.recipe, output=replace(resized.recipe.output, width=size, height=size))
            cached = render_sequence(resized, frames=[0], cache=tmp_path / "resized")
            fresh = render_sequence(resized, frames=[0], cache=None)
            assert cached.frame_blobs == fresh.frame_blobs
            if size == 32:
                assert (-1, "label.compile") not in cached.observed_recompute
            else:
                assert (0, "label.composite") in cached.observed_recompute
        compose = ms._composite_native_label_layers
        def fail_after_composition(*args, **kwargs):
            compose(*args, **kwargs)
            raise RuntimeError("text compositor failure control")
        monkeypatch.setattr(ms, "_composite_native_label_layers", fail_after_composition)
        with pytest.raises(RuntimeError, match="text compositor failure control"):
            render_sequence(_scenes("Peak")[1], frames=[0], cache=tmp_path / "failed")
        monkeypatch.setattr(ms, "_composite_native_label_layers", compose)
        recovered = render_sequence(_scenes("P")[1], frames=[0], cache=tmp_path / "recovered")
        assert recovered.frame_blobs == render_sequence(_scenes("P")[1], frames=[0], cache=None).frame_blobs

        # Same-path image and metrics edits must retire decoded atlas values,
        # even if the filesystem timestamps are restored after the edit.
        from PIL import Image
        from forge3d.text_atlas import default_latin_atlas_paths
        atlas_png, atlas_json = default_latin_atlas_paths()
        image_path, metrics_path = tmp_path / "atlas.png", tmp_path / "atlas.json"
        image_path.write_bytes(atlas_png.read_bytes())
        metrics_path.write_bytes(atlas_json.read_bytes())
        atlas_scene = _scenes("Peak")[1]
        layer = replace(atlas_scene.recipe.layers[0],
                        glyph_atlas={"image_path": str(image_path), "metrics_path": str(metrics_path)},
                        typography={"font_size": 12.0})
        atlas_scene.recipe = replace(atlas_scene.recipe, layers=(layer,))
        previous = render_sequence(atlas_scene, frames=[0], cache=tmp_path / "atlas-cache")
        assert (0, "label.composite") in previous.observed_recompute
        for path in (image_path, metrics_path):
            stamp = path.stat()
            if path == image_path:
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert("RGBA")).copy()
                pixels[..., :3] = 255 - pixels[..., :3]
                Image.fromarray(pixels).save(path)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["font_size"] *= 1.25
                path.write_text(json.dumps(data), encoding="utf-8")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            changed = render_sequence(atlas_scene, frames=[0], cache=tmp_path / "atlas-cache")
            independent = render_sequence(atlas_scene, frames=[0], cache=None)
            assert changed.frame_blobs == independent.frame_blobs
            assert changed.frame_hashes != previous.frame_hashes
            assert (0, "label.composite") in changed.observed_recompute
            previous = changed


def test_recognized_unsupported_scene_executes_fresh_instead_of_reference(tmp_path, monkeypatch):
    """A CPU dispatch fixture; this test is not physical rendering evidence."""
    from forge3d.helpers.offscreen import save_png_deterministic
    from test_chronos_flythrough import _synthetic_scene

    scene = _synthetic_scene(output_overrides={"samples": 2})
    calls = []
    expected = np.full((64, 64, 4), 255, dtype=np.uint8)

    def render(self, path, **kwargs):
        calls.append(path)
        save_png_deterministic(path, expected)

    monkeypatch.setattr(f3d.MapScene, "render", render)
    cache = tmp_path / "unused-cache"
    for _ in ("first", "repeat"):
        result = render_sequence(scene, frames=[0], cache=cache, capabilities={"backend": "unit-cpu", "granted": [], "limits": {}})
        assert result.frame_blobs[0].startswith(b"\x89PNG\r\n\x1a\n")
        assert result.observed_recompute == [(0, "frame.output")]
        assert result.bypass_reason == "scene_accumulation"
        assert not result.cache_report.hits and result.cache_report.bytes_written == 0
    assert len(calls) == 2 and not cache.exists()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="600 production scene frames require a physical GPU")
def test_public_scene_600_frame_acceptance(tmp_path, monkeypatch):
    """The public driver uses its built-in native adapter, with no executors."""
    import forge3d.map_scene as ms
    from test_chronos_flythrough import _synthetic_scene
    from forge3d._native import get_native_module

    def scenes(text):
        result = {frame: _synthetic_scene() for frame in range(600)}
        for frame, scene in result.items():
            scene.recipe = replace(scene.recipe, camera=replace(scene.recipe.camera, azimuth_deg=20.0 + frame * 0.25))
        result[250].recipe = replace(result[250].recipe, layers=_scenes(text)[1].recipe.layers)
        return result

    phase, counts = ["original"], dict.fromkeys(("original", "incremental", "cold"), 0)
    native_render = ms._render_native_offscreen_result
    native = get_native_module()
    native.clear_native_degradations()

    def observe(recipe, compiled, **kwargs):
        counts[phase[0]] += 1
        if counts[phase[0]] % 100 == 0:
            print(f"ANAMNESIS public scene {phase[0]} {counts[phase[0]]}/600", flush=True)
        return native_render(recipe, compiled, **kwargs)

    monkeypatch.setattr(ms, "_render_native_offscreen_result", observe)
    original = render_sequence(scenes("Peak"), cache=tmp_path / "shared")
    phase[0] = "incremental"
    incremental = render_sequence(scenes("Hill"), cache=tmp_path / "shared")
    phase[0] = "cold"
    cold = render_sequence(scenes("Hill"), cache=None)
    expected = {(-1, "label.compile"), (250, "label.composite"), (250, "frame.output")}
    result = {
        "matching_frames": sum(a == b for a, b in zip(incremental.frame_blobs, cold.frame_blobs, strict=True)),
        "predicted": sorted(incremental.predicted_recompute),
        "observed": sorted(incremental.observed_recompute),
        "native_calls": counts,
        "cold_seconds": cold.elapsed_seconds,
        "incremental_seconds": incremental.elapsed_seconds,
        "speedup": cold.elapsed_seconds / incremental.elapsed_seconds,
        "adapter": f3d.device_probe(),
        "degradations": native.native_degradations(),
    }
    (tmp_path / "public-scene-600.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if destination := os.environ.get("FORGE3D_ANAMNESIS_SCENE_RESULT_PATH"):
        from pathlib import Path
        Path(destination).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    assert result["matching_frames"] == 600
    assert set(incremental.predicted_recompute) == set(incremental.observed_recompute) == expected
    assert counts == {"original": 600, "incremental": 0, "cold": 600}
    assert not cold.cache_report.hits and cold.cache_report.bytes_written == 0
    assert original.frame_hashes[250] != incremental.frame_hashes[250]
    assert len(set(original.frame_hashes)) > 1, "flythrough must contain visible camera motion"
    assert not [entry for entry in result["degradations"] if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
    assert result["adapter"]["software_fallback"] is False
    assert result["speedup"] >= 20.0


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="fresh-process acceptance requires a physical GPU")
def test_public_scene_600_frame_fresh_process_acceptance(tmp_path):
    from anamnesis_subprocess_acceptance import run_acceptance

    result = run_acceptance(tmp_path)
    phases = result["phases"]
    expected = {(-1, "label.compile"), (250, "label.composite"), (250, "frame.output")}
    assert all(phase["pid"] != os.getpid() for phase in phases.values())
    assert result["matching_frames"] == 600
    assert set(map(tuple, phases["incremental"]["predicted"])) == expected
    assert set(map(tuple, phases["incremental"]["observed"])) == expected
    assert {name: phase["native_calls"] for name, phase in phases.items()} == {"original": 600, "incremental": 0, "cold": 600}
    assert phases["cold"]["hits"] == phases["cold"]["bytes_written"] == 0
    assert phases["original"]["blob_hashes"][250] != phases["incremental"]["blob_hashes"][250]
    assert len(set(phases["original"]["blob_hashes"])) > 1
    for phase in phases.values():
        assert phase["adapter"]["software_fallback"] is False
        assert not [entry for entry in phase["degradations"] if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
        assert phase["frame_hashes"] == phase["blob_hashes"]
    assert result["incremental_seconds"] <= result["cold_seconds"] / 20.0


def test_frozen_frame_exclusions_exactly_match_the_proof_ledger():
    from copy import deepcopy
    from pathlib import Path
    from _toml_compat import load_toml
    from forge3d._anamnesis_scene import _TERRAIN_FRAME_EXCLUSIONS, _terrain_frame_projection

    ledger = load_toml(Path(__file__).with_name("anamnesis_irrelevant_inputs.toml"))
    proofs = {item["field"]: item for item in ledger["terrain_frame_exclusions"]}
    before = {"labels": [1], "label_set_hash": "label", "scene_hash": "scene",
              "frame_seed": 7, "unknown": 9,
              "scene": {"labels": [2], "label_fade_frames": 1, "retained": 3,
                        "scene": {"layers": [4], "terrain": 5}}}
    projected = _terrain_frame_projection(deepcopy(before))

    def leaves(node, path=""):
        if isinstance(node, dict):
            return {key for name, child in node.items() for key in leaves(child, f"{path}.{name}" if path else name)}
        return {path}

    assert leaves(before) - leaves(projected) == set(proofs) == set(_TERRAIN_FRAME_EXCLUSIONS)
    assert all(item["proof"] and item["test"] == "test_frozen_label_fields_do_not_change_native_terrain" for item in proofs.values())
    assert projected == {"frame_seed": 7, "unknown": 9, "scene": {"retained": 3, "scene": {"terrain": 5}}}


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="exclusion proofs require a physical GPU")
@pytest.mark.parametrize("field", ["labels", "label_set_hash", "scene_hash", "scene.labels", "scene.label_fade_frames", "scene.scene.layers"])
def test_frozen_label_fields_do_not_change_native_terrain(tmp_path, field):
    from copy import deepcopy
    from forge3d._anamnesis_scene import _SceneExecution, _terrain_frame_projection
    from forge3d.chronos import _prepare_frame_scene, _prepare_frozen_frame, _accepted_labels
    from forge3d.map_scene import _shared_terrain_render_context

    scene = _prepare_frame_scene(_scenes("Peak")[1], 1, 7, 1, tmp_path / "unused.png")
    events = {key: (None, None) for key in _accepted_labels(scene)}
    frozen = _prepare_frozen_frame(scene, 1, 7, 1, label_events=events)
    original = json.loads(frozen.to_json())
    changed_scene = deepcopy(original["scene"])
    if field in {"labels", "label_set_hash", "scene.labels"}:
        changed_scene["labels"][0]["payload"]["text"] = "Hill"
    elif field == "scene.label_fade_frames":
        changed_scene["label_fade_frames"] += 1
    else:
        changed_scene["scene"]["layers"].append({"kind": "label", "layer_id": "unused", "labels": []})
    # Compile valid frames: derived label/scene hashes must change together
    # with their source. Forged standalone hash edits are correctly rejected.
    changed = f3d.compile_frame(1, 7, 1, json.dumps(original["camera"]), json.dumps(changed_scene))
    revised = json.loads(changed.to_json())

    def at(node, path):
        for part in path.split("."):
            node = node[part]
        return node

    assert at(original, field) != at(revised, field)
    assert _terrain_frame_projection(deepcopy(original)) == _terrain_frame_projection(deepcopy(revised))
    f3d.clear_native_degradations()
    with _shared_terrain_render_context():
        baseline_adapter = _SceneExecution({1: scene}, [1], tmp_path)
        baseline = baseline_adapter.terrain(None, 1, [])
        scene.compiled_plan = replace(scene.compiled_plan, frame=changed)
        changed_adapter = _SceneExecution({1: scene}, [1], tmp_path)
        result = changed_adapter.terrain(None, 1, [])
    assert baseline == result
    assert baseline_adapter.base_states[1] == changed_adapter.base_states[1]
    assert f3d.device_probe()["software_fallback"] is False
    assert np.count_nonzero(np.frombuffer(result, dtype=np.uint8).reshape(64, 64, 4)[..., :3]) > 0


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="default identity parity requires a physical GPU")
def test_default_resource_identity_matches_initialized_renderer(tmp_path):
    import forge3d.map_scene as ms
    from forge3d.determinism import write_canonical_hdr

    hdr = tmp_path / "environment.hdr"
    write_canonical_hdr(hdr)
    resources = ms._TerrainRenderResources(f3d)
    environment = resources.env_maps(str(hdr))
    expected = resources.anamnesis_resource_fingerprint(environment)
    assert resources._renderer is None and resources._session is None
    actual = resources.renderer._anamnesis_resource_fingerprint(resources.material_set, environment)
    assert expected == actual
    resources.renderer.register_material_vt_source(0, "albedo", np.full((248, 248, 4), 255, dtype=np.uint8), (248, 248), None)
    assert resources.anamnesis_resource_fingerprint(environment) is None


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="material preparation ownership requires a physical GPU")
def test_prepared_material_upload_owns_pixels_before_context_wait(tmp_path, monkeypatch):
    from PIL import Image
    import forge3d.map_scene as ms
    from forge3d.determinism import write_canonical_hdr
    from forge3d._native import get_native_module
    native = get_native_module()
    monkeypatch.setenv("FORGE3D_MATERIAL_DIR", str(tmp_path))
    texture = tmp_path / "rock_albedo.png"
    Image.new("RGBA", (13, 7), (255, 0, 0, 255)).save(texture)
    hdr = tmp_path / "environment.hdr"
    write_canonical_hdr(hdr)
    original = ms._TerrainRenderResources(f3d)
    environment = original.env_maps(str(hdr))
    expected = original.anamnesis_resource_fingerprint(environment)
    candidate = ms._TerrainRenderResources(f3d)
    before = dict(native.global_memory_metrics())
    waits = []

    class Ready:
        def result(self):
            waits.append(True)
            # CPU preparation has not allocated a texture or buffer yet.
            assert dict(native.global_memory_metrics()) == before
            Image.new("RGBA", (13, 7), (0, 0, 255, 255)).save(texture)

    actual = candidate.anamnesis_resource_fingerprint(environment, context_ready=Ready())
    assert waits == [True]
    assert actual == expected
    assert candidate.renderer._anamnesis_resource_fingerprint(candidate.material_set, environment) == expected
    changed = ms._TerrainRenderResources(f3d)
    assert changed.anamnesis_resource_fingerprint(environment) != expected


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="material initialization failure requires the native extension")
def test_material_preparation_propagates_context_failure_without_gpu_allocation(tmp_path):
    import forge3d.map_scene as ms
    from forge3d.determinism import write_canonical_hdr
    from forge3d._native import get_native_module
    native = get_native_module()
    hdr = tmp_path / "environment.hdr"
    write_canonical_hdr(hdr)
    resources = ms._TerrainRenderResources(f3d)
    environment = resources.env_maps(str(hdr))
    before = dict(native.global_memory_metrics())
    waits = []

    class Failed:
        def result(self):
            waits.append(True)
            raise RuntimeError("original context initialization failed")

    with pytest.raises(RuntimeError, match="original context initialization failed"):
        resources.anamnesis_resource_fingerprint(environment, context_ready=Failed())
    assert waits == [True]
    assert dict(native.global_memory_metrics()) == before
    assert resources._renderer is None and resources._session is None


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="cached live-validation control requires a physical GPU")
def test_cached_scene_still_runs_live_validation(tmp_path, monkeypatch):
    import forge3d._map_scene_compiler as compiler
    from test_chronos_flythrough import _synthetic_scene

    scene = _synthetic_scene()
    render_sequence(scene, frames=[0], cache=tmp_path / "cache")
    assert scene.compiled_plan is None, "sequence compilation must own its snapshot"
    def reject(*args, **kwargs):
        raise RuntimeError("live validation control")
    monkeypatch.setattr(compiler, "_validate_scene_recipe", reject)
    with pytest.raises(RuntimeError, match="live validation control"):
        render_sequence(scene, frames=[0], cache=tmp_path / "cache")


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="default material identity requires a physical GPU")
@pytest.mark.parametrize("change", ["directory", "contents", "working_directory"])
def test_default_resource_key_tracks_material_sources(tmp_path, monkeypatch, change):
    from PIL import Image
    from test_chronos_flythrough import _synthetic_scene

    first, second = tmp_path / "first", tmp_path / "second"
    for directory, color in ((first, (255, 0, 0, 255)), (second, (0, 0, 255, 255))):
        directory.mkdir()
        Image.new("RGBA", (2, 2), color).save(directory / "rock_albedo.png")
    if change == "working_directory":
        # The native default root is relative to cwd when the variable is absent.
        import shutil
        for directory in (first, second):
            (directory / "assets/materials").mkdir(parents=True)
            shutil.copyfile(directory / "rock_albedo.png", directory / "assets/materials/rock_albedo.png")
        monkeypatch.delenv("FORGE3D_MATERIAL_DIR", raising=False)
        monkeypatch.chdir(first)
    else:
        monkeypatch.setenv("FORGE3D_MATERIAL_DIR", str(first))
    cache = tmp_path / "cache"
    before = render_sequence(_synthetic_scene(), frames=[0], cache=cache)
    repeated = render_sequence(_synthetic_scene(), frames=[0], cache=cache)
    assert repeated.pass_keys == before.pass_keys
    assert not repeated.observed_recompute
    if change == "directory":
        monkeypatch.setenv("FORGE3D_MATERIAL_DIR", str(second))
    elif change == "contents":
        Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(first / "rock_albedo.png")
    else:
        monkeypatch.chdir(second)
    after = render_sequence(_synthetic_scene(), frames=[0], cache=cache)
    assert before.pass_keys["0:terrain.render"] != after.pass_keys["0:terrain.render"]
    assert (0, "terrain.render") in after.observed_recompute
    fresh = render_sequence(_synthetic_scene(), frames=[0], cache=None)
    assert after.frame_blobs == fresh.frame_blobs



@pytest.mark.parametrize("carrier", ["object", "object_array", "copyreg"])
def test_snapshot_hooks_can_change_later_copy_protocol(monkeypatch, carrier):
    import copy
    import copyreg
    import forge3d.map_scene as ms
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    def run(copier):
        with monkeypatch.context() as patch:
            calls = []

            def camera_copy(value, memo):
                result = ms.OrbitCamera(distance=value.distance + 1.0)
                memo[id(value)] = result
                return result

            def install():
                calls.append("install")
                patch.setattr(ms.OrbitCamera, "__deepcopy__", camera_copy, raising=False)

            class Hook:
                def __deepcopy__(self, memo):
                    install()
                    return {}

            class Reduced:
                pass

            def reduce(value):
                install()
                return dict, ()

            if carrier == "copyreg":
                patch.setitem(copyreg.dispatch_table, Reduced, reduce)
                value = Reduced()
            else:
                value = Hook()
                if carrier == "object_array":
                    value = np.array([value], dtype=object)
            recipe = _synthetic_scene().recipe
            recipe.terrain.metadata = {"installer": value}
            snapshots = copier({0: recipe, 1: recipe})
            return [snapshots[frame].camera.distance for frame in (0, 1)], calls

    expected = run(lambda recipes: {frame: copy.deepcopy(recipe) for frame, recipe in recipes.items()})
    assert expected[1] == ["install", "install"]
    assert run(_snapshot_recipes) == expected


def test_snapshot_hooks_observe_later_state_values():
    import copy
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    def run(copier):
        recipe = _synthetic_scene().recipe
        target = recipe.camera.target

        class Edit:
            def __deepcopy__(self, memo):
                recipe.camera.distance += 1.0
                return copy.deepcopy(target, memo)

        recipe.camera.target = Edit()
        snapshots = copier({0: recipe, 1: recipe})
        return [snapshots[frame].camera.distance for frame in (0, 1)]

    expected = run(lambda recipes: {frame: copy.deepcopy(recipe) for frame, recipe in recipes.items()})
    assert expected[1] == expected[0] + 1.0
    assert run(_snapshot_recipes) == expected


def test_snapshot_hooks_preserve_state_dictionary_size_errors():
    import copy
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    for copier in (lambda recipes: {frame: copy.deepcopy(recipe) for frame, recipe in recipes.items()}, _snapshot_recipes):
        recipe = _synthetic_scene().recipe
        target = recipe.camera.target

        class Edit:
            def __deepcopy__(self, memo):
                recipe.camera.added_during_copy = True
                return target

        recipe.camera.target = Edit()
        with pytest.raises(RuntimeError, match="dictionary changed size"):
            copier({0: recipe})


def test_snapshot_respects_modified_container_dispatch(monkeypatch):
    import copy
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    original = copy._deepcopy_dispatch[list]

    def tagged(value, memo):
        result = original(value, memo)
        result.append("custom")
        return result

    recipe = _synthetic_scene().recipe
    recipe.terrain.metadata = {"values": [1]}
    monkeypatch.setitem(copy._deepcopy_dispatch, list, tagged)
    expected = copy.deepcopy(recipe).terrain.metadata["values"]
    assert expected == [1, "custom"]
    assert _snapshot_recipes({0: recipe})[0].terrain.metadata["values"] == expected


def test_snapshot_respects_replaced_deepcopy(monkeypatch):
    import copy
    import forge3d.map_scene as ms
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    original = copy.deepcopy

    def tagged(value, *args):
        result = original(value, *args)
        if isinstance(value, ms.SceneRecipe):
            result.camera.distance += 1.0
        return result

    recipe = _synthetic_scene().recipe
    monkeypatch.setattr(copy, "deepcopy", tagged)
    expected = copy.deepcopy(recipe).camera.distance
    assert expected == recipe.camera.distance + 1.0
    assert _snapshot_recipes({0: recipe})[0].camera.distance == expected


def test_snapshot_fallback_unpacks_mapping_items():
    import copy
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    recipe = _synthetic_scene().recipe

    class Pair(tuple):
        def __iter__(self):
            return iter((0, recipe))

        def __getitem__(self, index):
            raise AssertionError("Python unpacking uses the iterator")

    class Recipes(dict):
        def items(self):
            return [Pair(())]

    source = Recipes()
    expected = {frame: copy.deepcopy(value) for frame, value in source.items()}
    actual = _snapshot_recipes(source)
    assert actual[0].camera.distance == expected[0].camera.distance
    assert actual[0] is not recipe


def test_snapshot_recipe_dictionary_changes_raise_python_error():
    import copy
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    for copier in (lambda recipes: {frame: copy.deepcopy(recipe) for frame, recipe in recipes.items()}, _snapshot_recipes):
        recipe = _synthetic_scene().recipe
        recipes = {0: recipe}

        class Edit:
            def __deepcopy__(self, memo):
                recipes[1] = recipe
                return {}

        recipe.terrain.metadata = {"edit": Edit()}
        with pytest.raises(RuntimeError, match="dictionary changed size"):
            copier(recipes)


@pytest.mark.parametrize("hook", ["__getnewargs__", "__getnewargs_ex__"])
def test_snapshot_preserves_newargs_hooks(monkeypatch, hook):
    import copy
    import forge3d.map_scene as ms
    from forge3d._anamnesis_scene import _snapshot_recipes
    from test_chronos_flythrough import _synthetic_scene

    calls = []

    def newargs(value):
        calls.append(value.distance)
        return ((), {}) if hook == "__getnewargs_ex__" else ()

    monkeypatch.setattr(ms.OrbitCamera, hook, newargs, raising=False)
    recipe = _synthetic_scene().recipe
    expected = copy.deepcopy(recipe)
    expected_calls = calls[:]
    calls.clear()
    actual = _snapshot_recipes({0: recipe})[0]
    assert expected_calls and calls == expected_calls
    assert actual.camera.distance == expected.camera.distance


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="MapScene content identity requires a physical GPU")
@pytest.mark.parametrize("source", ["height_array", "height_file", "height_georef", "no_size_metadata", "sky_file", "styled_raster"])
def test_mapscene_cache_owns_content_and_matches_independent_render(tmp_path, source):
    """Exercise the public method, not a parallel key implementation."""
    import hashlib
    from pathlib import Path
    from PIL import Image
    import forge3d.map_scene as ms
    from test_chronos_flythrough import _synthetic_scene

    f3d.clear_native_degradations()
    scene = _synthetic_scene()
    values = scene.recipe.terrain.data.copy()
    asset = tmp_path / ("environment.hdr" if source == "sky_file" else "input.npy")
    def write_geotiff(x):
        import rasterio
        from rasterio.transform import from_origin
        assert not getattr(rasterio, "__forge3d_stub__", False), "real rasterio is required"
        with rasterio.open(asset, "w", driver="GTiff", width=64, height=64, count=1,
                           dtype="float32", crs="EPSG:32610",
                           transform=from_origin(x, 4100064.0, 1.0, 1.0)) as target:
            target.write(values, 1)

    if source in {"height_file", "no_size_metadata"}:
        np.save(asset, values)
        scene.recipe.terrain = replace(scene.recipe.terrain, data=None, path=asset,
            metadata={} if source == "no_size_metadata" else scene.recipe.terrain.metadata)
    elif source == "height_georef":
        asset = tmp_path / "dem.tif"
        write_geotiff(500000.0)
        scene.recipe.terrain = replace(scene.recipe.terrain, data=None, path=asset,
                                      metadata={"width": 64, "height": 64})
        overlay = tmp_path / "bounded.png"
        Image.fromarray(np.tile(np.array([255, 0, 0, 255], np.uint8), (16, 16, 1))).save(overlay)
        scene.recipe.layers = [f3d.RasterOverlay("bounded", path=overlay, crs="EPSG:32610",
            metadata={"bounds": [500000.0, 4100000.0, 500032.0, 4100064.0]})]
    elif source == "sky_file":
        ms._write_minimal_hdr(asset)
        scene.recipe.lighting = replace(scene.recipe.lighting, settings={"ibl": {"path": str(asset)}})
    elif source == "styled_raster":
        from test_raster_styles import bivariate
        primary = tmp_path / "primary.npy"
        np.save(primary, np.full(values.shape, 1.5, np.float32))
        np.save(asset, np.zeros(values.shape, np.float32))
        scene.recipe.layers = [f3d.RasterOverlay(
            "thematic", path=primary, secondary_path=asset,
            style=bivariate(), crs=scene.recipe.terrain.crs)]

    cache, output = tmp_path / "cache", tmp_path / "cached.png"
    scene.render(str(output), cache=cache)
    original = output.read_bytes()
    assert scene.last_render_metadata["cache_hit"] is False
    scene.render(str(output), cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    assert output.read_bytes() == original

    if source == "height_georef":
        write_geotiff(500032.0)
    elif source.startswith("height") or source == "no_size_metadata":
        changed = np.ascontiguousarray(np.flipud(values) * 1.7, dtype=np.float32)
        if source in {"height_file", "no_size_metadata"}:
            np.save(asset, changed)
        else:
            scene.recipe.terrain.data[:] = changed
    elif source == "sky_file":
        header, payload = asset.read_bytes().rsplit(b"-Y 2 +X 2\n", 1)
        assert len(payload) == 16
        asset.write_bytes(header + b"-Y 2 +X 2\n" + bytes([240, 50, 20, 137]) * 4)
    else:
        np.save(asset, np.full(values.shape, 30.0, np.float32))

    scene.render(str(output), cache=cache)
    assert scene.last_render_metadata["cache_hit"] is False
    revised = output.read_bytes()
    independent = f3d.MapScene(recipe=scene.recipe)
    fresh = tmp_path / "fresh.png"
    independent.render(str(fresh), cache=None)
    from forge3d._anamnesis_scene import _reject_gpu_errors
    _reject_gpu_errors()
    assert revised == fresh.read_bytes()
    assert revised != original, "the production input control must change visible pixels"
    scene.render(str(output), cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    assert output.read_bytes() == revised
    assert np.count_nonzero(np.asarray(Image.open(output))[..., :3]) > 0
    adapter = f3d.device_probe()
    assert adapter["software_fallback"] is False
    errors = [entry for entry in f3d.native_degradations()
              if entry.get("kind") in {"validation_error", "uncaptured_gpu_error"}]
    assert not errors
    (tmp_path / "content-control.json").write_text(json.dumps({
        "source": source, "adapter": adapter, "gpu_errors": errors,
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "revised_sha256": hashlib.sha256(revised).hexdigest(),
        "fresh_sha256": hashlib.sha256(fresh.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")


def test_sequence_rejects_device_loss_before_reusing_native_output(monkeypatch):
    from types import SimpleNamespace
    import forge3d._native as native_boundary
    from forge3d._anamnesis_scene import _reject_gpu_errors
    monkeypatch.setattr(native_boundary, "get_native_module", lambda: SimpleNamespace(
        native_degradations=lambda: [{"kind": "device_lost", "message": "reset"}]))
    with pytest.raises(RuntimeError, match="refuses invalid native GPU output"):
        _reject_gpu_errors()


@pytest.mark.parametrize("frame", [0, 250])
@pytest.mark.parametrize("negative_zero", [False, True])
def test_prepared_native_uniform_fragments_preserve_exact_canonical_bytes(frame, negative_zero):
    from forge3d._anamnesis_scene import _uniform_resource_fragments
    zero = -0.0 if negative_zero else 0.0
    from forge3d._canonical_json import canonical_json_bytes
    state = {"environment": {"native_resources": None, "quoted": '"native_resources":null'},
             "native_resources": None, "height": {"zero": zero},
             "recipe": {"nested": {"native_resources": ""}, "values": [zero, -0.001, "é"]}}
    prefix, suffix = _uniform_resource_fragments(frame, state)
    identity = bytes(range(32)).hex()
    state["native_resources"] = identity
    assert prefix + b'"' + identity.encode("ascii") + b'"' + suffix == canonical_json_bytes(
        {"frame": frame, "state": state}, error_context="test")


def test_completed_resource_future_uses_the_ordinary_warm_declaration_path(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import forge3d._anamnesis_scene as adapter_module
    from test_chronos_flythrough import _synthetic_scene
    resource = Future()
    resource.set_result(b"already-owned")
    environment = tmp_path / "owned.hdr"
    environment.write_bytes(b"owned; no renderer is called in this CPU contract test")
    monkeypatch.setattr(adapter_module, "_default_environment_identity",
                        lambda request: (environment, resource, (environment,)))
    with adapter_module._LazyScratch() as scratch:
        adapter = adapter_module._SceneExecution({0: _synthetic_scene()}, [0], scratch, cache_requested=True)
    assert adapter.base_states[0]["native_resources"] == b"already-owned".hex()
    assert not adapter._pending_resources
    assert not adapter._prepared_uniforms
    assert adapter._semantic_passes is None


@pytest.mark.parametrize("label", ["Peak", "é雪", '{"frame":-1,"state":}'])
def test_prepared_label_payload_matches_independent_canonical_serialization(tmp_path, monkeypatch, label):
    from concurrent.futures import Future
    import forge3d._anamnesis_scene as adapter_module
    from forge3d._canonical_json import canonical_json_bytes
    resource = Future()
    environment = tmp_path / "owned.hdr"
    environment.write_bytes(b"owned; this contract never renders")
    monkeypatch.setattr(adapter_module, "_default_environment_identity",
                        lambda request: (environment, resource, (environment,)))
    scenes = _scenes("Peak")
    scenes[1].recipe.layers[0].labels[0]["id"] = label
    with adapter_module._LazyScratch() as scratch:
        adapter = adapter_module._SceneExecution(scenes, [0, 1], scratch, cache_requested=True)
        state = adapter.label_state
        assert adapter._prepared_label_blob is not None
        expected = canonical_json_bytes(state, error_context="independent frozen plans")
        assert adapter.compile_labels(state, -1, ()) == expected
        alternate = {"changed": {"label": label, "zero": -0.0}}
        assert adapter.compile_labels(alternate, -1, ()) == canonical_json_bytes(alternate, error_context="independent alternate")


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="native compositor parity requires a physical GPU")
@pytest.mark.parametrize("size", [(64, 64), (31, 19)])
@pytest.mark.parametrize("base_alpha", [0, 127, 255])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("rotation", [0.0, 0.4])
def test_private_label_compositor_matches_scene_and_clears_reused_quads(size, base_alpha, channels, rotation):
    from forge3d._native import get_native_module
    native = get_native_module()
    width, height = size
    full = native.Scene(width, height)
    full.disable_terrain()
    lean = native._anamnesis_prepare_label_scene(width, height)
    assert isinstance(lean, native._LabelCompositor)
    assert not hasattr(lean, "enable_terrain")
    rng = np.random.default_rng(46)
    base = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    base[:, :, 3] = base_alpha
    atlas = rng.integers(0, 256, (16, 16, channels), dtype=np.uint8)
    for scene in (full, lean):
        scene.set_raster_overlay(base, 1.0, None, None)
        scene.set_native_text_atlas(atlas, channels, 1.0)
        scene.enable_native_text()
        for x, y in [(3., 4.), (-5., 2.), (width - 4., height - 5.)]:
            scene.add_native_text_rect_uv_halo(x, y, 18., 20., 0., 0., 1., 1.,
                0.8, 0.2, 0.4, 0.8, 0.1, 0.7, 0.2, 0.7, 1.3, rotation)
    np.testing.assert_array_equal(lean.render_rgba(), full.render_rgba())
    for scene in (full, lean):
        scene.clear_native_text()
    np.testing.assert_array_equal(lean.render_rgba(), full.render_rgba())


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="native compositor ownership requires a physical GPU")
def test_private_label_compositor_reports_real_work_and_releases_allocations():
    import gc
    from forge3d._native import get_native_module
    native = get_native_module()
    native._anamnesis_prepare_context()
    baseline = dict(native.global_memory_metrics())
    compositor = native._anamnesis_prepare_label_scene(64, 64)
    assert native.global_memory_metrics()["texture_bytes"] > baseline["texture_bytes"]
    compositor.set_raster_overlay(np.full((64, 64, 4), 255, dtype=np.uint8), 1., None, None)
    compositor.set_native_text_atlas(np.full((8, 8, 1), 255, dtype=np.uint8), 1, 1.)
    compositor.enable_native_text()
    compositor.add_native_text_rect_uv_halo(3., 4., 18., 20., 0., 0., 1., 1.,
        0.8, 0.2, 0.4, 0.8, 0.1, 0.7, 0.2, 0.7, 1.3)
    compositor.render_rgba()
    report = json.loads(native.render_execution_report())
    assert [item["label"] for item in report["passes"]] == ["label.clear", "label.composite", "label.readback_copy"]
    assert [item["draw_calls"] for item in report["passes"]] == [0, 2, 1]
    assert all(item["gpu_ms"] >= 0 for item in report["passes"])
    if "timestamp_query" in report["capabilities"]["granted"]:
        assert any(item["gpu_ms"] > 0 for item in report["passes"])
    assert set(report["engine"]["wgsl_module_hashes"]) == {"overlays_shader", "text_overlay_shader"}
    labels = report["allocations"]["by_label"]
    assert "label-compositor-color" in labels
    assert "label-compositor-readback" in labels
    assert report["allocations"]["peak_host_visible_bytes"] > 0
    assert report["allocations"]["peak_device_local_bytes"] > 0
    compositor.clear_native_text()
    compositor.render_rgba()
    cleared = json.loads(native.render_execution_report())
    assert set(cleared["engine"]["wgsl_module_hashes"]) == {"overlays_shader"}
    assert cleared["passes"][1]["draw_calls"] == 1
    del compositor
    gc.collect()
    after = dict(native.global_memory_metrics())
    for name in ("buffer_count", "texture_count", "buffer_bytes", "texture_bytes", "host_visible_bytes", "total_bytes"):
        assert after[name] == baseline[name], name


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="native compositor input validation requires a physical GPU")
def test_private_label_compositor_rejects_invalid_inputs_before_rendering():
    from forge3d._native import get_native_module
    native = get_native_module()
    compositor = native._anamnesis_prepare_label_scene(64, 64)
    for channels in (0, 2, 5):
        with pytest.raises(ValueError, match="overlay channels must be 3 or 4"):
            compositor.set_raster_overlay(np.zeros((8, 8, channels), dtype=np.uint8), 1., None, None)
        with pytest.raises(ValueError, match="atlas channels must be 1, 3, or 4"):
            compositor.set_native_text_atlas(np.zeros((8, 8, channels), dtype=np.uint8), channels, 1.)
    with pytest.raises(TypeError, match="numpy uint8"):
        compositor.set_raster_overlay(np.zeros((8, 8, 4), dtype=np.float32), 1., None, None)
    with pytest.raises(ValueError, match="rotation must be finite"):
        compositor.add_native_text_rect_uv_halo(3., 4., 18., 20., 0., 0., 1., 1.,
            0.8, 0.2, 0.4, 0.8, 0.1, 0.7, 0.2, 0.7, 1.3, float('nan'))
