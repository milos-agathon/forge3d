"""GPU-free MapScene render-cache key and store contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d as f3d
from forge3d.helpers.offscreen import save_png_deterministic


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
def fake_renderer(monkeypatch) -> list[float]:
    calls: list[float] = []

    def render_impl(self, path=None, **_kwargs):
        calls.append(float(self.recipe.camera.azimuth_deg))
        target = Path(path or self.recipe.output.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., 0] = int(round(self.recipe.camera.azimuth_deg)) % 256
        rgba[..., 3] = 255
        save_png_deterministic(target, rgba)
        self.last_render_path = str(target)
        self.last_render_metadata = {}
        return self._compiled_plan_for_current_recipe().validation_report

    monkeypatch.setattr(f3d.MapScene, "_render_impl", render_impl)
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
    import forge3d.anamnesis as anamnesis

    class ContextCaptured(Exception):
        pass

    captured = []

    def capture_sequence(recipe, **kwargs):
        captured.append((recipe, kwargs["render_frame_context"]))
        raise ContextCaptured

    with monkeypatch.context() as patch:
        patch.setattr(anamnesis, "render_sequence", capture_sequence)
        for scene in (
            _scene(tmp_path / "first.png"),
            _scene(tmp_path / "second.png"),
            _scene(tmp_path / "third.png", azimuth_deg=73.0),
        ):
            with pytest.raises(ContextCaptured):
                scene.render(cache=tmp_path / "cache")

    assert captured[0][0] == captured[1][0]
    assert captured[0][1] == captured[1][1]
    assert captured[0][1] != captured[2][1]
    for recipe, _context in captured:
        assert all(
            name not in recipe["recipe"]["output"]
            for name in ("path", "directory", "filename")
        )
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


def _capture_context(monkeypatch, scene: f3d.MapScene, output: Path) -> bytes:
    import forge3d.anamnesis as anamnesis

    class ContextCaptured(Exception):
        pass

    captured: list[bytes] = []

    def capture_sequence(*_args, **kwargs):
        captured.append(kwargs["render_frame_context"])
        raise ContextCaptured

    with monkeypatch.context() as patch:
        patch.setattr(anamnesis, "render_sequence", capture_sequence)
        with pytest.raises(ContextCaptured):
            scene.render(str(output), cache=output.parent / "cache")
    assert len(captured) == 1
    return captured[0]


def _path_free_scene(scene: f3d.MapScene) -> dict:
    value = scene.to_dict()
    for name in ("path", "directory", "filename"):
        value["recipe"]["output"].pop(name, None)
    return value


def test_frame_free_context_is_the_path_free_scene(monkeypatch, tmp_path):
    from forge3d._canonical_json import canonical_json_bytes

    scene = _scene(tmp_path / "frame.png")
    context = _capture_context(monkeypatch, scene, tmp_path / "frame.png")
    assert context == canonical_json_bytes(
        _path_free_scene(scene), error_context="MapScene ANAMNESIS callback context"
    )


def test_compiled_chronos_frame_changes_context_for_same_recipe(monkeypatch, tmp_path):
    output = tmp_path / "frame.png"
    scene = _scene(output)
    plan = scene.compile_plan()
    contexts = []
    for frame_json in ('{"frame":0}', '{"frame":1}'):
        scene.compiled_plan = replace(
            plan, frame=SimpleNamespace(to_json=lambda value=frame_json: value)
        )
        contexts.append(_capture_context(monkeypatch, scene, output))
    assert contexts[0] != contexts[1]
    assert json.loads(contexts[0]) == {
        "scene": _path_free_scene(scene),
        "chronos_frame": '{"frame":0}',
    }
