"""CHRONOS deterministic flythrough: structural + gated physical-render tests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d as f3d
from forge3d.chronos import FlythroughManifest, render_flythrough
from _terrain_runtime import terrain_rendering_available

CHRONOS_NATIVE_SYMBOLS = (
    "frame_seed",
    "compile_frame",
    "render_compiled_frame",
    "CompiledFrame",
)


def _chronos_native() -> bool:
    return all(hasattr(f3d, name) for name in CHRONOS_NATIVE_SYMBOLS)


def _require_chronos_native() -> None:
    if not _chronos_native():
        pytest.skip("CHRONOS native symbols not present in this build")


CAMERA_JSON = json.dumps(
    {
        "kind": "orbit_camera",
        "target": [0.0, 0.0, 0.0],
        "distance": 120.0,
        "azimuth_deg": 30.0,
        "elevation_deg": 45.0,
        "fov_deg": 45.0,
        "near": None,
        "far": None,
    },
    sort_keys=True,
    separators=(",", ":"),
)


def _scene_json(**overrides) -> str:
    scene = {
        "labels": [],
        "clipmap": None,
        "virtual_texture": None,
        "scene": {"kind": "scene_recipe"},
    }
    scene.update(overrides)
    return json.dumps(scene, sort_keys=True, separators=(",", ":"))


class TestFrameSeed:
    def test_known_vectors(self):
        _require_chronos_native()
        assert f3d.frame_seed(0, 0) == 0xE220A8397B1DCDAF
        assert f3d.frame_seed(0, 1) == 0x910A2DEC89025CC1

    def test_adjacent_seeds_distinct(self):
        _require_chronos_native()
        seeds = {f3d.frame_seed(7, i) for i in range(256)}
        assert len(seeds) == 256


class TestCompileFrame:
    def test_compile_byte_identity(self):
        _require_chronos_native()
        a = f3d.compile_frame(3, 11, 4, CAMERA_JSON, _scene_json())
        b = f3d.compile_frame(3, 11, 4, CAMERA_JSON, _scene_json())
        assert a.to_json() == b.to_json()
        payload = json.loads(a.to_json())
        assert payload["schema"] == "forge3d.chronos.compiled_frame/1"
        assert payload["frame_index"] == 3
        assert payload["base_seed"] == 11
        assert payload["frame_seed"] == f3d.frame_seed(11, 3)
        assert payload["samples"] == 4
        for field in (
            "camera_hash",
            "scene_hash",
            "label_set_hash",
            "residency_hash",
            "lod_hash",
            "engine_revision",
        ):
            assert isinstance(payload[field], str) and payload[field]

    def test_compile_key_order_insensitive(self):
        _require_chronos_native()
        scene_a = _scene_json(scene={"a": 1, "b": 2})
        scene_b = '{"virtual_texture": null, "scene": {"b": 2, "a": 1}, "labels": [], "clipmap": null}'
        a = f3d.compile_frame(0, 1, 1, CAMERA_JSON, scene_a)
        b = f3d.compile_frame(0, 1, 1, CAMERA_JSON, scene_b)
        assert a.to_json() == b.to_json()

    def test_distinct_frames_distinct_payloads(self):
        _require_chronos_native()
        a = f3d.compile_frame(0, 5, 1, CAMERA_JSON, _scene_json())
        b = f3d.compile_frame(1, 5, 1, CAMERA_JSON, _scene_json())
        assert a.frame_seed != b.frame_seed
        assert a.to_json() != b.to_json()

    def test_from_json_round_trip_and_tamper(self):
        _require_chronos_native()
        compiled = f3d.compile_frame(2, 9, 1, CAMERA_JSON, _scene_json())
        restored = f3d.CompiledFrame.from_json(compiled.to_json())
        assert restored.to_json() == compiled.to_json()
        tampered = json.loads(compiled.to_json())
        tampered["frame_seed"] = 12345
        with pytest.raises(Exception):
            f3d.CompiledFrame.from_json(json.dumps(tampered))
        tampered = json.loads(compiled.to_json())
        tampered["lod_hash"] = "00" * 32
        with pytest.raises(Exception):
            f3d.CompiledFrame.from_json(json.dumps(tampered))

    def test_label_alpha_records(self):
        _require_chronos_native()
        label = {
            "layer_id": "cities",
            "label_id": "summit",
            "visible": True,
            "appearance_frame": 4,
            "disappearance_frame": 12,
            "payload": {"text": "Summit"},
        }
        def labels_payload(frame: int):
            return json.loads(
                f3d.compile_frame(frame, 0, 1, CAMERA_JSON, _scene_json(labels=[label]))
                .to_json()
            )["labels"]
        (record,) = labels_payload(4)
        assert record["visible"] is True
        assert abs(record["alpha"] - 1.0 / 256.0) < 1e-9
        (record,) = labels_payload(8)
        assert abs(record["alpha"] - 4.0 / 256.0) < 1e-9
        (record,) = labels_payload(3)
        assert record["alpha"] == 0.0
        (record,) = labels_payload(12)
        assert record["alpha"] == 0.0
        (record,) = labels_payload(300)
        assert record["alpha"] == 0.0
        persistent = dict(label, disappearance_frame=None)
        (record,) = json.loads(
            f3d.compile_frame(300, 0, 1, CAMERA_JSON, _scene_json(labels=[persistent])).to_json()
        )["labels"]
        assert record["alpha"] == 1.0
        hidden = dict(label, visible=False)
        (record,) = json.loads(
            f3d.compile_frame(6, 0, 1, CAMERA_JSON, _scene_json(labels=[hidden])).to_json()
        )["labels"]
        assert record["alpha"] == 0.0

    def test_compile_rejects_zero_samples(self):
        _require_chronos_native()
        with pytest.raises(Exception, match="samples"):
            f3d.compile_frame(0, 0, 0, CAMERA_JSON, _scene_json())

    def test_compile_rejects_nonobject_camera(self):
        _require_chronos_native()
        with pytest.raises(Exception, match="object"):
            f3d.compile_frame(0, 0, 1, "[1, 2]", _scene_json())

    def test_compile_rejects_invalid_label_interval(self):
        _require_chronos_native()
        label = {
            "layer_id": "cities",
            "label_id": "summit",
            "visible": True,
            "appearance_frame": 9,
            "disappearance_frame": 9,
            "payload": {},
        }
        with pytest.raises(Exception, match="appearance_frame"):
            f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json(labels=[label]))
        label["appearance_frame"] = 12
        label["disappearance_frame"] = 9
        with pytest.raises(Exception, match="appearance_frame"):
            f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json(labels=[label]))

    def test_compile_rejects_duplicate_labels(self):
        _require_chronos_native()
        label = {
            "layer_id": "cities",
            "label_id": "summit",
            "visible": True,
            "appearance_frame": None,
            "disappearance_frame": None,
            "payload": {},
        }
        with pytest.raises(Exception, match="duplicate"):
            f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json(labels=[label, dict(label)]))

    def test_lod_records(self):
        _require_chronos_native()
        clipmap = {
            "ring_count": 3,
            "morph_range": 0.25,
            "terrain_span": 1024.0,
            "camera_distance": 896.0,
        }
        payload = json.loads(
            f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json(clipmap=clipmap)).to_json()
        )
        lod = payload["lod"]
        assert [record["tile_id"] for record in lod] == [
            "clipmap:ring:0",
            "clipmap:ring:1",
            "clipmap:ring:2",
        ]
        assert [record["lod"] for record in lod] == [0, 1, 2]
        assert abs(lod[0]["morph"] - 0.5) < 1e-9
        assert lod[1]["morph"] == 1.0

    def test_residency_sorted_and_tamper_rejected(self):
        _require_chronos_native()
        vt = {
            "virtual_size_px": [1024, 1024],
            "tile_size": 128,
            "max_mip_levels": 4,
            "sources": [
                {"family_slot": 1, "material_index": 0},
                {"family_slot": 0, "material_index": 0},
            ],
            "render_size": [256, 256],
            "terrain_span": 1000.0,
            "camera_mode": "map",
            "camera_target": [0.0, 0.0, 0.0],
            "camera_distance": 100.0,
            "fov_deg": 45.0,
        }
        compiled = f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json(virtual_texture=vt))
        payload = json.loads(compiled.to_json())
        residency = payload["residency"]
        assert residency, "expected non-empty required residency"
        keys = [
            (r["family_slot"], r["material_index"], r["x"], r["y"], r["mip_level"])
            for r in residency
        ]
        assert keys == sorted(keys)
        tampered = json.loads(compiled.to_json())
        tampered["residency"] = tampered["residency"][1:]
        with pytest.raises(Exception):
            f3d.CompiledFrame.from_json(json.dumps(tampered))


class TestRenderCompiledFrame:
    def test_provenance_fields(self):
        _require_chronos_native()
        compiled = f3d.compile_frame(1, 2, 1, CAMERA_JSON, _scene_json())
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        provenance = json.loads(f3d.render_compiled_frame(compiled, rgba))
        for field in (
            "frame_index",
            "base_seed",
            "frame_seed",
            "samples",
            "residency_hash",
            "label_set_hash",
            "lod_hash",
            "engine_revision",
            "pixel_hash",
            "camera_hash",
            "scene_hash",
            "compiled_frame",
        ):
            assert field in provenance
        assert provenance["frame_index"] == 1
        assert provenance["compiled_frame"]["frame_index"] == 1

    def test_bad_shape_rejected(self):
        _require_chronos_native()
        compiled = f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json())
        with pytest.raises(Exception):
            f3d.render_compiled_frame(compiled, np.zeros((4, 4, 3), dtype=np.uint8))


def _frame_record(compiled, frame_index: int = 0) -> dict:
    payload = json.loads(compiled.to_json())
    record = {
        "frame_index": frame_index,
        "image": f"frame_{frame_index:08d}.png",
        "provenance": f"frame_{frame_index:08d}.provenance.json",
        "pixel_hash": "ab" * 32,
        "width": 64,
        "height": 64,
        "compiled_frame": compiled.to_json(),
    }
    for name in (
        "base_seed",
        "frame_seed",
        "samples",
        "residency_hash",
        "label_set_hash",
        "lod_hash",
        "engine_revision",
        "camera_hash",
        "scene_hash",
    ):
        record[name] = payload[name]
    return record


class TestManifestRoundTrip:
    def test_canonical_manifest_round_trip(self, tmp_path):
        _require_chronos_native()
        compiled = f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json())
        record = _frame_record(compiled)
        manifest = FlythroughManifest(base_seed=0, samples=1, frames=(record,))
        path = tmp_path / "flythrough_manifest.json"
        manifest.save(path)
        blob_a = path.read_bytes()
        manifest.save(path)
        assert path.read_bytes() == blob_a
        loaded = FlythroughManifest.load(path)
        assert loaded.to_dict() == manifest.to_dict()
        assert loaded.frame_record(0)["compiled_frame"] == record["compiled_frame"]
        payload = json.loads(compiled.to_json())
        for name in (
            "base_seed",
            "frame_seed",
            "samples",
            "residency_hash",
            "label_set_hash",
            "lod_hash",
            "engine_revision",
            "camera_hash",
            "scene_hash",
        ):
            assert loaded.frame_record(0)[name] == payload[name]

    def test_manifest_rejects_unknown_schema(self):
        _require_chronos_native()
        compiled = f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json())
        record = _frame_record(compiled)
        with pytest.raises(Exception, match="schema"):
            FlythroughManifest(base_seed=0, samples=1, frames=(record,), schema="bogus/9")
        with pytest.raises(Exception, match="schema"):
            FlythroughManifest.from_dict(
                {"schema": "bogus/9", "base_seed": 0, "samples": 1, "frames": [record]}
            )

    def test_manifest_rejects_mismatched_frame_seed(self):
        _require_chronos_native()
        compiled = f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json())
        record = _frame_record(compiled)
        record["frame_seed"] = int(record["frame_seed"]) + 1
        with pytest.raises(Exception, match="frame_seed"):
            FlythroughManifest(base_seed=0, samples=1, frames=(record,))

    def test_manifest_rejects_unsorted_or_duplicate_indices(self):
        _require_chronos_native()
        record0 = _frame_record(f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json()))
        record1 = _frame_record(f3d.compile_frame(1, 0, 1, CAMERA_JSON, _scene_json()), 1)
        with pytest.raises(Exception, match="sorted"):
            FlythroughManifest(base_seed=0, samples=1, frames=(record1, record0))
        with pytest.raises(Exception, match="sorted"):
            FlythroughManifest(base_seed=0, samples=1, frames=(record0, record0))

    def test_manifest_rejects_supplied_compiled_frame_mismatch(self):
        _require_chronos_native()
        record0 = _frame_record(f3d.compile_frame(0, 0, 1, CAMERA_JSON, _scene_json()))
        compiled_1 = f3d.compile_frame(1, 0, 1, CAMERA_JSON, _scene_json())
        with pytest.raises(Exception):
            FlythroughManifest(
                base_seed=0, samples=1, frames=(record0,), _compiled_frames=(compiled_1,)
            )
        with pytest.raises(Exception, match="count"):
            FlythroughManifest(
                base_seed=0,
                samples=1,
                frames=(record0,),
                _compiled_frames=(compiled_1, compiled_1),
            )


class TestFailClosedResidency:
    def test_tampered_residency_rejected(self):
        _require_chronos_native()
        if not hasattr(f3d, "TerrainRenderParams"):
            pytest.skip("TerrainRenderParams native class not present")
        vt = {
            "virtual_size_px": [512, 512],
            "tile_size": 128,
            "max_mip_levels": 3,
            "sources": [{"family_slot": 0, "material_index": 0}],
            "render_size": [64, 64],
            "terrain_span": 64.0,
            "camera_mode": "map",
            "camera_target": [0.0, 0.0, 0.0],
            "camera_distance": 100.0,
            "fov_deg": 45.0,
        }
        compiled = f3d.compile_frame(0, 3, 1, CAMERA_JSON, _scene_json(virtual_texture=vt))
        payload = json.loads(compiled.to_json())
        assert payload["residency"], "expected non-empty required residency"
        tampered = json.loads(compiled.to_json())
        tampered["residency"] = [
            r for r in tampered["residency"] if r["mip_level"] == tampered["residency"][0]["mip_level"]
        ]
        tampered_json = json.dumps(tampered)
        with pytest.raises(Exception):
            f3d.CompiledFrame.from_json(tampered_json)

    def test_seed_mismatch_rejected_by_params(self):
        _require_chronos_native()
        if not hasattr(f3d, "TerrainRenderParams"):
            pytest.skip("TerrainRenderParams native class not present")
        from forge3d.terrain_params import make_terrain_params_config

        compiled = f3d.compile_frame(0, 3, 1, CAMERA_JSON, _scene_json())
        config = make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=64.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 1.0),
            cam_radius=100.0,
            cam_phi_deg=30.0,
            cam_theta_deg=45.0,
            fov_y_deg=45.0,
            aa_samples=1,
            aa_seed=int(f3d.frame_seed(3, 0)) + 1,
        )
        config.chronos_frame_json = compiled.to_json()
        with pytest.raises(Exception):
            f3d.TerrainRenderParams(config)


class TestOutputContract:
    def test_rejects_non_png_format(self, tmp_path):
        _require_chronos_native()
        scene = _synthetic_scene(output_overrides={"format": "exr"})
        with pytest.raises(Exception, match="PNG"):
            render_flythrough([scene], base_seed=0, samples=1, out_dir=tmp_path)

    def test_rejects_non8_bit_depth(self, tmp_path):
        _require_chronos_native()
        scene = _synthetic_scene(output_overrides={"bit_depth": 16})
        with pytest.raises(Exception, match="bit_depth"):
            render_flythrough([scene], base_seed=0, samples=1, out_dir=tmp_path)

    def test_rejects_hdr(self, tmp_path):
        _require_chronos_native()
        scene = _synthetic_scene(output_overrides={"hdr": True})
        with pytest.raises(Exception, match="hdr"):
            render_flythrough([scene], base_seed=0, samples=1, out_dir=tmp_path)

    def test_rejects_aovs(self, tmp_path):
        _require_chronos_native()
        scene = _synthetic_scene(output_overrides={"aovs": ("depth",)})
        with pytest.raises(Exception, match="AOV"):
            render_flythrough([scene], base_seed=0, samples=1, out_dir=tmp_path)


_RUN_GOLDENS = os.environ.get("FORGE3D_RUN_TERRAIN_GOLDENS") == "1"


def _physical_render_available() -> bool:
    if not _RUN_GOLDENS or not _chronos_native():
        return False
    if not hasattr(f3d, "has_gpu") or not f3d.has_gpu():
        return False
    return all(
        hasattr(f3d, name)
        for name in ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    )


def _synthetic_scene(
    width: int = 64,
    height: int = 64,
    output_overrides: dict | None = None,
    terrain_metadata: dict | None = None,
) -> "f3d.MapScene":
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, 64, dtype=np.float32),
        np.linspace(0.0, 1.0, 64, dtype=np.float32),
        indexing="ij",
    )
    dem = (0.5 + 0.3 * x + 0.2 * np.sin(6.0 * y)).astype(np.float32)
    output_spec = dict(width=width, height=height, format="png", samples=1)
    output_spec.update(output_overrides or {})
    metadata = {
        "width": 64,
        "height": 64,
        "source_id": "chronos-synthetic",
        "clipmap": {
            "mode": "clipmap",
            "ring_count": 2,
            "ring_resolution": 16,
            "center_resolution": 16,
            "morph_range": 0.3,
            "skirt_depth": 10.0,
        },
    }
    metadata.update(terrain_metadata or {})
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=dem,
            crs="EPSG:32610",
            metadata=metadata,
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=75.0, azimuth_deg=35.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(**output_spec),
    )


def _rec709_luminance(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _load_rgba(path: Path) -> np.ndarray:
    from forge3d._png import load_png_rgba

    return np.ascontiguousarray(load_png_rgba(path))


def _require_terrain_rendering() -> None:
    if not terrain_rendering_available():
        pytest.skip("CHRONOS flythrough requires a hardware terrain renderer")


def _capture_cache_context(monkeypatch, scene: "f3d.MapScene", output: Path) -> bytes:
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


def _path_free_scene(scene: "f3d.MapScene") -> dict:
    value = scene.to_dict()
    for name in ("path", "directory", "filename"):
        value["recipe"]["output"].pop(name, None)
    return value


def test_frame_free_cache_context_preserves_existing_bytes(monkeypatch, tmp_path):
    from forge3d._canonical_json import canonical_json_bytes

    scene = _synthetic_scene()
    context = _capture_cache_context(monkeypatch, scene, tmp_path / "frame.png")
    assert context == canonical_json_bytes(
        _path_free_scene(scene), error_context="MapScene ANAMNESIS callback context"
    )


def test_compiled_frame_changes_cache_context_for_same_recipe(monkeypatch, tmp_path):
    scene = _synthetic_scene()
    plan = scene.compile_plan()
    output = tmp_path / "frame.png"
    contexts = []
    for frame_json in ('{"frame":0}', '{"frame":1}'):
        scene.compiled_plan = replace(
            plan, frame=SimpleNamespace(to_json=lambda value=frame_json: value)
        )
        contexts.append(_capture_cache_context(monkeypatch, scene, output))
    assert contexts[0] != contexts[1]
    assert json.loads(contexts[0]) == {
        "scene": _path_free_scene(scene),
        "chronos_frame": '{"frame":0}',
    }


def test_flythrough_default_keywords_preserve_artifact_bytes(tmp_path):
    _require_terrain_rendering()
    scene = _synthetic_scene()
    camera_path = {0: scene, 1: scene}
    omitted = tmp_path / "omitted"
    explicit = tmp_path / "explicit"
    render_flythrough(camera_path, base_seed=7, samples=1, out_dir=omitted)
    render_flythrough(
        camera_path,
        base_seed=7,
        samples=1,
        out_dir=explicit,
        certificate=False,
        cache=None,
    )
    names = sorted(path.name for path in omitted.iterdir())
    assert names == sorted(path.name for path in explicit.iterdir())
    assert "flythrough_manifest.json" in names
    assert any(name.endswith(".provenance.json") for name in names)
    assert any(name.endswith(".png") for name in names)
    for name in names:
        assert (omitted / name).read_bytes() == (explicit / name).read_bytes(), name


def test_flythrough_certificate_writes_one_sidecar_per_frame(tmp_path):
    _require_terrain_rendering()
    scene = _synthetic_scene()
    camera_path = {0: scene, 1: scene}
    plain = tmp_path / "plain"
    certified = tmp_path / "certified"
    render_flythrough(camera_path, base_seed=7, samples=1, out_dir=plain)
    render_flythrough(
        camera_path, base_seed=7, samples=1, out_dir=certified, certificate=True
    )
    assert (plain / "flythrough_manifest.json").read_bytes() == (
        certified / "flythrough_manifest.json"
    ).read_bytes()
    expected = {f"frame_{index:08d}.certificate.json" for index in camera_path}
    observed = {path.name for path in certified.glob("*.certificate.json")}
    assert observed == expected
    for name in expected:
        assert json.loads((certified / name).read_text("utf-8"))


def test_flythrough_repeat_run_hits_cache_and_preserves_pixel_hashes(
    monkeypatch, tmp_path
):
    _require_terrain_rendering()
    scene = _synthetic_scene()
    camera_path = {0: scene, 1: scene}
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    original_render = f3d.MapScene.render
    hits: list[tuple[int, bool | None]] = []

    def record_render(self, *args, **kwargs):
        report = original_render(self, *args, **kwargs)
        frame = json.loads(self.compiled_plan.frame.to_json())
        hits.append((frame["frame_index"], self.last_render_metadata.get("cache_hit")))
        return report

    monkeypatch.setattr(f3d.MapScene, "render", record_render)
    first = render_flythrough(
        camera_path, base_seed=7, samples=1, out_dir=output, cache=cache
    )
    first_manifest = (output / "flythrough_manifest.json").read_bytes()
    second = render_flythrough(
        camera_path, base_seed=7, samples=1, out_dir=output, cache=cache
    )
    moved_output = tmp_path / "moved-output"
    third = render_flythrough(
        camera_path, base_seed=7, samples=1, out_dir=moved_output, cache=cache
    )
    assert hits == [
        (0, False), (1, False), (0, True), (1, True), (0, True), (1, True)
    ]
    assert (output / "flythrough_manifest.json").read_bytes() == first_manifest
    for manifest, directory in ((second, output), (third, moved_output)):
        for record in manifest.frames:
            index = int(record["frame_index"])
            compiled = f3d.CompiledFrame.from_json(record["compiled_frame"])
            rgba = _load_rgba(directory / record["image"])
            provenance = json.loads(f3d.render_compiled_frame(compiled, rgba))
            assert provenance["pixel_hash"] == first.frame_record(index)["pixel_hash"]


def test_different_cameras_do_not_share_cache_entry(tmp_path):
    from forge3d.chronos import _camera_json, _prepare_frame_scene, _scene_json

    _require_terrain_rendering()
    first = _synthetic_scene()
    second = f3d.MapScene(
        replace(first.recipe, camera=replace(first.recipe.camera, azimuth_deg=73.0))
    )
    output = tmp_path / "shared.png"
    cache = tmp_path / "cache"
    camera_hashes = []
    for source in (first, second):
        frame_scene = _prepare_frame_scene(source, 0, 7, 1, output)
        compiled = f3d.compile_frame(
            0, 7, 1, _camera_json(frame_scene), _scene_json(frame_scene, [])
        )
        frame_scene.compiled_plan = replace(frame_scene.compiled_plan, frame=compiled)
        frame_scene.render(str(output), cache=cache)
        assert frame_scene.last_render_metadata["cache_hit"] is False
        camera_hashes.append(json.loads(compiled.to_json())["camera_hash"])
    assert camera_hashes[0] != camera_hashes[1]


@pytest.mark.skipif(
    not _RUN_GOLDENS,
    reason="CHRONOS physical render runs only in the GPU golden lane",
)
class TestFlythroughPhysical:
    def test_static_path_hash_identical_and_replay(self, tmp_path):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")

        frame_count = 3
        source_scene = _synthetic_scene()
        camera_path = {index: source_scene for index in range(frame_count)}

        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        manifest_a = render_flythrough(camera_path, base_seed=7, samples=1, out_dir=run_a)
        manifest_b = render_flythrough(camera_path, base_seed=7, samples=1, out_dir=run_b)

        identical = 0
        rgba_frames_a = []
        for record in manifest_a.frames:
            index = int(record["frame_index"])
            record_a = manifest_a.frame_record(index)
            record_b = manifest_b.frame_record(index)
            rgba_a = _load_rgba(run_a / record_a["image"])
            rgba_b = _load_rgba(run_b / record_b["image"])
            rgba_frames_a.append(rgba_a)
            sidecar_a = json.loads((run_a / record_a["provenance"]).read_text("utf-8"))
            sidecar_b = json.loads((run_b / record_b["provenance"]).read_text("utf-8"))
            raw_hash_a = hashlib.sha256(rgba_a.tobytes()).hexdigest()
            raw_hash_b = hashlib.sha256(rgba_b.tobytes()).hexdigest()
            assert raw_hash_a == record_a["pixel_hash"] == sidecar_a["pixel_hash"]
            assert raw_hash_b == record_b["pixel_hash"] == sidecar_b["pixel_hash"]
            assert record_a["pixel_hash"] == record_b["pixel_hash"]
            for name in (
                "base_seed",
                "frame_seed",
                "samples",
                "residency_hash",
                "label_set_hash",
                "lod_hash",
                "engine_revision",
                "camera_hash",
                "scene_hash",
            ):
                assert record_a[name] == sidecar_a[name]
                assert record_b[name] == sidecar_b[name]
            identical += 1
        print(f"{identical}/{len(manifest_a.frames)} frames hash-identical on re-render")
        assert identical == len(manifest_a.frames)

        luminance_a = [_rec709_luminance(rgba) for rgba in rgba_frames_a]
        adjacent_deltas = [
            float(np.abs(luminance_a[i + 1] - luminance_a[i]).max())
            for i in range(len(luminance_a) - 1)
        ]
        static_metric = max(adjacent_deltas, default=0.0)
        print(f"max static-pixel luminance delta: {static_metric}")
        # The 256-frame fade gate requires below one 8-bit step (< 1/255).
        assert static_metric < 1.0 / 255.0

        loaded = FlythroughManifest.load(run_a / "flythrough_manifest.json")
        replay_index = frame_count - 1
        replay_path = tmp_path / "replay" / f"frame_{replay_index:08d}.png"
        provenance = loaded.replay_frame(source_scene, replay_index, replay_path)
        record = loaded.frame_record(replay_index)
        assert provenance["pixel_hash"] == record["pixel_hash"]

    def test_vt_required_residency_render(self, tmp_path):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")

        vt_metadata = {
            "virtual_texture": {
                "enabled": True,
                "virtual_size_px": (128, 128),
                "layers": [
                    {
                        "family": "albedo",
                        "virtual_size_px": (128, 128),
                        "tile_size": 64,
                        "tile_border": 4,
                    }
                ],
                "procedural_sources": True,
                "source_count": 1,
                "source_size": 128,
                "atlas_size": 2304,
                "residency_budget_mb": 8.0,
                "max_mip_levels": 2,
                "use_feedback": False,
            }
        }
        source_scene = _synthetic_scene(terrain_metadata=vt_metadata)
        manifest = render_flythrough(
            {0: source_scene}, base_seed=11, samples=1, out_dir=tmp_path
        )
        record = manifest.frame_record(0)
        payload = json.loads(record["compiled_frame"])
        assert payload["residency"], "expected non-empty compiled residency"
        sidecar = json.loads((tmp_path / record["provenance"]).read_text("utf-8"))
        assert sidecar["residency_hash"] == record["residency_hash"]
        assert sidecar["residency_hash"] == payload["residency_hash"]
