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
        assert f3d.frame_seed(0, 0) == 0x48218226FF3CD4BF
        assert f3d.frame_seed(0, 1) == 0xA706DD2F4D197E6F

    def test_adjacent_seeds_distinct(self):
        _require_chronos_native()
        seeds = {f3d.frame_seed(7, i) for i in range(256)}
        assert len(seeds) == 256

    def test_base_seed_is_mixed_before_frame_index(self):
        # A plain base_seed + frame_index combination makes neighbouring base
        # seeds replay each other's frames shifted by one.
        _require_chronos_native()
        for base in (0, 1, 7, 2**32, 2**63, 2**64 - 2):
            for index in (1, 2, 17, 255):
                assert f3d.frame_seed(base, index) != f3d.frame_seed(base + 1, index - 1)


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
    rendered: list[int] = []

    # The flythrough cache is one ANAMNESIS sequence: a hit replays stored
    # frame bytes without calling MapScene.render at all.
    def record_render(self, *args, **kwargs):
        report = original_render(self, *args, **kwargs)
        frame = json.loads(self.compiled_plan.frame.to_json())
        rendered.append(frame["frame_index"])
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
    # Only the first run renders; the repeat and moved-output runs hit.
    assert rendered == [0, 1]
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

    def test_certificate_and_cache_contracts(self, tmp_path, monkeypatch):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")
        import forge3d.chronos as chronos_module
        from forge3d import certificate
        from forge3d.diagnostics import capabilities

        source_scene = _synthetic_scene()
        camera_path = {index: source_scene for index in range(2)}
        cache_dir = tmp_path / "cache"

        caps = capabilities()
        absent_caps = set(caps["requested"]) - set(caps["granted"])
        certified = render_flythrough(
            camera_path, base_seed=11, samples=1, out_dir=tmp_path / "cert", certificate=True
        )
        for record in certified.frames:
            name = f"frame_{int(record['frame_index']):08d}.certificate.json"
            cert_path = tmp_path / "cert" / name
            cert = json.loads(cert_path.read_text("utf-8"))
            assert certificate.verify(cert_path, cert["signature"]["pubkey"]) is True
            # Adapter-honest: only capabilities this adapter lacks may degrade
            # (hosted Metal has no timestamp/pipeline-statistics queries). With
            # every requested feature granted this stays an empty-list check.
            assert all(
                item["kind"] == "capability_absent" for item in cert["degradations"]
            ), cert["degradations"]
            assert {item["name"] for item in cert["degradations"]} <= absent_caps
            assert cert["passes"], "certificate must record the frame's executed passes"

        cold = render_flythrough(
            camera_path, base_seed=11, samples=1, out_dir=tmp_path / "cold", cache=cache_dir
        )
        fresh_renders = []
        original = chronos_module.MapScene.render

        def counting_render(self, *args, **kwargs):
            fresh_renders.append(args)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(chronos_module.MapScene, "render", counting_render)
        warm = render_flythrough(
            camera_path, base_seed=11, samples=1, out_dir=tmp_path / "warm", cache=cache_dir
        )
        assert fresh_renders == [], "identical compiled frames must replay from the cache"
        assert [r["pixel_hash"] for r in warm.frames] == [r["pixel_hash"] for r in cold.frames]
        assert [r["pixel_hash"] for r in cold.frames] == [
            r["pixel_hash"] for r in certified.frames
        ]

        # A different base seed compiles different frames: no stale cache hit.
        render_flythrough(
            camera_path, base_seed=12, samples=1, out_dir=tmp_path / "reseeded", cache=cache_dir
        )
        assert len(fresh_renders) == 2

        with pytest.raises(TypeError, match="certificate"):
            render_flythrough(
                camera_path, base_seed=11, samples=1, out_dir=tmp_path / "bad", certificate="x"
            )

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
                "source_count": 4,
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

    def test_multisample_frames_are_deterministic_and_seeded(self, tmp_path):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")

        source_scene = _synthetic_scene()
        camera_path = {index: source_scene for index in range(2)}

        def hashes(base_seed: int, samples: int, name: str) -> list[str]:
            manifest = render_flythrough(
                camera_path, base_seed=base_seed, samples=samples, out_dir=tmp_path / name
            )
            out = []
            for record in manifest.frames:
                rgba = _load_rgba(tmp_path / name / record["image"])
                assert hashlib.sha256(rgba.tobytes()).hexdigest() == record["pixel_hash"]
                assert int(record["samples"]) == samples
                out.append(record["pixel_hash"])
            return out

        multi_a = hashes(7, 4, "multi_a")
        multi_b = hashes(7, 4, "multi_b")
        single = hashes(7, 1, "single")
        reseeded = hashes(8, 4, "reseeded")
        # Re-rendering the same compiled multi-sample frames is byte-identical.
        assert multi_a == multi_b
        # samples > 1 is executed accumulation, not a relabelled single sample.
        assert multi_a[0] != single[0]
        # The derived per-frame seed reaches the pixels: same camera, different
        # frame index (and different base seed) change the jitter phases.
        assert multi_a[0] != multi_a[1]
        assert multi_a[0] != reseeded[0]

    def test_label_fade_on_held_camera_path_does_not_pop(self, tmp_path):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")

        base = _synthetic_scene(width=96, height=64)
        labelled = f3d.MapScene(
            replace(
                base.recipe,
                layers=(
                    f3d.LabelLayer(
                        layer_id="labels",
                        labels=[
                            {
                                "id": "peak",
                                "text": "Peak",
                                "geometry": {"type": "Point", "coordinates": (48.0, 32.0, 0.0)},
                                "typography": {
                                    "color": [1.0, 1.0, 1.0, 1.0],
                                    "halo_color": [0.0, 0.0, 0.0, 1.0],
                                    "halo_width_px": 2.0,
                                },
                            }
                        ],
                        glyph_atlas={"glyphs": sorted(set("Peak"))},
                        occlusion="none",
                    ),
                ),
            )
        )
        appearance = 100
        fade_end = appearance + 255  # alpha reaches 1.0 on this frame
        windows = [
            range(appearance - 1, appearance + 5),
            range(appearance + 126, appearance + 131),
            range(fade_end - 3, fade_end + 3),
        ]
        indices = [index for window in windows for index in window]
        # The camera is held for the whole path; only the label changes, and it
        # enters at `appearance`, crossing its full fade window.
        camera_path = {
            index: (base if index < appearance else labelled) for index in indices
        }
        manifest = render_flythrough(camera_path, base_seed=5, samples=1, out_dir=tmp_path)

        alphas = {}
        codes = {}
        for record in manifest.frames:
            index = int(record["frame_index"])
            payload = json.loads(record["compiled_frame"])
            alphas[index] = [item["alpha"] for item in payload["labels"] if item["visible"]]
            codes[index] = _load_rgba(tmp_path / record["image"])[..., :3].astype(np.int64)
        assert alphas[appearance - 1] == []
        assert alphas[appearance] == [pytest.approx(1.0 / 256.0)]
        assert alphas[fade_end] == [1.0]

        # Rec. 709 luminance in exact integer units of 1/10000 of an 8-bit code
        # (2126 + 7152 + 722 == 10000), so "one code" is compared without float
        # rounding: a 1-code step in all three channels is exactly 10000.
        weights = np.array([2126, 7152, 722], dtype=np.int64)
        luminance = {index: rgb @ weights for index, rgb in codes.items()}
        one_code = int(weights.sum())

        footprint = np.abs(luminance[fade_end] - luminance[appearance - 1]) > 0
        assert footprint.any(), "the label must be drawn at full opacity"
        assert not footprint.all(), "the path must hold static terrain pixels"

        worst_step = 0
        worst_static = 0
        worst_channel = 0
        for window in windows:
            ordered = list(window)
            for previous, current in zip(ordered, ordered[1:]):
                delta = np.abs(luminance[current] - luminance[previous])
                worst_step = max(worst_step, int(delta.max()))
                worst_static = max(worst_static, int(delta[~footprint].max()))
                worst_channel = max(
                    worst_channel, int(np.abs(codes[current] - codes[previous]).max())
                )
        print(
            f"max per-frame luminance step {worst_step / one_code:.4f} code(s); "
            f"max channel step {worst_channel} code(s); held-static {worst_static}"
        )
        assert worst_static == 0, "held-static pixels must not change between frames"
        # Regression: a pixel blended once per overlapping glyph quad at the fade
        # alpha is rounded twice and jumps 2 codes in one frame.
        assert worst_channel <= 1, "a fade step moved a channel by 2 or more 8-bit codes"
        # One fade step moves a pixel by at most one 8-bit step (<= 1/255): with
        # 8-bit output, a 1/256 alpha step can round all three channels at once.
        assert worst_step <= one_code

    def test_dolly_across_clipmap_morph_band_does_not_pop(self, tmp_path):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")
        from forge3d.chronos import _clipmap_descriptor

        def dolly(morph_range: float) -> dict[int, "f3d.MapScene"]:
            clipmap = {
                "mode": "clipmap",
                "ring_count": 2,
                "ring_resolution": 16,
                "center_resolution": 16,
                "morph_range": morph_range,
                "skirt_depth": 10.0,
            }
            base = _synthetic_scene(width=96, height=64, terrain_metadata={"clipmap": clipmap})
            return {
                index: f3d.MapScene(
                    replace(base.recipe, camera=replace(base.recipe.camera, distance=distance))
                )
                for index, distance in enumerate(distances)
            }

        # Ring 0's morph band is [span * (1 - 0.3), span]. The slow dolly starts
        # inside the finer side of the band and leaves it on the coarse side.
        descriptor = _clipmap_descriptor(_synthetic_scene().recipe)
        outer = descriptor["terrain_span"]
        band_start = outer * (1.0 - descriptor["morph_range"])
        distances = [band_start - 0.8 + 0.75 * step for step in range(30)]
        assert distances[0] < band_start and distances[-1] > outer

        def luminance(camera_path: dict, name: str) -> tuple[list, list]:
            manifest = render_flythrough(
                camera_path, base_seed=9, samples=1, out_dir=tmp_path / name
            )
            frames, morphs = [], []
            for record in manifest.frames:
                payload = json.loads(record["compiled_frame"])
                morphs.append({int(item["lod"]): float(item["morph"]) for item in payload["lod"]})
                frames.append(_rec709_luminance(_load_rgba(tmp_path / name / record["image"])))
            return frames, morphs

        morphing, morphing_lod = luminance(dolly(0.3), "morphing")
        frozen, frozen_lod = luminance(dolly(0.0), "frozen")
        # The morphing path really crosses ring 0's band; the reference holds
        # every ring at the finer level for the whole dolly.
        ring0 = [lod[0] for lod in morphing_lod]
        assert ring0[0] == 0.0 and ring0[-1] == 1.0
        assert any(0.0 < value < 1.0 for value in ring0)
        assert all(value == 0.0 for lod in frozen_lod for value in lod.values())
        # Non-vacuous: the compiled morph reaches the pixels inside the band.
        assert any(
            np.abs(morphing[index] - frozen[index]).max() > 0.0
            for index, value in enumerate(ring0)
            if 0.0 < value < 1.0
        ), "ring-0 geomorph never changed a pixel; the comparison would be vacuous"

        worst_excess = -1.0
        for index in range(len(distances) - 1):
            moving = float(np.abs(morphing[index + 1] - morphing[index]).max())
            reference = float(np.abs(frozen[index + 1] - frozen[index]).max())
            worst_excess = max(worst_excess, moving - reference)
            assert moving <= reference + 1.0 / 255.0, (index, moving, reference)
        print(f"worst morph-over-frozen adjacent luminance excess: {worst_excess:.6f}")

    def test_vt_render_samples_only_compiled_residency(self, tmp_path):
        if not _physical_render_available():
            pytest.skip("CHRONOS physical render requires GPU-backed native module")
        from forge3d.chronos import _camera_json, _prepare_frame_scene, _scene_json
        from forge3d.map_scene import _ACTIVE_TERRAIN_RESOURCES, _shared_terrain_render_context

        def vt_scene(size: int, budget_mb: float) -> "f3d.MapScene":
            return _synthetic_scene(
                width=size,
                height=size,
                terrain_metadata={
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
                        "source_count": 4,
                        "source_size": 128,
                        "atlas_size": 2304,
                        "residency_budget_mb": budget_mb,
                        "max_mip_levels": 2,
                        "use_feedback": False,
                    }
                },
            )

        def render_compiled(source: "f3d.MapScene", index: int) -> tuple[dict, dict, set]:
            output = tmp_path / f"frame_{index}.png"
            frame_scene = _prepare_frame_scene(source, index, 3, 1, output)
            compiled = f3d.compile_frame(
                index, 3, 1, _camera_json(frame_scene), _scene_json(frame_scene, [])
            )
            frame_scene.compiled_plan = replace(frame_scene.compiled_plan, frame=compiled)
            frame_scene.render(str(output))
            payload = json.loads(compiled.to_json())
            (resources,) = _ACTIVE_TERRAIN_RESOURCES.holder
            resident = {
                tuple(key) for key in resources.renderer.read_material_vt_resident_pages_for_test()
            }
            return payload, dict(frame_scene.last_render_metadata["material_vt_stats"]), resident

        def page_key(record: dict) -> tuple[int, int, int, int, int]:
            return (
                int(record["family_slot"]),
                int(record["material_index"]),
                int(record["mip_level"]),
                int(record["x"]),
                int(record["y"]),
            )

        # One shared renderer: the finer frame leaves more pages resident than
        # the coarser frame compiles, so a stale page would be observable.
        with _shared_terrain_render_context():
            fine, fine_stats, fine_resident = render_compiled(vt_scene(128, 8.0), 0)
            coarse, coarse_stats, coarse_resident = render_compiled(vt_scene(64, 8.0), 1)
        fine_keys = {page_key(item) for item in fine["residency"]}
        coarse_keys = {page_key(item) for item in coarse["residency"]}
        assert len(fine_keys) == len(fine["residency"])
        assert len(coarse_keys) == len(coarse["residency"])
        assert coarse_keys and fine_keys - coarse_keys, (fine_keys, coarse_keys)
        # Set equality, not just cardinality: exactly the compiled pages are
        # resident, so no page left over from the finer frame survives.
        assert fine_resident == fine_keys
        assert coarse_resident == coarse_keys
        assert int(fine_stats["resident_pages"]) == len(fine_keys)
        assert int(coarse_stats["resident_pages"]) == len(coarse_keys)

        # A budget that cannot hold the compiled set fails with a diagnostic
        # instead of rendering from a partial residency.
        with pytest.raises(Exception, match="chronos residency"):
            render_compiled(vt_scene(128, 0.25), 2)
