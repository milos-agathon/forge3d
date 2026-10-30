from __future__ import annotations
from anamnesis_packed_store import metadata as packed_metadata, read_blob

import os
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d as f3d
from forge3d.anamnesis import render_sequence
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)
from forge3d.helpers.offscreen import render_offscreen_rgba
from forge3d.terrain_params import AovSettings, ShadowSettings


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="requires a physical GPU")
@pytest.mark.parametrize("enabled,strength", [(False, 0.3), (True, 0.0)])
def test_inactive_detail_map_does_not_open_missing_file(tmp_path, enabled, strength):
    from forge3d.terrain_params import DetailSettings
    f3d.clear_native_degradations()
    hdr = tmp_path / "env.hdr"
    write_canonical_hdr(str(hdr))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.terrain_default()
    env = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.detail = DetailSettings(enabled=enabled, detail_strength=strength,
                                   detail_normal_path=str(tmp_path / "absent.png"))
    params = f3d.TerrainRenderParams(config)
    height = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    fresh = renderer.render_terrain_pbr_pom(material, env, params, height, cache=None).to_numpy()
    cached = renderer.render_terrain_pbr_pom(material, env, params, height, cache=tmp_path / "cache").to_numpy()
    restored = renderer.render_terrain_pbr_pom(material, env, params, height, cache=tmp_path / "cache").to_numpy()
    assert fresh.tobytes() == cached.tobytes() == restored.tobytes()
    assert renderer.last_anamnesis_cache_report["hits"]
    assert f3d.device_probe()["software_fallback"] is False
    assert not [e for e in f3d.native_degradations() if e["kind"] in {"validation_error", "uncaptured_gpu_error"}]


def test_native_recompute_control_is_required_on_physical_gpu_ci():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    production_job = workflow.split("  test-anamnesis-production:", 1)[1].split("\n  test-anamnesis-hermeticity:", 1)[0]
    native_job = workflow.split("  test-anamnesis-hermeticity:", 1)[1].split("\n  test-tessella-gpu:", 1)[0]
    for fragment in (
        "test_native_production_10000_mutations_match_independent_fresh_frames",
        "test_native_hdr_cache_preserves_eight_byte_texels",
        "test_public_scene_600_frame_acceptance",
        "test_mapscene_cache_owns_content_and_matches_independent_render",
        "test_inactive_detail_map_does_not_open_missing_file",
        "pip install pytest numpy rasterio",
        "test_cached_material_identity_tracks_uploaded_resources",
        "test_retained_parameter_snapshot_cannot_alias_new_native_state",
        "test_native_provenance_feedback_survives_normal_derivatives",
        "anamnesis-native-mutations.json",
        "scripts/assert_junit_zero_skips.py",
    ):
        assert fragment in native_job
    required_fragments = (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]",
        "WGPU_BACKEND: vulkan",
        "FORGE3D_RUN_GPU_ANAMNESIS: '1'",
        "test_native_terrain_cache_restores_all_graph_passes",
        "--junitxml",
        "scripts/assert_junit_zero_skips.py",
        "anamnesis-p0-adapter.json",
        '$actual | Set-Content "$env:RUNNER_TEMP/anamnesis-checked-out-head.txt"',
        "Get-PSDrive -PSProvider FileSystem",
        "Sort-Object Free -Descending",
        '$scratchScope = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT-$env:GITHUB_JOB"',
        "FORGE3D_ANAMNESIS_SCRATCH_DIR",
        "FORGE3D_ANAMNESIS_BASE_TEMP",
        '--basetemp="$env:FORGE3D_ANAMNESIS_BASE_TEMP"',
        "name: Clean ANAMNESIS scratch",
        "(Split-Path -Leaf $parentDir) -ne 'forge3d-ci-scratch'",
    )
    for fragment in required_fragments:
        assert fragment in production_job
    assert production_job.count("--basetemp=") == 1
    assert native_job.count("--basetemp=") == 1
    assert production_job.index(
        "name: Upload ANAMNESIS production evidence"
    ) < production_job.index("name: Clean ANAMNESIS scratch")


def test_cache_none_is_byte_identical_to_enabled_cold_render(tmp_path):
    recipe = {
        "terrain": {"dem_sha256": "ab" * 32},
        "camera": {"path": "inertness"},
        "layers": [{"kind": "label", "text": "A", "visible_frames": [1]}],
        "output": {"width": 32, "height": 32, "samples": 2},
    }
    uncached = render_sequence(recipe, frames=range(4), cache=None, reference_only=True)
    cached = render_sequence(recipe, frames=range(4), cache=tmp_path, reference_only=True)
    assert uncached.frame_blobs == cached.frame_blobs
    assert uncached.frame_hashes == cached.frame_hashes
    assert uncached.cache_report.hits == []


def test_callback_requires_explicit_renderer_fingerprint(tmp_path):
    recipe = {"terrain": {}, "output": {}}
    try:
        render_sequence(recipe, frames=[0], cache=tmp_path, render_frame=lambda _r, _f: b"x")
    except ValueError as error:
        assert "render_frame_fingerprint" in str(error)
    else:
        raise AssertionError("hidden callback code identity must not be cacheable")
    with pytest.raises(ValueError, match="render_frame_context"):
        render_sequence(
            recipe,
            frames=[0],
            cache=tmp_path,
            render_frame=lambda _r, _f: b"x",
            render_frame_fingerprint=b"renderer",
        )


@pytest.mark.parametrize("incomplete", [False, True])
def test_unavailable_capabilities_recompute_without_touching_store(tmp_path, monkeypatch, incomplete):
    import forge3d.anamnesis as anamnesis

    def unavailable():
        if incomplete:
            return {"granted": [], "limits": {}}
        raise RuntimeError("device capability probe unavailable")

    monkeypatch.setattr(anamnesis, "get_native_module", lambda: SimpleNamespace(capabilities=unavailable))
    calls = []

    def renderer(_recipe, _frame):
        calls.append(len(calls))
        return bytes([len(calls)])

    results = [render_sequence(
        {}, frames=[0], cache=tmp_path / "cache", render_frame=renderer,
        render_frame_fingerprint=b"renderer", render_frame_context=b"context",
    ) for _ in range(2)]
    assert len(calls) == 2
    assert results[0].frame_blobs != results[1].frame_blobs
    assert all(not result.cache_report.hits for result in results)
    assert all(result.bypass_reason == "capability_fingerprint_unavailable" for result in results)
    assert all(result.cache_report.bytes_written == 0 for result in results)
    assert not (tmp_path / "cache").exists()


def test_explicit_capability_backend_is_preserved_without_probing(tmp_path, monkeypatch):
    import forge3d.anamnesis as anamnesis

    probes = []

    def unexpected_probe():
        probes.append(True)
        raise AssertionError("explicit external capabilities must not probe a local GPU")

    monkeypatch.setattr(anamnesis, "get_native_module", lambda: SimpleNamespace(capabilities=unexpected_probe))
    calls = []

    def renderer(_recipe, _frame):
        calls.append(True)
        return b"same pixels"

    results = [render_sequence(
        {"anamnesis_state": {"backend": "declared-external-renderer"}},
        frames=[0], cache=tmp_path, render_frame=renderer,
        render_frame_fingerprint=b"renderer", render_frame_context=b"context",
        capabilities={"backend": backend, "granted": [], "limits": {}},
    ) for backend in ("vulkan", "dx12", "dx12")]
    assert not probes
    assert len(calls) == 2
    assert results[0].pass_keys["0:frame.output"] != results[1].pass_keys["0:frame.output"]
    assert results[1].pass_keys == results[2].pass_keys
    assert results[2].observed_recompute == []


def test_opaque_renderer_recipe_change_alone_cannot_serve_stale_hit(tmp_path):
    calls: list[str] = []

    def renderer(recipe, _frame):
        identity = str(recipe["scene"]["identity"])
        calls.append(identity)
        return identity.encode("ascii")

    recipe_a = {"scene": {"identity": "A"}, "output": {"format": "png"}}
    recipe_b = {"scene": {"identity": "B"}, "output": {"format": "png"}}
    first = render_sequence(
        recipe_a,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-external-context",
    )
    second = render_sequence(
        recipe_b,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-external-context",
    )

    assert first.frame_blobs == [b"A"]
    assert second.frame_blobs == [b"B"]
    assert calls == ["A", "B"]
    assert (0, "frame.output") in second.observed_recompute


def test_opaque_renderer_fingerprint_change_alone_cannot_serve_stale_hit(tmp_path):
    calls: list[str] = []

    def renderer_a(_recipe, _frame):
        calls.append("a")
        return b"A"

    def renderer_b(_recipe, _frame):
        calls.append("b")
        return b"B"

    recipe = {"scene": {"identity": "same"}, "output": {"format": "png"}}
    first = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_a,
        render_frame_fingerprint=b"renderer-A",
        render_frame_context=b"same-external-context",
    )
    second = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_b,
        render_frame_fingerprint=b"renderer-B",
        render_frame_context=b"same-external-context",
    )

    assert first.frame_blobs == [b"A"]
    assert second.frame_blobs == [b"B"]
    assert calls == ["a", "b"]
    assert (0, "frame.output") in second.observed_recompute


def test_opaque_renderer_context_change_alone_cannot_serve_stale_hit(tmp_path):
    calls: list[str] = []

    def renderer_a(_recipe, _frame):
        calls.append("a")
        return b"A"

    def renderer_b(_recipe, _frame):
        calls.append("b")
        return b"B"

    recipe = {"scene": {"identity": "same"}, "output": {"format": "png"}}
    first = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_a,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"captured-input-A",
    )
    second = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_b,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"captured-input-B",
    )

    assert first.frame_blobs == [b"A"]
    assert second.frame_blobs == [b"B"]
    assert calls == ["a", "b"]
    assert (0, "frame.output") in second.observed_recompute


def test_reference_work_factor_change_cannot_serve_stale_hit(tmp_path):
    recipe = {
        "terrain": {"dem": [0, 1]},
        "camera": {"eye": [1, 2, 3]},
        "output": {"format": "png"},
    }
    first = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        reference_work_factor=0,
     reference_only=True)
    second = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        reference_work_factor=1,
     reference_only=True)
    uncached_second = render_sequence(
        recipe,
        frames=[0],
        cache=None,
        reference_work_factor=1,
     reference_only=True)

    assert first.frame_blobs != second.frame_blobs
    assert second.frame_blobs == uncached_second.frame_blobs
    assert (0, "frame.output") in second.observed_recompute


def test_output_destination_is_proven_irrelevant_to_pixel_keys(tmp_path):
    calls: list[str] = []

    def renderer(_recipe, _frame):
        calls.append("render")
        return b"same pixels"

    first = {
        "terrain": {"dem": [0, 1]},
        "output": {"format": "png", "path": "first/result.png"},
    }
    second = {
        "terrain": {"dem": [0, 1]},
        "output": {"format": "png", "path": "second/result.png"},
    }
    render_sequence(
        first,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-captured-inputs",
        capabilities={"backend": "test-cpu", "granted": [], "limits": {}},
    )
    rerender = render_sequence(
        second,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-captured-inputs",
        capabilities={"backend": "test-cpu", "granted": [], "limits": {}},
    )
    assert calls == ["render"]
    assert rerender.observed_recompute == []
    assert rerender.predicted_recompute == []


def test_offscreen_helper_cache_is_inert_and_serves_identical_bytes(tmp_path):
    uncached = render_offscreen_rgba(8, 6, seed=9, frames=1, cache=None)
    first = render_offscreen_rgba(8, 6, seed=9, frames=1, cache=str(tmp_path))
    second = render_offscreen_rgba(8, 6, seed=9, frames=1, cache=str(tmp_path))
    assert uncached.tobytes() == first.tobytes() == second.tobytes()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_terrain_cache_restores_all_graph_passes(tmp_path):
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    params = f3d.TerrainRenderParams(_canonical_params_config()(64, 64))
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    uncached = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=None
    ).to_numpy()
    uncached_report = dict(renderer.last_anamnesis_cache_report)
    first = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native"
    ).to_numpy()
    first_report = dict(renderer.last_anamnesis_cache_report)
    second = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native"
    ).to_numpy()
    second_report = dict(renderer.last_anamnesis_cache_report)

    camera_config = _canonical_params_config()(64, 64)
    camera_config.cam_phi_deg = 55.0
    changed_camera_params = f3d.TerrainRenderParams(camera_config)
    camera_changed = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        changed_camera_params,
        heightmap,
        cache=tmp_path / "native",
    ).to_numpy()
    camera_changed_report = dict(renderer.last_anamnesis_cache_report)
    camera_restored = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        changed_camera_params,
        heightmap,
        cache=tmp_path / "native",
    ).to_numpy()
    camera_restored_report = dict(renderer.last_anamnesis_cache_report)

    changed_heightmap = heightmap.copy()
    changed_heightmap[0, 0] += np.float32(0.25)
    changed = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        params,
        changed_heightmap,
        cache=tmp_path / "native",
    ).to_numpy()
    changed_report = dict(renderer.last_anamnesis_cache_report)

    assert uncached.tobytes() == first.tobytes() == second.tobytes()
    assert uncached_report["hits"] == []
    assert uncached_report["misses"] == []
    assert uncached_report["bytes_read"] == 0
    assert uncached_report["bytes_written"] == 0
    assert uncached_report["graph_command_submissions"] == 3
    pass_labels = [
        "terrain.prepare",
        "terrain.shadow",
        "terrain.forward",
        "terrain.resolve",
    ]
    assert first_report["hits"] == []
    assert first_report["misses"] == pass_labels
    assert set(first_report["pass_keys"]) == set(pass_labels)
    assert first_report["pass_keys"] == second_report["pass_keys"]
    assert uncached_report["pass_keys"] == {}
    assert first_report["bytes_written"] > 0
    assert first_report["hit_rate"] == 0.0
    assert first_report["graph_command_submissions"] == 5
    assert second_report["hits"] == pass_labels
    assert second_report["misses"] == []
    assert second_report["bytes_read"] > 0
    assert second_report["bytes_written"] == 0
    assert second_report["hit_rate"] == 1.0
    assert second_report["graph_command_submissions"] == 0
    assert camera_changed_report["hits"] == []
    assert camera_changed_report["misses"] == pass_labels
    assert camera_changed_report["bytes_read"] == 0
    assert camera_changed_report["bytes_written"] > 0
    assert camera_changed_report["graph_command_submissions"] == 5
    assert camera_changed.tobytes() != second.tobytes()
    assert camera_restored_report["hits"] == pass_labels
    assert camera_restored_report["misses"] == []
    assert camera_restored_report["graph_command_submissions"] == 0
    assert camera_restored.tobytes() == camera_changed.tobytes()
    assert changed_report["hits"] == []
    assert changed_report["misses"] == pass_labels
    assert changed_report["bytes_written"] > 0
    assert changed_report["hit_rate"] == 0.0
    assert changed_report["graph_command_submissions"] == 5
    assert changed.tobytes() != second.tobytes()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
@pytest.mark.parametrize("technique", ["VSM", "EVSM", "MSM"])
def test_native_terrain_cache_recomputes_moment_shadow_techniques(tmp_path, technique):
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.shadows = ShadowSettings(
        True, technique, 512, 2, 250.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 9.0, 0.9
    )
    params = f3d.TerrainRenderParams(config)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    uncached = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=None
    ).to_numpy()
    cached = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native-moment"
    ).to_numpy()
    assert uncached.tobytes() == cached.tobytes()
    report = dict(renderer.last_anamnesis_cache_report)
    assert report["hits"] == []
    assert report["bytes_read"] == report["bytes_written"] == 0
    assert report["graph_command_submissions"] > 0
    assert not (tmp_path / "native-moment").exists()
    assert report["bypass_reason"] == "moment_shadows"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_terrain_cache_bypasses_participating_media(tmp_path):
    from forge3d.media import Medium

    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.media = Medium((0.01, 0.01, 0.01), (0.02, 0.02, 0.02), 1.0)
    config.camera_mode = "mesh:yup"
    params = f3d.TerrainRenderParams(config)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native-media"
    )
    report = dict(renderer.last_anamnesis_cache_report)
    assert report["hits"] == []
    assert report["bytes_read"] == report["bytes_written"] == 0
    assert report["graph_command_submissions"] > 0
    assert not (tmp_path / "native-media").exists()
    assert report["bypass_reason"] == "participating_media"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_aov_cache_option_recomputes_every_attachment(tmp_path):
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.aov = AovSettings(enabled=True)
    params = f3d.TerrainRenderParams(config)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    cold, cold_aov = renderer.render_with_aov(material_set, env_maps, params, heightmap, cache=None)
    fresh, fresh_aov = renderer.render_with_aov(material_set, env_maps, params, heightmap, cache=tmp_path / "aov-cache")
    assert cold.to_numpy().tobytes() == fresh.to_numpy().tobytes()
    for channel in ("albedo", "normal", "depth"):
        assert getattr(cold_aov, channel)().tobytes() == getattr(fresh_aov, channel)().tobytes()
    report = dict(renderer.last_anamnesis_cache_report)
    assert report["hits"] == [] and report["pass_keys"] == {}
    assert report["bytes_read"] == report["bytes_written"] == 0
    assert not (tmp_path / "aov-cache").exists()
    assert report["bypass_reason"] == "render_with_aov"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_cached_material_identity_tracks_uploaded_resources(tmp_path, monkeypatch):
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    monkeypatch.setenv("FORGE3D_MATERIAL_DIR", str(texture_dir))

    def write_textures(color):
        pixels = np.full((2, 2, 4), color, dtype=np.uint8)
        for material in ("rock", "grass", "dirt", "snow"):
            f3d.numpy_to_png(texture_dir / f"{material}_albedo.png", pixels)

    write_textures((255, 0, 0, 255))
    original_material = f3d.MaterialSet.terrain_default()
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    environment = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.camera_mode = "mesh"
    config.albedo_mode = "material"
    config.colormap_strength = 0.0
    params = f3d.TerrainRenderParams(config)
    height = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    def render(material, cache):
        frame = renderer.render_terrain_pbr_pom(material, environment, params, height, cache=cache)
        return frame.to_numpy().tobytes(), dict(renderer.last_anamnesis_cache_report)

    original, _ = render(original_material, None)
    write_textures((0, 0, 255, 255))
    resident, resident_report = render(original_material, tmp_path / "cache")
    assert resident == original, "the existing material owns its already uploaded textures"
    revised_material = f3d.MaterialSet.terrain_default()
    revised, _ = render(revised_material, None)
    assert revised != resident, "material revision control must visibly change pixels"
    restored, revised_report = render(revised_material, tmp_path / "cache")
    assert restored == revised, "new material served stale bytes from another GPU resource"
    assert revised_report["pass_keys"]["terrain.resolve"] != resident_report["pass_keys"]["terrain.resolve"]
    recalled, recalled_report = render(original_material, tmp_path / "cache")
    assert recalled == original
    assert recalled_report["pass_keys"] == resident_report["pass_keys"]
    assert len(recalled_report["hits"]) == 4 and not recalled_report["misses"]

    # The material loader's existing Ultra memory tier caps a dimension at
    # 4096. A solid 4097x1 source and its 4096x1 resolved image upload the same
    # mip chain; source dimensions must not describe a different GPU resource.
    def solid_strip(width):
        pixels = np.full((1, width, 4), (0, 0, 255, 255), dtype=np.uint8)
        for name in ("rock", "grass", "dirt", "snow"):
            f3d.numpy_to_png(texture_dir / f"{name}_albedo.png", pixels)
        return f3d.MaterialSet.terrain_default()

    oversized, oversized_report = render(solid_strip(4097), tmp_path / "resized-cache")
    resolved_material = solid_strip(4096)
    resolved_cold, _ = render(resolved_material, None)
    resolved, resolved_report = render(resolved_material, tmp_path / "resized-cache")
    assert oversized == resolved == resolved_cold
    assert oversized_report["pass_keys"] == resolved_report["pass_keys"]
    assert len(resolved_report["hits"]) == 4 and not resolved_report["misses"]


def _hdr_cache_worker(root):
    from dataclasses import replace
    from forge3d._native import get_native_module
    native = get_native_module()
    native.clear_native_degradations()
    root = Path(root)
    hdr = root / "environment.hdr"
    write_canonical_hdr(str(hdr))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.terrain_default()
    environment = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    config = replace(_canonical_params_config()(64, 64), camera_mode="mesh")
    height = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    params = f3d.TerrainRenderParams(config)
    frames, reports, hdr_exports = [], [], []
    for cache in (None, root / "cache", root / "cache"):
        frame = renderer.render_terrain_pbr_pom(material, environment, params, height, cache=cache)
        assert frame.format() == "Rgba16Float"
        frames.append(frame.to_numpy().tobytes())
        # Terrain Frame exposes lossless float export through EXR, while
        # to_numpy_f32 belongs to the separate HdrFrame type.
        export = root / f"readback-{len(frames)}.exr"
        frame.save(str(export))
        hdr_exports.append(export.read_bytes())
        reports.append(dict(renderer.last_anamnesis_cache_report))
    assert frames[0] == frames[1] == frames[2]
    assert hdr_exports[0] == hdr_exports[1] == hdr_exports[2]
    assert np.count_nonzero(np.frombuffer(frames[0], dtype=np.uint8).reshape(64, 64, 4)[..., :3]) > 0
    moved_config = replace(config, cam_phi_deg=config.cam_phi_deg + 20.0)
    moved = renderer.render_terrain_pbr_pom(material, environment, f3d.TerrainRenderParams(moved_config), height)
    assert frames[0] != moved.to_numpy().tobytes(), "camera control must visibly change HDR output"
    errors = [entry for entry in native.native_degradations() if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
    assert not errors, errors
    assert reports[2]["hits"] == reports[1]["misses"]
    assert not reports[2]["misses"] and reports[2]["graph_command_submissions"] == 0
    for label in ("terrain.forward", "terrain.resolve"):
        key = reports[1]["pass_keys"][label]
        blob = read_blob(root / "cache", key)
        assert int.from_bytes(blob[8:12], "little") == 4  # RGBA16F blob kind.
        assert len(blob) == 32 + config.size_px[0] * config.size_px[1] * 8
        rgba = np.frombuffer(blob, dtype="<f2", offset=32).reshape(64, 64, 4)
        assert np.isfinite(rgba).all() and np.count_nonzero(rgba[..., :3]) > 0
    print(json.dumps({"format": "Rgba16Float", "adapter": f3d.device_probe(), "restored_passes": reports[2]["hits"]}))


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_hdr_cache_preserves_eight_byte_texels(tmp_path):
    # The determinism policy is process-latched, so select it before importing
    # the extension in an independent process rather than mutate a live device.
    environment = dict(os.environ, FORGE3D_DETERMINISTIC="1")
    script = "import sys; sys.path.insert(0, 'tests'); from test_anamnesis_inertness import _hdr_cache_worker; _hdr_cache_worker(sys.argv[1])"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["adapter"]["software_fallback"] is False
    assert result["format"] == "Rgba16Float"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="VT bypass requires a physical GPU")
@pytest.mark.parametrize("active", [False, True])
def test_native_material_vt_cache_bypass_is_reported(tmp_path, active):
    from forge3d.terrain_params import TerrainVTSettings, VTLayerFamily

    hdr = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.terrain_default()
    environment = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    pixels = np.full((248, 248, 4), (128, 128, 255, 255), dtype=np.uint8)
    for material_index in range(material.material_count):
        renderer.register_material_vt_source(material_index, "albedo", pixels, (248, 248), None)
    config = _canonical_params_config()(64, 64)
    if active:
        config.vt = TerrainVTSettings(enabled=True, atlas_size=256,
            layers=[VTLayerFamily(family="albedo", virtual_size_px=(248, 248))])
    params = f3d.TerrainRenderParams(config)
    height = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    f3d.clear_native_degradations()
    fresh = renderer.render_terrain_pbr_pom(material, environment, params, height, cache=None).to_numpy().tobytes()
    cached = renderer.render_terrain_pbr_pom(material, environment, params, height, cache=tmp_path / "unused").to_numpy().tobytes()
    report = dict(renderer.last_anamnesis_cache_report)
    assert fresh == cached
    assert report["bypass_reason"] == "material_virtual_textures"
    assert report["hits"] == [] and report["pass_keys"] == {}
    assert report["bytes_read"] == report["bytes_written"] == 0
    assert not (tmp_path / "unused").exists()
    assert renderer._anamnesis_resource_fingerprint(material, environment) is None
    errors = [entry for entry in f3d.native_degradations() if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
    assert not errors, errors


@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="resident provenance feedback requires a physical GPU")
def test_native_provenance_feedback_survives_normal_derivatives(monkeypatch):
    """The derivative-safe forward/AOV path retains actual resident-page keys."""
    import test_provenance_veritas as vt

    monkeypatch.setenv("VF_SPEC_AA_ENABLED", "1")
    fixture = vt.provenance_render_env.__wrapped__()
    env = next(fixture)
    try:
        f3d.clear_native_degradations()
        renderer = env[0]
        render = vt.TestVeritasProvenanceDoD()._render_triple
        rgba, normal, depth, source_map, tiles = render(env)
        _, _, _, reversed_map, reversed_tiles = render(env, tuple(reversed(vt.RESIDENCY_SCHEDULE)))
        assert np.any(rgba[..., :3])
        np.testing.assert_array_equal(reversed_map, source_map)
        assert reversed_tiles == tiles
        expected = vt._expected_ids_from_authored_layout(normal, depth)
        covered = expected != vt.prov.SOURCE_ID_NONE
        assert int(covered.sum()) >= 1024  # Existing VERITAS authored-coverage gate.
        np.testing.assert_array_equal(source_map[covered], expected[covered])
        assert set(np.unique(source_map[covered]).tolist()) == {1, 2}
        albedo = [tile for tile in tiles if tile["family"] == "albedo"]
        assert {tile["mip_level"] for tile in albedo} == {0, 1}
        assert {1, 2} <= {tile["source_id"] for tile in albedo}
        assert renderer.get_material_vt_stats()["feedback_overflow"] == 0.0
        assert not [entry for entry in f3d.native_degradations()
                    if entry["kind"] in {"validation_error", "uncaptured_gpu_error"}]
    finally:
        fixture.close()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1", reason="native profile identity requires a physical GPU")
def test_unqualified_native_profile_retains_real_device_fingerprint(tmp_path, monkeypatch):
    from forge3d._native import get_native_module
    profile = "terra-determinata-native-portable-v1"
    monkeypatch.setenv("FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE", profile)
    native = get_native_module()
    native.clear_native_degradations()
    hdr = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.terrain_default()
    environment = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    params = f3d.TerrainRenderParams(_canonical_params_config()(64, 64))
    height = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    cache = tmp_path / "native"
    def render():
        rgba = renderer.render_terrain_pbr_pom(material, environment, params, height, cache=cache).to_numpy()
        return rgba.copy(), dict(renderer.last_anamnesis_cache_report)
    first, cold = render()
    restored, warm = render()
    assert first.tobytes() == restored.tobytes()
    assert np.count_nonzero(first[..., :3]) > 0
    assert not cold["hits"] and cold["graph_command_submissions"] == 5
    assert not warm["misses"] and warm["hit_rate"] == 1.0
    assert warm["graph_command_submissions"] == 0
    key = cold["pass_keys"]["terrain.forward"]
    metadata = packed_metadata(cache, key)
    capability = bytes.fromhex(metadata["derivation"]["material"]["capability_fingerprint_hex"])
    prefix = b"forge3d.anamnesis.compatibility-profile/v1\0" + profile.encode() + b"\0"
    assert capability.startswith(prefix)
    fingerprint = json.loads(capability[len(prefix):])
    adapter = f3d.device_probe()
    assert adapter["software_fallback"] is False
    assert fingerprint["backend"] == adapter["backend"].lower()
    assert fingerprint["limits"] and "dx12_compiler" in fingerprint
    import struct
    blob = read_blob(cache, key)
    assert blob[:8] == b"F3DANM02"
    assert struct.unpack_from("<I", blob, 8)[0] == 4  # BlobKind::Rgba16Float
    assert struct.unpack_from("<Q", blob, 24)[0] == 64 * 64 * 8
    monkeypatch.setenv("FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE", profile + "-mismatch")
    independent, mismatch = render()
    assert first.tobytes() == independent.tobytes()
    assert not mismatch["hits"] and mismatch["graph_command_submissions"] == 5
    import hashlib
    evidence = {
        "adapter": adapter, "deterministic": os.environ.get("FORGE3D_DETERMINISTIC"),
        "profile": profile, "fingerprint": fingerprint,
        "rgba_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "nonzero_rgb": int(np.count_nonzero(first[..., :3])),
        "hdr_texel_bytes": 8, "cold": cold, "restored": warm, "mismatch": mismatch,
        "gpu_errors": [entry for entry in native.native_degradations()
                       if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}],
    }
    (tmp_path / "profile-control.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"adapter": adapter, "deterministic": evidence["deterministic"],
                      "hdr_texel_bytes": 8, "warm_submissions": warm["graph_command_submissions"],
                      "mismatch_submissions": mismatch["graph_command_submissions"]}))
    assert not [entry for entry in native.native_degradations()
                if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
