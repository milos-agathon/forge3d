from __future__ import annotations

import os

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


def test_cache_none_is_byte_identical_to_enabled_cold_render(tmp_path):
    recipe = {
        "terrain": {"dem_sha256": "ab" * 32},
        "camera": {"path": "inertness"},
        "layers": [{"kind": "label", "text": "A", "visible_frames": [1]}],
        "output": {"width": 32, "height": 32, "samples": 2},
    }
    uncached = render_sequence(recipe, frames=range(4), cache=None)
    cached = render_sequence(recipe, frames=range(4), cache=tmp_path)
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
def test_native_terrain_cache_skips_encoder_and_invalidates_dem(tmp_path):
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
    first = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native"
    ).to_numpy()
    first_report = dict(renderer.last_anamnesis_cache_report)
    second = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native"
    ).to_numpy()
    second_report = dict(renderer.last_anamnesis_cache_report)

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
    assert first_report["misses"] == ["terrain.one_shot.final"]
    assert second_report["hits"] == ["terrain.one_shot.final"]
    assert changed_report["misses"] == ["terrain.one_shot.final"]
    assert changed.tobytes() != second.tobytes()
