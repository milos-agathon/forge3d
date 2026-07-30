from __future__ import annotations

import os

import numpy as np
import pytest

import forge3d as f3d
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)
from forge3d.terrain_params import ShadowSettings


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_vsm_shadow_cache_regenerates_moments_from_restored_depth(tmp_path):
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    session = f3d.Session(window=False)
    seed_renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.shadows = ShadowSettings(
        True, "VSM", 512, 2, 250.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 9.0, 0.9
    )
    seed_params = f3d.TerrainRenderParams(config)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    cache = tmp_path / "native-vsm"

    seed_renderer.render_terrain_pbr_pom(
        material_set, env_maps, seed_params, heightmap, cache=cache
    ).to_numpy()
    cold_report = dict(seed_renderer.last_anamnesis_cache_report)

    changed_config = _canonical_params_config()(64, 64)
    changed_config.cam_phi_deg = 55.0
    changed_config.shadows = config.shadows
    changed_params = f3d.TerrainRenderParams(changed_config)
    restored_renderer = f3d.TerrainRenderer(session)
    restored = restored_renderer.render_terrain_pbr_pom(
        material_set, env_maps, changed_params, heightmap, cache=cache
    ).to_numpy()
    restored_report = dict(restored_renderer.last_anamnesis_cache_report)
    reference_renderer = f3d.TerrainRenderer(session)
    reference = reference_renderer.render_terrain_pbr_pom(
        material_set, env_maps, changed_params, heightmap, cache=None
    ).to_numpy()

    assert cold_report["misses"] == [
        "terrain.prepare",
        "terrain.shadow",
        "terrain.forward",
        "terrain.resolve",
    ]
    assert restored_report["hits"] == ["terrain.shadow"]
    assert restored_report["misses"] == [
        "terrain.prepare",
        "terrain.forward",
        "terrain.resolve",
    ]
    assert restored_report["graph_command_submissions"] > 0
    assert restored.tobytes() == reference.tobytes()
