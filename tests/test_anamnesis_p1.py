from __future__ import annotations

import os
from pathlib import Path

import forge3d as f3d
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _job(workflow: str, name: str, next_name: str) -> str:
    return workflow.split(f"  {name}:", 1)[1].split(f"\n  {next_name}:", 1)[0]


def test_production_renderers_consume_real_framegraph_barriers():
    framegraph = (ROOT / "src/core/framegraph_impl/mod.rs").read_text(encoding="utf-8")
    forward = (ROOT / "src/offscreen/forward.rs").read_text(encoding="utf-8")
    terrain = (ROOT / "src/terrain/renderer/draw/mod.rs").read_text(encoding="utf-8")
    terrain_aov = (ROOT / "src/terrain/renderer/aov.rs").read_text(encoding="utf-8")
    diagnostics = (ROOT / "src/py_functions/diagnostics.rs").read_text(encoding="utf-8")

    assert "pub fn execute_with_barriers" in framegraph
    assert 'execute_with_barriers("offscreen.forward"' in forward
    assert 'execute_with_barriers("offscreen.readback"' in forward
    assert "offscreen readback lost its compiled color transition" in forward
    assert 'execute_with_barriers("terrain.forward"' in terrain
    assert 'execute_with_barriers("terrain.resolve"' in terrain
    assert "lost its compiled shadow transition" in terrain
    assert "lost its compiled beauty transition" in terrain
    assert 'execute_with_barriers("terrain.forward_aov"' in terrain_aov
    assert 'execute_with_barriers("terrain.resolve_aov"' in terrain_aov
    assert "last_renderer_graph_report" in diagnostics
    assert "RendererGraphBuilder" not in diagnostics
    assert "compile_renderer_graph" not in diagnostics

    direct_constructors = []
    for source in (ROOT / "src").rglob("*.rs"):
        if source == ROOT / "src/core/framegraph_impl/mod.rs":
            continue
        if "FrameGraph::new()" in source.read_text(encoding="utf-8"):
            direct_constructors.append(source.relative_to(ROOT).as_posix())
    assert direct_constructors == []


def test_p1_rejecting_controls_cover_recompute_and_complete_disk_bound():
    inertness = (ROOT / "tests/test_anamnesis_inertness.py").read_text(encoding="utf-8")
    incremental = (ROOT / "tests/test_anamnesis_incremental.py").read_text(
        encoding="utf-8"
    )
    adversarial = (ROOT / "tests/test_anamnesis_adversarial_keys.py").read_text(
        encoding="utf-8"
    )

    assert "test_opaque_renderer_recipe_change_alone_cannot_serve_stale_hit" in inertness
    assert "test_output_destination_is_proven_irrelevant_to_pixel_keys" in inertness
    assert "set(incremental.predicted_recompute)" in incremental
    assert "set(incremental.observed_recompute)" in incremental
    assert "test_complete_store_footprint_is_hard_bounded" in adversarial
    assert "test_tiny_budget_rejects_unrepresentable_self_describing_entry" in adversarial


def test_p1_portability_and_production_lanes_fail_closed():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix = (ROOT / ".github/workflows/determinism-matrix.yml").read_text(
        encoding="utf-8"
    )

    hosted_seed = _job(matrix, "anamnesis-seed", "anamnesis-portability")
    hosted_consumer = _job(matrix, "anamnesis-portability", "diff")
    for job in (hosted_seed, hosted_consumer):
        assert "scripts/terrain_ci_probe.py" in job
        assert "probe_status" in job
        assert 'exit "$probe_status"' in job
        assert "grep -Eq" not in job

    physical_seed = _job(
        ci, "test-anamnesis-portability-seed", "test-anamnesis-portability"
    )
    physical_consumer = _job(
        ci, "test-anamnesis-portability", "test-anamnesis-production"
    )
    production = _job(ci, "test-anamnesis-production", "build-docs")
    for job in (physical_seed, physical_consumer, production):
        assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
        assert "Prove exact-head checkout" in job
        assert "ANAMNESIS.ABSENT" not in job
    assert "--require-nvidia-vulkan" in physical_seed
    assert '--json "anamnesis-portability-producer-adapter.json"' in physical_seed
    assert "RUNNER_TEMP/anamnesis-portability-producer-adapter.json" not in physical_seed
    assert "WGPU_BACKEND: dx12" in physical_consumer
    assert '--json "anamnesis-portability-consumer-adapter.json"' in physical_consumer
    assert "RUNNER_TEMP/anamnesis-portability-consumer-adapter.json" not in physical_consumer
    assert "anamnesis-portability-result.json" in physical_consumer
    assert "test_real_gpu_600_frame_acceptance" in production
    assert "test_public_gpu_graph_cache_restores_intermediate_texture" in production
    assert "scripts/assert_junit_zero_skips.py" in production


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_public_gpu_graph_cache_restores_intermediate_texture(tmp_path):
    _, cold_raster, cold_meta = f3d.render_adjudication_pair(
        16, 12, 1, cache=tmp_path
    )
    _, warm_raster, warm_meta = f3d.render_adjudication_pair(
        16, 12, 1, cache=tmp_path
    )

    cold_cache = dict(cold_meta["cache"])
    warm_cache = dict(warm_meta["cache"])
    assert cold_cache["hits"] == []
    assert cold_cache["misses"] == ["offscreen.forward", "offscreen.readback"]
    assert warm_cache["hits"] == ["offscreen.forward"]
    assert warm_cache["misses"] == ["offscreen.readback"]
    assert warm_cache["bytes_read"] > 0
    assert warm_cache["wall_ms_saved"] > 0.0
    np.testing.assert_array_equal(
        np.asarray(warm_raster),
        np.asarray(cold_raster),
        err_msg="rehydrated GPU texture differs from the cold rendered resource",
    )
