from __future__ import annotations

import copy
import os

import pytest

from forge3d.anamnesis import render_sequence


def _recipe(text: str) -> dict:
    return {
        "terrain": {"dem_sha256": "11" * 32, "z_scale": 1.5},
        "atmosphere": {"model": "clear"},
        "lighting": {"azimuth": 215.0, "elevation": 35.0},
        "camera": {"path": "deterministic-flythrough-v1"},
        "layers": [
            {
                "kind": "label",
                "id": "summit",
                "text": text,
                "visible_frames": list(range(250, 260)),
            }
        ],
        "output": {"width": 64, "height": 64, "samples": 4},
    }


def test_600_frame_incremental_matches_cold_and_recomputes_exact_set(tmp_path):
    warm_store = tmp_path / "warm"
    cold_store = tmp_path / "cold-modified"
    original = _recipe("Old summit")
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"

    render_sequence(original, cache=warm_store)
    incremental = render_sequence(modified, cache=warm_store)
    cold_modified = render_sequence(modified, cache=cold_store)

    assert incremental.frame_hashes == cold_modified.frame_hashes
    assert len(incremental.frame_hashes) == 600
    assert incremental.prediction_matches
    assert set(incremental.predicted_recompute) == set(incremental.observed_recompute)
    assert len(incremental.observed_recompute) == 21
    labels = {label for _, label in incremental.observed_recompute}
    assert labels == {"label.compile", "label.composite", "frame.output"}
    assert incremental.cache_report.hit_rate > 0.99


def test_dry_run_predicts_without_writing(tmp_path):
    store = tmp_path / "dry"
    result = render_sequence(_recipe("Summit"), frames=range(3), cache=store, dry_run=True)
    assert result.predicted_recompute
    assert result.observed_recompute == []
    assert not any(path.name == "blob" for path in store.rglob("blob"))


def test_600_frame_incremental_speedup_is_at_least_20x(tmp_path):
    original = _recipe("Old summit")
    original["layers"][0]["visible_frames"] = [250]
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    options = {"verify_reads": False, "reference_work_factor": 10_000}

    render_sequence(original, cache=tmp_path / "warm-speed", **options)
    incremental = render_sequence(modified, cache=tmp_path / "warm-speed", **options)
    cold = render_sequence(modified, cache=tmp_path / "cold-speed", **options)

    assert incremental.frame_hashes == cold.frame_hashes
    assert incremental.prediction_matches
    speedup = cold.elapsed_seconds / incremental.elapsed_seconds
    assert speedup >= 20.0, (
        f"cold={cold.elapsed_seconds:.3f}s incremental={incremental.elapsed_seconds:.3f}s "
        f"speedup={speedup:.2f}x"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_real_gpu_600_frame_acceptance(tmp_path):
    from anamnesis_gpu_acceptance import run_acceptance

    result = run_acceptance(tmp_path)
    assert result["matching_frames"] == 600
    assert result["symmetric_difference"] == []
    assert result["speedup"] >= 20.0
