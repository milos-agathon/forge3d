"""Device-free checks of the NEPHELE acceptance producers and gate verdicts."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts import nephele_acceptance as acceptance

FIXTURE = acceptance.FIXTURE


def _medium() -> dict:
    return json.loads((FIXTURE / "medium.json").read_text(encoding="utf-8"))


def test_majorant_probe_covers_domain_and_never_undercuts_the_medium() -> None:
    medium = _medium()
    pairs = acceptance.majorant_probe(medium, count=200_000)
    assert pairs.shape == (200_000, 2) and pairs.dtype == np.float32
    assert np.all(pairs[:, 0] <= pairs[:, 1])
    # A majorant grid that is too small must be caught.
    medium["majorant_cells"] = [value * 0.5 for value in medium["majorant_cells"]]
    lowered = acceptance.majorant_probe(medium, count=200_000)
    assert np.any(lowered[:, 0] > lowered[:, 1])


def test_comparator_matches_the_analytic_column_mean_for_a_uniform_medium() -> None:
    medium = _medium()
    medium["density_r16"] = [65535] * len(medium["density_r16"])
    span = medium["domain"]["bounds_max"][2] - medium["domain"]["bounds_min"][2]
    sigma_t = max(a + s for a, s in zip(medium["sigma_a"], medium["sigma_s"]))
    expected = math.exp(-sigma_t * span * medium["density_scale"])
    samples = 400_000
    se = math.sqrt(expected * (1 - expected) / samples)
    assert abs(acceptance.comparator_survival(medium, samples) - expected) < 4 * se


def _synthetic_artifact(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    slab = json.loads((acceptance.ROOT / "tests/nephele/homogeneous-slab.json").read_text(encoding="utf-8"))
    tau = slab["sigma_t_max_channel"] * slab["density"] * slab["distance"]
    n = acceptance.GATE1_SAMPLES
    np.save(tmp_path / "homogeneous-ratio-samples.npy", (rng.random(n) < math.exp(-tau)).astype(np.float64))
    heterogeneous = (rng.random(n) < 0.43).astype(np.float64)
    np.save(tmp_path / "heterogeneous-ratio-samples.npy", heterogeneous)
    (tmp_path / "heterogeneous-comparator.json").write_text(json.dumps(
        {"samples": acceptance.COMPARATOR_SAMPLES, "mean": float(heterogeneous.mean())}), encoding="utf-8")
    oracle = (1 - math.exp(-tau)) * slab["sigma_s"][0] / slab["sigma_t_max_channel"]
    rr = (rng.random(n) < oracle).astype(np.float64)
    np.save(tmp_path / "rr-contributions.npy", np.column_stack([rr, rr]))
    np.save(tmp_path / "majorant-probe-pairs.npy", np.tile(np.float32([[1.0, 2.0]]), (acceptance.MAJORANT_PROBES, 1)))
    outcome = rng.integers(0, 3, acceptance.GATE2_SAMPLES)
    outcomes = np.stack([outcome == k for k in range(3)], axis=1).astype(np.uint8)
    outcomes[:1000] = 0  # independent ensembles do not close sample by sample
    outcomes[1000:2000] = 1
    np.save(tmp_path / "gate2-energy-outcomes.npy", outcomes)
    reference = np.load(FIXTURE / "reference-rgb.npy")
    for name in ("realtime-rgb.npy", "froxel-frame-run-a.npy", "froxel-frame-run-b.npy"):
        np.save(tmp_path / name, reference)
    np.save(tmp_path / "medium-disabled-rgb.npy", np.zeros_like(reference))
    np.save(tmp_path / "terrain-occlusion-disabled-rgb.npy", 255 - reference)
    slices = np.load(FIXTURE / "reference-terrain-slice.npy")
    termination = np.floor(np.clip(np.where(np.isfinite(slices), slices, 0), 0, 63)).astype(np.float32)
    shaft = np.load(FIXTURE / "shaft-mask.npy")
    termination[shaft] = np.floor(slices[shaft])
    np.save(tmp_path / "realtime-termination-slice.npy", termination)
    for name in ("transmittance.npy", "in-scatter.npy", "cloud-shadow-aov.npy", "optical-depth.npy"):
        np.save(tmp_path / name, np.ones((64, 64, 3), dtype=np.float32))
    (tmp_path / "gate5-static.json").write_text(json.dumps({
        "unresolved_source_expressions": [], "naga_validated_assemblies": 2, "resolved_source_sha256": {"a": "", "b": ""},
        "executed_assembled_naga_contracts": ["a", "b"], "compute_entry_texture_sample_compare_calls": 0,
        "stale_disabled_shadow_comments": 0,
    }), encoding="utf-8")
    diagnostics = {
        "backend": "Vulkan", "terrain_shading_model": "lambert_physical", "temporal_history_decision": "rejected",
        "majorant_valid": True, "executed_multi_scatter": True, "f16_flushed_max_value": 0.0,
        "f16_flushed_sample_count": 0, "step_count": int(termination.sum() + termination.size),
        "sun_transmittance_method": acceptance.SUN_TRANSMITTANCE_METHOD,
        "sun_transmittance_bias": acceptance.SUN_TRANSMITTANCE_BIAS, "sun_transmittance_max_segment_length": 1.0,
        "sun_transmittance_executed_steps": 4, "sun_transmittance_max_abs_error": 0.01,
    }
    for label, pid in (("a", 1), ("b", 2)):
        (tmp_path / f"run-{label}.json").write_text(json.dumps({"process_id": pid, "diagnostics": diagnostics}), encoding="utf-8")
    (tmp_path / "memory.json").write_text(json.dumps({
        "output_dimensions": [64, 64], "froxel_depth": 64, "readback_bytes": 64 * 64 * 64,
        "peak_host_visible_bytes": 1 << 20,
    }), encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _synthetic_artifact(tmp_path_factory.mktemp("nephele-artifact"))


@pytest.mark.parametrize("name", ["gate1", "gate2", "gate3", "gate5", "gate6"])
def test_consistent_evidence_passes(artifact: Path, name: str) -> None:
    assert acceptance.GATES[name](artifact)["failures"] == []


def test_gate4_reports_each_failed_criterion(artifact: Path) -> None:
    # Identical shadows pass; the inverted ablation drops SSIM; the black
    # no-medium frame changes every terrain pixel.
    assert acceptance.gate4(artifact)["failures"] == []


def test_failures_are_named(tmp_path: Path, artifact: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(artifact, broken)
    np.save(broken / "froxel-frame-run-b.npy", np.zeros((64, 64, 3), dtype=np.uint8))
    np.save(broken / "terrain-occlusion-disabled-rgb.npy", np.load(broken / "realtime-rgb.npy"))
    assert "frames bitwise identical" in acceptance.gate6(broken)["failures"]
    assert acceptance.gate4(broken)["failures"] == ["disabling terrain occlusion drops god-ray ROI SSIM below 0.80"]
    for name in ("verification-report.json", "lane-ran.json"):
        (broken / name).write_text("{}", encoding="utf-8")
    assert not acceptance.report(broken, "0" * 40)
    assert (broken / "verification-error.json").is_file()
    assert not (broken / "lane-ran.json").exists()
    assert not (broken / "verification-report.json").exists()
