from __future__ import annotations

import copy
import pathlib
import tomllib

import pytest

from forge3d._native import NATIVE_AVAILABLE

if not NATIVE_AVAILABLE:
    pytest.skip("shader proof tests require the compiled native extension", allow_module_level=True)

from forge3d import verify


ROOT = pathlib.Path(__file__).resolve().parent.parent


def _stable(report: dict) -> dict:
    report = copy.deepcopy(report)
    for verdict in report["verdicts"]:
        verdict["timing_ms"] = 0.0
    report["ablations"]["height_range_div"]["timing_ms"] = 0.0
    report["ablations"]["pt_shade_delete_guard"]["timing_ms"] = 0.0
    report["unsafe_fixture"]["timing_ms"] = 0.0
    return report


def test_proven_list_clean_and_large_enough():
    report = verify.shader_report()
    assert report["status"] == "ok"
    assert report["proven_count"] >= 10
    assert report["module_count"] >= 8
    assert report["suppressions"] == []
    assert all(v["proof_status"] == "proven" for v in report["verdicts"])
    assert sum(v["timing_ms"] for v in report["verdicts"]) < 300_000


def test_unsafe_fixture_rejected_and_accumulation_proved():
    report = verify.shader_report()
    unsafe = report["unsafe_fixture"]
    assert unsafe["proof_status"] == "unproven"
    assert unsafe["alarms"][0]["kind"] == "possible_zero_division"
    assert "denom == 0" in unsafe["alarms"][0]["detail"]

    hybrid = next(v for v in report["verdicts"] if v["module"] == "hybrid_terrain_traversal")
    assert hybrid["proof_status"] == "proven"
    assert hybrid["alarms"] == []
    source = (ROOT / "src/shaders/hybrid_terrain_traversal.wgsl").read_text(encoding="utf-8")
    assert "prev.a + 1.0" in source
    assert "acc.rgb / acc.a" in source


def test_seeded_ablations_are_caught():
    report = verify.shader_report()
    div = report["ablations"]["height_range_div"]
    assert div["proof_status"] == "unproven"
    assert any(a["kind"] == "possible_zero_division" for a in div["alarms"])

    guard = report["ablations"]["pt_shade_delete_guard"]
    assert guard["proof_status"] == "unproven"
    assert any(a["kind"] == "missing_guard" for a in guard["alarms"])


def test_runtime_contract_assert_mode_reports_pass_and_failure_fixture():
    assert verify.shader_report()["runtime_assert"]["status"] == "passed"
    falsified = verify.shader_report("falsified_contract")
    assert falsified["runtime_assert"]["status"] == "failed"
    assert "falsified contract" in falsified["runtime_assert"]["alarm"]


def test_verdicts_are_deterministic_and_containment_checked():
    first = _stable(verify.shader_report())
    second = _stable(verify.shader_report())
    assert first == second
    assert first["containment"]["status"] == "passed"
    assert first["containment"]["samples_per_op"] == 1_000_000


def test_ledger_is_exhaustive_and_unexpired():
    report = verify.shader_report()
    assert report["ledger"]["missing"] == []
    assert report["ledger"]["expired"] == []

    ledger = tomllib.loads((ROOT / "tests/shader_proofs_ledger.toml").read_text(encoding="utf-8"))
    for row in ledger["unproven"]:
        assert row["path"].startswith("src/shaders/")
        assert row["reason"]
        assert row["owner"]
        assert row["expiry"] >= "2026-07-17"
