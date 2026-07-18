"""EUCLIDEA hard gate for adaptive exact planar predicates."""

from __future__ import annotations

import pytest

from forge3d._native import NATIVE_AVAILABLE, get_native_module

if not NATIVE_AVAILABLE:
    pytest.skip("exact predicate gate requires the native extension", allow_module_level=True)

_native = get_native_module()


def test_ten_million_random_and_one_million_adversarial_predicates() -> None:
    report = _native._euclidea_predicate_report()
    assert report["random_cases"] == 10_000_000
    assert report["adversarial_cases"] == 1_000_000
    assert report["orientation_cases"] + report["incircle_cases"] == 11_000_000
    assert report["exact_errors"] == 0, report
    assert report["filter_rejects"] > 0, report
    # This negative control proves the adversarial family exercises a real
    # robustness problem rather than merely replaying easy random inputs.
    assert report["ablation_errors"] >= 10_000, report
