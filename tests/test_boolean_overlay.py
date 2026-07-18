"""EUCLIDEA deterministic boolean-topology and source hard gates."""

from __future__ import annotations

from pathlib import Path

import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE, get_native_module

from _geomfuzz import SeededPolygonCorpus, shrink_failure

if not NATIVE_AVAILABLE:
    pytest.skip("boolean overlay gate requires the native extension", allow_module_level=True)

_native = get_native_module()
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORPUS_SHA256 = "d2fe056ce87439e5ac8d5eff579bc183d2bc5fda263df04f30d94960d54601bd"


def test_one_hundred_thousand_pairs_are_valid_conservative_and_deterministic() -> None:
    report = _native._euclidea_boolean_fuzz_report()
    assert report["cases"] == 100_000
    assert report["operations"] == 400_000
    assert report["errors"] == 0, report
    assert report["panics"] == 0, report
    assert report["invalid_outputs"] == 0, report
    assert report["snap_bound_violations"] == 0, report
    assert report["conservation_violations"] == 0, report
    assert report["hash_a"] == report["hash_b"], report
    # The same fixed digest is asserted by every OS/Python wheel lane, turning
    # per-platform replay into a cross-platform byte-determinism gate.
    assert report["hash_a"] == EXPECTED_CORPUS_SHA256, report
    if report["oracle_enabled"]:
        assert report["oracle_cases"] == 2_000
        assert report["oracle_disagreements"] == 0, report
        assert report["benchmark_ratio"] <= 3.0, report


@pytest.mark.parametrize("operation", [gis.union, gis.intersection, gis.difference])
def test_seeded_python_generators_cover_pathological_polygon_families(operation) -> None:
    for left, right in SeededPolygonCorpus().pairs(120):
        if not gis.is_valid(left)["valid"] or not gis.is_valid(right)["valid"]:
            continue
        output = operation([left, right], crs="EPSG:3857")
        assert output["geometry"] is None or gis.is_valid(output["geometry"])["valid"]


def test_shrinker_negative_control_reaches_triangles() -> None:
    pair = next(SeededPolygonCorpus().pairs(1))
    shrunk = shrink_failure(pair, lambda _pair: True)
    for geometry in shrunk:
        assert len(geometry["coordinates"][0]) == 4


def test_validity_checker_rejects_seeded_star_self_intersections() -> None:
    for star in SeededPolygonCorpus().invalid_stars(100):
        report = gis.is_valid(star)
        assert not report["valid"], report
        assert report["reasons"], report


def test_overlay_geometry_decisions_use_exact_ordering_and_ordered_maps() -> None:
    source_dir = ROOT / "src" / "geometry" / "overlay"
    decision_files = [
        source_dir / name
        for name in ("faces.rs", "rings.rs", "sweep.rs", "validity.rs")
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in decision_files)
    assert "HashMap" not in combined
    assert "partial_cmp" not in combined
    assert "orient2d(" in combined
    assert "exact_cross(" in combined
    # The sole raw shoelace cross is isolated in verification.rs for area
    # measurement; topology decisions must never use this sign pattern.
    assert ".x *" not in combined or ".y -" not in combined
