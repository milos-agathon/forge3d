"""TERMINUS torture-atlas gate."""

from __future__ import annotations

from collections import Counter
import json

import pytest

from forge3d._native import NATIVE_AVAILABLE

from _torture import (
    TORTURE_ROOT,
    TortureWorker,
    evaluate_case,
    format_scoreboard,
    load_cases,
    scoreboard,
)
from tests._fuzz import failure_preserving_shrink_proof, replay_case, shrink_validation_transcript
from tests._torture_materiality import derive_coverage, validate_materiality


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="TERMINUS torture atlas requires the compiled forge3d native extension",
)


def test_torture_manifest_is_complete():
    manifest = json.loads((TORTURE_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    cases = load_cases()
    counts = Counter(case["family"] for case in cases)

    assert len(cases) == 500
    assert dict(sorted(counts.items())) == dict(sorted(manifest["families"].items()))
    assert manifest["total_cases"] == 500
    assert manifest["semantic_value_cases"] >= 50
    assert sum(1 for case in cases if case.get("semantic")) == manifest["semantic_value_cases"]
    assert all(
        case.get("expect", {}).get("type")
        for case in cases
        if case.get("expect", {}).get("class") in {"error", "structured_error"}
    ), "every expected error must lock its Python exception type"

    signatures = {
        json.dumps(
            {
                "operation": case["operation"],
                "payload": case.get("payload", {}),
                "expect": case.get("expect", {}),
            },
            sort_keys=True,
        )
        for case in cases
    }
    assert len(signatures) == len(cases), "torture descriptors must exercise distinct inputs"


def _coverage():
    return json.loads((TORTURE_ROOT / "COVERAGE.json").read_text(encoding="utf-8"))["cases"]


def test_every_torture_case_has_material_executable_coverage():
    errors = validate_materiality(load_cases(), _coverage())
    assert errors == [], "TERMINUS materiality gate failed:\n" + "\n".join(errors)


def _materiality_case(
    case_id,
    payload,
    *,
    operation="gis_validate_geometry",
    expect=None,
    notes="specific test rationale",
):
    return {
        "id": case_id,
        "family": "geometry",
        "operation": operation,
        "payload": payload,
        "expect": expect or {"class": "ok"},
        "notes": notes,
    }


def _coverage_entries(*cases):
    return [derive_coverage(case) for case in cases]


def test_materiality_rejects_translated_rotated_or_reversed_polygons():
    square = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
    }
    translated = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[11, 21], [11, 20], [10, 20], [10, 21], [11, 21]]],
        }
    }
    cases = [_materiality_case("square-a", square), _materiality_case("square-b", translated)]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_irrelevant_constant_dem_fill_changes():
    cases = [
        {
            "id": "dem-a",
            "family": "dems",
            "operation": "dem_derive_water_mask",
            "payload": {"array": {"shape": [4, 4], "fill": 0.25, "dtype": "float32"}},
            "expect": {"class": "ok"},
            "notes": "positive finite constant DEM",
        },
        {
            "id": "dem-b",
            "family": "dems",
            "operation": "dem_derive_water_mask",
            "payload": {"array": {"shape": [8, 8], "fill": 0.75, "dtype": "float32"}},
            "expect": {"class": "ok"},
            "notes": "another positive finite constant DEM",
        },
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_note_or_identifier_only_uniqueness():
    payload = {"geometry": {"type": "Point", "coordinates": [0, 0]}}
    cases = [
        _materiality_case("metadata-a", payload, notes="first explanation"),
        _materiality_case("metadata-b", payload, notes="second explanation"),
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_numeric_string_only_uniqueness():
    base = {
        "src_crs": "EPSG:4326",
        "dst_crs": "EPSG:3857",
        "x": 1.0,
        "y": 2.0,
    }
    equivalent = {**base, "x": "1.0", "y": "2.0"}
    cases = [
        _materiality_case("numeric-a", base, operation="gis_transform_point"),
        _materiality_case("numeric-b", equivalent, operation="gis_transform_point"),
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_raw_array_only_uniqueness():
    raw = {"positions": [[0.0, 0.0], [1.0, 1.0]], "point_size": 4.0}
    values = {
        "positions": {"dtype": "float64", "values": [[0.0, 0.0], [1.0, 1.0]]},
        "point_size": 4.0,
    }
    cases = [
        _materiality_case("array-a", raw, operation="vector_add_points"),
        _materiality_case("array-b", values, operation="vector_add_points"),
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_error_message_match_only_uniqueness():
    payload = {"geometry": {"type": "Point", "coordinates": ["nan", 0.0]}}
    cases = [
        _materiality_case(
            "match-a",
            payload,
            expect={"class": "error", "type": "ValueError", "match": "invalid"},
        ),
        _materiality_case(
            "match-b",
            payload,
            expect={"class": "error", "type": "ValueError", "match": "invalid_argument"},
        ),
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_unreachable_vector_point_positions():
    cases = [
        _materiality_case(
            "point-size-a",
            {"positions": {"dtype": "float64", "values": [[0.0, 0.0]]}, "point_size": 0.0},
            operation="vector_add_points",
        ),
        _materiality_case(
            "point-size-b",
            {"positions": {"dtype": "float64", "values": [[180.0, 0.0]]}, "point_size": -1.0},
            operation="vector_add_points",
        ),
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_rejects_unreachable_vector_line_paths():
    cases = [
        _materiality_case(
            "line-width-a",
            {
                "path": {"dtype": "float64", "values": [[0.0, 0.0], [1.0, 1.0]]},
                "stroke_width": 0.0,
            },
            operation="vector_add_lines",
        ),
        _materiality_case(
            "line-width-b",
            {
                "path": {"dtype": "float64", "values": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]},
                "stroke_width": -1.0,
            },
            operation="vector_add_lines",
        ),
    ]
    errors = validate_materiality(cases, _coverage_entries(*cases))
    assert any("materiality collision" in error for error in errors)


def test_materiality_accepts_different_pathology_boundary_and_oracle():
    valid = _materiality_case(
        "valid-square",
        {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]]],
            }
        },
    )
    invalid = {
        **_materiality_case(
            "non-finite-point",
            {"geometry": {"type": "Point", "coordinates": ["nan", 0]}},
        ),
        "expect": {"class": "error", "type": "ValueError", "match": "InvalidGeometry"},
    }
    errors = validate_materiality([valid, invalid], _coverage_entries(valid, invalid))
    assert errors == []


def test_materiality_rejects_generic_rationale_even_for_unique_payload():
    case = _materiality_case(
        "generic",
        {"geometry": {"type": "Point", "coordinates": [0, 0]}},
        notes="distinct adversarial input 17",
    )
    errors = validate_materiality([case], _coverage_entries(case))
    assert any("generic padding rationale" in error for error in errors)


def test_torture_atlas_all_cases_green(tmp_path, capsys):
    cases = load_cases()
    outcomes = []
    failures = []

    with TortureWorker() as worker:
        for case in cases:
            outcome = evaluate_case(case, tmp_path=tmp_path, worker=worker)
            if outcome["class"] == "wrong_value":
                failures.append(
                    (case["id"], case.get("_path"), outcome["expectation_errors"], outcome)
                )
            outcomes.append(outcome)

    board = scoreboard(outcomes)
    print("TERMINUS atlas scoreboard:", format_scoreboard(board))
    print(
        "TERMINUS family scoreboard:",
        dict(sorted(Counter(case["family"] for case in cases).items())),
    )

    assert board["total"] == 500
    assert board["panic"] == 0
    assert board["hang"] == 0
    assert board["wrong_value"] == 0, failures[:5]
    assert not failures


def test_torture_atlas_deterministic(tmp_path):
    cases = load_cases()
    with TortureWorker() as worker:
        first = [evaluate_case(case, tmp_path=tmp_path, worker=worker) for case in cases]
    with TortureWorker() as worker:
        second = [evaluate_case(case, tmp_path=tmp_path, worker=worker) for case in cases]
    assert scoreboard(first) == scoreboard(second)
    assert first == second


def test_torture_worker_classifies_timeout_and_process_death(tmp_path):
    with TortureWorker() as worker:
        hang = worker.classify(
            {
                "operation": "_harness_sleep",
                "payload": {"seconds": 1.0},
                "watchdog_seconds": 0.05,
            },
            tmp_path,
        )
        crash = worker.classify(
            {"operation": "_harness_exit", "payload": {"code": 86}},
            tmp_path,
        )
    assert hang["class"] == "hang"
    assert crash["class"] == "panic"
    assert "exit_code=86" in crash["message"]


def test_fuzzer_seed_replay_and_shrinker_are_deterministic():
    assert replay_case(42, 137) == replay_case(42, 137)
    first = failure_preserving_shrink_proof(42, 137)
    second = failure_preserving_shrink_proof(42, 137)
    assert first == second
    assert first["initial"]["payload"]["array"]["values"] != [[2.0]]
    assert first["minimal"]["payload"]["array"]["values"] == [[2.0]]
    assert all(
        any(float(value) > 1.0 for row in case["payload"]["array"]["values"] for value in row)
        for case in first["accepted"]
    )
    transcript = shrink_validation_transcript()
    assert "shrink before=" in transcript
    assert "shrink after=" in transcript
    assert '"values": [[2.0]]' in transcript
    assert "failure_preserved=true" in transcript


def test_fuzzer_projection_cases_stay_inside_area_of_use():
    for index in range(2_000):
        case = replay_case(42, index)
        if case["family"] != "crs":
            continue
        payload = case["payload"]
        assert payload["src_crs"] == "EPSG:4326"
        x = float(payload["x"])
        y = float(payload["y"])
        if payload["dst_crs"] == "EPSG:3857":
            assert -170.0 <= x <= 170.0
            assert -80.0 <= y <= 80.0
        elif payload["dst_crs"] == "EPSG:32631":
            assert 0.0 <= x <= 6.0
            assert 0.0 <= y <= 80.0
        elif payload["dst_crs"] == "EPSG:32733":
            assert 12.0 <= x <= 18.0
            assert -80.0 <= y <= 0.0
        else:  # pragma: no cover - generator contract guard
            raise AssertionError(f"unexpected fuzz destination CRS: {payload['dst_crs']}")
