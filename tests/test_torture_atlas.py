"""TERMINUS torture-atlas gate."""

from __future__ import annotations

from collections import Counter
import json

import pytest

from forge3d._native import NATIVE_AVAILABLE

from _torture import (
    TORTURE_ROOT,
    classify_case,
    expectation_errors,
    format_scoreboard,
    load_cases,
    scoreboard,
)


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


def test_torture_atlas_all_cases_green(tmp_path, capsys):
    cases = load_cases()
    outcomes = []
    failures = []

    for case in cases:
        outcome = classify_case(case, tmp_path=tmp_path)
        errors = expectation_errors(case, outcome)
        if errors:
            outcome = {**outcome, "class": "wrong_value", "expectation_errors": errors}
            failures.append((case["id"], case.get("_path"), errors, outcome))
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
    first = [classify_case(case, tmp_path=tmp_path) for case in cases]
    second = [classify_case(case, tmp_path=tmp_path) for case in cases]
    assert scoreboard(first) == scoreboard(second)
    assert [case["class"] for case in first] == [case["class"] for case in second]
