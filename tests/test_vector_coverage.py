"""Numerical gates for LIMES analytic vector coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _coverage_ref import (
    REFERENCE_SAMPLES_PER_AXIS,
    line_records_for_rings,
    supersample_coverage,
)

_SHEET_PATH = Path(__file__).parent / "data" / "vector_torture" / "cases.json"


def _load_sheet() -> dict:
    return json.loads(_SHEET_PATH.read_text(encoding="utf-8"))


def test_committed_torture_sheet_has_every_required_primitive_class():
    sheet = _load_sheet()
    assert sheet["reference_samples_per_axis"] == REFERENCE_SAMPLES_PER_AXIS
    cases = {case["name"]: case for case in sheet["cases"]}
    assert {
        "thin_sliver_0_1px",
        "near_horizontal_slope_1e_4",
        "self_intersecting_star_nonzero",
        "self_intersecting_star_evenodd",
        "subpixel_triangles",
        "shared_edge_mosaic_100",
        "hairline_stroke_0_5px",
        "collinear_round_joins",
        "fold_180_round",
        "spiral_round_joins",
    } <= cases.keys()
    mosaic = cases["shared_edge_mosaic_100"]
    assert mosaic["layers"][0]["name"] == "mosaic-background"
    grid = mosaic["layers"][1]["polygon_grid"]
    assert grid["columns"] * grid["rows"] == 100
    assert cases["hairline_stroke_0_5px"]["layers"][0]["polylines"][0]["width"] == 0.5


def test_numpy_reference_counts_4096_samples_per_pixel():
    records = line_records_for_rings(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
    )
    coverage = supersample_coverage(
        records,
        width=1,
        height=1,
        fill_rules=["nonzero"],
    )
    # Sample centers on a 64x64 grid include 2016 points below x+y=1.
    assert coverage.shape == (1, 1, 1)
    assert coverage[0, 0, 0] == 2016 / 4096


def test_numpy_reference_distinguishes_nonzero_and_evenodd_star_center():
    star = [[16.0, 2.0], [24.23, 27.33], [2.68, 11.67], [29.32, 11.67], [7.77, 27.33]]
    records = line_records_for_rings([star])
    nonzero = supersample_coverage(
        records,
        width=32,
        height=32,
        fill_rules=["nonzero"],
        samples_per_axis=8,
    )
    evenodd = supersample_coverage(
        records,
        width=32,
        height=32,
        fill_rules=["evenodd"],
        samples_per_axis=8,
    )
    assert not np.array_equal(nonzero, evenodd)
    assert nonzero[0, 15, 15] > evenodd[0, 15, 15]
