"""Numerical gates for LIMES analytic vector coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import forge3d as f3d
import forge3d._forge3d as _native

from _coverage_ref import (
    REFERENCE_SAMPLES_PER_AXIS,
    deterministic_coverage_hash,
    error_stats,
    line_records_for_rings,
    supersample_coverage,
)

_SHEET_PATH = Path(__file__).parent / "data" / "vector_torture" / "cases.json"
_MEAN_ERROR_GATE = 1.0e-3
_MAX_ERROR_GATE = 0.5 / 255.0


def _load_sheet() -> dict:
    return json.loads(_SHEET_PATH.read_text(encoding="utf-8"))


def _scene_payload(case: dict) -> dict:
    return {
        "width": case["width"],
        "height": case["height"],
        "layers": case["layers"],
    }


def _materialized(case: dict) -> dict:
    return f3d.vector_coverage_primitives_py(
        json.dumps(
            _scene_payload(case),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _coverage_from_rgba(rgba: np.ndarray) -> np.ndarray:
    return np.asarray(rgba, dtype=np.float32)[np.newaxis, :, :, 3] / 255.0


def _passes_gpu_ms(report: dict) -> float:
    values = [float(entry["gpu_ms"]) for entry in report["passes"]]
    assert values and all(value > 0.0 for value in values), report
    return sum(values)


def _expand_grid(layer: dict) -> list[dict]:
    grid = layer.get("polygon_grid")
    if not grid:
        return list(layer.get("polygons", []))
    polygons = list(layer.get("polygons", []))
    for row in range(grid["rows"]):
        for column in range(grid["columns"]):
            x = grid["origin"][0] + column * grid["cell_size"][0]
            y = grid["origin"][1] + row * grid["cell_size"][1]
            dx, dy = grid["cell_size"]
            polygons.append(
                {
                    "exterior": [
                        [x, y],
                        [x + dx, y],
                        [x + dx, y + dy],
                        [x, y + dy],
                    ],
                    "holes": [],
                }
            )
    return polygons


def _throughput_scenes() -> tuple[dict, dict, dict]:
    width, height = 1920, 1080
    columns, rows = 500, 200
    cell_x = width / columns
    cell_y = height / rows
    roads = []
    for index in range(columns * rows):
        column = index % columns
        row = index // columns
        x = column * cell_x + 0.2
        y = row * cell_y + 0.4
        roads.append(
            {
                "path": [[x, y], [x + 0.72 * cell_x, y + 0.12 * cell_y]],
                "width": 0.5,
                "cap": "round",
                "join": "round",
            }
        )

    torture_polygons = []
    torture_lines = []
    for case in _load_sheet()["cases"]:
        for layer in case["layers"]:
            torture_polygons.extend(_expand_grid(layer))
            torture_lines.extend(layer.get("polylines", []))

    line_layer = {
        "name": "100k-road-plus-stroke-torture",
        "quality": "analytic",
        "fill_rule": "nonzero",
        "color": [1.0, 1.0, 1.0, 1.0],
        "polygons": [],
        "polylines": roads + torture_lines,
    }
    fill_layer = {
        "name": "fill-torture",
        "quality": "analytic",
        "fill_rule": "nonzero",
        "color": [1.0, 1.0, 1.0, 1.0],
        "polygons": torture_polygons,
        "polylines": [],
    }
    analytic = {"width": width, "height": height, "layers": [fill_layer, line_layer]}
    current_fill = {"width": width, "height": height, "layers": [fill_layer]}
    current_line = {"width": width, "height": height, "layers": [line_layer]}
    return analytic, current_fill, current_line


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
    assert all(
        layer["quality"] == "analytic"
        for case in sheet["cases"]
        for layer in case["layers"]
    )
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


def test_snapshot_quality_kwarg_routes_an_explicit_analytic_layer(monkeypatch):
    from forge3d import vector

    captured = {}

    def fake_render(scene, *, certificate=False):
        captured["scene"] = scene
        captured["certificate"] = certificate
        return np.zeros((8, 12, 4), dtype=np.uint8)

    monkeypatch.setattr(vector, "render_analytic", fake_render)
    scene = vector.VectorScene()
    scene.add_polyline([(1.0, 2.0), (9.0, 6.0)], width=0.5)
    rgba = scene.render_snapshot(12, 8, quality="analytic", certificate=True)
    assert rgba.shape == (8, 12, 4)
    assert captured["certificate"] is True
    assert captured["scene"]["layers"][0]["quality"] == "analytic"
    assert captured["scene"]["layers"][0]["polylines"][0]["width"] == 0.5


def test_coverage_report_wrapper_returns_parsed_execution_report(monkeypatch):
    from forge3d import vector

    monkeypatch.setattr(
        f3d,
        "vector_render_analytic_py",
        lambda *_args, **_kwargs: {
            "report": {
                "quality": "analytic",
                "execution_report_json": '{"schema":"forge3d.render_certificate/1"}',
            }
        },
    )
    scene = {
        "width": 1,
        "height": 1,
        "layers": [
            {
                "name": "one",
                "quality": "analytic",
                "fill_rule": "nonzero",
                "color": [1.0, 1.0, 1.0, 1.0],
                "polygons": [],
                "polylines": [],
            }
        ],
    }
    report = vector.coverage_report(scene)
    assert report["quality"] == "analytic"
    assert report["execution_report"]["schema"] == "forge3d.render_certificate/1"


@pytest.mark.parametrize("case", _load_sheet()["cases"], ids=lambda case: case["name"])
def test_native_ingest_materializes_reference_compatible_records(case):
    materialized = _materialized(case)
    records = materialized["records"]
    assert records
    assert [record["stable_id"] for record in records] == list(range(len(records)))
    assert all(record["kind"] in {0, 1} for record in records)
    reference = supersample_coverage(
        records,
        width=case["width"],
        height=case["height"],
        fill_rules=materialized["fill_rules"],
        samples_per_axis=2,
    )
    assert reference.shape == (len(case["layers"]), case["height"], case["width"])
    assert np.isfinite(reference).all()
    assert np.all((reference >= 0.0) & (reference <= 1.0))


@pytest.mark.skipif(not f3d.has_gpu(), reason="LIMES numerical gate requires a GPU adapter")
@pytest.mark.parametrize("case", _load_sheet()["cases"], ids=lambda case: case["name"])
def test_analytic_coverage_meets_committed_reference_gate(case):
    scene_json = json.dumps(
        _scene_payload(case),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result = f3d.vector_render_analytic_py(
        scene_json,
        include_coverage=True,
        include_records=True,
        certificate=True,
    )
    report = result["report"]
    assert report["written_memberships"] == report["measured_memberships"]
    assert 0 < report["resolve_pixel_count"] <= report["active_pixel_count"]
    assert report["structured_errors"] == [0, 0, 0, 0]
    actual = np.asarray(result["coverage"], dtype=np.float32)
    reference = supersample_coverage(
        result["records"],
        width=case["width"],
        height=case["height"],
        fill_rules=[layer["fill_rule"] for layer in case["layers"]],
    )
    stats = error_stats(actual, reference)
    print(
        f"LIMES_ERROR case={case['name']} "
        f"mean={stats['mean_abs_error']:.9g} max={stats['max_abs_error']:.9g}"
    )
    assert stats["mean_abs_error"] < _MEAN_ERROR_GATE
    assert stats["max_abs_error"] < _MAX_ERROR_GATE


@pytest.mark.skipif(not f3d.has_gpu(), reason="LIMES ablation requires a GPU adapter")
@pytest.mark.parametrize(
    "case_name",
    [
        "thin_sliver_0_1px",
        "near_horizontal_slope_1e_4",
        "hairline_stroke_0_5px",
    ],
)
def test_current_and_real_msaa4_both_fail_the_analytic_gate(case_name):
    case = next(item for item in _load_sheet()["cases"] if item["name"] == case_name)
    scene_json = json.dumps(
        _scene_payload(case), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    materialized = _materialized(case)
    reference = supersample_coverage(
        materialized["records"],
        width=case["width"],
        height=case["height"],
        fill_rules=materialized["fill_rules"],
    )
    rows = {}
    for label, sample_count in (("current", 1), ("msaa4", 4)):
        rgba = _native._vector_render_coverage_ablation_py(
            scene_json, sample_count, certificate=True
        )
        stats = error_stats(_coverage_from_rgba(rgba), reference)
        rows[label] = stats
        passed = (
            stats["mean_abs_error"] < _MEAN_ERROR_GATE
            and stats["max_abs_error"] < _MAX_ERROR_GATE
        )
        assert not passed, f"{label} unexpectedly passed LIMES gate for {case_name}: {stats}"
    print(
        f"LIMES_ABLATION case={case_name} "
        f"current_mean={rows['current']['mean_abs_error']:.9g} "
        f"current_max={rows['current']['max_abs_error']:.9g} "
        f"msaa4_mean={rows['msaa4']['mean_abs_error']:.9g} "
        f"msaa4_max={rows['msaa4']['max_abs_error']:.9g}"
    )


@pytest.mark.skipif(not f3d.has_gpu(), reason="LIMES mosaic gate requires a GPU adapter")
def test_shared_edge_mosaic_has_no_interior_seam():
    case = next(
        item
        for item in _load_sheet()["cases"]
        if item["name"] == "shared_edge_mosaic_100"
    )
    result = f3d.vector_render_analytic_py(
        json.dumps(_scene_payload(case), sort_keys=True, separators=(",", ":")),
        include_coverage=True,
        certificate=True,
    )
    mosaic_coverage = np.asarray(result["coverage"], dtype=np.float32)[1]
    max_deviation = float(np.max(np.abs(1.0 - mosaic_coverage)))
    print(f"LIMES_MOSAIC_MAX_DEVIATION={max_deviation:.9g}")
    assert max_deviation <= 1.0 / 255.0


@pytest.mark.skipif(not f3d.has_gpu(), reason="LIMES determinism gate requires a GPU adapter")
def test_analytic_output_is_byte_identical_across_two_runs():
    case = next(
        item for item in _load_sheet()["cases"] if item["name"] == "spiral_round_joins"
    )
    scene_json = json.dumps(
        _scene_payload(case), sort_keys=True, separators=(",", ":")
    )
    first = f3d.vector_render_analytic_py(
        scene_json, include_coverage=True, certificate=True
    )
    second = f3d.vector_render_analytic_py(
        scene_json, include_coverage=True, certificate=True
    )
    first_coverage = np.asarray(first["coverage"], dtype=np.float32)
    second_coverage = np.asarray(second["coverage"], dtype=np.float32)
    first_hash = deterministic_coverage_hash(first_coverage)
    second_hash = deterministic_coverage_hash(second_coverage)
    print(f"LIMES_DETERMINISM first={first_hash} second={second_hash}")
    assert first_hash == second_hash
    assert np.array_equal(first["rgba"], second["rgba"])
    report = json.loads(second["report"]["execution_report_json"])
    assert set(report["engine"]["wgsl_module_hashes"]) == {
        "vector_coverage_bin.wgsl",
        "vector_coverage_raster.wgsl",
        "vector_coverage_resolve.wgsl",
    }


@pytest.mark.skipif(
    not (f3d.has_gpu() and os.environ.get("RUN_LIMES_GPU_CI") == "1"),
    reason="LIMES throughput gate runs on the designated physical-GPU lane",
)
def test_torture_plus_100k_road_segments_is_within_twice_default_gpu_time():
    analytic, current_fill, current_line = _throughput_scenes()

    _native._vector_render_coverage_ablation_py(
        json.dumps(current_fill, separators=(",", ":"), allow_nan=False),
        1,
        certificate=True,
    )
    current_fill_report = json.loads(f3d.render_execution_report())
    _native._vector_render_coverage_ablation_py(
        json.dumps(current_line, separators=(",", ":"), allow_nan=False),
        1,
        certificate=True,
    )
    current_line_report = json.loads(f3d.render_execution_report())
    current_ms = _passes_gpu_ms(current_fill_report) + _passes_gpu_ms(
        current_line_report
    )

    result = f3d.vector_render_analytic_py(
        json.dumps(analytic, separators=(",", ":"), allow_nan=False),
        certificate=True,
    )
    analytic_report = json.loads(result["report"]["execution_report_json"])
    analytic_ms = _passes_gpu_ms(analytic_report)
    ratio = analytic_ms / current_ms
    print(
        "LIMES_THROUGHPUT "
        f"adapter={analytic_report['adapter']['device']!r} "
        f"segments=100000 current_gpu_ms={current_ms:.6f} "
        f"analytic_gpu_ms={analytic_ms:.6f} ratio={ratio:.6f}"
    )
    assert ratio <= 2.0
