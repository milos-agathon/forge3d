"""Deterministic in-tree public-API fuzzer for TERMINUS.

Usage:
    python -m tests._fuzz --cases 10000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

try:
    from forge3d._native import NATIVE_AVAILABLE
except Exception:  # pragma: no cover
    NATIVE_AVAILABLE = False

from ._torture import TortureWorker, evaluate_case, format_scoreboard, scoreboard


def _point(rng: np.random.Generator) -> list[float]:
    lon = float(rng.uniform(-220.0, 220.0))
    lat = float(rng.uniform(-95.0, 95.0))
    if rng.random() < 0.04:
        lon = float(rng.choice([np.nan, np.inf, -np.inf, np.finfo(np.float32).max]))
    if rng.random() < 0.04:
        lat = float(rng.choice([np.nan, np.inf, -np.inf, 90.0, -90.0]))
    return [lon, lat]


def _polygon_case(rng: np.random.Generator, case_id: int) -> dict[str, Any]:
    count = int(rng.integers(0, 18))
    if rng.random() < 0.45:
        west, east = 170.0 + float(rng.uniform(0, 9)), -170.0 - float(rng.uniform(0, 9))
        south, north = sorted([float(rng.uniform(-70, 70)), float(rng.uniform(-70, 70))])
        coords = [[west, south], [east, south], [east, north], [west, north], [west, south]]
        return {
            "id": f"fuzz-geometry-{case_id:06d}",
            "family": "geometry",
            "operation": "gis_geometry_wrap_invariance",
            "payload": {
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "crs": "EPSG:4326",
                "shift": float(rng.choice([-360.0, 360.0])),
            },
            "expect": {"class": "error_or_value"},
            "property": "area_wrap_invariance",
            "property_tolerance": 1.0e-8,
        }
    else:
        coords = [_point(rng) for _ in range(count)]
        if coords and rng.random() < 0.75:
            coords.append(coords[0])
    return {
        "id": f"fuzz-geometry-{case_id:06d}",
        "family": "geometry",
        "operation": "gis_validate_geometry",
        "payload": {"geometry": {"type": "Polygon", "coordinates": [coords]}},
        "expect": {"class": "error_or_value"},
    }


def _crs_case(rng: np.random.Generator, case_id: int) -> dict[str, Any]:
    # Round-trip closure is only meaningful inside the destination CRS's
    # documented area of use.  In particular, projecting arbitrary longitudes
    # into a distant UTM zone produces numerically unstable (but not invalid)
    # values and tests the generator rather than the public API contract.
    destination, lon_bounds, lat_bounds = (
        (
            "EPSG:3857",
            (-170.0, 170.0),
            (-80.0, 80.0),
        ),
        (
            "EPSG:32631",
            (0.0, 6.0),
            (0.0, 80.0),
        ),
        (
            "EPSG:32733",
            (12.0, 18.0),
            (-80.0, 0.0),
        ),
    )[int(rng.integers(0, 3))]
    lon = float(rng.uniform(*lon_bounds))
    lat = float(rng.uniform(*lat_bounds))
    return {
        "id": f"fuzz-crs-{case_id:06d}",
        "family": "crs",
        "operation": "gis_transform_roundtrip",
        "payload": {"src_crs": "EPSG:4326", "dst_crs": destination, "x": lon, "y": lat},
        "expect": {"class": "error_or_value"},
        "property": "projection_roundtrip",
        "property_tolerance": 1.0e-5,
    }


def _raster_case(rng: np.random.Generator, case_id: int) -> dict[str, Any]:
    rows = int(rng.integers(0, 8))
    cols = int(rng.integers(0, 8))
    data = rng.normal(size=(rows, cols)).astype(float).tolist()
    if rows and cols and rng.random() < 0.25:
        data[int(rng.integers(0, rows))][int(rng.integers(0, cols))] = "nan"
    return {
        "id": f"fuzz-raster-{case_id:06d}",
        "family": "rasters",
        "operation": "gis_normalize_raster",
        "payload": {
            "array": {"values": data, "dtype": "float32"},
            "bins": [-1.0, 0.0, 1.0],
            "method": "minmax",
        },
        "expect": {"class": "error_or_value"},
        "property": "normalized_unit_interval",
    }


def _label_case(rng: np.random.Generator, case_id: int) -> dict[str, Any]:
    text = str(rng.choice(["", "A", "terminus", "בדיקה", "مرحبا"]))
    size = float(rng.choice([0.0, 1.0, 12.0, 256.0]))
    return {
        "id": f"fuzz-label-{case_id:06d}",
        "family": "labels",
        "operation": "text_shape",
        "payload": {"text": text, "size": size},
        "expect": {"class": "error_or_value"},
    }


GENERATORS: tuple[Callable[[np.random.Generator, int], dict[str, Any]], ...] = (
    _polygon_case,
    _crs_case,
    _raster_case,
    _label_case,
)


def _property_failure(case: dict[str, Any], tmp_dir: Path, worker: TortureWorker) -> str | None:
    outcome = evaluate_case(case, tmp_path=tmp_dir, worker=worker)
    if outcome["class"] in {"panic", "hang", "wrong_value"}:
        return json.dumps(outcome, sort_keys=True)
    return None


def _shrink_polygon(case: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    geom = case.get("payload", {}).get("geometry", {})
    rings = geom.get("coordinates") if isinstance(geom, dict) else None
    if not rings or not isinstance(rings, list) or not rings[0]:
        return case
    coords = list(rings[0])
    changed = True
    while changed and len(coords) > 1:
        changed = False
        for index in range(len(coords)):
            trial = coords[:index] + coords[index + 1 :]
            candidate = json.loads(json.dumps(case))
            candidate["payload"]["geometry"]["coordinates"] = [trial]
            if predicate(candidate):
                coords = trial
                case = candidate
                changed = True
                break
    for precision in (6, 3, 1, 0):
        trial = [[round(float(x), precision), round(float(y), precision)] for x, y in coords]
        candidate = json.loads(json.dumps(case))
        candidate["payload"]["geometry"]["coordinates"] = [trial]
        if predicate(candidate):
            coords = trial
            case = candidate
    return case


def _shrink_raster(case: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    values = case.get("payload", {}).get("array", {}).get("values")
    if not isinstance(values, list):
        return case
    rows = [list(row) for row in values if isinstance(row, list)]
    for row_count in range(max(1, len(rows) - 1), 0, -1):
        candidate = json.loads(json.dumps(case))
        candidate["payload"]["array"]["values"] = rows[:row_count]
        if predicate(candidate):
            case = candidate
            rows = rows[:row_count]
    if rows:
        width = min((len(row) for row in rows), default=0)
        for col_count in range(max(1, width - 1), 0, -1):
            candidate = json.loads(json.dumps(case))
            candidate["payload"]["array"]["values"] = [row[:col_count] for row in rows]
            if predicate(candidate):
                case = candidate
                rows = [row[:col_count] for row in rows]
    for precision in (3, 1, 0):
        rounded = [
            [round(float(value), precision) if isinstance(value, (int, float)) else value for value in row]
            for row in rows
        ]
        candidate = json.loads(json.dumps(case))
        candidate["payload"]["array"]["values"] = rounded
        if predicate(candidate):
            case = candidate
            rows = rounded
    return case


def _shrink_crs(case: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    for key in ("x", "y"):
        original = float(case.get("payload", {}).get(key, 0.0))
        for value in (round(original, 3), round(original, 0), original / 2.0, 0.0):
            candidate = json.loads(json.dumps(case))
            candidate["payload"][key] = value
            if predicate(candidate):
                case = candidate
    return case


def _shrink_label(case: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    text = str(case.get("payload", {}).get("text", ""))
    while len(text) > 1:
        candidate = json.loads(json.dumps(case))
        candidate["payload"]["text"] = text[: max(1, len(text) // 2)]
        if not predicate(candidate):
            break
        case = candidate
        text = candidate["payload"]["text"]
    for size in (12.0, 1.0, 0.0):
        candidate = json.loads(json.dumps(case))
        candidate["payload"]["size"] = size
        if predicate(candidate):
            case = candidate
    return case


def shrink_case(case: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    if case.get("family") == "geometry":
        return _shrink_polygon(case, predicate)
    if case.get("family") == "rasters":
        return _shrink_raster(case, predicate)
    if case.get("family") == "crs":
        return _shrink_crs(case, predicate)
    if case.get("family") == "labels":
        return _shrink_label(case, predicate)
    return case


def run_session(cases: int, seed: int, tmp_dir: Path) -> tuple[dict[str, int], list[str], str]:
    rng = np.random.default_rng(seed)
    outcomes = []
    transcripts: list[str] = []
    with TortureWorker() as worker:
        for index in range(cases):
            generator = GENERATORS[int(rng.integers(0, len(GENERATORS)))]
            case = generator(rng, index)
            outcome = evaluate_case(case, tmp_path=tmp_dir, worker=worker)
            outcomes.append(outcome)
            if outcome["class"] in {"panic", "hang", "wrong_value"}:
                before = json.loads(json.dumps(case))
                shrunk = shrink_case(
                    case,
                    lambda candidate: _property_failure(candidate, tmp_dir, worker) is not None,
                )
                transcripts.append(
                    "shrink before="
                    + json.dumps(before, sort_keys=True)
                    + "\nshrink after="
                    + json.dumps(shrunk, sort_keys=True)
                    + "\nreason="
                    + json.dumps(outcome, sort_keys=True)
                )
    digest = hashlib.sha256(
        json.dumps(outcomes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return scoreboard(outcomes), transcripts, digest


def replay_case(seed: int, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("replay index must be non-negative")
    rng = np.random.default_rng(seed)
    case: dict[str, Any] | None = None
    for current in range(index + 1):
        generator = GENERATORS[int(rng.integers(0, len(GENERATORS)))]
        case = generator(rng, current)
    assert case is not None
    return case


def shrink_validation_transcript() -> str:
    proof = failure_preserving_shrink_proof(seed=42, index=137)
    return (
        "shrink before="
        + json.dumps(proof["initial"], sort_keys=True)
        + "\nshrink after="
        + json.dumps(proof["minimal"], sort_keys=True)
        + "\nfailure_preserved=true"
    )


def _shrinker_self_test_case(seed: int, index: int) -> dict[str, Any]:
    """Return a deterministic deliberately-failing shrinker fixture.

    This is not a production defect. The test property says every raster value
    must be <= 1, while the seeded fixture deliberately retains a 2 at [0, 0].
    """

    rng = np.random.default_rng(seed)
    noise = rng.integers(-1, 2, size=(index % 3 + 2, index % 4 + 2)).astype(float)
    noise[0, 0] = 2.0
    return {
        "id": f"shrinker-self-test-{seed}-{index}",
        "family": "rasters",
        "operation": "gis_normalize_raster",
        "payload": {
            "array": {"values": noise.tolist(), "dtype": "float32"},
            "method": "minmax",
        },
        "expect": {"class": "error_or_value"},
    }


def _violates_shrinker_self_test_property(case: dict[str, Any]) -> bool:
    values = case["payload"]["array"]["values"]
    return any(float(value) > 1.0 for row in values for value in row)


def failure_preserving_shrink_proof(seed: int, index: int) -> dict[str, Any]:
    initial = _shrinker_self_test_case(seed, index)
    accepted: list[dict[str, Any]] = []

    def still_fails(candidate: dict[str, Any]) -> bool:
        failing = _violates_shrinker_self_test_property(candidate)
        if failing:
            accepted.append(json.loads(json.dumps(candidate)))
        return failing

    assert _violates_shrinker_self_test_property(initial)
    minimal = shrink_case(json.loads(json.dumps(initial)), still_fails)
    assert accepted and all(_violates_shrinker_self_test_property(case) for case in accepted)
    assert minimal["payload"]["array"]["values"] == [[2.0]]
    return {"initial": initial, "minimal": minimal, "accepted": accepted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp-dir", type=Path, default=Path("target/terminus-fuzz"))
    parser.add_argument("--replay-index", type=int)
    args = parser.parse_args(argv)

    if not NATIVE_AVAILABLE:
        print("TERMINUS fuzz requires the native extension", file=sys.stderr)
        return 2

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.replay_index is not None:
        case = replay_case(args.seed, args.replay_index)
        with TortureWorker() as worker:
            outcome = evaluate_case(case, tmp_path=args.tmp_dir, worker=worker)
        print("TERMINUS replay case:", json.dumps(case, sort_keys=True))
        print("TERMINUS replay outcome:", json.dumps(outcome, sort_keys=True))
        return 1 if outcome["class"] in {"panic", "hang", "wrong_value"} else 0

    first, transcripts, first_digest = run_session(args.cases, args.seed, args.tmp_dir)
    second, _, second_digest = run_session(args.cases, args.seed, args.tmp_dir)
    print(
        "TERMINUS fuzz summary:",
        format_scoreboard(first),
        f"seed={args.seed}",
        f"cases={args.cases}",
        f"digest={first_digest}",
    )
    print("TERMINUS fuzz determinism:", format_scoreboard(second), f"digest={second_digest}")
    if transcripts:
        print("TERMINUS shrink transcript:")
        print(transcripts[0])
    else:
        print("TERMINUS shrink transcript: no failing seed remained; shrinker validation below")
        print(shrink_validation_transcript())

    if first != second or first_digest != second_digest:
        print("fuzz outcomes are not deterministic", file=sys.stderr)
        return 1
    if first.get("panic", 0) or first.get("hang", 0) or first.get("wrong_value", 0):
        print("fuzz found panic/hang/property failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
