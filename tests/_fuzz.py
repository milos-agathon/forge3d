"""Deterministic in-tree public-API fuzzer for TERMINUS.

Usage:
    python -m tests._fuzz --cases 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

try:
    from forge3d._native import NATIVE_AVAILABLE
except Exception:  # pragma: no cover
    NATIVE_AVAILABLE = False

from ._torture import classify_case, expectation_errors, format_scoreboard, scoreboard


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
    if rng.random() < 0.20:
        west, east = 170.0 + float(rng.uniform(0, 9)), -170.0 - float(rng.uniform(0, 9))
        south, north = sorted([float(rng.uniform(-70, 70)), float(rng.uniform(-70, 70))])
        coords = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    else:
        coords = [_point(rng) for _ in range(count)]
        if coords and rng.random() < 0.75:
            coords.append(coords[0])
    return {
        "id": f"fuzz-geometry-{case_id:06d}",
        "family": "geometry",
        "operation": "gis_validate_geometry",
        "payload": {"geometry": {"type": "Polygon", "coordinates": [coords]}},
        "expect": {"class": "ok"},
    }


def _crs_case(rng: np.random.Generator, case_id: int) -> dict[str, Any]:
    src, dst = rng.choice(["EPSG:4326", "EPSG:3857", "EPSG:32631", "EPSG:3413"], size=2)
    lon, lat = _point(rng)
    return {
        "id": f"fuzz-crs-{case_id:06d}",
        "family": "crs",
        "operation": "gis_transform_point",
        "payload": {"src_crs": str(src), "dst_crs": str(dst), "x": lon, "y": lat},
        "expect": {"class": "ok"},
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
        "operation": str(rng.choice(["gis_normalize_raster", "gis_classify_raster", "dem_derive_water_mask"])),
        "payload": {
            "array": {"values": data, "dtype": "float32"},
            "bins": [-1.0, 0.0, 1.0],
        },
        "expect": {"class": "ok"},
    }


def _label_case(rng: np.random.Generator, case_id: int) -> dict[str, Any]:
    text = str(rng.choice(["", "A", "terminus", "בדיקה", "مرحبا"]))
    size = float(rng.choice([0.0, 1.0, 12.0, 256.0]))
    return {
        "id": f"fuzz-label-{case_id:06d}",
        "family": "labels",
        "operation": "text_shape",
        "payload": {"text": text, "size": size},
        "expect": {"class": "ok"},
    }


GENERATORS: tuple[Callable[[np.random.Generator, int], dict[str, Any]], ...] = (
    _polygon_case,
    _crs_case,
    _raster_case,
    _label_case,
)


def _property_failure(case: dict[str, Any], tmp_dir: Path) -> str | None:
    outcome = classify_case(case, tmp_path=tmp_dir)
    if outcome["class"] in {"panic", "hang"}:
        return json.dumps(outcome, sort_keys=True)
    if outcome["class"] == "structured_error":
        return None
    errors = expectation_errors(case, outcome)
    return "; ".join(errors) if errors else None


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


def shrink_case(case: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    if case.get("operation") == "gis_validate_geometry":
        return _shrink_polygon(case, predicate)
    return case


def run_session(cases: int, seed: int, tmp_dir: Path) -> tuple[dict[str, int], list[str]]:
    rng = np.random.default_rng(seed)
    outcomes = []
    transcripts: list[str] = []
    for index in range(cases):
        generator = GENERATORS[int(rng.integers(0, len(GENERATORS)))]
        case = generator(rng, index)
        outcome = classify_case(case, tmp_path=tmp_dir)
        outcomes.append(outcome)
        if outcome["class"] in {"panic", "hang"}:
            before = json.loads(json.dumps(case))
            shrunk = shrink_case(
                case,
                lambda candidate: _property_failure(candidate, tmp_dir) is not None,
            )
            transcripts.append(
                "shrink before="
                + json.dumps(before, sort_keys=True)
                + "\nshrink after="
                + json.dumps(shrunk, sort_keys=True)
                + "\nreason="
                + json.dumps(outcome, sort_keys=True)
            )
    return scoreboard(outcomes), transcripts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp-dir", type=Path, default=Path("target/terminus-fuzz"))
    args = parser.parse_args(argv)

    if not NATIVE_AVAILABLE:
        print("TERMINUS fuzz skipped: native extension unavailable")
        return 0

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    first, transcripts = run_session(args.cases, args.seed, args.tmp_dir)
    second, _ = run_session(args.cases, args.seed, args.tmp_dir)
    print("TERMINUS fuzz summary:", format_scoreboard(first), f"seed={args.seed}", f"cases={args.cases}")
    print("TERMINUS fuzz determinism:", format_scoreboard(second))
    if transcripts:
        print("TERMINUS shrink transcript:")
        print(transcripts[0])
    else:
        print("TERMINUS shrink transcript: no failing seed remained; synthetic shrink smoke below")
        demo = {
            "id": "synthetic-shrink-demo",
            "family": "geometry",
            "operation": "gis_validate_geometry",
            "payload": {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.2345, 0.0], [2.0, 2.0], [0.0, 0.0]]],
                }
            },
        }
        shrunk = shrink_case(demo, lambda candidate: len(candidate["payload"]["geometry"]["coordinates"][0]) >= 3)
        print("shrink before=" + json.dumps(demo, sort_keys=True))
        print("shrink after=" + json.dumps(shrunk, sort_keys=True))

    if first != second:
        print("fuzz scoreboard is not deterministic", file=sys.stderr)
        return 1
    if first.get("panic", 0) or first.get("hang", 0):
        print("fuzz found panic/hang failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
