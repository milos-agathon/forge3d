"""Shared TERMINUS torture-atlas execution helpers.

Descriptors are intentionally data-only.  The executor below is the single
place that maps a descriptor operation onto forge3d's public Python surface.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TORTURE_ROOT = ROOT / "tests" / "torture"
DEFAULT_WATCHDOG_SECONDS = 10.0


class TortureTimeout(TimeoutError):
    """Raised when a torture case exceeds its watchdog budget."""


def load_cases(root: Path = TORTURE_ROOT) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(root.glob("**/*.json")):
        if path.name == "MANIFEST.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path
        cases.append(data)
    return cases


@contextmanager
def watchdog(seconds: float | None):
    if seconds is None or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _timeout(_signum: int, _frame: Any) -> None:
        raise TortureTimeout(f"torture case exceeded {seconds:.3f}s watchdog")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, str):
        table = {
            "nan": math.nan,
            "inf": math.inf,
            "+inf": math.inf,
            "-inf": -math.inf,
            "f32_max": np.finfo(np.float32).max,
            "-f32_max": -np.finfo(np.float32).max,
            "subnormal": np.nextafter(np.float32(0.0), np.float32(1.0)).item(),
            "-subnormal": np.nextafter(np.float32(0.0), np.float32(-1.0)).item(),
        }
        if value in table:
            return table[value]
    return value


def _decode_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_nested(item) for key, item in value.items()}
    return _decode_scalar(value)


def _array(spec: Any, *, default_dtype: str = "float64") -> np.ndarray:
    if isinstance(spec, dict):
        dtype = str(spec.get("dtype", default_dtype))
        if "values" in spec:
            return np.asarray(_decode_nested(spec["values"]), dtype=dtype)
        shape = tuple(int(v) for v in spec.get("shape", ()))
        if "fill" in spec:
            return np.full(shape, _decode_scalar(spec["fill"]), dtype=dtype)
        if "arange" in spec:
            start, stop, step = spec["arange"]
            return np.arange(start, stop, step, dtype=dtype).reshape(shape)
        if spec.get("pattern") == "checker":
            rows, cols = shape
            yy, xx = np.mgrid[:rows, :cols]
            return ((xx + yy) % 2).astype(dtype)
        if spec.get("pattern") == "ramp":
            rows, cols = shape
            return np.linspace(0.0, 1.0, rows * cols, dtype=dtype).reshape(shape)
    return np.asarray(_decode_nested(spec), dtype=default_dtype)


def _geometry(payload: dict[str, Any]) -> dict[str, Any]:
    if "geometry" in payload:
        return _decode_nested(payload["geometry"])
    generated = payload.get("generated")
    if generated == "dateline_box":
        west = float(payload.get("west", 170.0))
        east = float(payload.get("east", -170.0))
        south = float(payload.get("south", -10.0))
        north = float(payload.get("north", 10.0))
        return {
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        }
    if generated == "degenerate_line":
        value = float(payload.get("value", 0.0))
        return {"type": "LineString", "coordinates": [[value, value], [value, value]]}
    if generated == "regular_polygon":
        count = int(payload.get("vertices", 32))
        radius = float(payload.get("radius", 1.0))
        lon0 = float(payload.get("lon0", 0.0))
        lat0 = float(payload.get("lat0", 0.0))
        points = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            points.append([lon0 + radius * math.cos(angle), lat0 + radius * math.sin(angle)])
        points.append(points[0])
        return {"type": "Polygon", "coordinates": [points]}
    raise KeyError(f"unknown geometry payload: {payload}")


def _feature_collection(geometry: dict[str, Any], crs: str | None = None) -> dict[str, Any]:
    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": geometry}],
    }
    if crs:
        payload["info"] = {
            "path": "",
            "driver": "GeoJSON",
            "layer_name": None,
            "layer_count": 1,
            "geometry_type": geometry.get("type", "Unknown"),
            "feature_count": 1,
            "schema": [],
            "crs_wkt": None,
            "crs_authority": {"name": "EPSG", "code": str(crs).split(":")[-1]},
            "bounds": None,
            "is_georeferenced": True,
            "warnings": [],
        }
    return payload


def _default_font() -> str:
    return str(ROOT / "assets" / "fonts" / "NotoSans-subset.ttf")


def execute_payload(operation: str, payload: dict[str, Any], tmp_path: Path | None = None) -> Any:
    import forge3d as f3d
    import forge3d.gis as gis

    if operation == "gis_validate_geometry":
        return gis.validate_geometry(_geometry(payload))
    if operation == "gis_geometry_measure":
        return gis.geometry_measure(_geometry(payload), crs=payload.get("crs", "EPSG:4326"))
    if operation == "gis_geometry_centroid":
        return gis.geometry_centroid(_geometry(payload), crs=payload.get("crs"))
    if operation == "gis_representative_point":
        return gis.representative_point(_geometry(payload), crs=payload.get("crs"))
    if operation == "gis_interpolate_line":
        return gis.interpolate_line(
            _geometry(payload),
            float(payload.get("distance", 0.5)),
            normalized=bool(payload.get("normalized", True)),
            crs=payload.get("crs"),
        )
    if operation == "gis_simplify_geometry":
        return gis.simplify_geometry(
            _geometry(payload),
            float(payload.get("tolerance", 0.01)),
            preserve_topology=bool(payload.get("preserve_topology", True)),
            crs=payload.get("crs"),
        )
    if operation == "gis_buffer_geometry":
        return gis.buffer_geometry(
            _geometry(payload),
            float(payload.get("distance", 1.0)),
            quad_segs=int(payload.get("quad_segs", 8)),
            crs=payload.get("crs"),
        )
    if operation == "gis_transform_point":
        transformer = gis.create_crs_transformer(
            payload["src_crs"],
            payload["dst_crs"],
            always_xy=bool(payload.get("always_xy", True)),
        )
        return {"point": transformer.transform_point(float(payload["x"]), float(payload["y"]))}
    if operation == "gis_transform_roundtrip":
        forward = gis.create_crs_transformer(
            payload["src_crs"],
            payload["dst_crs"],
            always_xy=bool(payload.get("always_xy", True)),
        )
        backward = gis.create_crs_transformer(
            payload["dst_crs"],
            payload["src_crs"],
            always_xy=bool(payload.get("always_xy", True)),
        )
        x, y = float(payload["x"]), float(payload["y"])
        e, n = forward.transform_point(x, y)
        rx, ry = backward.transform_point(e, n)
        return {"forward": (e, n), "roundtrip": (rx, ry)}
    if operation == "gis_transform_bounds":
        return {
            "bounds": gis.transform_bounds(
                payload["src_crs"],
                payload["dst_crs"],
                tuple(float(v) for v in payload["bounds"]),
                densify=payload.get("densify"),
            )
        }
    if operation == "gis_parse_crs":
        return gis.parse_crs(_decode_nested(payload["crs"]))
    if operation == "gis_slippy_tile_index":
        return gis.slippy_tile_index(
            tuple(float(v) for v in payload["bounds"]),
            int(payload["zoom"]),
            payload.get("crs", "EPSG:4326"),
        )
    if operation == "gis_transform_from_bounds":
        transform = gis.transform_from_bounds(
            tuple(float(v) for v in payload["bounds"]),
            int(payload["width"]),
            int(payload["height"]),
        )
        return {"transform": transform}
    if operation == "gis_array_bounds":
        return {
            "bounds": gis.array_bounds(
                int(payload["height"]),
                int(payload["width"]),
                tuple(float(v) for v in payload["transform"]),
            )
        }
    if operation == "gis_rowcol_xy":
        transform = tuple(float(v) for v in payload["transform"])
        row, col = gis.rowcol(transform, float(payload["x"]), float(payload["y"]))
        x, y = gis.xy(transform, row, col)
        return {"rowcol": (row, col), "xy": (x, y)}
    if operation == "gis_normalize_raster":
        result = gis.normalize_raster(
            _array(payload["array"], default_dtype=str(payload.get("dtype", "float32"))),
            method=payload.get("method", "minmax"),
            nodata=payload.get("nodata"),
            clip=payload.get("clip"),
        )
        return {"array": result["array"], "info": result.get("info")}
    if operation == "gis_classify_raster":
        result = gis.classify_raster(
            _array(payload["array"], default_dtype=str(payload.get("dtype", "float32"))),
            bins=payload.get("bins"),
            labels=payload.get("labels"),
            dtype=payload.get("out_dtype", "uint16"),
            nodata=payload.get("nodata"),
        )
        return {"array": result["array"], "classes": result.get("classes")}
    if operation == "gis_apply_nodata":
        return {"array": gis.apply_nodata(_array(payload["array"], default_dtype="float32"), payload.get("nodata"))}
    if operation == "gis_write_raster":
        if tmp_path is None:
            tmp_path = ROOT / "target" / "terminus-tmp"
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / f"{payload.get('name', 'case')}.tif"
        info = gis.write_raster(
            path,
            _array(payload["array"], default_dtype=str(payload.get("dtype", "float32"))),
            crs=payload.get("crs"),
            transform=tuple(_decode_scalar(v) for v in payload["transform"]) if payload.get("transform") else None,
            nodata=payload.get("nodata"),
            overwrite=True,
        )
        return {"width": info.width, "height": info.height, "band_count": info.band_count}
    if operation == "cog_dataset_open":
        from forge3d.cog import CogDataset

        if tmp_path is None:
            tmp_path = ROOT / "target" / "terminus-tmp"
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / f"{payload.get('name', 'case')}.tif"
        path.write_bytes(bytes.fromhex(str(payload["hex"])))
        dataset = CogDataset(
            "file://" + str(path),
            cache_size_mb=int(payload.get("cache_size_mb", 1)),
            cache_budget_mb=int(payload.get("cache_budget_mb", 1)),
        )
        return {"bounds": dataset.bounds, "overview_count": dataset.overview_count}
    if operation == "dem_derive_water_mask":
        return {"array": gis.derive_water_mask(_array(payload["array"], default_dtype="float32"))}
    if operation == "vector_add_points":
        import forge3d.vector as vector

        vector.clear_vectors()
        ids = vector.add_points(_array(payload["positions"], default_dtype="float64"), point_size=float(payload.get("point_size", 4.0)))
        return {"ids": ids, "counts": vector.get_vector_counts()}
    if operation == "vector_add_lines":
        import forge3d.vector as vector

        vector.clear_vectors()
        ids = vector.add_lines(_array(payload["path"], default_dtype="float64"), stroke_width=float(payload.get("stroke_width", 1.0)))
        return {"ids": ids, "counts": vector.get_vector_counts()}
    if operation == "vector_add_polygons":
        import forge3d.vector as vector

        vector.clear_vectors()
        ids = vector.add_polygons(_array(payload["exterior"], default_dtype="float64"), stroke_width=float(payload.get("stroke_width", 1.0)))
        return {"ids": ids, "counts": vector.get_vector_counts()}
    if operation == "vector_add_graph":
        import forge3d.vector as vector

        vector.clear_vectors()
        graph_id = vector.add_graph(
            _array(payload["nodes"], default_dtype="float64"),
            _array(payload["edges"], default_dtype="uint32"),
        )
        return {"id": graph_id, "counts": vector.get_vector_counts()}
    if operation == "text_shape":
        import forge3d.text as text

        fonts = payload.get("fonts") or [_default_font()]
        shaped = text.shape(
            str(payload.get("text", "")),
            [str(ROOT / item) if not str(item).startswith("/") else str(item) for item in fonts],
            float(payload.get("size", 12.0)),
            payload.get("script"),
            payload.get("language"),
            payload.get("features"),
        )
        data = shaped.to_dict() if hasattr(shaped, "to_dict") else {}
        return {"glyph_count": len(data.get("glyphs", ())), "text": data.get("text")}
    if operation == "output_spec":
        output = f3d.OutputSpec(
            width=int(payload["width"]),
            height=int(payload["height"]),
            format=payload.get("format", "png"),
            samples=int(payload.get("samples", 1)),
            bit_depth=int(payload.get("bit_depth", 8)),
            aovs=tuple(payload.get("aovs", ())),
            denoiser=payload.get("denoiser", "none"),
        )
        return output.to_dict()
    if operation == "terrain_source":
        terrain = f3d.TerrainSource(
            data=_array(payload["array"], default_dtype="float32"),
            crs=payload.get("crs"),
            metadata=payload.get("metadata"),
            dtype=payload.get("dtype", "float32"),
            nodata_policy=payload.get("nodata_policy", "fill"),
        )
        return terrain.to_dict()
    if operation == "map_scene_validate":
        terrain = f3d.TerrainSource(
            data=_array(payload["terrain"], default_dtype="float32"),
            crs=payload.get("crs", "EPSG:4326"),
            metadata=payload.get("metadata", {"source_id": "terminus", "width": 2, "height": 2}),
        )
        scene = f3d.MapScene(
            terrain=terrain,
            layers=tuple(),
            output=f3d.OutputSpec(width=int(payload.get("width", 32)), height=int(payload.get("height", 32))),
        )
        report = scene.validate()
        return {"ok": report.ok, "diagnostic_count": len(report.diagnostics)}
    if operation == "viewer_set_msaa":
        import forge3d.viewer as viewer

        return {"samples": viewer.set_msaa(int(payload["samples"]))}
    raise KeyError(f"unknown torture operation: {operation}")


def _is_panic_exception(exc: BaseException) -> bool:
    return type(exc).__module__ == "pyo3_runtime" and type(exc).__name__ == "PanicException"


def classify_case(case: dict[str, Any], tmp_path: Path | None = None) -> dict[str, Any]:
    timeout = float(case.get("watchdog_seconds", DEFAULT_WATCHDOG_SECONDS))
    try:
        with watchdog(timeout):
            result = execute_payload(case["operation"], case.get("payload", {}), tmp_path=tmp_path)
        return {"class": "ok", "result": normalize_result(result)}
    except TortureTimeout as exc:
        return {"class": "hang", "error_type": type(exc).__name__, "message": str(exc)}
    except BaseException as exc:  # noqa: BLE001 - classification is the contract.
        if _is_panic_exception(exc):
            cls = "panic"
        else:
            cls = "structured_error"
        return {
            "class": cls,
            "error_type": type(exc).__name__,
            "module": type(exc).__module__,
            "message": str(exc),
            "diagnostics": normalize_result(getattr(exc, "diagnostics", None)),
        }


def normalize_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.asarray([])
        summary: dict[str, Any] = {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
        }
        if finite.size:
            summary.update(
                {
                    "finite_count": int(finite.size),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "sum": float(np.sum(finite)),
                }
            )
        else:
            summary["finite_count"] = 0
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_result(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def get_path(value: Any, path: Iterable[Any]) -> Any:
    current = value
    for part in path:
        if isinstance(current, dict):
            current = current[part]
        else:
            current = current[int(part)]
    return current


def expectation_errors(case: dict[str, Any], outcome: dict[str, Any]) -> list[str]:
    expect = case.get("expect", {"class": "ok"})
    errors: list[str] = []
    expected_class = expect.get("class", "ok")
    if expected_class == "error":
        expected_class = "structured_error"
    if outcome.get("class") != expected_class:
        return [f"class {outcome.get('class')} != expected {expected_class}: {outcome}"]

    if expected_class == "structured_error":
        error_type = expect.get("type")
        if error_type and outcome.get("error_type") != error_type:
            errors.append(f"error_type {outcome.get('error_type')} != {error_type}")
        match = expect.get("match")
        if match and match not in outcome.get("message", ""):
            errors.append(f"error message does not contain {match!r}: {outcome.get('message')!r}")

    for check in expect.get("checks", ()):
        actual = get_path(outcome["result"], check["path"])
        if "equals" in check and actual != check["equals"]:
            errors.append(f"{check['path']} actual {actual!r} != {check['equals']!r}")
        if "contains" in check and check["contains"] not in actual:
            errors.append(f"{check['path']} actual {actual!r} lacks {check['contains']!r}")
        if "approx" in check:
            tolerance = float(check.get("tolerance", 1.0e-9))
            expected = float(check["approx"])
            if abs(float(actual) - expected) > tolerance:
                errors.append(f"{check['path']} actual {actual!r} not within {tolerance} of {expected!r}")
        if "min" in check and float(actual) < float(check["min"]):
            errors.append(f"{check['path']} actual {actual!r} < {check['min']!r}")
        if "max" in check and float(actual) > float(check["max"]):
            errors.append(f"{check['path']} actual {actual!r} > {check['max']!r}")
    return errors


def scoreboard(outcomes: Iterable[dict[str, Any]]) -> dict[str, int]:
    board = {"total": 0, "ok": 0, "structured_error": 0, "panic": 0, "hang": 0, "wrong_value": 0}
    for outcome in outcomes:
        board["total"] += 1
        cls = str(outcome.get("class"))
        board[cls] = board.get(cls, 0) + 1
    return board


def format_scoreboard(board: dict[str, int]) -> str:
    keys = ("total", "ok", "structured_error", "panic", "hang", "wrong_value")
    return " ".join(f"{key}={board.get(key, 0)}" for key in keys)


if __name__ == "__main__":  # pragma: no cover - convenience for manual replay.
    for raw in sys.argv[1:]:
        case = json.loads(Path(raw).read_text(encoding="utf-8"))
        print(json.dumps(classify_case(case), indent=2, sort_keys=True))
