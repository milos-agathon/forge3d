"""Executable material-distinctness contract for the TERMINUS atlas.

Coverage metadata never participates in the semantic signature. Every entry is
re-derived from the operation, payload, property, and oracle, so identifiers or
notes cannot make otherwise equivalent cases pass.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import re
from typing import Any, Iterable

REQUIRED_COVERAGE_FIELDS = {
    "case_id",
    "family",
    "operation",
    "input_partition",
    "pathology",
    "boundary",
    "oracle_kind",
    "expected_outcome",
    "distinguishing_feature",
}
GENERIC_RATIONALE = re.compile(
    r"(?:distinct\s+(?:adversarial\s+)?input|distinct\s+.*pathology\s+variant|variant\s+\d+)",
    re.I,
)


def _number_category(value: Any) -> str:
    if isinstance(value, str) and value in {
        "nan",
        "inf",
        "+inf",
        "-inf",
        "f32_max",
        "-f32_max",
        "subnormal",
        "-subnormal",
    }:
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return type(value).__name__
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    absolute = abs(number)
    if number == 0:
        return "zero"
    if absolute < 1.0e-37:
        return "subnormal"
    if absolute < 1.0e-6:
        return "near_zero"
    if absolute < 1.0:
        return "fractional"
    if absolute == 1.0:
        return "unit"
    if absolute <= 90.0:
        return "ordinary"
    if absolute <= 180.0:
        return "geographic_limit"
    if absolute <= 1.0e6:
        return "large"
    if absolute <= 3.5e38:
        return "f32_extreme"
    return "beyond_f32"


def _shape_partition(shape: list[int] | tuple[int, ...]) -> str:
    dims = tuple(int(item) for item in shape)
    if not dims or any(item == 0 for item in dims):
        return "empty"
    if len(dims) == 1:
        return "vector_singleton" if dims[0] == 1 else "vector"
    if len(dims) != 2:
        return f"rank_{len(dims)}"
    rows, cols = dims
    if rows == cols == 1:
        return "singleton_2d"
    if rows == 1:
        return "single_row"
    if cols == 1:
        return "single_column"
    if rows == cols:
        return "square_small" if rows <= 16 else "square_large"
    return "rectangular_small" if max(rows, cols) <= 16 else "rectangular_large"


def _array_signature(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        return {"representation": type(spec).__name__}
    dtype = str(spec.get("dtype", "float64"))
    if "shape" in spec:
        shape = [int(value) for value in spec["shape"]]
    elif "values" in spec and isinstance(spec["values"], list):
        values = spec["values"]
        if values and isinstance(values[0], list):
            shape = [len(values), min((len(row) for row in values), default=0)]
        else:
            shape = [len(values)]
    else:
        shape = []
    signature: dict[str, Any] = {"dtype": dtype, "shape": _shape_partition(shape)}
    if "pattern" in spec:
        signature["variability"] = str(spec["pattern"])
    elif "fill" in spec:
        signature["variability"] = "constant"
        signature["value_class"] = _number_category(spec["fill"])
    elif "arange" in spec:
        signature["variability"] = "arange"
        signature["range_classes"] = [_number_category(value) for value in spec["arange"]]
    elif "values" in spec:
        flat: list[Any] = []

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
            else:
                flat.append(value)

        visit(spec["values"])
        classes = [_number_category(value) for value in flat]
        unique = sorted(set(classes))
        if not flat:
            variability = "empty"
        elif all(item in {"nan", "+inf", "-inf", "inf"} for item in classes):
            variability = "all_non_finite"
        elif any(item in {"nan", "+inf", "-inf", "inf"} for item in classes):
            variability = "mixed_finiteness"
        elif len({json.dumps(value, sort_keys=True) for value in flat}) == 1:
            variability = "constant"
        else:
            variability = "variable"
        signature.update(variability=variability, value_classes=unique)
    else:
        signature["variability"] = "implicit"
    return signature


def _geographic_locus(points: list[tuple[float, float]]) -> str:
    if not points:
        return "empty"
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if any(abs(value) >= 89.999 for value in ys):
        if max(ys) >= 89.999 and min(ys) <= -89.999:
            return "both_poles"
        return "north_pole" if max(ys) >= 89.999 else "south_pole"
    if max(xs) - min(xs) > 180.0 or any(abs(value) >= 179.0 for value in xs):
        return "antimeridian"
    return "ordinary"


def _canonical_path(points: list[tuple[float, float]], *, ring: bool) -> list[list[float]]:
    if not points:
        return []
    work = points[:-1] if ring and len(points) > 1 and points[0] == points[-1] else points[:]
    min_x = min(point[0] for point in work)
    min_y = min(point[1] for point in work)
    relative = [(round(x - min_x, 9), round(y - min_y, 9)) for x, y in work]
    if ring and relative:
        variants = []
        for oriented in (relative, list(reversed(relative))):
            variants.extend(oriented[index:] + oriented[:index] for index in range(len(oriented)))
        relative = min(variants)
    elif len(relative) > 1:
        relative = min(relative, list(reversed(relative)))
    return [[x, y] for x, y in relative]


def _geometry_signature(geometry: Any) -> dict[str, Any]:
    if not isinstance(geometry, dict):
        return {"representation": type(geometry).__name__}
    geometry_type = str(geometry.get("type", "missing"))
    coords = geometry.get("coordinates")
    points: list[tuple[float, float]] = []
    special: list[str] = []

    def visit(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and not isinstance(value[0], list)
            and not isinstance(value[1], list)
        ):
            categories = [_number_category(value[0]), _number_category(value[1])]
            try:
                x, y = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                special.extend(categories)
                return
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
            else:
                special.extend(categories)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif value is not None:
            special.append(type(value).__name__)

    visit(coords)
    signature: dict[str, Any] = {
        "type": geometry_type,
        "point_count": len(points),
        "locus": _geographic_locus(points),
        "special_coordinates": sorted(set(special)),
    }
    if points and not special:
        signature["translation_invariant_shape"] = _canonical_path(
            points,
            ring=geometry_type in {"Polygon", "MultiPolygon"},
        )
    return signature


def _text_partition(text: str) -> str:
    if not text:
        return "empty"
    if "\n" in text:
        return "multiline"
    if len(text) >= 64:
        return "long_run"
    codepoints = [ord(char) for char in text]
    if any(0x1F300 <= value <= 0x1FAFF for value in codepoints):
        return "emoji"
    if any(0x0590 <= value <= 0x05FF for value in codepoints):
        return "hebrew"
    if any(0x0600 <= value <= 0x06FF for value in codepoints):
        return "arabic"
    if any(0x0900 <= value <= 0x097F for value in codepoints):
        return "devanagari"
    return "ascii_single" if len(text) == 1 else "ascii_word"


def _bounds_partition(bounds: Any) -> dict[str, Any]:
    if not isinstance(bounds, list) or len(bounds) != 4:
        return {"shape": "malformed"}
    categories = [_number_category(value) for value in bounds]
    try:
        values = [float(value) for value in bounds]
    except (TypeError, ValueError):
        return {"shape": "non_numeric", "classes": categories}
    if not all(math.isfinite(value) for value in values):
        return {"shape": "non_finite", "classes": categories}
    left, bottom, right, top = values
    if bottom < -90 or top > 90:
        shape = "latitude_outside_world"
    elif left > right:
        shape = "antimeridian_crossing"
    elif left == right or bottom == top:
        shape = "zero_extent"
    elif bottom < -85.05112878 or top > 85.05112878:
        shape = "web_mercator_clamp"
    elif left <= -180 and right >= 180:
        shape = "full_world"
    else:
        shape = "ordinary"
    return {"shape": shape, "classes": categories}


def _normalized_payload(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(payload.items()):
        if key in {"name", "source_id"}:
            continue
        if key in {"array", "terrain"}:
            result[key] = _array_signature(value)
        elif key in {"geometry", "exterior"}:
            result[key] = _geometry_signature(value if key == "geometry" else {"type": "Polygon", "coordinates": [value]})
        elif key in {"positions", "path", "nodes"} and isinstance(value, (dict, list)):
            result[key] = _array_signature(value)
        elif key == "bounds":
            result[key] = _bounds_partition(value)
        elif key == "text":
            result[key] = _text_partition(str(value))
        elif key == "fonts":
            fonts = value or []
            result[key] = [
                "missing" if "missing" in str(font) else str(font).rsplit("/", 1)[-1]
                for font in fonts
            ]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in {"zoom", "width", "height", "anchor_count", "vertices", "samples", "bit_depth"}:
                result[key] = value
            else:
                result[key] = _number_category(value)
        elif isinstance(value, list):
            result[key] = [
                _number_category(item) if isinstance(item, (int, float)) else item for item in value
            ]
        elif isinstance(value, dict):
            result[key] = {
                child_key: (
                    _number_category(child_value)
                    if isinstance(child_value, (int, float)) and not isinstance(child_value, bool)
                    else child_value
                )
                for child_key, child_value in sorted(value.items())
            }
        else:
            result[key] = value
    return result


def semantic_signature(case: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect", {"class": "ok"})
    oracle = {
        "class": "structured_error" if expect.get("class") == "error" else expect.get("class", "ok"),
        "type": expect.get("type"),
        "match": expect.get("match"),
        "checks": expect.get("checks", []),
        "property": case.get("property"),
        "property_tolerance": case.get("property_tolerance"),
    }
    return {
        "operation": case["operation"],
        "payload": _normalized_payload(case["operation"], case.get("payload", {})),
        "oracle": oracle,
    }


def derive_coverage(case: dict[str, Any]) -> dict[str, Any]:
    signature = semantic_signature(case)
    expect = case.get("expect", {"class": "ok"})
    if expect.get("checks"):
        oracle_kind = "exact_behavioral_checks"
    elif case.get("property"):
        oracle_kind = "property_oracle"
    elif expect.get("class") in {"error", "structured_error"}:
        oracle_kind = "typed_structured_error"
    else:
        oracle_kind = "completion_classification"
    payload_text = json.dumps(signature["payload"], sort_keys=True, separators=(",", ":"))
    pathology_tokens = sorted(
        token
        for token in {
            "non_finite" if any(mark in payload_text for mark in ('"nan"', '"+inf"', '"-inf"')) else "",
            "empty" if '"empty"' in payload_text else "",
            "malformed" if expect.get("class") in {"error", "structured_error"} else "",
        }
        if token
    )
    boundary_tokens = sorted(
        token
        for token in (
            "antimeridian" if "antimeridian" in payload_text else "",
            "pole" if "pole" in payload_text else "",
            "clamp" if "clamp" in payload_text else "",
            "safe_limit" if any(key in case.get("payload", {}) for key in ("anchor_count", "vertices")) else "",
        )
        if token
    )
    return {
        "case_id": case["id"],
        "family": case["family"],
        "operation": case["operation"],
        "input_partition": f"{case['operation']}:{payload_text}",
        "pathology": "+".join(pathology_tokens) or "none",
        "boundary": "+".join(boundary_tokens) or "ordinary",
        "oracle_kind": oracle_kind,
        "expected_outcome": "structured_error"
        if expect.get("class") == "error"
        else expect.get("class", "ok"),
        "distinguishing_feature": payload_text,
    }


def validate_materiality(
    cases: Iterable[dict[str, Any]], coverage_entries: Iterable[dict[str, Any]]
) -> list[str]:
    cases = list(cases)
    coverage_entries = list(coverage_entries)
    errors: list[str] = []
    case_by_id = {case["id"]: case for case in cases}
    coverage_by_id: dict[str, dict[str, Any]] = {}
    for entry in coverage_entries:
        case_id = entry.get("case_id")
        if case_id in coverage_by_id:
            errors.append(f"duplicate coverage entry for {case_id}")
        coverage_by_id[case_id] = entry
    missing = sorted(set(case_by_id) - set(coverage_by_id))
    orphaned = sorted(set(coverage_by_id) - set(case_by_id))
    if missing:
        errors.append(f"descriptors without coverage: {missing}")
    if orphaned:
        errors.append(f"coverage without descriptors: {orphaned}")

    clusters: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        notes = str(case.get("notes", ""))
        if GENERIC_RATIONALE.search(notes):
            errors.append(f"{case['id']}: generic padding rationale {notes!r}")
        entry = coverage_by_id.get(case["id"])
        if entry is not None:
            absent = sorted(REQUIRED_COVERAGE_FIELDS - set(entry))
            if absent:
                errors.append(f"{case['id']}: coverage missing fields {absent}")
            expected = derive_coverage(case)
            if entry != expected:
                errors.append(f"{case['id']}: coverage does not match executable descriptor")
            for value in entry.values():
                if isinstance(value, str) and GENERIC_RATIONALE.search(value):
                    errors.append(f"{case['id']}: generic coverage rationale {value!r}")
        signature = json.dumps(semantic_signature(case), sort_keys=True, separators=(",", ":"))
        clusters[signature].append(case["id"])
    for signature, ids in clusters.items():
        if len(ids) > 1:
            operation = json.loads(signature)["operation"]
            errors.append(f"materiality collision operation={operation} ids={sorted(ids)}")
    return errors
