"""Dependency-free canonical JSON helpers shared by signed artifacts."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping


def canonical_json_value(value: Any, *, error_context: str) -> Any:
    """Normalize finite JSON values for byte-stable serialization."""
    if isinstance(value, Mapping):
        return {
            str(key): canonical_json_value(item, error_context=error_context)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [canonical_json_value(item, error_context=error_context) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{error_context} does not permit NaN or infinite floats")
        return 0.0 if value == 0.0 else value
    return value


def canonical_json_bytes(value: Any, *, error_context: str) -> bytes:
    """Serialize a normalized JSON value with stable key and separator rules."""
    normalized = canonical_json_value(value, error_context=error_context)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
