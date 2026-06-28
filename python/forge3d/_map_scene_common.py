"""Shared private helpers for MapScene recipe modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value.keys(), key=str):
            result[str(key)] = _json_safe(value[key])
        return result
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"MapScene recipe values must be JSON-serializable, got {type(value).__name__}")


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return _json_safe(dict(value or {}))


def _stable_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sequence(value: Sequence[Any] | None) -> list[Any]:
    return [_json_safe(item) for item in (value or ())]


def _layer_id(layer: Any, fallback: str) -> str:
    return str(getattr(layer, "layer_id", None) or getattr(layer, "name", None) or fallback)


def _same_crs(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    try:
        from .crs import _crs_equal

        return bool(_crs_equal(str(left), str(right)))
    except Exception:
        return str(left).strip().lower() == str(right).strip().lower()


def _metadata_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _has_explicit_crs_policy(layer: Any) -> bool:
    metadata = _metadata_dict(getattr(layer, "metadata", None))
    policy = str(metadata.get("crs_policy", "")).lower()
    return bool(metadata.get("crs_transform")) or policy in {
        "compatible",
        "explicit_transform",
        "transform_provided",
    }
