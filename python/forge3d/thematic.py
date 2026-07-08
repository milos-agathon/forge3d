"""Vector-value thematic classification helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["classify", "apply_palette"]


def _valid_values(values: np.ndarray, nodata: float | None) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != float(nodata)
    return values[valid].astype(np.float64), valid


def _validate_k(k: int, count: int) -> int:
    k = int(k)
    if k < 2:
        raise ValueError("k must be at least 2")
    if count == 0:
        raise ValueError("empty thematic input: no finite valid values")
    if k > count:
        raise ValueError("k must not exceed the number of valid values")
    return k


def _equal_interval_bins(valid: np.ndarray, k: int) -> np.ndarray:
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if lo == hi:
        raise ValueError("equal_interval requires at least two distinct values")
    return np.linspace(lo, hi, k + 1, dtype=np.float64)[1:-1]


def _quantile_bins(valid: np.ndarray, k: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, k + 1, dtype=np.float64)[1:-1]
    bins = np.quantile(valid, qs)
    if np.unique(bins).size != bins.size:
        raise ValueError("quantile produced duplicate bins; reduce k or provide more varied values")
    return bins.astype(np.float64)


def _jenks_bins(valid: np.ndarray, k: int) -> np.ndarray:
    data = np.sort(valid.astype(np.float64))
    n = data.size
    lower = np.zeros((n + 1, k + 1), dtype=np.int32)
    variance = np.full((n + 1, k + 1), np.inf, dtype=np.float64)

    for classes in range(1, k + 1):
        lower[1, classes] = 1
        variance[1, classes] = 0.0
        for row in range(2, n + 1):
            variance[row, classes] = np.inf

    for end in range(2, n + 1):
        sum_values = 0.0
        sum_squares = 0.0
        weight = 0.0
        for start_offset in range(1, end + 1):
            start = end - start_offset + 1
            value = data[start - 1]
            weight += 1.0
            sum_values += value
            sum_squares += value * value
            group_variance = sum_squares - (sum_values * sum_values) / weight
            previous = start - 1
            if previous == 0:
                continue
            for classes in range(2, k + 1):
                candidate = group_variance + variance[previous, classes - 1]
                if variance[end, classes] >= candidate:
                    lower[end, classes] = start
                    variance[end, classes] = candidate
        lower[end, 1] = 1
        variance[end, 1] = group_variance

    bins = np.zeros(k - 1, dtype=np.float64)
    end = n
    for classes in range(k, 1, -1):
        start = lower[end, classes]
        bins[classes - 2] = data[start - 2]
        end = start - 1
    if np.unique(bins).size != bins.size:
        raise ValueError("jenks produced duplicate bins; reduce k or provide more varied values")
    return bins


def classify(
    values: Any,
    *,
    scheme: str = "quantile",
    k: int = 5,
    nodata: float | None = None,
    right: bool = False,
) -> dict[str, Any]:
    """Classify numeric values into 1-based class IDs, reserving 0 for nodata."""
    array = np.asarray(values)
    valid_values, valid_mask = _valid_values(array.astype(np.float64, copy=False), nodata)
    k = _validate_k(k, valid_values.size)
    scheme = str(scheme).strip().lower().replace("-", "_")

    if scheme == "equal_interval":
        bins = _equal_interval_bins(valid_values, k)
    elif scheme == "quantile":
        bins = _quantile_bins(valid_values, k)
    elif scheme in {"jenks", "natural_breaks"}:
        bins = _jenks_bins(valid_values, k)
        scheme = "jenks"
        right = True
    else:
        raise ValueError("scheme must be one of: equal_interval, quantile, jenks")

    classes = np.zeros(array.shape, dtype=np.uint16)
    classes[valid_mask] = np.digitize(valid_values, bins, right=right).astype(np.uint16) + 1
    table = [
        {
            "class_id": class_id,
            "left": None if class_id == 1 else float(bins[class_id - 2]),
            "right": None if class_id == k else float(bins[class_id - 1]),
            "count": int(np.count_nonzero(classes == class_id)),
        }
        for class_id in range(1, k + 1)
    ]
    return {
        "scheme": scheme,
        "k": k,
        "bins": bins,
        "classes": classes,
        "class_table": table,
        "valid_count": int(valid_values.size),
        "nodata_count": int(array.size - valid_values.size),
    }


def apply_palette(
    classes: Any,
    colors: Sequence[Sequence[int]],
    *,
    nodata_color: Sequence[int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Map 0-based nodata and 1-based thematic classes to RGBA8 colors."""
    class_array = np.asarray(classes)
    rgba = np.zeros(class_array.shape + (4,), dtype=np.uint8)
    rgba[class_array == 0] = _rgba(nodata_color)
    for class_id, color in enumerate(colors, start=1):
        rgba[class_array == class_id] = _rgba(color)
    return rgba


def _rgba(color: Sequence[int]) -> tuple[int, int, int, int]:
    if len(color) == 3:
        r, g, b = color
        a = 255
    elif len(color) == 4:
        r, g, b, a = color
    else:
        raise ValueError("colors must be RGB or RGBA sequences")
    return (int(r), int(g), int(b), int(a))
