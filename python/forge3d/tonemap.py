"""
Tone mapping utilities for Milestone 5 (Linear/Reinhard/ACES).

Provides numpy-friendly helpers for applying tone curves in linear space,
including the Narkowicz ACES Filmic approximation with the official
input/output matrices and RRT+ODT fit.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

TONEMAP_LINEAR = "linear"
TONEMAP_REINHARD = "reinhard"
TONEMAP_ACES = "aces"

TONEMAP_MODES = (TONEMAP_LINEAR, TONEMAP_REINHARD, TONEMAP_ACES)

_ACES_INPUT = np.array(
    [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ],
    dtype=np.float32,
)

_ACES_OUTPUT = np.array(
    [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ],
    dtype=np.float32,
)


def _as_float_array(color: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(color, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB data with shape (..., 3); got {arr.shape}")
    return arr


def _clip_positive(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, None)


def tonemap_linear(color: np.ndarray | Iterable[float]) -> np.ndarray:
    """Linear (no tone map) path."""
    return _clip_positive(_as_float_array(color))


def tonemap_reinhard(color: np.ndarray | Iterable[float]) -> np.ndarray:
    """Reinhard per-channel tone map."""
    arr = _clip_positive(_as_float_array(color))
    return arr / (arr + 1.0)


def _rrt_odt_fit(v: np.ndarray) -> np.ndarray:
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / np.where(np.abs(b) < 1e-8, 1.0, b)


def tonemap_aces(color: np.ndarray | Iterable[float]) -> np.ndarray:
    """ACES Filmic tone map using Narkowicz's approximation."""
    arr = _clip_positive(_as_float_array(color))
    flat = arr.reshape(-1, 3)
    aces = flat @ _ACES_INPUT.T
    aces = _rrt_odt_fit(aces)
    out = aces @ _ACES_OUTPUT.T
    return np.clip(out.reshape(arr.shape), 0.0, 1.0)


def apply_tonemap(color: np.ndarray | Iterable[float], mode: str) -> np.ndarray:
    """Apply specified tone mapping curve."""
    normalized = mode.lower()
    if normalized not in TONEMAP_MODES:
        raise ValueError(f"Unsupported tone mapping mode '{mode}'. Expected one of {TONEMAP_MODES}.")
    if normalized == TONEMAP_LINEAR:
        return tonemap_linear(color)
    if normalized == TONEMAP_REINHARD:
        return tonemap_reinhard(color)
    return tonemap_aces(color)


__all__ = [
    "TONEMAP_LINEAR",
    "TONEMAP_REINHARD",
    "TONEMAP_ACES",
    "TONEMAP_MODES",
    "tonemap_linear",
    "tonemap_reinhard",
    "tonemap_aces",
    "apply_tonemap",
]
