# python/forge3d/viewer.py
# Viewer control utilities for Workstream B anti-aliasing toggles.
# Exists to expose MSAA configuration hooks to Python clients.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/pbr.py,tests/test_b1_msaa.py,shaders/tone_map.wgsl

from __future__ import annotations

import warnings

try:
    from . import _forge3d  # type: ignore[attr-defined]
except ImportError:
    _forge3d = None

from . import Renderer, _SUPPORTED_MSAA


def set_msaa(samples: int) -> int:
    """Set the default MSAA sample count for newly created renderers."""
    if samples not in _SUPPORTED_MSAA:
        raise ValueError(f"Unsupported MSAA sample count: {samples} (allowed: {_SUPPORTED_MSAA})")

    Renderer._set_default_msaa(samples)

    if _forge3d is not None:
        setter = getattr(_forge3d, "set_msaa_samples", None)
        if setter is not None:
            try:
                setter(samples)
            except Exception as exc:  # pragma: no cover - optional binding path
                warnings.warn(f"forge3d.set_msaa_samples failed: {exc}")

    return samples
