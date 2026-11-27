# python/forge3d/mem.py
# Memory telemetry facade combining native metrics with Python fallbacks
# Exists to expose budget helpers to higher level APIs without recursion bugs
# RELEVANT FILES: python/forge3d/_memory.py, src/core/memory_tracker.rs, python/forge3d/__init__.py, tests/test_memory_budget.py

from __future__ import annotations

from typing import Dict

from . import _memory
from ._poster_memory import MemoryAdmissionRefused, estimate_poster as _estimate_poster

MEMORY_LIMIT_BYTES: int = _memory.MEMORY_LIMIT_BYTES
aligned_row_size = _memory.aligned_row_size

__all__ = [
    "estimate",
    "MemoryAdmissionRefused",
    "MEMORY_LIMIT_BYTES",
    "aligned_row_size",
    "memory_metrics",
    "update_memory_usage",
    "enforce_memory_budget",
    "set_budget_policy",
    "budget_policy",
    "get_budget_policy",
    "budget_remaining",
    "utilization_ratio",
    "override_memory_limit",
]


def estimate(job: dict) -> dict:
    """Preflight a poster, MapScene.render (scene=...), or render_flythrough job."""
    from inspect import signature
    from .path_tracing import render_terrain_poster

    options = dict(job)
    route = options.pop("entry_point", None)
    if route in ("MapScene.render", "render_flythrough"):
        from .map_scene import MapScene
        from .chronos import render_flythrough
        from ._scene_memory import estimate_scene_job
        if route == "MapScene.render":
            if "scene" in options:
                options["self"] = options.pop("scene")
            bound = signature(MapScene.render).bind(**options)
        else:
            bound = signature(render_flythrough).bind(**options)
        return estimate_scene_job(route, bound.arguments)
    if route != "render_terrain_poster":
        raise ValueError(f"unsupported memory estimation entry point: {route!r}")
    bound = signature(render_terrain_poster).bind(**options)
    if extra := bound.arguments.get("kwargs"):
        raise TypeError(f"unexpected poster arguments: {', '.join(extra)}")
    parameters = signature(_estimate_poster).parameters
    return _estimate_poster(**{key: value for key, value in options.items() if key in parameters})


def memory_metrics() -> Dict[str, float]:
    metrics = dict(_memory.memory_metrics())
    metrics.setdefault("resident_tiles", 0)
    metrics.setdefault("resident_tile_bytes", 0)
    metrics.setdefault("staging_bytes_in_flight", 0)
    metrics.setdefault("staging_ring_count", 0)
    metrics.setdefault("staging_buffer_size", 0)
    metrics.setdefault("staging_buffer_stalls", 0)
    metrics.setdefault("budget_policy", get_budget_policy())
    return metrics


def set_budget_policy(policy: str) -> str:
    return _memory.set_budget_policy(policy)


def budget_policy(policy: str) -> str:
    """Spec-named opt-in alias for :func:`set_budget_policy`.

    Sets the host-visible memory budget behavior to ``'enforce'`` (default) or
    ``'warn'`` and returns the normalized policy.
    """
    return _memory.set_budget_policy(policy)


def get_budget_policy() -> str:
    return _memory.get_budget_policy()


def enforce_memory_budget() -> None:
    _memory.enforce_memory_budget()


def update_memory_usage(
    *,
    buffer_bytes_delta: int = 0,
    texture_bytes_delta: int = 0,
    buffer_count_delta: int = 0,
    texture_count_delta: int = 0,
) -> None:
    _memory.update_memory_usage(
        buffer_bytes_delta=buffer_bytes_delta,
        texture_bytes_delta=texture_bytes_delta,
        buffer_count_delta=buffer_count_delta,
        texture_count_delta=texture_count_delta,
    )
    enforce_memory_budget()


def budget_remaining() -> int:
    metrics = memory_metrics()
    limit = int(metrics.get("limit_bytes", MEMORY_LIMIT_BYTES))
    remaining = limit - int(metrics.get("host_visible_bytes", 0))
    return max(0, remaining)


def utilization_ratio() -> float:
    metrics = memory_metrics()
    limit = float(metrics.get("limit_bytes", MEMORY_LIMIT_BYTES) or 1)
    return float(metrics.get("host_visible_bytes", 0)) / limit


def override_memory_limit(limit_bytes: int) -> None:
    global MEMORY_LIMIT_BYTES
    MEMORY_LIMIT_BYTES = int(limit_bytes)
    _memory.MEMORY_LIMIT_BYTES = MEMORY_LIMIT_BYTES
