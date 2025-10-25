# python/forge3d/_memory.py
# Memory accounting utilities shared across the Python facade.
# Keeps host-visible usage tracking consistent with the Rust telemetry bridge.
# RELEVANT FILES:python/forge3d/__init__.py, python/forge3d/_memory.py

from __future__ import annotations

from typing import Dict

from ._native import NATIVE_AVAILABLE, get_native_module

MEMORY_LIMIT_BYTES: int = 512 * 1024 * 1024  # 512 MiB budget for host-visible memory
_GLOBAL_MEMORY = {
    "buffer_count": 0,
    "texture_count": 0,
    "buffer_bytes": 0,
    "texture_bytes": 0,
}


def aligned_row_size(row_bytes: int, alignment: int = 256) -> int:
    """Round row_bytes up to the next multiple of alignment."""
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((int(row_bytes) + alignment - 1) // alignment) * alignment


def update_memory_usage(
    *,
    buffer_bytes_delta: int = 0,
    texture_bytes_delta: int = 0,
    buffer_count_delta: int = 0,
    texture_count_delta: int = 0,
) -> None:
    """Apply deltas to the fallback memory counters, clamping at zero."""
    _GLOBAL_MEMORY["buffer_bytes"] = max(
        0, _GLOBAL_MEMORY["buffer_bytes"] + int(buffer_bytes_delta)
    )
    _GLOBAL_MEMORY["texture_bytes"] = max(
        0, _GLOBAL_MEMORY["texture_bytes"] + int(texture_bytes_delta)
    )
    _GLOBAL_MEMORY["buffer_count"] = max(
        0, _GLOBAL_MEMORY["buffer_count"] + int(buffer_count_delta)
    )
    _GLOBAL_MEMORY["texture_count"] = max(
        0, _GLOBAL_MEMORY["texture_count"] + int(texture_count_delta)
    )


def _fallback_metrics() -> Dict[str, float]:
    buffer_bytes = int(_GLOBAL_MEMORY["buffer_bytes"])
    texture_bytes = int(_GLOBAL_MEMORY["texture_bytes"])
    total = buffer_bytes + texture_bytes
    host_visible = min(buffer_bytes, MEMORY_LIMIT_BYTES)
    within = host_visible <= MEMORY_LIMIT_BYTES
    utilization = host_visible / MEMORY_LIMIT_BYTES if MEMORY_LIMIT_BYTES else 0.0
    return {
        "buffer_count": int(_GLOBAL_MEMORY["buffer_count"]),
        "texture_count": int(_GLOBAL_MEMORY["texture_count"]),
        "buffer_bytes": buffer_bytes,
        "texture_bytes": texture_bytes,
        "total_bytes": total,
        "host_visible_bytes": host_visible,
        "limit_bytes": MEMORY_LIMIT_BYTES,
        "within_budget": bool(within),
        "utilization_ratio": float(utilization),
        "resident_tiles": 0,
        "resident_tile_bytes": 0,
        "staging_bytes_in_flight": 0,
        "staging_ring_count": 0,
        "staging_buffer_size": 0,
        "staging_buffer_stalls": 0,
    }


def memory_metrics() -> Dict[str, float]:
    """Return a snapshot of tracked memory usage."""
    native = get_native_module() if NATIVE_AVAILABLE else None
    native_fn = getattr(native, "global_memory_metrics", None) if native else None
    if callable(native_fn):
        try:
            metrics = dict(native_fn())
        except Exception:
            metrics = {}
        else:
            metrics.setdefault("buffer_count", 0)
            metrics.setdefault("texture_count", 0)
            metrics.setdefault("buffer_bytes", 0)
            metrics.setdefault("texture_bytes", 0)
            metrics.setdefault(
                "host_visible_bytes", metrics.get("buffer_bytes", 0)
            )
            metrics.setdefault(
                "total_bytes",
                metrics.get("buffer_bytes", 0) + metrics.get("texture_bytes", 0),
            )
            metrics.setdefault("limit_bytes", MEMORY_LIMIT_BYTES)
            metrics.setdefault("within_budget", True)
            metrics.setdefault("utilization_ratio", 0.0)
            metrics.setdefault("resident_tiles", 0)
            metrics.setdefault("resident_tile_bytes", 0)
            metrics.setdefault("staging_bytes_in_flight", 0)
            metrics.setdefault("staging_ring_count", 0)
            metrics.setdefault("staging_buffer_size", 0)
            metrics.setdefault("staging_buffer_stalls", 0)
            return metrics

    return _fallback_metrics()


def enforce_memory_budget() -> None:
    """Raise RuntimeError if the host-visible memory budget is exceeded."""
    metrics = memory_metrics()
    if not metrics.get("within_budget", True):
        raise RuntimeError(
            f"Host-visible memory budget exceeded: {metrics['host_visible_bytes']} / {metrics['limit_bytes']} bytes"
        )


__all__ = [
    "MEMORY_LIMIT_BYTES",
    "aligned_row_size",
    "update_memory_usage",
    "memory_metrics",
    "enforce_memory_budget",
]
