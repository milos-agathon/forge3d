# python/forge3d/_memory.py
# Memory accounting utilities shared across the Python facade.
# Keeps host-visible usage tracking consistent with the Rust telemetry bridge.
# RELEVANT FILES:python/forge3d/__init__.py, python/forge3d/_memory.py

from __future__ import annotations

from typing import Dict

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
    """Apply deltas to the global memory counters, clamping at zero."""
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


def memory_metrics() -> Dict[str, float]:
    """Return a snapshot of tracked memory usage."""
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
    }


__all__ = [
    "MEMORY_LIMIT_BYTES",
    "aligned_row_size",
    "update_memory_usage",
    "memory_metrics",
]
