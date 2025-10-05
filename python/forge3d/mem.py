# python/forge3d/mem.py
# Memory telemetry facade combining native metrics with Python fallbacks
# Exists to expose budget helpers to higher level APIs without recursion bugs
# RELEVANT FILES: python/forge3d/_memory.py, src/core/memory_tracker.rs, python/forge3d/__init__.py, tests/test_memory_budget.py

from __future__ import annotations

from typing import Dict

from . import _memory

MEMORY_LIMIT_BYTES: int = _memory.MEMORY_LIMIT_BYTES
aligned_row_size = _memory.aligned_row_size

__all__ = [
    "MEMORY_LIMIT_BYTES",
    "aligned_row_size",
    "memory_metrics",
    "update_memory_usage",
    "enforce_memory_budget",
    "budget_remaining",
    "utilization_ratio",
    "override_memory_limit",
]


def memory_metrics() -> Dict[str, float]:
    metrics = dict(_memory.memory_metrics())
    metrics.setdefault("resident_tiles", 0)
    metrics.setdefault("resident_tile_bytes", 0)
    metrics.setdefault("staging_bytes_in_flight", 0)
    metrics.setdefault("staging_ring_count", 0)
    metrics.setdefault("staging_buffer_size", 0)
    metrics.setdefault("staging_buffer_stalls", 0)
    return metrics


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