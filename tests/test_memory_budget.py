# tests/test_memory_budget.py
# Validate Python memory telemetry helpers and budget enforcement
# Exists to ensure Python layer mirrors native counters and raises when exceeded
# RELEVANT FILES: python/forge3d/mem.py, python/forge3d/_memory.py, src/core/memory_tracker.rs, tests/test_b15_memory_integration.py

import pytest

import forge3d.mem as mem
import forge3d._memory as native_mem


def test_memory_metrics_includes_native_fields(monkeypatch):
    """Ensure memory_metrics always includes the extended telemetry keys."""
    monkeypatch.setattr(native_mem, "memory_metrics", lambda: {
        "buffer_count": 1,
        "texture_count": 2,
        "buffer_bytes": 16,
        "texture_bytes": 32,
        "host_visible_bytes": 8,
        "total_bytes": 48,
        "limit_bytes": mem.MEMORY_LIMIT_BYTES,
        "within_budget": True,
        "utilization_ratio": 0.25,
    })
    metrics = mem.memory_metrics()
    for key in (
        "resident_tiles",
        "resident_tile_bytes",
        "staging_bytes_in_flight",
        "staging_ring_count",
        "staging_buffer_size",
        "staging_buffer_stalls",
    ):
        assert key in metrics
        assert metrics[key] == 0


def test_update_memory_usage_enforces_budget(monkeypatch):
    """update_memory_usage should raise when usage exceeds the configured limit."""
    original_limit = mem.MEMORY_LIMIT_BYTES
    original_state = native_mem._GLOBAL_MEMORY.copy()

    mem.override_memory_limit(128)
    for key in native_mem._GLOBAL_MEMORY:
        native_mem._GLOBAL_MEMORY[key] = 0

    with pytest.raises(RuntimeError):
        mem.update_memory_usage(buffer_bytes_delta=256)

    mem.override_memory_limit(original_limit)
    native_mem._GLOBAL_MEMORY.update(original_state)