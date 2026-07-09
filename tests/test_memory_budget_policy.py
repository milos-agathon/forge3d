import importlib

import pytest


def test_default_budget_policy_is_enforce():
    # CENSOR contract change: the honest default host-visible budget policy is
    # 'enforce', not 'warn'. Reload the module to read its pristine default,
    # independent of any runtime mutation by other tests in the session.
    memory = importlib.reload(importlib.import_module("forge3d._memory"))
    assert memory.BUDGET_POLICY == "enforce"


def test_set_budget_policy_validates_values():
    mem = importlib.import_module("forge3d.mem")

    assert mem.set_budget_policy("warn") == "warn"
    assert mem.set_budget_policy("enforce") == "enforce"

    with pytest.raises(ValueError, match="budget policy"):
        mem.set_budget_policy("ignore")


def test_budget_policy_alias_delegates_to_set_budget_policy():
    mem = importlib.import_module("forge3d.mem")

    # Spec-named opt-in alias mirrors set_budget_policy.
    assert mem.budget_policy("warn") == "warn"
    assert mem.budget_policy("enforce") == "enforce"


def test_fallback_enforce_policy_raises_when_budget_exceeded(monkeypatch):
    memory = importlib.import_module("forge3d._memory")
    mem = importlib.import_module("forge3d.mem")

    monkeypatch.setattr(memory, "NATIVE_AVAILABLE", False)
    memory._GLOBAL_MEMORY.update(
        {"buffer_count": 0, "texture_count": 0, "buffer_bytes": 0, "texture_bytes": 0}
    )
    mem.override_memory_limit(1024)
    mem.set_budget_policy("enforce")

    with pytest.raises(RuntimeError, match="Host-visible memory budget exceeded"):
        mem.update_memory_usage(buffer_bytes_delta=2048, buffer_count_delta=1)


def test_fallback_warn_policy_reports_over_budget_without_raising(monkeypatch):
    memory = importlib.import_module("forge3d._memory")
    mem = importlib.import_module("forge3d.mem")

    monkeypatch.setattr(memory, "NATIVE_AVAILABLE", False)
    memory._GLOBAL_MEMORY.update(
        {"buffer_count": 0, "texture_count": 0, "buffer_bytes": 0, "texture_bytes": 0}
    )
    mem.override_memory_limit(1024)
    mem.set_budget_policy("warn")

    mem.update_memory_usage(buffer_bytes_delta=2048, buffer_count_delta=1)

    metrics = mem.memory_metrics()
    assert metrics["host_visible_bytes"] == 2048
    assert metrics["within_budget"] is False
    assert metrics["budget_policy"] == "warn"


def test_native_memory_metrics_expose_policy_when_available():
    native = importlib.import_module("forge3d._native").get_native_module()
    if native is None or not hasattr(native, "global_memory_metrics"):
        pytest.skip("native memory metrics unavailable")

    metrics = dict(native.global_memory_metrics())

    assert metrics["budget_policy"] in {"enforce", "warn"}
    assert "peak_total_bytes" in metrics
    assert "peak_host_visible_bytes" in metrics
