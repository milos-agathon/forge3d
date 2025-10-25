# tests/test_device_probe.py
# Validate Python GPU probe helpers for native and fallback modes
# Exists to guarantee enumerate_adapters and device_probe behave without hardware
# RELEVANT FILES: python/forge3d/_gpu.py, python/forge3d/__init__.py, tests/conftest.py

import forge3d
import forge3d._gpu as gpu


def test_device_probe_with_native(monkeypatch):
    """When native callbacks are present the probe should report success."""

    def fake_native_functions():
        def enum():
            return [{"status": "ok", "backend": "Mock"}]

        def probe(backend=None):
            data = {"status": "ok", "backend": backend or "Mock"}
            return data

        def get_device():
            return object()

        return enum, probe, get_device

    monkeypatch.setattr(gpu, "_native_functions", fake_native_functions)

    adapters = forge3d.enumerate_adapters()
    assert adapters and adapters[0]["status"] == "ok"

    probe = forge3d.device_probe()
    assert probe["status"] == "ok"


def test_device_probe_without_native(monkeypatch):
    """Fallback behaviour should return conservative defaults."""

    def fake_native_functions():
        return None, None, None

    monkeypatch.setattr(gpu, "_native_functions", fake_native_functions)

    adapters = forge3d.enumerate_adapters()
    assert adapters == []

    probe = forge3d.device_probe()
    assert probe["status"] == "unavailable"

    assert forge3d.has_gpu() is False