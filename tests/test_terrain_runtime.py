from __future__ import annotations

import _terrain_runtime as terrain_runtime


def test_running_on_unsupported_hosted_macos_ci_detects_github_actions(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Darwin")

    assert terrain_runtime._running_on_unsupported_hosted_macos_ci() is True


def test_running_on_unsupported_hosted_macos_ci_ignores_local_macos(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Darwin")

    assert terrain_runtime._running_on_unsupported_hosted_macos_ci() is False


def test_terrain_rendering_available_short_circuits_on_hosted_macos_ci(monkeypatch) -> None:
    terrain_runtime.terrain_rendering_available.cache_clear()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Darwin")

    def fail_if_called():
        raise AssertionError("terrain GPU probe should not run on hosted macOS CI")

    monkeypatch.setattr(terrain_runtime.f3d, "has_gpu", fail_if_called)

    try:
        assert terrain_runtime.terrain_rendering_available() is False
    finally:
        terrain_runtime.terrain_rendering_available.cache_clear()
