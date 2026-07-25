"""Verified double-float precision tools for absolute GPU coordinates."""

from __future__ import annotations

from typing import Any

from ._native import get_native_module, native_import_error


def _native() -> Any:
    module = get_native_module()
    if module is None:
        cause = native_import_error()
        raise RuntimeError(f"forge3d native precision API is unavailable: {cause!r}")
    return module


def dd_selftest() -> dict[str, Any]:
    """Run the backend exactness canary and return its evidence report."""
    return _native().dd_selftest()


def dd_harness(operation: str, n: int = 100_000_000) -> dict[str, Any]:
    """Prove a DD operation over ``n`` generated plus 1M adversarial vectors."""
    return _native().dd_harness(operation, n)


def dd_jitter_demo(frames: int = 1_000) -> dict[str, Any]:
    """Measure DD and raw-f32 absolute-coordinate jitter at Everest ECEF."""
    return _native().dd_jitter_demo(frames)


__all__ = ["dd_selftest", "dd_harness", "dd_jitter_demo"]
