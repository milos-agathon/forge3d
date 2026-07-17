"""Shader proof reports."""

from __future__ import annotations

from typing import Any

from ._native import get_native_module, native_import_error


def shader_report(mode: str | None = None) -> dict[str, Any]:
    """Return PROBATUM WGSL value-safety proof verdicts."""
    native = get_native_module()
    if native is None or not hasattr(native, "shader_report"):
        cause = native_import_error()
        detail = f": {cause!r}" if cause is not None else ""
        raise RuntimeError(f"forge3d._forge3d shader_report is unavailable{detail}")
    return native.shader_report(mode)
