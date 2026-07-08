# python/forge3d/_native.py
# Provide shared access to the compiled _forge3d extension module.
# Ensures other modules can query native availability without import cycles.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/_gpu.py,python/forge3d/_memory.py

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional


# Root cause of the most recent failed native import, preserved so downstream
# "native module is not available" messages can surface the real reason
# (ABI mismatch, missing DLL, wrong interpreter, ...) instead of losing it.
NATIVE_IMPORT_ERROR: Optional[BaseException] = None


def _load_native() -> Optional[ModuleType]:
    global NATIVE_IMPORT_ERROR
    try:
        module = importlib.import_module("forge3d._forge3d")
        sys.modules.setdefault("python.forge3d._forge3d", module)
        NATIVE_IMPORT_ERROR = None
        return module
    except Exception as exc:
        NATIVE_IMPORT_ERROR = exc
        return None


NATIVE_MODULE: Optional[ModuleType] = _load_native()
NATIVE_AVAILABLE: bool = NATIVE_MODULE is not None


def get_native_module() -> Optional[ModuleType]:
    """Expose the cached PyO3 extension module (if available)."""
    return NATIVE_MODULE


def native_import_error() -> Optional[BaseException]:
    """Return the exception raised by the most recent ``forge3d._forge3d`` import attempt.

    Returns ``None`` when the native extension imported successfully. When the
    extension is unavailable this preserves the root cause (e.g. a missing DLL
    or ABI mismatch) that a bare "native module is not available" message loses.
    """
    return NATIVE_IMPORT_ERROR


def refresh_native_module() -> Optional[ModuleType]:
    """Reload the PyO3 extension and update global availability flags."""
    global NATIVE_MODULE, NATIVE_AVAILABLE
    NATIVE_MODULE = _load_native()
    NATIVE_AVAILABLE = NATIVE_MODULE is not None
    return NATIVE_MODULE
