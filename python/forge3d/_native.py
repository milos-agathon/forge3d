# python/forge3d/_native.py
# Provide shared access to the compiled _forge3d extension module.
# Ensures other modules can query native availability without import cycles.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/_gpu.py,python/forge3d/_memory.py

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional


def _load_native() -> Optional[ModuleType]:
    try:
        module = importlib.import_module("forge3d._forge3d")
        sys.modules.setdefault("python.forge3d._forge3d", module)
        return module
    except Exception:
        return None


NATIVE_MODULE: Optional[ModuleType] = _load_native()
NATIVE_AVAILABLE: bool = NATIVE_MODULE is not None


def get_native_module() -> Optional[ModuleType]:
    """Expose the cached PyO3 extension module (if available)."""
    return NATIVE_MODULE


def refresh_native_module() -> Optional[ModuleType]:
    """Reload the PyO3 extension and update global availability flags."""
    global NATIVE_MODULE, NATIVE_AVAILABLE
    NATIVE_MODULE = _load_native()
    NATIVE_AVAILABLE = NATIVE_MODULE is not None
    return NATIVE_MODULE
