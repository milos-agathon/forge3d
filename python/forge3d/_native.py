# python/forge3d/_native.py
# Provide shared access to the compiled _forge3d extension module.
# Ensures other modules can query native availability without import cycles.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/_gpu.py,python/forge3d/_memory.py

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Optional


# Root cause of the most recent failed native import, preserved so downstream
# "native module is not available" messages can surface the real reason
# (ABI mismatch, missing DLL, wrong interpreter, ...) instead of losing it.
NATIVE_IMPORT_ERROR: Optional[BaseException] = None


def _load_native() -> Optional[ModuleType]:
    global NATIVE_IMPORT_ERROR
    try:
        module = importlib.import_module("forge3d._forge3d")
        sys.modules["python.forge3d._forge3d"] = module
        NATIVE_IMPORT_ERROR = None
        return module
    except Exception as exc:
        sys.modules.pop("python.forge3d._forge3d", None)
        NATIVE_IMPORT_ERROR = exc
        return None


NATIVE_MODULE: Optional[ModuleType] = _load_native()
NATIVE_AVAILABLE: bool = NATIVE_MODULE is not None


def get_native_module() -> Optional[ModuleType]:
    """Return the current extension snapshot without importing or probing a GPU.

    Runtime callers must resolve this at the point of use, rather than retain
    the result or a copied ``NATIVE_AVAILABLE`` flag across an explicit refresh.
    Import-time class exports intentionally retain their original identities.
    """
    return NATIVE_MODULE


def get_native_symbol(name: str, default: Any = None) -> Any:
    """Resolve an optional symbol against the current extension snapshot."""
    return getattr(get_native_module(), name, default)


def require_native_module(
    message: str = "forge3d native module is not available; build the extension first",
) -> ModuleType:
    """Resolve the current extension or raise with the preserved import cause."""
    module = get_native_module()
    if module is None:
        raise RuntimeError(message) from native_import_error()
    return module


def native_import_error() -> Optional[BaseException]:
    """Return the exception raised by the most recent ``forge3d._forge3d`` import attempt.

    Returns ``None`` when the native extension imported successfully. When the
    extension is unavailable this preserves the root cause (e.g. a missing DLL
    or ABI mismatch) that a bare "native module is not available" message loses.
    """
    return NATIVE_IMPORT_ERROR


def refresh_native_module() -> Optional[ModuleType]:
    """Retry the extension import and replace the shared capability snapshot.

    This does not reload an already imported binary or rebind exported classes
    and existing native objects. Ordinary lookups never retry an import. Python's
    import caches are invalidated so a newly installed extension can be found.
    ``NATIVE_MODULE``, ``NATIVE_AVAILABLE`` and the import error remain available
    for compatibility; values imported by value are intentional snapshots.
    """
    global NATIVE_MODULE, NATIVE_AVAILABLE
    importlib.invalidate_caches()
    NATIVE_MODULE = _load_native()
    NATIVE_AVAILABLE = NATIVE_MODULE is not None
    return NATIVE_MODULE
