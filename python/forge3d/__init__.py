# Lightweight Python shim that re-exports the native PyO3 module
# This file is intentionally tiny to keep cold-start fast.
from ._forge3d import *  # noqa: F401,F403

try:
    from ._forge3d import __version__  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [n for n in dir() if not n.startswith("_")]