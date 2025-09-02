# Python shim: expose the native PyO3 module as forge3d._forge3d
from ._forge3d import *  # noqa: F401,F403
try:
    from ._forge3d import __version__  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
__all__ = [n for n in dir() if not n.startswith("_")]