# Python shim: expose the native PyO3 module as forge3d._forge3d
from ._forge3d import *  # noqa: F401,F403

# Expose package version for runtime checks
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    __version__ = _pkg_version("forge3d")
except PackageNotFoundError:  # fallback during editable builds
    __version__ = "0.5.0"
del _pkg_version, PackageNotFoundError

__all__ = [n for n in dir() if not n.startswith("_")]