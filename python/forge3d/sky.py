"""Observation-driven viewer sky: IAU positions and declared night light models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
import weakref

from .astro import _utc_text

if TYPE_CHECKING:
    from .viewer import ViewerHandle

_active_viewer: weakref.ReferenceType[ViewerHandle] | None = None


def _register_viewer(viewer: ViewerHandle) -> None:
    global _active_viewer
    _active_viewer = weakref.ref(viewer)


def set_observation(datetime_utc: datetime, lat: float, lon: float, *, viewer: ViewerHandle | None = None) -> None:
    """Apply one UTC/site observation to Sun, Moon, planets and bright stars.

    The most recently opened asynchronous viewer is used unless ``viewer`` is
    passed explicitly. A viewer is required because this changes a GPU scene.
    """
    utc = _utc_text(datetime_utc)
    target = viewer if viewer is not None else (_active_viewer() if _active_viewer is not None else None)
    if target is None:
        raise RuntimeError("open a viewer or pass viewer= before setting a sky observation")
    target._send_command({"cmd": "set_sky_observation", "utc": utc, "latitude_deg": float(lat), "longitude_deg": float(lon)})


__all__ = ["set_observation"]
