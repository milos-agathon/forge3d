"""Offline, reference-checked positions for the 2000–2050 UTC window.

Angles are degrees; distance is kilometres. The native implementation uses
IAU 2006/2000B frames, VSOP87D planets, ELP2000-82B Moon, and WGS84 sites.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ._native import get_native_module


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime_utc must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _native_astro():
    native = get_native_module()
    if native is None:
        raise RuntimeError("SIDERA requires the forge3d native extension")
    return native


def body_position(body: str, datetime_utc: datetime, lat: float, lon: float) -> tuple[float, float, float]:
    """Return airless topocentric (azimuth°, altitude°, distance km)."""
    return _native_astro().astro_body_position(body, _utc_text(datetime_utc), float(lat), float(lon))


def body_position_refracted(body: str, datetime_utc: datetime, lat: float, lon: float) -> tuple[float, float, float]:
    """Return topocentric position with standard-atmosphere apparent altitude."""
    return _native_astro().astro_body_position_refracted(body, _utc_text(datetime_utc), float(lat), float(lon))


def moon_phase(datetime_utc: datetime, lat: float | None = None, lon: float | None = None) -> float:
    """Return illuminated fraction; pass both site angles for topocentric phase."""
    utc = _utc_text(datetime_utc)
    if (lat is None) != (lon is None):
        raise ValueError("lat and lon must be provided together")
    if lat is None:
        return _native_astro().astro_moon_phase(utc)
    return _native_astro().astro_moon_phase_at(utc, float(lat), float(lon))


__all__ = ["body_position", "body_position_refracted", "moon_phase"]
