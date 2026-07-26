"""Geodetic solar geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict

from ._native import get_native_module

UtcTuple = tuple[int, int, int, int, int, int | float]


class SolarVector(TypedDict):
    zenith_deg: float
    azimuth_deg: float
    apparent_elevation_deg: float
    true_elevation_deg: float
    distance_au: float
    equation_of_time_min: float


def solar_position(
    utc: UtcTuple,
    lat: float,
    lon: float,
    elev_m: float = 0.0,
    *,
    tz_offset_hours: float = 0.0,
    delta_t_seconds: float = 69.0,
    pressure_mbar: float = 1013.25,
    temperature_c: float = 15.0,
) -> SolarVector:
    """Return the NREL-SPA solar vector.

    ``utc`` is a timezone-free civil tuple. The caller supplies its UTC offset
    and ΔT explicitly; no timezone or leap-second database is consulted.
    """
    native = get_native_module()
    if native is None:
        raise RuntimeError("forge3d native extension is required")
    return native.solar_position(
        utc,
        lat,
        lon,
        elev_m,
        tz_offset_hours=tz_offset_hours,
        delta_t_seconds=delta_t_seconds,
        pressure_mbar=pressure_mbar,
        temperature_c=temperature_c,
    )


@dataclass(frozen=True)
class SolarTime:
    utc: UtcTuple
    observer_lat: float
    observer_lon: float
    observer_elev_m: float = 0.0
    tz_offset_hours: float = 0.0
    delta_t_seconds: float = 69.0
    pressure_mbar: float = 1013.25
    temperature_c: float = 15.0

    def position(self) -> SolarVector:
        return solar_position(
            self.utc,
            self.observer_lat,
            self.observer_lon,
            self.observer_elev_m,
            tz_offset_hours=self.tz_offset_hours,
            delta_t_seconds=self.delta_t_seconds,
            pressure_mbar=self.pressure_mbar,
            temperature_c=self.temperature_c,
        )

    def to_native(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["SolarTime", "SolarVector", "solar_position"]
