from __future__ import annotations

import csv
from pathlib import Path

import pytest

from forge3d.geo import SolarTime, solar_position


REFERENCE = Path(__file__).parent / "data" / "spa_reference.csv"
ANGLE_TOLERANCE_DEG = 0.0003


def _angle_error(actual: float, expected: float) -> float:
    delta = abs(actual - expected) % 360.0
    return min(delta, 360.0 - delta)


def _reference_rows() -> list[dict[str, float]]:
    with REFERENCE.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def test_spa_worked_example_matches_nrel() -> None:
    result = solar_position(
        (2003, 10, 17, 12, 30, 30),
        39.742476,
        -105.1786,
        1830.14,
        tz_offset_hours=-7,
        delta_t_seconds=67,
        pressure_mbar=820,
        temperature_c=11,
    )

    assert abs(result["zenith_deg"] - 50.11162) <= ANGLE_TOLERANCE_DEG
    assert _angle_error(result["azimuth_deg"], 194.34024) <= ANGLE_TOLERANCE_DEG


def test_spa_matches_committed_reference_rows() -> None:
    rows = _reference_rows()
    assert len(rows) >= 20
    assert min(row["lat"] for row in rows) <= -80
    assert max(row["lat"] for row in rows) >= 80
    assert min(row["year"] for row in rows) <= 1900
    assert max(row["year"] for row in rows) >= 2100

    for row in rows:
        result = solar_position(
            tuple(int(row[key]) for key in ("year", "month", "day", "hour", "minute", "second")),
            row["lat"],
            row["lon"],
            row["elev_m"],
            tz_offset_hours=row["tz_offset_hours"],
            delta_t_seconds=row["delta_t_seconds"],
            pressure_mbar=row["pressure_mbar"],
            temperature_c=row["temperature_c"],
        )
        label = f"{int(row['year'])}-{int(row['month']):02}-{int(row['day']):02}@{row['lat']},{row['lon']}"
        assert abs(result["zenith_deg"] - row["zenith_deg"]) <= ANGLE_TOLERANCE_DEG, label
        assert _angle_error(result["azimuth_deg"], row["azimuth_deg"]) <= ANGLE_TOLERANCE_DEG, label
        assert abs(result["true_elevation_deg"] - row["true_elevation_deg"]) <= ANGLE_TOLERANCE_DEG, label
        assert abs(result["distance_au"] - row["distance_au"]) <= 1e-10, label
        assert abs(result["equation_of_time_min"] - row["equation_of_time_min"]) <= 1e-7, label


def test_solar_time_is_an_explicit_timezone_free_contract() -> None:
    solar_time = SolarTime(
        utc=(2025, 6, 21, 12, 0, 0),
        observer_lat=48.2082,
        observer_lon=16.3738,
        observer_elev_m=171,
        tz_offset_hours=2,
        delta_t_seconds=74.5,
        pressure_mbar=1000,
        temperature_c=25,
    )

    assert solar_time.position()["azimuth_deg"] == pytest.approx(150.7305003541, abs=ANGLE_TOLERANCE_DEG)
    assert solar_time.to_native()["utc"] == (2025, 6, 21, 12, 0, 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", 90.0001),
        ("longitude", 180.0001),
        ("pressure_mbar", 0.0),
        ("temperature_c", -273.15),
        ("tz_offset_hours", 19.0),
    ],
)
def test_spa_rejects_invalid_physical_inputs(field: str, value: float) -> None:
    kwargs = {
        "tz_offset_hours": 0.0,
        "delta_t_seconds": 74.0,
        "pressure_mbar": 1013.25,
        "temperature_c": 15.0,
    }
    latitude = 45.0
    longitude = 5.0
    if field == "latitude":
        latitude = value
    elif field == "longitude":
        longitude = value
    else:
        kwargs[field] = value

    with pytest.raises(ValueError):
        solar_position((2025, 1, 1, 12, 0, 0), latitude, longitude, 0.0, **kwargs)
