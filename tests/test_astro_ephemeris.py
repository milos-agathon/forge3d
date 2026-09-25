"""SIDERA's frozen independent Horizons observer-position acceptance oracle."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re

import pytest

pytest.importorskip("forge3d._forge3d")

from forge3d import astro, sky, sun_position_utc

ORACLE = Path(__file__).resolve().parent / "data/horizons_vectors.dat"
ROOT = Path(__file__).resolve().parent.parent


def angular_error_arcsec(az_a: float, alt_a: float, az_b: float, alt_b: float) -> float:
    a, b, c, d = map(math.radians, (az_a, alt_a, az_b, alt_b))
    cosine = math.sin(b) * math.sin(d) + math.cos(b) * math.cos(d) * math.cos(a - c)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine)))) * 3600.0


def test_horizons_40_epoch_site_combinations() -> None:
    with ORACLE.open(encoding="ascii", newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    assert len(rows) == 280
    assert len({(row["utc"], row["site"]) for row in rows}) == 40
    worst = {body: (0.0, "") for body in ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn")}
    phase_worst = semi_worst = 0.0
    noaa_worst = 0.0
    for row in rows:
        when = datetime.fromisoformat(row["utc"].replace("Z", "+00:00"))
        lat, lon = float(row["latitude_deg"]), float(row["longitude_deg"])
        body = row["body"]
        az, alt, distance_km = astro.body_position(body, when, lat, lon)
        error = angular_error_arcsec(az, alt, float(row["azimuth_deg"]), float(row["altitude_deg"]))
        if error > worst[body][0]:
            worst[body] = (error, f"{row['utc']} {row['site']}")
        if body == "Moon":
            phase_worst = max(phase_worst, abs(astro.moon_phase(when, lat, lon) - float(row["illuminated_fraction"])))
            semi_arcsec = math.degrees(math.asin(1737.4 / distance_km)) * 3600.0
            semi_worst = max(semi_worst, abs(semi_arcsec - float(row["semidiameter_arcsec"])))
        if body == "Sun":
            noaa = sun_position_utc(lat, lon, when.year, when.month, when.day, when.hour, when.minute, when.second)
            noaa_worst = max(noaa_worst, angular_error_arcsec(noaa.azimuth, noaa.elevation, float(row["azimuth_deg"]), float(row["altitude_deg"])))
    print("SIDERA worst arcseconds:", worst)
    print("NOAA baseline worst arcseconds:", noaa_worst)
    print("Moon fraction/semidiameter worst:", phase_worst, semi_worst)
    assert worst["Sun"][0] < 10.0
    assert worst["Moon"][0] < 30.0
    for body in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn"):
        assert worst[body][0] < 60.0
    assert phase_worst < 0.005
    assert semi_worst < 1.0


def test_utc_window_and_timezone_contract() -> None:
    for year in (1999, 2051):
        with pytest.raises(ValueError, match="2000"):
            astro.body_position("Sun", datetime(year, 1, 1, tzinfo=timezone.utc), 0.0, 0.0)
    with pytest.raises(ValueError, match="timezone-aware"):
        astro.moon_phase(datetime(2026, 9, 25))
    with pytest.raises(ValueError, match="unsupported"):
        astro.body_position("Pluto", datetime(2026, 9, 25, tzinfo=timezone.utc), 0.0, 0.0)


def test_one_call_observation_payload() -> None:
    class Viewer:
        def __init__(self) -> None:
            self.command = None

        def _send_command(self, command):
            self.command = command

    viewer = Viewer()
    sky.set_observation(datetime(2026, 9, 25, 22, tzinfo=timezone.utc), 52.37, 4.9, viewer=viewer)
    assert viewer.command == {
        "cmd": "set_sky_observation", "utc": "2026-09-25T22:00:00.000000Z",
        "latitude_deg": 52.37, "longitude_deg": 4.9,
    }


def test_sourced_assets_match_manifest_and_budget() -> None:
    source = ROOT / "data/sidera"
    manifest = (source / "MANIFEST.md").read_text(encoding="utf-8")
    pinned = re.findall(r"^\| `([^`]+\.bin)` \| (\d+) \| `([0-9a-f]{64})` \|", manifest, re.M)
    assert len(pinned) == 5
    for filename, length, digest in pinned:
        contents = (source / filename).read_bytes()
        assert len(contents) == int(length)
        assert hashlib.sha256(contents).hexdigest() == digest
    assert (source / "ybsc5.bin").stat().st_size <= 1024 * 1024
    fixtures = [*(source.glob("*.bin")), source / "MANIFEST.md", source / "ERFA-LICENSE",
                ORACLE, ROOT / "tests/golden/sidera/night.png",
                ROOT / "tests/golden/sidera/night_certificate.json"]
    assert sum(path.stat().st_size for path in fixtures) <= 2 * 1024 * 1024
