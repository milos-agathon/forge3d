"""SELENE planetary-datum CPU gates."""

from pathlib import Path

import pytest

from forge3d import crs


def _areoid_reference_points():
    path = Path(__file__).parent / "data" / "mars_areoid_reference.txt"
    return [
        (float(lat), float(lon), float(value), source)
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
        for lat, lon, value, source in [line.split()]
    ]


def test_body_info_reports_reference_surfaces_and_units():
    earth = crs.body_info("earth")
    moon = crs.body_info("MOON")
    mars = crs.body_info("Mars")

    assert earth["name"] == "Earth"
    assert earth["semi_major_m"] == 6_378_137.0
    assert earth["gravity_surface"] == "EGM96"
    assert moon["name"] == "Moon"
    assert moon["semi_major_m"] == 1_737_400.0
    assert moon["flattening"] == 0.0
    assert moon["gravity_surface"] is None
    assert mars["name"] == "Mars"
    assert mars["semi_major_m"] == 3_396_190.0
    assert 1.0 / mars["flattening"] == pytest.approx(169.894_447_223_612)
    assert mars["gravity_surface"] == "Mars areoid"
    assert set(mars) == {
        "name",
        "semi_major_m",
        "semi_minor_m",
        "flattening",
        "prime_meridian_w0_deg",
        "rotation_rate_deg_per_day",
        "gravity_surface",
    }


def test_body_info_rejects_unknown_body_without_earth_fallback():
    with pytest.raises(ValueError, match="unsupported body 'ceres'.*earth, moon, or mars"):
        crs.body_info("ceres")


def test_mars_areoid_matches_committed_pds_gmm3_map_below_half_metre():
    points = _areoid_reference_points()
    assert len(points) >= 20
    errors = [
        abs(crs.areoid_undulation(lat, lon) - expected)
        for lat, lon, expected, _source in points
    ]
    assert max(errors) < 0.5


@pytest.mark.parametrize(
    "lat,lon",
    [(90.1, 0.0), (-90.1, 0.0), (float("nan"), 0.0), (0.0, float("inf"))],
)
def test_mars_areoid_rejects_invalid_coordinates(lat, lon):
    with pytest.raises(ValueError):
        crs.areoid_undulation(lat, lon)
