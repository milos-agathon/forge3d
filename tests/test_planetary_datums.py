"""SELENE planetary-datum CPU gates."""

import pytest

from forge3d import crs


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
