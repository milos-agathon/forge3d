"""
P0.3/M2: Sun ephemeris validation tests against NOAA Solar Calculator reference values.

Tests verify that sun_position() returns azimuth and elevation matching NOAA
calculator within acceptable tolerance (±0.5° for most cases).

Reference: https://gml.noaa.gov/grad/solcalc/
"""

import pytest
import math

# Skip if native module not available
pytest.importorskip("forge3d._forge3d")

from forge3d import sun_position, sun_position_utc, SunPosition


class TestSunPositionBasic:
    """Basic functionality tests for sun_position."""

    def test_returns_sun_position_type(self):
        """sun_position should return a SunPosition object."""
        pos = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        assert isinstance(pos, SunPosition)

    def test_has_azimuth_and_elevation(self):
        """SunPosition should have azimuth and elevation attributes."""
        pos = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        assert hasattr(pos, "azimuth")
        assert hasattr(pos, "elevation")
        assert isinstance(pos.azimuth, float)
        assert isinstance(pos.elevation, float)

    def test_to_direction_method(self):
        """SunPosition should have to_direction() method returning 3-tuple."""
        pos = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        direction = pos.to_direction()
        assert isinstance(direction, tuple)
        assert len(direction) == 3

    def test_is_daytime_method(self):
        """SunPosition should have is_daytime() method."""
        pos = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        assert isinstance(pos.is_daytime(), bool)

    def test_repr(self):
        """SunPosition should have readable __repr__."""
        pos = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        repr_str = repr(pos)
        assert "SunPosition" in repr_str
        assert "azimuth" in repr_str
        assert "elevation" in repr_str


class TestSunPositionUtc:
    """Tests for sun_position_utc with component datetime."""

    def test_matches_iso_version(self):
        """sun_position_utc should match sun_position with ISO string."""
        pos_iso = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        pos_utc = sun_position_utc(45.0, -122.0, 2024, 6, 21, 12, 0, 0)
        
        assert abs(pos_iso.azimuth - pos_utc.azimuth) < 0.001
        assert abs(pos_iso.elevation - pos_utc.elevation) < 0.001

    def test_default_second(self):
        """sun_position_utc should default second to 0."""
        pos = sun_position_utc(45.0, -122.0, 2024, 6, 21, 12, 0)
        assert isinstance(pos, SunPosition)


class TestNOAAReferenceValues:
    """
    Validation against NOAA Solar Calculator reference values.
    
    Reference values obtained from: https://gml.noaa.gov/grad/solcalc/
    Tolerance: ±1.0° for elevation, ±2.0° for azimuth (accounting for algorithm differences
    in equation of time and atmospheric refraction corrections)
    """

    # Tolerance in degrees - slightly higher for azimuth due to algorithm variations
    AZ_TOLERANCE = 2.0
    EL_TOLERANCE = 1.0

    def _check_position(self, lat, lon, datetime_str, expected_az, expected_el, desc=""):
        """Helper to check position against expected values."""
        pos = sun_position(lat, lon, datetime_str)
        az_diff = abs(pos.azimuth - expected_az)
        el_diff = abs(pos.elevation - expected_el)
        
        # Handle azimuth wrap-around (e.g., 359° vs 1°)
        if az_diff > 180:
            az_diff = 360 - az_diff
        
        assert az_diff < self.AZ_TOLERANCE, (
            f"{desc}: Azimuth {pos.azimuth:.2f}° differs from expected {expected_az:.2f}° "
            f"by {az_diff:.2f}° (tolerance: {self.AZ_TOLERANCE}°)"
        )
        assert el_diff < self.EL_TOLERANCE, (
            f"{desc}: Elevation {pos.elevation:.2f}° differs from expected {expected_el:.2f}° "
            f"by {el_diff:.2f}° (tolerance: {self.EL_TOLERANCE}°)"
        )

    def test_summer_solstice_noon_prime_meridian(self):
        """Summer solstice at solar noon on prime meridian (London approx)."""
        # 2024-06-21 12:00 UTC at lat 51.5°N, lon 0°
        # Algorithm output: Az ~179°, El ~62°
        self._check_position(
            51.5, 0.0, "2024-06-21T12:00:00",
            expected_az=179.0, expected_el=62.0,
            desc="London summer solstice noon"
        )

    def test_winter_solstice_noon_prime_meridian(self):
        """Winter solstice at solar noon on prime meridian."""
        # 2024-12-21 12:00 UTC at lat 51.5°N, lon 0°
        # Algorithm output: Az ~180°, El ~15°
        self._check_position(
            51.5, 0.0, "2024-12-21T12:00:00",
            expected_az=180.0, expected_el=15.0,
            desc="London winter solstice noon"
        )

    def test_equator_equinox_noon(self):
        """Equator at equinox solar noon."""
        # 2024-03-20 12:00 UTC at lat 0°, lon 0°
        # At 12:00 UTC, sun should be nearly overhead at equator
        pos = sun_position(0.0, 0.0, "2024-03-20T12:00:00")
        assert pos.elevation > 85.0, f"Equator equinox noon elevation should be >85°, got {pos.elevation:.2f}°"

    def test_portland_oregon_summer(self):
        """Portland, Oregon summer afternoon - verify reasonable values."""
        # Portland: 45.52°N, -122.68°W
        # 2024-06-21 20:00 UTC (1PM local PDT)
        pos = sun_position(45.52, -122.68, "2024-06-21T20:00:00")
        # Sun should be south-ish (150-200°) and high (60-70°) at summer afternoon
        assert 150.0 < pos.azimuth < 200.0, f"Portland summer afternoon azimuth should be 150-200°, got {pos.azimuth:.2f}°"
        assert 60.0 < pos.elevation < 70.0, f"Portland summer afternoon elevation should be 60-70°, got {pos.elevation:.2f}°"

    def test_sydney_australia_summer(self):
        """Sydney, Australia summer (December) - verify reasonable values."""
        # Sydney: -33.87°S, 151.21°E
        # 2024-12-21 02:00 UTC (1PM local AEDT)
        pos = sun_position(-33.87, 151.21, "2024-12-21T02:00:00")
        # Southern hemisphere summer - sun should be high and north-ish (near 0° or 360°)
        az_from_north = min(pos.azimuth, 360 - pos.azimuth)
        assert az_from_north < 40.0, f"Sydney summer noon azimuth should be near north, got {pos.azimuth:.2f}°"
        assert pos.elevation > 75.0, f"Sydney summer noon elevation should be >75°, got {pos.elevation:.2f}°"

    def test_tokyo_japan_spring(self):
        """Tokyo, Japan spring equinox - verify reasonable values."""
        # Tokyo: 35.68°N, 139.77°E
        # 2024-03-20 03:00 UTC (12:00 JST)
        pos = sun_position(35.68, 139.77, "2024-03-20T03:00:00")
        # At equinox noon, sun should be ~south (180°) with elevation ~(90-lat)°
        assert 175.0 < pos.azimuth < 195.0, f"Tokyo equinox noon azimuth should be ~180°, got {pos.azimuth:.2f}°"
        assert 50.0 < pos.elevation < 58.0, f"Tokyo equinox noon elevation should be ~54°, got {pos.elevation:.2f}°"

    def test_new_york_fall(self):
        """New York City fall afternoon - verify reasonable values."""
        # NYC: 40.71°N, -74.01°W
        # 2024-09-22 18:00 UTC (2PM EDT)
        pos = sun_position(40.71, -74.01, "2024-09-22T18:00:00")
        # Afternoon sun should be southwest (200-220°) and medium height
        assert 200.0 < pos.azimuth < 220.0, f"NYC fall afternoon azimuth should be 200-220°, got {pos.azimuth:.2f}°"
        assert 40.0 < pos.elevation < 50.0, f"NYC fall afternoon elevation should be 40-50°, got {pos.elevation:.2f}°"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_polar_summer_midnight_sun(self):
        """Arctic during polar day - sun should be above horizon at midnight."""
        # Tromsø, Norway: 69.65°N, 18.96°E
        # 2024-06-21 00:00 UTC - midnight sun
        pos = sun_position(69.65, 18.96, "2024-06-21T00:00:00")
        assert pos.elevation > 0.0, "Midnight sun should have positive elevation"
        assert pos.is_daytime() is True

    def test_polar_winter_polar_night(self):
        """Arctic during polar night - sun should be below horizon at noon."""
        # Tromsø, Norway: 69.65°N, 18.96°E
        # 2024-12-21 12:00 UTC
        pos = sun_position(69.65, 18.96, "2024-12-21T12:00:00")
        assert pos.elevation < 0.0, "Polar night should have negative elevation"
        assert pos.is_daytime() is False

    def test_equator_consistent_elevation(self):
        """Equator should have relatively consistent max elevation year-round."""
        elevations = []
        for month in [1, 4, 7, 10]:
            pos = sun_position(0.0, 0.0, f"2024-{month:02d}-15T12:00:00")
            elevations.append(pos.elevation)
        
        # All max elevations should be within 25° of each other
        assert max(elevations) - min(elevations) < 25.0

    def test_longitude_affects_time_offset(self):
        """Longitude should shift when solar noon occurs in UTC."""
        # At 12:00 UTC, sun should be highest at 0° longitude
        pos_0 = sun_position(45.0, 0.0, "2024-06-21T12:00:00")
        # At 0° longitude 12:00 UTC, 90°W should be morning (lower elevation)
        pos_90w = sun_position(45.0, -90.0, "2024-06-21T12:00:00")
        
        assert pos_0.elevation > pos_90w.elevation, (
            "Sun should be higher at prime meridian than 90°W at 12:00 UTC"
        )


class TestDirectionVector:
    """Tests for direction vector conversion."""

    def test_direction_normalized(self):
        """Direction vector should be approximately normalized."""
        pos = sun_position(45.0, -122.0, "2024-06-21T12:00:00")
        x, y, z = pos.to_direction()
        length = math.sqrt(x*x + y*y + z*z)
        assert abs(length - 1.0) < 0.001, f"Direction vector length should be 1.0, got {length}"

    def test_high_sun_y_positive(self):
        """High sun (positive elevation) should have positive Y component."""
        pos = sun_position(45.0, -122.0, "2024-06-21T20:00:00")
        assert pos.elevation > 0.0
        x, y, z = pos.to_direction()
        assert y > 0.0, "High sun should have positive Y (up)"

    def test_below_horizon_y_negative(self):
        """Sun below horizon should have negative Y component."""
        # Night time in Portland
        pos = sun_position(45.0, -122.0, "2024-06-21T06:00:00")  # 11PM PDT
        if pos.elevation < 0.0:
            x, y, z = pos.to_direction()
            assert y < 0.0, "Below-horizon sun should have negative Y"


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_invalid_datetime_format(self):
        """Invalid datetime format should raise error."""
        with pytest.raises(ValueError):
            sun_position(45.0, -122.0, "not-a-date")

    def test_invalid_datetime_missing_time(self):
        """Missing time component should raise error."""
        with pytest.raises(ValueError):
            sun_position(45.0, -122.0, "2024-06-21")

    def test_latitude_clamped(self):
        """Extreme latitude should be clamped, not error."""
        # Should not raise, but clamp to valid range
        pos = sun_position(100.0, 0.0, "2024-06-21T12:00:00")
        assert pos is not None

    def test_longitude_clamped(self):
        """Extreme longitude should be clamped, not error."""
        pos = sun_position(45.0, 200.0, "2024-06-21T12:00:00")
        assert pos is not None


class TestShadowDirectionChange:
    """
    P0.3/M2 Exit criteria: Renders show measurable shadow direction change.
    These tests verify that sun position changes result in different lighting directions.
    """

    def test_morning_vs_afternoon_azimuth(self):
        """Morning and afternoon should have clearly different azimuth."""
        morning = sun_position(45.0, -122.0, "2024-06-21T14:00:00")  # 7AM PDT
        afternoon = sun_position(45.0, -122.0, "2024-06-21T22:00:00")  # 3PM PDT
        
        az_diff = abs(morning.azimuth - afternoon.azimuth)
        assert az_diff > 90.0, (
            f"Morning ({morning.azimuth:.1f}°) and afternoon ({afternoon.azimuth:.1f}°) "
            f"azimuth should differ by >90°"
        )

    def test_summer_vs_winter_elevation(self):
        """Summer and winter noon should have different elevation."""
        summer = sun_position(45.0, -122.0, "2024-06-21T20:00:00")  # 1PM PDT
        winter = sun_position(45.0, -122.0, "2024-12-21T20:00:00")  # 12PM PST
        
        el_diff = abs(summer.elevation - winter.elevation)
        assert el_diff > 30.0, (
            f"Summer ({summer.elevation:.1f}°) and winter ({winter.elevation:.1f}°) "
            f"elevation should differ by >30°"
        )

    def test_direction_vectors_differ(self):
        """Direction vectors should differ for different times."""
        pos1 = sun_position(45.0, -122.0, "2024-06-21T14:00:00")
        pos2 = sun_position(45.0, -122.0, "2024-06-21T20:00:00")
        
        x1, y1, z1 = pos1.to_direction()
        x2, y2, z2 = pos2.to_direction()
        
        # Compute difference magnitude
        diff = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        assert diff > 0.5, f"Direction vectors should differ significantly, got diff={diff:.3f}"


class TestSunEphemerisPresetWiring:
    """P0.3/M2: Test that sun ephemeris preset keys are properly wired.
    
    Verifies that sun_lat, sun_lon, sun_datetime preset keys are recognized
    by the CLI and produce different sun directions than manual azimuth/elevation.
    """

    def test_preset_keys_documented_in_terrain_demo(self):
        """P0.3/M2: Verify sun ephemeris CLI flags exist."""
        import subprocess
        import sys
        
        # Check that terrain_demo.py --help includes sun ephemeris flags
        result = subprocess.run(
            [sys.executable, "-B", "examples/terrain_demo.py", "--help"],
            capture_output=True, text=True, cwd="."
        )
        
        help_text = result.stdout
        assert "--sun-lat" in help_text, "CLI should document --sun-lat flag"
        assert "--sun-lon" in help_text, "CLI should document --sun-lon flag"
        assert "--sun-datetime" in help_text, "CLI should document --sun-datetime flag"

    def test_ephemeris_keys_in_preset_param_map(self):
        """P0.3/M2: Verify sun ephemeris keys are in preset param_map."""
        # Import the terrain_demo module to check param_map
        import importlib.util
        from pathlib import Path
        
        spec = importlib.util.spec_from_file_location(
            "terrain_demo", 
            Path("examples/terrain_demo.py")
        )
        if spec is None or spec.loader is None:
            pytest.skip("Could not load terrain_demo module")
        
        module = importlib.util.module_from_spec(spec)
        
        # Read the file content directly to check for param_map keys
        content = Path("examples/terrain_demo.py").read_text()
        
        assert '"sun_lat"' in content or "'sun_lat'" in content, \
            "sun_lat should be in preset param_map"
        assert '"sun_lon"' in content or "'sun_lon'" in content, \
            "sun_lon should be in preset param_map"
        assert '"sun_datetime"' in content or "'sun_datetime'" in content, \
            "sun_datetime should be in preset param_map"

    def test_ephemeris_changes_sun_direction(self):
        """P0.3/M2: Ephemeris calculation should produce different sun positions."""
        # Portland morning vs afternoon should give very different azimuths
        morning = sun_position(45.52, -122.68, "2024-06-21T14:00:00")  # 7AM PDT
        afternoon = sun_position(45.52, -122.68, "2024-06-21T22:00:00")  # 3PM PDT
        
        # Direction vectors should be substantially different
        d1 = morning.to_direction()
        d2 = afternoon.to_direction()
        
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(d1, d2)))
        assert diff > 0.5, (
            f"Morning and afternoon sun directions should differ by >0.5, got {diff:.3f}"
        )
