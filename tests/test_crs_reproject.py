# tests/test_crs_reproject.py
# Unit tests for P3-reproject: CRS reprojection utilities
"""
Tests for the forge3d.crs module which provides coordinate reprojection.

Covers:
- proj_available() detection
- transform_coords() for coordinate arrays
- crs_to_epsg() parsing
- Edge cases (empty arrays, same CRS, invalid CRS)
"""

import numpy as np
import pytest

from forge3d.crs import (
    proj_available,
    transform_coords,
    crs_to_epsg,
)


class TestProjAvailable:
    """Tests for proj_available() function."""

    def test_returns_bool(self):
        """proj_available returns a boolean."""
        result = proj_available()
        assert isinstance(result, bool)


class TestCrsToEpsg:
    """Tests for crs_to_epsg() function."""

    def test_epsg_4326(self):
        """Parse EPSG:4326 (WGS84)."""
        assert crs_to_epsg("EPSG:4326") == 4326

    def test_epsg_32654(self):
        """Parse EPSG:32654 (UTM zone 54N)."""
        assert crs_to_epsg("EPSG:32654") == 32654

    def test_lowercase_epsg(self):
        """Parse lowercase epsg prefix."""
        assert crs_to_epsg("epsg:4326") == 4326

    def test_invalid_format(self):
        """Non-EPSG string returns None."""
        assert crs_to_epsg("WGS84") is None

    def test_invalid_number(self):
        """EPSG with non-numeric code returns None."""
        assert crs_to_epsg("EPSG:invalid") is None


class TestTransformCoords:
    """Tests for transform_coords() function."""

    def test_empty_array(self):
        """Empty coordinate array returns empty array."""
        coords = np.array([]).reshape(0, 2)
        result = transform_coords(coords, "EPSG:4326", "EPSG:32654")
        assert result.shape == (0, 2)

    def test_same_crs_returns_copy(self):
        """Same source and target CRS returns a copy."""
        coords = np.array([[138.73, 35.36], [138.74, 35.37]])
        result = transform_coords(coords, "EPSG:4326", "EPSG:4326")
        np.testing.assert_array_equal(result, coords)
        # Must be a copy, not the same object
        assert result is not coords

    def test_invalid_shape_raises(self):
        """Invalid coordinate shape raises ValueError."""
        coords = np.array([1.0, 2.0, 3.0])  # 1D array
        with pytest.raises(ValueError, match="shape"):
            transform_coords(coords, "EPSG:4326", "EPSG:32654")

    @pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
    def test_wgs84_to_utm(self):
        """Transform WGS84 to UTM zone 54N."""
        # Mt. Fuji approximate location
        coords = np.array([[138.7274, 35.3606]])
        result = transform_coords(coords, "EPSG:4326", "EPSG:32654")

        # UTM zone 54N coordinates should be roughly in these ranges
        assert result.shape == (1, 2)
        assert 300_000 < result[0, 0] < 500_000, f"X out of range: {result[0, 0]}"
        assert 3_900_000 < result[0, 1] < 4_000_000, f"Y out of range: {result[0, 1]}"

    @pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
    def test_roundtrip(self):
        """WGS84 -> UTM -> WGS84 roundtrip preserves coordinates."""
        original = np.array([[138.7274, 35.3606], [138.73, 35.37]])
        utm = transform_coords(original, "EPSG:4326", "EPSG:32654")
        back = transform_coords(utm, "EPSG:32654", "EPSG:4326")

        np.testing.assert_allclose(back, original, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
    def test_multiple_points(self):
        """Transform multiple points at once."""
        coords = np.array([
            [138.70, 35.35],
            [138.72, 35.36],
            [138.74, 35.37],
            [138.76, 35.38],
        ])
        result = transform_coords(coords, "EPSG:4326", "EPSG:32654")
        assert result.shape == (4, 2)
        # All X values should be similar (same longitude range)
        assert np.std(result[:, 0]) < 10_000  # Within 10km
        # Y values should increase with latitude
        assert all(result[i, 1] < result[i + 1, 1] for i in range(3))

    def test_no_backend_raises(self):
        """If no backend available and different CRS, should raise or return original."""
        # This test documents behavior when neither pyproj nor native proj is available
        # The actual behavior depends on the environment
        coords = np.array([[138.73, 35.36]])
        try:
            result = transform_coords(coords, "EPSG:4326", "EPSG:32654")
            # If it succeeds, a backend was available
            assert result.shape == coords.shape
        except RuntimeError as e:
            # Expected if no backend is available
            assert "backend" in str(e).lower()


class TestEdgeCases:
    """Edge case tests for CRS module."""

    def test_float32_input(self):
        """Float32 input is handled correctly."""
        coords = np.array([[138.73, 35.36]], dtype=np.float32)
        result = transform_coords(coords, "EPSG:4326", "EPSG:4326")
        assert result.dtype == np.float64  # Output is always float64

    def test_list_input(self):
        """List input is converted to array."""
        coords = [[138.73, 35.36], [138.74, 35.37]]
        result = transform_coords(coords, "EPSG:4326", "EPSG:4326")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    @pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
    def test_large_batch(self):
        """Handle large coordinate arrays efficiently."""
        # Generate 10,000 random points in WGS84
        np.random.seed(42)
        coords = np.column_stack([
            np.random.uniform(138.0, 139.0, 10_000),
            np.random.uniform(35.0, 36.0, 10_000),
        ])
        result = transform_coords(coords, "EPSG:4326", "EPSG:32654")
        assert result.shape == (10_000, 2)
        # All points should be in valid UTM range
        assert np.all(result[:, 0] > 200_000)
        assert np.all(result[:, 1] > 3_800_000)
