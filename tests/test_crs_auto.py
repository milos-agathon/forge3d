# tests/test_crs_auto.py
# Integration tests for P3-reproject: Auto-reprojection in render_polygons
"""
Tests for automatic CRS reprojection when rendering vector data.

Covers:
- render_polygons with target_crs parameter
- Auto-detection and reprojection of GeoJSON/GeoPackage data
- TerrainRenderParams.terrain_crs field
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from forge3d import render_polygons
from forge3d.terrain_params import TerrainRenderParams, make_terrain_params_config
from forge3d.crs import proj_available, crs_to_epsg


class TestTerrainCrsField:
    """Tests for TerrainRenderParams.terrain_crs field."""

    def test_terrain_crs_default_none(self):
        """terrain_crs defaults to None."""
        params = make_terrain_params_config(
            size_px=(800, 600),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0, 1000),
        )
        assert params.terrain_crs is None

    def test_terrain_crs_can_be_set(self):
        """terrain_crs can be set to a CRS string."""
        params = make_terrain_params_config(
            size_px=(800, 600),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0, 1000),
            terrain_crs="EPSG:32654",
        )
        assert params.terrain_crs == "EPSG:32654"

    def test_terrain_crs_epsg_4326(self):
        """terrain_crs accepts WGS84."""
        params = make_terrain_params_config(
            size_px=(800, 600),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0, 1000),
            terrain_crs="EPSG:4326",
        )
        assert params.terrain_crs == "EPSG:4326"


class TestRenderPolygonsTargetCrs:
    """Tests for render_polygons with target_crs parameter."""

    def test_ndarray_input_no_reprojection(self):
        """Array input renders correctly (no CRS to reproject)."""
        # Simple square polygon
        polygon = np.array([
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200],
            [100, 100],
        ], dtype=np.float64)

        img = render_polygons(
            polygon,
            size=(400, 300),
            fill_rgba=(0.5, 0.5, 0.5, 1.0),
            target_crs="EPSG:4326",  # Should be ignored for array input
        )

        assert img.shape == (300, 400, 4)
        assert img.dtype == np.uint8

    def test_dict_input_no_reprojection(self):
        """Dict input with exterior renders correctly."""
        polygon = {
            "exterior": np.array([
                [100, 100],
                [200, 100],
                [200, 200],
                [100, 200],
                [100, 100],
            ], dtype=np.float64),
        }

        img = render_polygons(
            polygon,
            size=(400, 300),
            fill_rgba=(0.5, 0.5, 0.5, 1.0),
            target_crs="EPSG:4326",  # Should be ignored for dict input
        )

        assert img.shape == (300, 400, 4)
        assert img.dtype == np.uint8


@pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
class TestRenderPolygonsAutoReproject:
    """Tests for auto-reprojection in render_polygons."""

    @pytest.fixture
    def geojson_wgs84(self, tmp_path):
        """Create a GeoJSON file in WGS84 (EPSG:4326)."""
        geojson_content = """{
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "test"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [138.70, 35.35],
                            [138.75, 35.35],
                            [138.75, 35.40],
                            [138.70, 35.40],
                            [138.70, 35.35]
                        ]]
                    }
                }
            ]
        }"""
        path = tmp_path / "test_wgs84.geojson"
        path.write_text(geojson_content)
        return path

    def test_geojson_no_target_crs(self, geojson_wgs84):
        """GeoJSON renders without target_crs (no reprojection)."""
        img = render_polygons(
            str(geojson_wgs84),
            size=(400, 300),
            fill_rgba=(0.2, 0.4, 0.8, 1.0),
        )
        assert img.shape == (300, 400, 4)

    def test_geojson_same_target_crs(self, geojson_wgs84):
        """GeoJSON with same target_crs (no reprojection needed)."""
        img = render_polygons(
            str(geojson_wgs84),
            size=(400, 300),
            fill_rgba=(0.2, 0.4, 0.8, 1.0),
            target_crs="EPSG:4326",
        )
        assert img.shape == (300, 400, 4)

    def test_geojson_different_target_crs(self, geojson_wgs84):
        """GeoJSON reprojected to different target_crs."""
        try:
            import geopandas  # Required for file-based rendering
        except ImportError:
            pytest.skip("geopandas not available")

        # Render in WGS84
        img_wgs84 = render_polygons(
            str(geojson_wgs84),
            size=(400, 300),
            fill_rgba=(0.2, 0.4, 0.8, 1.0),
            target_crs="EPSG:4326",
        )

        # Render reprojected to UTM
        img_utm = render_polygons(
            str(geojson_wgs84),
            size=(400, 300),
            fill_rgba=(0.2, 0.4, 0.8, 1.0),
            target_crs="EPSG:32654",
        )

        # Both should produce valid images
        assert img_wgs84.shape == (300, 400, 4)
        assert img_utm.shape == (300, 400, 4)

        # The polygon extent changes with reprojection, so the rendered
        # images may differ (the auto-fit transform adapts to the extent)
        # This is expected behavior


class TestCrsIntegration:
    """Integration tests for CRS workflow."""

    def test_crs_module_importable(self):
        """CRS module can be imported from forge3d."""
        from forge3d import proj_available, transform_coords, crs_to_epsg
        assert callable(proj_available)
        assert callable(transform_coords)
        assert callable(crs_to_epsg)

    def test_proj_available_accessible(self):
        """proj_available is accessible from top-level."""
        import forge3d
        result = forge3d.proj_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
    def test_transform_coords_accessible(self):
        """transform_coords is accessible from top-level."""
        import forge3d
        coords = np.array([[138.73, 35.36]])
        result = forge3d.transform_coords(coords, "EPSG:4326", "EPSG:4326")
        np.testing.assert_array_equal(result, coords)
