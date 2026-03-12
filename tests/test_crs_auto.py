"""Tests for CRS helpers and terrain CRS state after legacy vector render removal."""

import numpy as np
import pytest

from forge3d.crs import crs_to_epsg, proj_available
from forge3d.terrain_params import make_terrain_params_config


class TestTerrainCrsField:
    """Tests for TerrainRenderParams.terrain_crs field."""

    def test_terrain_crs_default_none(self):
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


class TestCrsIntegration:
    """Integration tests for CRS utility exposure."""

    def test_crs_module_importable(self):
        from forge3d import crs_to_epsg as top_level_crs_to_epsg
        from forge3d import proj_available as top_level_proj_available
        from forge3d import transform_coords

        assert callable(top_level_proj_available)
        assert callable(transform_coords)
        assert callable(top_level_crs_to_epsg)

    def test_proj_available_accessible(self):
        import forge3d

        result = forge3d.proj_available()
        assert isinstance(result, bool)

    def test_crs_to_epsg_parses_wgs84(self):
        assert crs_to_epsg("EPSG:4326") == 4326

    @pytest.mark.skipif(not proj_available(), reason="proj/pyproj not available")
    def test_transform_coords_identity(self):
        import forge3d

        coords = np.array([[138.73, 35.36]])
        result = forge3d.transform_coords(coords, "EPSG:4326", "EPSG:4326")
        np.testing.assert_array_equal(result, coords)


def test_legacy_render_polygons_removed():
    """The old CRS-via-render_polygons path no longer exists."""
    import forge3d

    assert not hasattr(forge3d, "render_polygons")
