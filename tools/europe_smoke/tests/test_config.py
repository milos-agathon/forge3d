# tools/europe_smoke/tests/test_config.py
import numpy as np
import pytest

from tools.europe_smoke import config


def test_area_edges_are_on_the_lattice():
    # §0.1: all four edges must be exact multiples of 0.4 so the ADS inward
    # snap (N/E floor, S/W ceil) is a no-op.
    for edge in config.AREA:
        assert config.on_lattice(edge), edge


def test_grid_shape_is_241_by_136():
    lon, lat = config.grid_axes()
    assert (lon.size, lat.size) == (241, 136)
    assert config.GRID_SHAPE == (136, 241)  # (lat, lon), matching NetCDF order


def test_grid_axes_orientation_and_step():
    lon, lat = config.grid_axes()
    assert lon[0] == pytest.approx(-38.0) and lon[-1] == pytest.approx(58.0)
    # latitude DESCENDS, as CAMS delivers it (§1.1)
    assert lat[0] == pytest.approx(78.0) and lat[-1] == pytest.approx(24.0)
    assert np.allclose(np.diff(lon), 0.4)
    assert np.allclose(np.diff(lat), -0.4)


def test_display_window_sits_strictly_inside_the_data_with_the_budgeted_bleed():
    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    n, w, s, e = config.AREA
    assert lon_min - w == pytest.approx(13.0)   # §0.1 margin budget
    assert e - lon_max == pytest.approx(13.0)
    assert lat_min - s == pytest.approx(6.0)
    assert n - lat_max == pytest.approx(6.0)


def test_display_block_is_175_by_106_cell_centres():
    lon, lat = config.grid_axes()
    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    sel_lon = lon[(lon >= lon_min) & (lon <= lon_max)]
    sel_lat = lat[(lat >= lat_min) & (lat <= lat_max)]
    assert (sel_lat.size, sel_lon.size) == config.DISPLAY_BLOCK_SHAPE == (106, 175)


def test_lattice_tolerance_admits_float_representation_error():
    # §0.1: the request literal itself is off by ~1e-14
    assert config.on_lattice(-25.2)
    assert not config.on_lattice(-25.3)


def test_max_advection_gain_is_the_validated_cap():
    # §5.5: gate 11(b) was measured at exactly 1.5x wind gain
    assert config.MAX_ADVECTION_GAIN == 1.5


def test_basemap_window_is_16_9_in_web_mercator():
    west, south, east, north = config.BASEMAP_WINDOW
    merc_w = config.mercator_x(east) - config.mercator_x(west)
    merc_h = config.mercator_y(north) - config.mercator_y(south)
    assert merc_w / merc_h == pytest.approx(16 / 9, rel=1e-12)
    assert config.BASEMAP_SIZE == (4000, 2250)
