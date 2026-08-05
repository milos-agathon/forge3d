# tools/europe_smoke/tests/test_population.py
import numpy as np
import pytest

from tools.europe_smoke import config, population

# Every test here reads config.GHSL_TIF (384 MB, off-tree on D:), the [ghsl]
# manifest entry. `ghsl_artifact` (conftest) skips with the manifest's own
# Finding when the raster is not on this machine; a raster that is present but
# does not match its pin is not absence and still runs, and fails.


def test_block_anchor_uses_the_edge_lattice_not_cell_centres(ghsl_artifact):
    lon, lat = config.grid_axes()
    import rasterio
    with rasterio.open(config.GHSL_TIF) as src:
        col0, row0, ncols, nrows = population.block_anchor(lon, lat, src.transform)
    # §1.3: west = lon[0] - 0.2, north = lat[0] + 0.2
    assert (col0, row0) == (17017, 1308)
    assert (ncols, nrows) == (241 * 48, 136 * 48) == (11568, 6528)


@pytest.mark.slow
def test_aggregate_shape_and_domain_total(ghsl_artifact):
    lon, lat = config.grid_axes()
    grid = population.aggregate(lon, lat)
    assert grid.shape == config.GRID_SHAPE == (136, 241)
    total = grid.sum()
    assert total / 1e6 == pytest.approx(1257.39, rel=2e-4), total


@pytest.mark.slow
def test_aggregate_conserves_mass_against_the_raw_window(ghsl_artifact):
    lon, lat = config.grid_axes()
    grid, raw_sum = population.aggregate(lon, lat, return_raw_sum=True)
    assert grid.sum() == pytest.approx(raw_sum, rel=1e-12)


@pytest.mark.slow
def test_display_block_is_the_number_the_page_prints(ghsl_artifact):
    lon, lat = config.grid_axes()
    block = population.display_block(population.aggregate(lon, lat))
    assert block.shape == config.DISPLAY_BLOCK_SHAPE == (106, 175)
    assert block.sum() / 1e6 == pytest.approx(1037.66, rel=2e-4)


@pytest.mark.slow
def test_largest_cell_forces_uint32_storage(ghsl_artifact):
    lon, lat = config.grid_axes()
    grid = population.aggregate(lon, lat)
    assert grid.max() == pytest.approx(22_693_914, rel=1e-4)
    assert grid.max() > np.iinfo(np.uint16).max, "uint16 would truncate Cairo 346x"


@pytest.mark.slow
def test_margin_ring_population_never_reaches_the_display_block(ghsl_artifact):
    lon, lat = config.grid_axes()
    grid = population.aggregate(lon, lat)
    ring = grid.sum() - population.display_block(grid).sum()
    assert ring / 1e6 == pytest.approx(219.74, rel=1e-3)
