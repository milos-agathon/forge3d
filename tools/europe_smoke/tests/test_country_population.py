# tools/europe_smoke/tests/test_country_population.py
import numpy as np
import pytest

from tools.europe_smoke import config, population


def test_largest_remainder_preserves_the_total_exactly():
    v = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
    out = population.largest_remainder(v, total=2)
    assert out.sum() == 2
    assert out.dtype.kind in "iu"


def test_largest_remainder_handles_sub_unit_values_without_losing_them():
    # the trap: naive int() would zero all of these and lose the whole mass
    v = np.array([0.6, 0.7, 0.8])
    out = population.largest_remainder(v, total=2)
    assert out.sum() == 2
    assert (out >= 0).all()


def test_largest_remainder_is_deterministic():
    v = np.array([1.5, 1.5, 1.5, 1.5])
    a = population.largest_remainder(v, total=6)
    b = population.largest_remainder(v.copy(), total=6)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_country_rasterisation_leaves_no_pixel_unassigned(ghsl_artifact):
    lon, lat = config.grid_axes()
    codes, names = population.rasterise_countries(config.CACHE_DIR, lon, lat)
    assert codes.min() >= 0, "nearest-country fill must leave nothing unassigned"
    assert len(names) > 40


@pytest.mark.slow
def test_country_cell_table_closes_on_the_domain_total(ghsl_artifact):
    rows = population.country_cell_table(config.CACHE_DIR)
    total = sum(r[3] for r in rows)
    lon, lat = config.grid_axes()
    grid_total = int(round(population.aggregate(lon, lat).sum()))
    assert total == grid_total, f"table {total} != grid {grid_total}"


@pytest.mark.slow
def test_country_cell_table_rows_are_positive_integers(ghsl_artifact):
    rows = population.country_cell_table(config.CACHE_DIR)
    assert all(isinstance(r[3], int) and r[3] > 0 for r in rows)
    assert len(rows) > 5000


@pytest.mark.slow
def test_return_stats_is_additive_and_reports_the_fill(ghsl_artifact):
    """The default return shape is unchanged; stats is opt-in."""
    rows, stats = population.country_cell_table(config.CACHE_DIR, return_stats=True)
    assert isinstance(rows, list) and isinstance(rows[0], tuple)
    assert stats["unassigned_pixels_after_fill"] == 0
    assert stats["unassigned_pixels_before_fill"] > 0
    assert 0.0 < stats["unassigned_people_before_fill"] / stats["attributed_people_float"] < 0.05
    assert stats["table_rows"] == len(rows)
    assert len(stats["countries"]) == stats["n_countries"]


# ------------------------------------------------------ ADDED BY THE VERIFIER


@pytest.mark.slow
def test_the_stats_dict_never_carries_the_internal_mask(ghsl_artifact):
    """It is a 75 MB bool array; a report writer must not meet it."""
    import json
    rows, stats = population.country_cell_table(config.CACHE_DIR, return_stats=True)
    assert not [k for k in stats if k.startswith("_")]
    json.dumps(stats)


@pytest.mark.slow
def test_both_ways_of_measuring_the_fill_agree(ghsl_artifact):
    """weights=... (published) and the low-memory mask path must agree."""
    lon, lat = config.grid_axes()
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(config.GHSL_TIF) as src:
        col0, row0, ncols, nrows = population.block_anchor(lon, lat, src.transform)
        pop = src.read(1, window=Window(col0, row0, ncols, nrows)).astype("float64")
    _, _, s = population.rasterise_countries(config.CACHE_DIR, lon, lat,
                                             weights=pop, return_stats=True)
    _, stats = population.country_cell_table(config.CACHE_DIR, return_stats=True)
    assert s["unassigned_people_before_fill"] == pytest.approx(
        stats["unassigned_people_before_fill"], rel=1e-12)
    assert s["unassigned_pixels_before_fill"] == stats["unassigned_pixels_before_fill"]
