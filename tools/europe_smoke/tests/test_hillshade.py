"""Gate 9, part 1 of 2: the pure numeric replacement.

Part 2 (does it run in production?) lives in test_basemap.py -- a pure-function
test alone is exactly how the previous revision shipped a patch that was never
installed.
"""
import math

import numpy as np
import pytest

from tools.europe_smoke import config, hillshade

# a mercator box spanning lat 30..72 (the display window)
LAT_S, LAT_N = 30.0, 72.0
EXAG, AZ, ALT = 5.5, 315.0, 34.0


def _merc_y(lat_deg):
    return math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2)) * hillshade.EARTH_R


def _bounds(nrows=1024, ncols=1024):
    return ((math.radians(-25.0) * hillshade.EARTH_R, _merc_y(LAT_S),
             math.radians(45.0) * hillshade.EARTH_R, _merc_y(LAT_N)), (nrows, ncols))


def _engine_dem_hillshade(h, exaggeration, azimuth_deg, altitude_deg):
    """The engine's own index-unit form, transcribed from the bytecode.

    [verified bit-identical to the real ``wildfire_smoke_engine._dem_hillshade``]
    """
    gy, gx = np.gradient(np.asarray(h).astype(np.float32))
    dzdx, dzdy = gx * exaggeration, gy * exaggeration
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    alt = math.radians(altitude_deg)
    az = math.radians(360.0 - azimuth_deg + 90.0)
    return np.clip(np.sin(alt) * np.cos(slope)
                   + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0.0, 1.0
                   ).astype(np.float32)


# ------------------------------------------------------------------ geometry
def test_row_ground_metres_shrinks_toward_the_pole():
    bounds, shape = _bounds()
    m = hillshade.row_ground_metres(bounds, shape)
    assert m.size == shape[0]
    # row 0 is the NORTH edge of a north-up raster
    assert m[0] < m[-1], "ground metres per pixel must shrink toward the pole"
    assert m[-1] / m[0] == pytest.approx(np.cos(np.radians(LAT_S))
                                         / np.cos(np.radians(LAT_N)), rel=0.02)


def test_row_latitudes_span_the_box_and_descend():
    bounds, shape = _bounds(nrows=512)
    lat = hillshade.row_latitudes(bounds, shape)
    assert (np.diff(lat) < 0).all()
    assert lat[0] < LAT_N and lat[0] > LAT_N - 0.2
    assert lat[-1] > LAT_S and lat[-1] < LAT_S + 0.2


def test_column_spacing_is_tracked_separately_from_row_spacing():
    # _build_terrarium_dem rounds out_h and out_w independently, so mercator
    # pixels are not exactly square and one axis cannot stand in for the other.
    bounds, shape = _bounds(nrows=1000, ncols=997)
    row = hillshade.row_ground_metres(bounds, shape)
    col = hillshade.col_ground_metres(bounds, shape)
    assert row.shape == col.shape == (1000,)
    assert not np.allclose(row, col)
    # both carry the identical cos(lat) factor: mercator is conformal
    assert np.allclose(col / row, (col[0] / row[0]) * np.ones_like(row))


# ---------------------------------------------------------------- the fix
def test_identical_terrain_reads_equally_rugged_at_both_ends():
    """Gate 9. Constant-GROUND-wavelength ridges must render with the same
    ruggedness at 30N and at 72N once the correction is applied."""
    bounds, shape = _bounds()
    metric = hillshade.worst_ruggedness_uniformity(bounds, shape, EXAG, AZ, ALT)
    assert metric["ground_metres_ratio"] == pytest.approx(2.80, rel=0.02)
    assert metric["px_per_wavelength_south"] > 45, "test field must not be aliased"
    assert metric["rel_diff"] < config.HILLSHADE_UNIFORMITY_MAX, metric


def test_the_engines_own_form_fails_the_same_check():
    """Proves the metric has teeth: index-unit gradients diverge. The engine's
    error is strongly amplitude-dependent -- it collapses once the slope term
    saturates -- which is exactly why the gate sweeps amplitude."""
    bounds, shape = _bounds()
    metric = hillshade.worst_ruggedness_uniformity(bounds, shape, EXAG, AZ, ALT,
                                                   shade=_engine_dem_hillshade)
    assert metric["rel_diff"] > 20 * config.HILLSHADE_UNIFORMITY_MAX, metric
    assert min(metric["sweep_rel_diff"].values()) < config.HILLSHADE_UNIFORMITY_MAX, (
        "a single-amplitude gate would have passed the buggy form: "
        f"{metric['sweep_rel_diff']}")


def test_gate_9_holds_across_the_whole_z_factor_sweep():
    """The section 2 exaggeration re-sweep cannot re-break the correction: equal
    gradients stay equal through the arctan, saturated or not."""
    bounds, shape = _bounds()
    for z in (1.0, 2.0, 5.5, 12.0, 20.0, 40.0, 100.0):
        metric = hillshade.worst_ruggedness_uniformity(bounds, shape, z, AZ, ALT)
        assert metric["rel_diff"] < config.HILLSHADE_UNIFORMITY_MAX, (z, metric)


def test_the_raw_gradient_bias_is_removed_exactly():
    """The defect, measured on the quantity it lives in rather than downstream
    of the arctan, which saturates and hides it."""
    bounds, shape = _bounds()
    row = hillshade.row_ground_metres(bounds, shape)
    col = hillshade.col_ground_metres(bounds, shape)
    lam = max(row[0], row[-1]) * 48
    raw, fixed = [], []
    for i in (0, -1):
        field = hillshade.ridge_patch(row[i], col[i], side_m=lam * 4,
                                      wavelength_m=lam, amplitude_m=200.0)
        gy, gx = np.gradient(field.astype(np.float64))
        raw.append(float(np.hypot(gx, gy).mean()))
        fixed.append(float(np.hypot(gx / col[i], gy / row[i]).mean()))
    assert raw[1] / raw[0] > 2.2, raw       # the engine's 2.5x bias
    assert fixed[1] / fixed[0] == pytest.approx(1.0, abs=0.02), fixed


# ----------------------------------------------------------- contract checks
def test_output_is_in_zero_one_and_finite():
    bounds, shape = _bounds(nrows=200, ncols=200)
    dem = np.random.default_rng(0).normal(0, 100, shape)
    shade = hillshade.make_coslat_hillshade(bounds, shape)(dem, EXAG, AZ, ALT)
    assert np.isfinite(shade).all()
    assert shade.min() >= 0.0 and shade.max() <= 1.0
    assert shade.dtype == np.float32


def test_the_replacement_counts_its_own_invocations():
    """The sentinel the production gate reads."""
    bounds, shape = _bounds(nrows=64, ncols=64)
    shade = hillshade.make_coslat_hillshade(bounds, shape)
    assert shade.is_coslat and shade.calls == 0
    shade(np.zeros(shape, np.float32), EXAG, AZ, ALT)
    shade(np.zeros(shape, np.float32), EXAG * 0.7, AZ, ALT + 8.0)
    assert shade.calls == 2
    assert shade.shape == shape


def test_a_shape_mismatch_raises_instead_of_silently_rescaling():
    """A wrong-shape array means the captured bounds no longer describe it;
    resampling the spacing to fit would apply a wrong correction quietly."""
    bounds, shape = _bounds(nrows=64, ncols=64)
    shade = hillshade.make_coslat_hillshade(bounds, shape)
    with pytest.raises(ValueError, match="built for"):
        shade(np.zeros((48, 64), np.float32), EXAG, AZ, ALT)


def test_only_the_gradient_denominator_differs_from_the_engine():
    """Everything except the spacing division must be the engine's own code:
    same azimuth expression, same aspect sign, same clip, same float32 chain.
    Feed the engine form a DEM pre-divided by the spacing and the two outputs
    must agree bit for bit."""
    dem = np.random.default_rng(7).normal(0, 50, (96, 96)).astype(np.float32)
    # a CONSTANT power-of-two spacing: then dividing the DEM commutes with
    # np.gradient exactly, and the division itself is exact in binary, so any
    # surviving difference would be a real formula difference
    sp = np.full(96, 1024.0)
    ours = hillshade.CosLatHillshade._from_spacings(sp, sp)(dem, EXAG, AZ, ALT)
    ref = _engine_dem_hillshade((dem / np.float32(1024.0)).astype(np.float32),
                                EXAG, AZ, ALT)
    assert np.array_equal(ours, ref)


def test_the_two_azimuth_spellings_agree_to_float32():
    """The engine writes radians(360 - az + 90); the obvious radians(90 - az) is
    the same angle mod 2*pi, but 2*pi is not exact in binary."""
    bounds, shape = _bounds(nrows=96, ncols=96)
    dem = np.random.default_rng(7).normal(0, 50, shape).astype(np.float32)
    ours = hillshade.make_coslat_hillshade(bounds, shape)(dem, EXAG, AZ, ALT)
    row = hillshade.row_ground_metres(bounds, shape).astype(np.float32)
    col = hillshade.col_ground_metres(bounds, shape).astype(np.float32)
    gy, gx = np.gradient(dem)
    dzdx = np.float32(EXAG) * gx / col[:, None]
    dzdy = np.float32(EXAG) * gy / row[:, None]
    slope, aspect = np.arctan(np.hypot(dzdx, dzdy)), np.arctan2(dzdy, -dzdx)
    alt, az = math.radians(ALT), math.radians(90.0 - AZ)
    alt_form = np.clip(np.sin(alt) * np.cos(slope)
                       + np.cos(alt) * np.sin(slope) * np.cos(az - aspect),
                       0.0, 1.0).astype(np.float32)
    assert not np.array_equal(ours, alt_form)
    assert np.allclose(ours, alt_form, atol=2e-7)


def test_ground_tiers_measure_h1_and_hit_the_display_share():
    dem = np.arange(100, dtype="float32").reshape(10, 10) * 10.0
    valid = np.ones_like(dem, dtype=bool)
    bounds = (-1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0)
    multiplier, h1, share = hillshade.ground_tiers(
        dem, valid, bounds, (-180.0, -85.0, 180.0, 85.0)
    )
    assert h1 > 300.0
    assert 0.18 <= share <= 0.24
    assert np.allclose(multiplier[dem < 300.0], 0.15)
    assert np.allclose(multiplier[dem >= h1], 1.0)


def test_ground_tiers_share_is_of_display_land_not_of_the_sea_too():
    """Europe's display window is ~half sea. A share target counted over ALL
    display pixels drags h1 down to the ~48th land percentile (206.6 m on the
    delivered z7 DEM [measured]), below the 300 m lowland ceiling by
    construction. The highland share is a share of display LAND.
    """
    dem = np.arange(100, dtype="float32").reshape(10, 10) * 10.0
    valid = np.ones_like(dem, dtype=bool)
    valid[:, 5:] = False                      # the right half is sea
    bounds = (-1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0)
    multiplier, h1, share = hillshade.ground_tiers(
        dem, valid, bounds, (-180.0, -85.0, 180.0, 85.0)
    )
    land_values = dem[valid]
    assert h1 == pytest.approx(
        float(np.quantile(land_values, 1.0 - round(0.21 * land_values.size)
                          / land_values.size)), rel=1e-6)
    assert 0.18 <= share <= 0.24
    assert np.allclose(multiplier[~valid], 0.0)


def test_ground_tiers_reject_an_h1_below_the_lowland_ceiling():
    dem = np.linspace(0.0, 250.0, 100, dtype="float32").reshape(10, 10)
    with pytest.raises(ValueError, match="h1"):
        hillshade.ground_tiers(
            dem, np.ones_like(dem, dtype=bool),
            (-1e6, -1e6, 1e6, 1e6), (-180.0, -85.0, 180.0, 85.0)
        )


def test_factory_requires_dem_and_valid_together():
    with pytest.raises(ValueError, match="together"):
        hillshade.make_coslat_hillshade(
            (-1e6, -1e6, 1e6, 1e6), (10, 10),
            dem=np.zeros((10, 10), dtype="float32"),
        )


def test_latitude_uniformity_separates_corrected_from_uncorrected():
    bounds, shape = _bounds(nrows=4241, ncols=4000)
    out = hillshade.latitude_uniformity(bounds, shape)
    assert out["verdict"] == "PASS"
    assert out["probe"]["resolved"] is True
    assert out["corrected"]["rel_diff"] < 0.05
    assert out["uncorrected"]["rel_diff"] > 0.20
    assert out["control_has_teeth"] is True


def test_latitude_uniformity_declines_on_an_unresolvable_raster():
    bounds, shape = _bounds(nrows=180, ncols=320)
    out = hillshade.latitude_uniformity(bounds, shape)
    assert out["verdict"] == "N/A"
    assert out["probe"]["resolved"] is False
