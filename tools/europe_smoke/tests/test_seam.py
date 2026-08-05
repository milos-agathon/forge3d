# tools/europe_smoke/tests/test_seam.py
import numpy as np
import pytest
import xarray as xr

from tools.europe_smoke import seam


def _ds(values, hours, arms):
    t = np.array([np.datetime64("2026-08-01T00") + np.timedelta64(h, "h") for h in hours],
                 dtype="datetime64[ns]")
    lat = np.array([60.0, 59.6, 59.2])
    lon = np.array([0.0, 0.4])
    data = np.stack([np.full((lat.size, lon.size), float(v), dtype="float32") for v in values])
    return xr.Dataset(
        {"omaod550": (("time", "latitude", "longitude"), data)},
        coords={"time": t, "latitude": lat, "longitude": lon,
                "arm": ("time", np.array(arms, dtype="<U2"))},
    )


def test_burden_is_cosine_weighted():
    ds = _ds([1.0], [0], ["an"])
    b = seam.burden(ds)
    expected = np.cos(np.radians(ds["latitude"].values)).sum() * ds["longitude"].size
    assert b[0] == pytest.approx(expected, rel=1e-6)


def test_rates_are_per_hour_and_normalised_by_the_mean_burden():
    times = np.array([np.datetime64("2026-08-01T00"), np.datetime64("2026-08-01T03")])
    b = np.array([100.0, 130.0])
    r = seam.rates(times, b)
    assert r[0] == pytest.approx(abs(130 - 100) / (0.5 * (100 + 130) * 3.0))


def test_a_smooth_join_passes():
    vals = [1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14]
    ds = _ds(vals, [0, 3, 6, 9, 12, 15, 18, 21], ["an"] * 4 + ["fc"] * 4)
    out = seam.evaluate(ds, np.datetime64("2026-08-01T09"))
    assert out["verdict"] == "PASS", out


def test_a_gross_misjoin_fails():
    vals = [1.0, 1.02, 1.04, 1.06, 5.0, 5.02, 5.04, 5.06]
    ds = _ds(vals, [0, 3, 6, 9, 12, 15, 18, 21], ["an"] * 4 + ["fc"] * 4)
    out = seam.evaluate(ds, np.datetime64("2026-08-01T09"))
    assert out["verdict"] == "FAIL", out


def test_a_frozen_seam_fails_even_though_the_rate_is_zero():
    vals = [1.0, 1.02, 1.04, 1.06, 1.06, 1.08, 1.10, 1.12]
    ds = _ds(vals, [0, 3, 6, 9, 12, 15, 18, 21], ["an"] * 4 + ["fc"] * 4)
    out = seam.evaluate(ds, np.datetime64("2026-08-01T09"))
    assert out["verdict"] == "FAIL"
    assert "identical" in out["reason"]


def test_reference_n_is_reported_because_p99_is_an_order_statistic():
    vals = [1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14]
    ds = _ds(vals, [0, 3, 6, 9, 12, 15, 18, 21], ["an"] * 4 + ["fc"] * 4)
    out = seam.evaluate(ds, np.datetime64("2026-08-01T09"))
    assert out["n_reference"] >= 1
    assert "power" in out["caveat"].lower()
