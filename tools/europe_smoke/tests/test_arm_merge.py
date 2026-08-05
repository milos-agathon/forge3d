# tools/europe_smoke/tests/test_arm_merge.py
import numpy as np
import pytest
import xarray as xr

from tools.europe_smoke import cams

# The real arms are 6-hourly (analysis) and 3-hourly (forecast) off a common
# t_init, so the overlap is more than lead 0. Lead 0 must be IDENTICAL -- it is
# the same field via a second request -- while the later overlapping steps have
# drifted, and those are where "analysis wins" is a meaningful claim.
T_INIT = "2026-08-03T12"
T_NOW = "2026-08-04T00"
AN_TIMES = ["2026-08-03T12", "2026-08-03T18", "2026-08-04T00"]
AN_VALUES = [0.10, 0.20, 0.30]
FC_TIMES = ["2026-08-03T12", "2026-08-03T15", "2026-08-03T18", "2026-08-03T21",
            "2026-08-04T00", "2026-08-04T03", "2026-08-04T06"]
#          lead0 == analysis   |------ drifted over the overlap ------|  future
FC_VALUES = [0.10, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

NLAT, NLON = 10, 12


def _arm(times, values, nlat=NLAT, nlon=NLON):
    """A one-variable arm. ``values`` is a scalar or one value per time step."""
    t = np.array(times, dtype="datetime64[ns]")
    lat = 78.0 - 0.4 * np.arange(nlat)
    lon = -38.0 + 0.4 * np.arange(nlon)
    vals = np.broadcast_to(np.asarray(values, dtype="float32").reshape(-1), (t.size,))
    data = np.repeat(vals, nlat * nlon).reshape(t.size, nlat, nlon).astype("float32")
    return xr.Dataset(
        {"omaod550": (("time", "latitude", "longitude"), data)},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )


def _plume_field(base, peak, n_hot, nlat=NLAT, nlon=NLON):
    """A sparse field: mostly clean air with a few bright plume cells."""
    f = np.full(nlat * nlon, float(base), dtype="float32")
    f[:n_hot] = float(peak)
    return f.reshape(nlat, nlon)


def _arm_from_fields(times, fields):
    t = np.array(times, dtype="datetime64[ns]")
    data = np.stack(fields).astype("float32")
    return xr.Dataset(
        {"omaod550": (("time", "latitude", "longitude"), data)},
        coords={"time": t,
                "latitude": 78.0 - 0.4 * np.arange(data.shape[1]),
                "longitude": -38.0 + 0.4 * np.arange(data.shape[2])},
    )


# ----------------------------------------------------------------- priority


def test_analysis_wins_every_overlapping_step_not_just_lead_zero():
    an = _arm(AN_TIMES, AN_VALUES)
    fc = _arm(FC_TIMES, FC_VALUES)
    merged, report = cams.merge_arms(an, fc, np.datetime64(T_NOW),
                                     np.datetime64(T_INIT))

    assert report["overlap"] == ["2026-08-03T12:00", "2026-08-03T18:00",
                                 "2026-08-04T00:00"]
    # lead 0 agrees by construction; 18Z and 00Z are the steps where the
    # forecast has drifted and the priority rule actually decides something.
    for stamp, expected in zip(AN_TIMES, AN_VALUES):
        got = merged.sel(time=np.datetime64(stamp))
        assert np.all(got["omaod550"].values == np.float32(expected)), stamp
        assert str(got["arm"].values) == "an", stamp
    # the 18Z and 00Z forecast values must be nowhere in the merged cube
    assert not np.any(merged["omaod550"].values == np.float32(0.60))
    assert not np.any(merged["omaod550"].values == np.float32(0.70))


def test_forecast_only_steps_at_or_before_t_now_are_discarded():
    # 15Z and 21Z exist only in the forecast but are <= t_now, so they are
    # inside the analysis's authority and must not be spliced in.
    an = _arm(AN_TIMES, AN_VALUES)
    fc = _arm(FC_TIMES, FC_VALUES)
    merged, report = cams.merge_arms(an, fc, np.datetime64(T_NOW),
                                     np.datetime64(T_INIT))
    times = set(str(t) for t in merged["time"].values.astype("datetime64[m]"))
    assert "2026-08-03T15:00" not in times
    assert "2026-08-03T21:00" not in times
    assert times == {"2026-08-03T12:00", "2026-08-03T18:00", "2026-08-04T00:00",
                     "2026-08-04T03:00", "2026-08-04T06:00"}
    assert report["n_merged"] == 5


def test_forecast_is_clipped_strictly_after_t_now():
    # t_init == t_now here (a 00Z run whose lead 0 is the newest analysis),
    # so lead 0 must still be identical -- 0.30 in both arms.
    an = _arm(["2026-08-03T18", "2026-08-04T00"], [0.20, 0.30])
    fc = _arm(["2026-08-04T00", "2026-08-04T03", "2026-08-04T06"],
              [0.30, 0.45, 0.50])
    merged, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                                     np.datetime64("2026-08-04T00"))
    times = merged["time"].values
    assert np.datetime64("2026-08-04T00", "ns") in times
    assert (merged.sel(time=np.datetime64("2026-08-04T03"))["omaod550"].values
            == np.float32(0.45)).all()
    assert len(np.unique(times)) == len(times)
    assert list(merged["arm"].values) == ["an", "an", "fc", "fc"]
    assert report["lead0_identity"]["max"] == pytest.approx(0.0)


def test_arm_tag_is_carried_so_the_seam_gate_can_find_the_join():
    an = _arm(AN_TIMES, AN_VALUES)
    fc = _arm(FC_TIMES, FC_VALUES)
    merged, _ = cams.merge_arms(an, fc, np.datetime64(T_NOW),
                                np.datetime64(T_INIT))
    assert list(merged["arm"].values) == ["an", "an", "an", "fc", "fc"]


def test_overlap_outside_t_init_t_now_is_an_error():
    an = _arm(["2026-08-01T00"], 1.0)
    fc = _arm(["2026-08-01T00"], 2.0)
    with pytest.raises(AssertionError, match="overlap"):
        cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-03T12"))


# ------------------------------------------- lead-0 identity gate conditioning


def test_lead0_identity_is_reported_with_the_threshold_it_was_judged_against():
    an = _arm(["2026-08-04T00"], 0.30)
    fc = _arm(["2026-08-04T00", "2026-08-04T03"], [0.30, 0.45])
    _, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                                np.datetime64("2026-08-04T00"))
    stats = report["lead0_identity"]
    assert stats["max"] == pytest.approx(0.0)
    assert stats["n_differing"] == 0
    assert stats["n_cells"] == NLAT * NLON
    assert stats["n_total"] == NLAT * NLON
    # the report must carry the yardstick, not just the measurement
    assert stats["threshold"] == pytest.approx(
        cams.LEAD0_ATOL + cams.LEAD0_RTOL * 0.30, rel=1e-6)


def test_a_gross_uniform_mismatch_raises():
    an = _arm(["2026-08-04T00"], 1.0)
    fc_bad = _arm(["2026-08-04T00", "2026-08-04T03"], 9.0)
    with pytest.raises(AssertionError, match="lead 0"):
        cams.merge_arms(an, fc_bad, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-04T00"))


def test_a_sparse_plume_mismatch_raises_even_though_the_median_is_zero():
    """The failure this gate exists to catch.

    Smoke is sparse: a wrong lead-0 field is wrong over the plumes and
    identical over the clean-air majority. A median-based gate is structurally
    blind to it -- median|d| is exactly 0.0 here. Measured step by step on the
    cached cubes, perturbing only the top 5% of cells by x3 + 0.5 gives
    median|d| = 0.000000 at max|d| = 0.6675..1.9163 (Greece, 28 of 546 cells)
    and 0.8541..4.3310 (Iberia, 41 of 816), every one of which the plan's
    `median(d) <= 0.05*p99` threshold (0.008838 / 0.023380) admits.
    """
    n_hot = 6                      # 6 of 120 cells = 5%
    good = _plume_field(0.05, 0.80, n_hot)
    bad = _plume_field(0.05, 2.90, n_hot)   # plumes tripled and offset
    d = np.abs(bad - good)
    assert np.median(d) == 0.0, "the fixture must be median-invisible"

    an = _arm_from_fields(["2026-08-04T00"], [good])
    fc = _arm_from_fields(["2026-08-04T00", "2026-08-04T03"], [bad, bad])
    with pytest.raises(AssertionError, match="lead 0"):
        cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-04T00"))


def test_a_small_smooth_assimilation_increment_is_consistent_not_fatal():
    """The identity premise fails on real ADS data: the analysis at t_init
    carries assimilation increments the same-hour forecast init does not.

    [measured on the 2026-08-05 delivery] max|d| = 0.0066 on a field of
    p99.9 = 0.156 (4.3% of scale), 93% of cells differing at the 1e-6 level,
    u10 up to 0.12 m/s -- smooth, field-wide, and far beyond the packing-noise
    identity threshold, yet unmistakably the same weather. A WRONG run is not
    this gate's last line of defence: fetch.run pins the forecast arm's axis to
    this run's request (test_run_stops_on_a_forecast_arm_from_an_older_run).
    """
    rng = np.random.default_rng(5)
    good = (0.05 + 0.25 * rng.random((NLAT, NLON))).astype("float32")
    drifted = (good * 1.03).astype("float32")     # a smooth 3% increment
    an = _arm_from_fields(["2026-08-04T00"], [good])
    fc = _arm_from_fields(["2026-08-04T00", "2026-08-04T03"], [drifted, drifted])
    merged, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                                     np.datetime64("2026-08-04T00"))
    stats = report["lead0_identity"]
    assert stats["max"] > stats["threshold"], "fixture must exceed identity"
    assert stats["verdict"] == "assimilation_increment"
    assert stats["max"] <= stats["increment_max_allowed"]
    assert int(merged.sizes["time"]) == 2


def test_a_clean_air_domain_does_not_false_alarm_on_rounding_noise():
    """The other end: an all-but-zero field must not make the gate hair-trigger.

    A relative-only threshold collapses toward 0 when the field is empty -- the
    plan's `0.05 * max(p99, 1e-12)` floor is 5e-14, far below float32 noise --
    so a 1e-7 unit-conversion residue would abort the build. The absolute floor
    prevents that.
    """
    an = _arm(["2026-08-04T00"], 0.0)
    fc = _arm(["2026-08-04T00", "2026-08-04T03"], 1e-7)
    _, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                                np.datetime64("2026-08-04T00"))
    stats = report["lead0_identity"]
    assert stats["scale_p99_9"] == pytest.approx(0.0)
    assert stats["threshold"] == pytest.approx(cams.LEAD0_ATOL)
    assert stats["max"] == pytest.approx(1e-7, abs=1e-9)


def test_int16_packing_round_trip_is_within_tolerance():
    """One arm packed as int16 and the other not is a legal ADS outcome; the
    quantum for a 0..2 AOD span is 2/65534 = 3.1e-5, well inside LEAD0_ATOL."""
    rng = np.random.default_rng(0)
    field = rng.uniform(0.0, 2.0, size=(NLAT, NLON)).astype("float32")
    # the CF convention ADS emits: value = packed * scale_factor + add_offset,
    # with the int16 range centred on the data range.
    scale_factor = (float(field.max()) - float(field.min())) / 65534.0
    add_offset = float(field.min()) + 32767.0 * scale_factor
    packed_i = np.round((field.astype("float64") - add_offset) / scale_factor)
    assert packed_i.min() >= -32768 and packed_i.max() <= 32767, "int16 overflow"
    packed = (packed_i.astype("int16").astype("float64") * scale_factor
              + add_offset).astype("float32")
    quantum = float(np.abs(packed.astype("float64")
                           - field.astype("float64")).max())
    assert 0.0 < quantum < cams.LEAD0_ATOL, quantum

    an = _arm_from_fields(["2026-08-04T00"], [field])
    fc = _arm_from_fields(["2026-08-04T00", "2026-08-04T03"], [packed, packed])
    _, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                                np.datetime64("2026-08-04T00"))
    assert report["lead0_identity"]["max"] == pytest.approx(quantum, rel=1e-6)
    assert report["lead0_identity"]["n_differing"] == 0


def test_an_all_nonfinite_lead_0_is_rejected_not_waved_through():
    """A max-based gate has a hole a median-based one did not.

    ``np.nanmax`` of an all-NaN difference is not a small number, it is *no*
    number, and the guarded ``0.0`` fallback would read as a perfect match.
    The plan's median gate rejected this by accident (median of NaN is NaN and
    ``NaN <= x`` is False); the replacement has to reject it on purpose.
    """
    an = _arm_from_fields(["2026-08-04T00"],
                          [np.full((NLAT, NLON), np.nan, dtype="float32")])
    fc = _arm(["2026-08-04T00", "2026-08-04T03"], 0.30)
    with pytest.raises(AssertionError, match="no finite cells"):
        cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-04T00"))


def test_a_few_masked_cells_do_not_defeat_the_gate():
    """Partial non-finiteness must be reported and ignored, not fatal -- but
    the finite remainder still has to pass on its own merits."""
    good = _plume_field(0.05, 0.80, 6)
    masked = good.copy()
    masked[0, 0] = np.nan
    an = _arm_from_fields(["2026-08-04T00"], [good])
    fc = _arm_from_fields(["2026-08-04T00", "2026-08-04T03"], [masked, masked])
    _, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                                np.datetime64("2026-08-04T00"))
    stats = report["lead0_identity"]
    assert stats["n_nonfinite"] == 1
    assert stats["n_cells"] == NLAT * NLON - 1
    assert stats["n_total"] == NLAT * NLON
    assert stats["max"] == pytest.approx(0.0)

    # ...and a masked cell must not buy the rest of the field an amnesty
    worse = masked.copy()
    worse[5, 5] = 2.90
    fc_bad = _arm_from_fields(["2026-08-04T00", "2026-08-04T03"], [worse, worse])
    with pytest.raises(AssertionError, match="lead 0"):
        cams.merge_arms(an, fc_bad, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-04T00"))


def test_lead0_gate_is_skipped_when_the_forecast_does_not_reach_back_to_t_init():
    # a run whose lead 0 predates the analysis window we asked for
    an = _arm(["2026-08-04T00", "2026-08-04T06"], [0.30, 0.35])
    fc = _arm(["2026-08-04T09", "2026-08-04T12"], [0.40, 0.45])
    _, report = cams.merge_arms(an, fc, np.datetime64("2026-08-04T06"),
                                np.datetime64("2026-08-03T12"))
    assert "lead0_identity" not in report
    assert report["overlap"] == []


def test_lead0_shape_mismatch_is_named_not_broadcast():
    an = _arm(["2026-08-04T00"], 0.30, nlat=NLAT, nlon=NLON)
    fc = _arm(["2026-08-04T00", "2026-08-04T03"], 0.30, nlat=NLAT, nlon=NLON + 1)
    with pytest.raises(Exception) as e:
        cams.merge_arms(an, fc, np.datetime64("2026-08-04T00"),
                        np.datetime64("2026-08-04T00"))
    assert "shape" in str(e.value).lower() or "align" in str(e.value).lower()


# ------------------------------------------------------------- determinism


def test_merge_is_independent_of_row_order_within_each_arm():
    """merge_arms must not inherit its answer from input row order.

    The plan's version compared a Dataset with `.copy()` of itself, which only
    proved the function is deterministic. Shuffling the rows is what actually
    exercises the lexsort/unique priority machinery.
    """
    an = _arm(AN_TIMES, AN_VALUES)
    fc = _arm(FC_TIMES, FC_VALUES)
    a, ra = cams.merge_arms(an, fc, np.datetime64(T_NOW), np.datetime64(T_INIT))

    rng = np.random.default_rng(7)
    an_shuf = an.isel(time=rng.permutation(an.sizes["time"]))
    fc_shuf = fc.isel(time=rng.permutation(fc.sizes["time"]))
    b, rb = cams.merge_arms(an_shuf, fc_shuf, np.datetime64(T_NOW),
                            np.datetime64(T_INIT))

    xr.testing.assert_identical(a, b)
    assert ra == rb


def test_merged_time_axis_is_strictly_ascending():
    an = _arm(AN_TIMES, AN_VALUES)
    fc = _arm(FC_TIMES, FC_VALUES)
    merged, _ = cams.merge_arms(an, fc, np.datetime64(T_NOW),
                                np.datetime64(T_INIT))
    t = merged["time"].values
    assert (np.diff(t) > np.timedelta64(0)).all()


def test_time_keys_survive_nanosecond_precision():
    """Regression: np.asarray(t, 'datetime64[ns]').tolist() yields ints, so a
    set built from it never matches a np.datetime64 and np.datetime64(int)
    raises 'Converting an integer to a NumPy datetime requires a specified
    unit'. The whole lead-0 gate was dead code because of it."""
    t = np.array(["2026-08-03T12", "2026-08-04T00"], dtype="datetime64[ns]")
    keys = cams._time_keys(t)
    assert keys.dtype == np.int64
    assert int(np.datetime64("2026-08-03T12", "ns").astype("int64")) in set(keys.tolist())
    assert cams._fmt(keys[0]) == "2026-08-03T12:00"
