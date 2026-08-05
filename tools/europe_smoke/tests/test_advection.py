# tools/europe_smoke/tests/test_advection.py
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import xarray as xr

from tools.europe_smoke import advection, config


# ------------------------------------------------------------------ helpers

def _cube(u_ms, v_ms, n_steps=2):
    """A CAMS-shaped dataset on the real Europe lattice with constant wind."""
    lon, lat = config.grid_axes()
    shape = (n_steps, lat.size, lon.size)
    return xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.full(shape, float(u_ms))),
         "v10": (("time", "latitude", "longitude"), np.full(shape, float(v_ms)))},
        coords={"time": np.arange(n_steps), "latitude": lat, "longitude": lon},
    )


# ------------------------------------------------------------------ knee/offsets

def test_soft_knee_is_exact_below_point_eight():
    for r in (0.0, 0.3, 0.5, 0.79):
        assert advection.soft_knee(r) == pytest.approx(r)


def test_soft_knee_is_continuous_at_the_join():
    lo = advection.soft_knee(0.8 - 1e-9)
    hi = advection.soft_knee(0.8 + 1e-9)
    assert abs(lo - hi) < 1e-6


def test_soft_knee_returns_0_9264_at_r_equals_one():
    assert advection.soft_knee(1.0) == pytest.approx(0.9264, abs=1e-4)


def test_soft_knee_never_exceeds_one():
    r = np.array([0.0, 1.0, 2.0, 5.0, 50.0, 1e6])
    assert (advection.soft_knee(r) <= 1.0 + 1e-12).all()


def test_degree_offsets_scale_longitude_by_one_over_cos_lat():
    u = np.array([[10.0]])
    v = np.array([[0.0]])
    dlon_30, _ = advection.degree_offsets(u, v, np.array([30.0]), hours=6.0)
    dlon_70, _ = advection.degree_offsets(u, v, np.array([70.0]), hours=6.0)
    ratio = abs(dlon_70[0, 0] / dlon_30[0, 0])
    assert ratio == pytest.approx(np.cos(np.radians(30)) / np.cos(np.radians(70)), rel=0.01)


def test_kernel_weights_sum_to_one_and_are_causal():
    assert sum(advection.KERNEL_WEIGHTS) == pytest.approx(1.0, abs=1e-3)
    assert all(lag >= 0 for lag in advection.KERNEL_LAGS)
    assert advection.KERNEL_LAGS == (0.0, 6.0, 12.0)


def test_fit_gain_clamps_at_the_validated_cap():
    out = advection.fit_gain_from_scores({0.5: 0.1, 1.0: 0.2, 1.5: 0.9, 3.0: 0.95},
                                         kmax=config.MAX_ADVECTION_GAIN)
    assert out["k"] == 1.5
    assert out["clamped"] is True


def test_fit_gain_reports_the_zero_gain_baseline():
    out = advection.fit_gain_from_scores({0.0: 0.30, 1.0: 0.62, 1.5: 0.55},
                                         kmax=config.MAX_ADVECTION_GAIN)
    assert out["k"] == 1.0
    assert out["baseline_k0"] == pytest.approx(0.30)
    assert out["clamped"] is False


def test_containment_thresholds_match_the_spec():
    assert advection.GATE_FRAC_OVER_D == 0.005
    assert advection.GATE_FRAC_OVER_08D == 0.020
    assert advection.D_LON == 11.6 and advection.D_LAT == 4.6


# ------------------------------------------------------------------ 11(a) static

def test_static_bound_reaches_12_8_and_5_8_degrees():
    res = advection.static_bound()
    assert res["reach_lon_deg"] == pytest.approx(11.6 + 0.4 + 0.8)
    assert res["reach_lat_deg"] == pytest.approx(4.6 + 0.4 + 0.8)
    assert res["terms"]["blur_tap"] == pytest.approx(0.8)


def test_static_bound_places_the_extreme_tap_with_half_a_texel_to_spare():
    edges = advection.static_bound()["edges"]
    assert edges["west"]["extreme_tap_deg"] == pytest.approx(-37.8)
    assert edges["east"]["extreme_tap_deg"] == pytest.approx(57.8)
    assert edges["south"]["extreme_tap_deg"] == pytest.approx(24.2)
    assert edges["north"]["extreme_tap_deg"] == pytest.approx(77.8)
    for name, e in edges.items():
        # 0.2 deg == half a 0.4 deg texel
        assert e["slack_deg"] == pytest.approx(0.2), name
        assert e["ok"] is True


def test_static_bound_asserts_all_four_inequalities_and_passes_on_the_shipped_area():
    res = advection.assert_static_bound()
    assert res["pass"] is True
    assert set(res["edges"]) == {"west", "east", "south", "north"}
    assert res["min_slack_deg"] == pytest.approx(0.2)


def test_static_bound_fails_when_the_bleed_is_short():
    # rev 1's 0.2 deg bleed: the thing gate 11(a) exists to catch
    thin = (72.2, -25.2, 29.8, 45.2)
    res = advection.static_bound(area=thin)
    assert res["pass"] is False
    assert all(not e["ok"] for e in res["edges"].values())
    with pytest.raises(AssertionError, match="gate 11.a. FAILED"):
        advection.assert_static_bound(area=thin)


def test_static_bound_would_also_pass_on_the_escalation_area():
    res = advection.static_bound(area=(79.2, -41.6, 22.8, 61.6),
                                 d_lon=15.2, d_lat=5.8)
    assert res["pass"] is True


def test_assert_static_bound_still_raises_under_python_dash_O():
    """A bare ``assert`` is compiled out by -O and the gate becomes a no-op.

    Same class of bug as the France sea-clamp fix ("not assert: must survive
    python -O", examples/population_ghsl/france_population_pt_3d.py). Run the
    check in a real -O subprocess, because nothing inside this process can.
    """
    import pathlib
    import_root = str(pathlib.Path(config.__file__).resolve().parents[2])
    src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {import_root!r})
        if __debug__:
            raise SystemExit("subprocess is not running under -O")
        from tools.europe_smoke import advection
        try:
            advection.assert_static_bound(area=(72.2, -25.2, 29.8, 45.2))
        except AssertionError as exc:
            print("RAISED", "gate 11(a) FAILED" in str(exc))
        else:
            print("NO-OP")
    """)
    out = subprocess.run([sys.executable, "-O", "-c", src],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "RAISED True", out.stdout + out.stderr


# ------------------------------------------------------------------ 11(b) weights

def test_mercator_screen_weights_sum_to_exactly_one():
    lon, lat = config.grid_axes()
    lat_sel = lat[(lat >= 30.0) & (lat <= 72.0)]
    w = advection.mercator_screen_weights(lat_sel, 175)
    assert w.shape == (106, 175)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_mercator_screen_weights_give_the_north_more_than_its_row_share():
    lon, lat = config.grid_axes()
    lat_sel = lat[(lat >= 30.0) & (lat <= 72.0)]
    w = advection.mercator_screen_weights(lat_sel, 175)
    band = lat_sel >= 66.0
    screen_share = float(w[band].sum())
    row_share = float(band.sum() / lat_sel.size)
    assert row_share == pytest.approx(0.1509, abs=5e-4)
    assert screen_share == pytest.approx(0.2400, abs=5e-4)
    assert screen_share > row_share


def test_mercator_screen_weights_are_flat_along_longitude():
    lon, lat = config.grid_axes()
    lat_sel = lat[(lat >= 30.0) & (lat <= 72.0)]
    w = advection.mercator_screen_weights(lat_sel, 175)
    assert np.allclose(w, w[:, :1])


def test_mercator_screen_weights_reject_a_degenerate_block():
    lon, lat = config.grid_axes()
    with pytest.raises(ValueError, match="n_lon must be positive"):
        advection.mercator_screen_weights(lat[:5], 0)
    with pytest.raises(ValueError, match="selects no rows"):
        advection.mercator_screen_weights(lat[:0], 175)


# ------------------------------------------------------------------ 11(b) gate

def test_containment_of_a_saturating_field_is_exactly_one_not_the_display_width():
    """The regression this whole finding is about.

    Weighting a boolean by a latitude-normalised vector broadcast across
    longitude summed to n_lon, so a field where EVERY sample exceeds D
    reported 175.0 instead of 1.0.
    """
    ds = _cube(u_ms=200.0, v_ms=200.0)      # ~ 100 deg of offset at 12 h
    res = advection.containment(ds, k=1.0)
    assert res["display_block"] == (106, 175)
    assert res["frac_over_D"] == pytest.approx(1.0, abs=1e-12)
    assert res["frac_over_0.8D"] == pytest.approx(1.0, abs=1e-12)
    assert res["frac_over_D"] < 1.0 + 1e-12    # never 175
    assert res["pass"] is False


def test_containment_of_a_calm_field_is_exactly_zero():
    res = advection.containment(_cube(0.0, 0.0), k=1.0)
    assert res["frac_over_D"] == 0.0
    assert res["frac_over_0.8D"] == 0.0
    assert res["pass"] is True


def test_containment_is_always_a_fraction_in_the_unit_interval():
    for u_ms, v_ms in ((0.0, 0.0), (5.0, 0.0), (25.0, 8.0), (1e4, 1e4)):
        res = advection.containment(_cube(u_ms, v_ms), k=1.5)
        for key in ("frac_over_D", "frac_over_0.8D"):
            assert 0.0 <= res[key] <= 1.0, (u_ms, v_ms, key, res[key])
        assert res["frac_over_D"] <= res["frac_over_0.8D"] + 1e-12


def test_containment_is_the_kernel_weighted_mean_over_the_engageable_lags():
    """12 h engaged, 6 h not: the answer must be w12/(w6+w12), not w12 and not 1."""
    # 12 h offset just over D_lat, 6 h offset just under 0.8 D_lat everywhere
    v = 4.7 * 111_320.0 / (12 * 3600.0)
    res = advection.containment(_cube(0.0, v), k=1.0)
    w6, w12 = advection.KERNEL_WEIGHTS[1], advection.KERNEL_WEIGHTS[2]
    assert res["per_lag"]["12h"]["frac_over_D"] == pytest.approx(1.0, abs=1e-12)
    assert res["per_lag"]["6h"]["frac_over_D"] == pytest.approx(0.0, abs=1e-12)
    assert res["frac_over_D"] == pytest.approx(w12 / (w6 + w12), abs=1e-12)
    assert res["frac_over_D"] == pytest.approx(0.30567, abs=1e-4)


def test_containment_weights_the_north_by_screen_share_not_row_count():
    """A field that only exceeds north of 66N must report the 24% screen share."""
    lon, lat = config.grid_axes()
    shape = (1, lat.size, lon.size)
    v = np.zeros(shape)
    v[:, lat >= 66.0, :] = 4.7 * 111_320.0 / (12 * 3600.0)
    ds = xr.Dataset({"u10": (("time", "latitude", "longitude"), np.zeros(shape)),
                     "v10": (("time", "latitude", "longitude"), v)},
                    coords={"time": [0], "latitude": lat, "longitude": lon})
    res = advection.containment(ds, k=1.0)
    w6, w12 = advection.KERNEL_WEIGHTS[1], advection.KERNEL_WEIGHTS[2]
    expected = (w12 / (w6 + w12)) * 0.2400
    assert res["frac_over_D"] == pytest.approx(expected, rel=3e-3)


def test_containment_scales_with_the_gain():
    v = 3.6 * 111_320.0 / (12 * 3600.0)      # 3.6 deg at 12 h: under D_lat
    calm = advection.containment(_cube(0.0, v), k=1.0)
    windy = advection.containment(_cube(0.0, v), k=1.5)   # 5.4 deg: over D_lat
    assert calm["frac_over_D"] == 0.0
    assert windy["frac_over_D"] > 0.0


def test_containment_rejects_a_display_window_outside_the_delivered_grid():
    with pytest.raises(ValueError, match="selects no cells"):
        advection.containment(_cube(0.0, 0.0), display_window=(100.0, 80.0, 120.0, 85.0))


def test_containment_flags_a_gain_above_the_validated_cap():
    assert advection.containment(_cube(0.0, 0.0), k=1.5)["k_over_validated_cap"] is False
    assert advection.containment(_cube(0.0, 0.0), k=3.0)["k_over_validated_cap"] is True


# ------------------------------------------------------------------ 11(c) debug

def test_debug_pass_finds_nothing_outside_for_a_calm_field():
    res = advection.debug_pass(_cube(0.0, 0.0), k=1.0)
    assert res["preclamp_outside"] == 0
    assert res["postclamp_outside"] == 0
    assert res["preclamp_clean"] is True
    assert res["pass"] is True
    # 0.2 deg of spare margin on both axes, reported as a negative overshoot
    assert res["worst_overshoot_deg"]["postclamp"]["lon"] == pytest.approx(-11.8)


def test_debug_pass_postclamp_is_zero_even_when_the_knee_is_saturated():
    """The invariant: no clamped sample can ever leave the fetched rectangle."""
    res = advection.debug_pass(_cube(200.0, 200.0), k=1.5)
    assert res["postclamp_outside"] == 0
    assert res["pass"] is True
    # ...and the spare margin is exactly gate 11(a)'s 0.2 deg
    w = res["worst_overshoot_deg"]["postclamp"]
    assert w["lon"] == pytest.approx(-0.2, abs=1e-9)
    assert w["lat"] == pytest.approx(-0.2, abs=1e-9)


def test_debug_pass_preclamp_counts_what_the_knee_saved():
    res = advection.debug_pass(_cube(200.0, 200.0), k=1.5)
    assert res["preclamp_outside"] == res["samples"]
    assert res["preclamp_clean"] is False
    assert res["preclamp_frac"] == pytest.approx(1.0)
    assert res["worst_overshoot_deg"]["preclamp"]["lon"] > 100.0


def test_debug_pass_sees_the_column_just_outside_the_display_block():
    """The window rim's bilinear partner is NOT in the display block.

    lon -25.0 is half a cell off the 0.4 lattice, so the outermost in-block
    column is -24.8 and the screen pixel at -25.0 interpolates it against the
    -25.2 column. Evaluating only the display block misses that column
    entirely: put the whole extreme wind on it and a block-only pass reports
    a clean zero.
    """
    lon, lat = config.grid_axes()
    ci = np.flatnonzero((lon >= -25.0 - 1e-9) & (lon <= 45.0 + 1e-9))
    assert lon[ci[0]] == pytest.approx(-24.8)      # 0.2 deg inside the window
    assert lon[ci[0] - 1] == pytest.approx(-25.2)  # the missed partner

    shape = (1, lat.size, lon.size)
    u = np.zeros(shape)
    u[:, :, ci[0] - 1] = -30.0 * 111_320.0 * np.cos(np.radians(40.0)) / (12 * 3600.0)
    ds = xr.Dataset({"u10": (("time", "latitude", "longitude"), u),
                     "v10": (("time", "latitude", "longitude"), np.zeros(shape))},
                    coords={"time": [0], "latitude": lat, "longitude": lon})

    assert advection.debug_pass(ds, k=1.5, halo=0)["preclamp_outside"] == 0
    assert advection.debug_pass(ds, k=1.5)["preclamp_outside"] == 18
    # the hard invariant is unaffected either way
    assert advection.debug_pass(ds, k=1.5)["postclamp_outside"] == 0


def test_debug_pass_halo_covers_the_window_and_nothing_further():
    """One ring, and the ring's dead cells are masked, not counted."""
    res = advection.debug_pass(_cube(0.0, 0.0), k=1.0)
    lat_rows, lon_cols = res["display_block"]
    assert (lat_rows, lon_cols) == (106, 175)
    # lat edges ARE on the lattice, so the lat halo rows clip to nothing;
    # lon edges are half a cell off, so both lon halo columns stay live.
    assert res["evaluated_cells"] == 106 * 177
    assert res["samples"] == 106 * 177 * 2 * 2      # x 2 lags x 2 steps
    assert res["halo"] == 1


def test_debug_pass_agrees_with_the_static_bound_margin():
    static = advection.static_bound()
    debug = advection.debug_pass(_cube(200.0, 200.0), k=1.5)
    assert debug["worst_overshoot_deg"]["postclamp"]["lon"] == pytest.approx(
        -static["edges"]["west"]["slack_deg"], abs=1e-9)


def test_gate11_runs_all_three_parts():
    res = advection.gate11(_cube(0.0, 0.0), k=1.0)
    assert set(res) == {"static", "empirical", "debug", "pass"}
    assert res["pass"] is True


def test_gate11_threads_D_through_to_the_escalation_area():
    """§0.1's escalation raises D to 15.2/5.8; gate11 must be able to say so."""
    esc_area = (79.2, -41.6, 22.8, 61.6)
    res = advection.gate11(_cube(0.0, 0.0), k=1.0, area=esc_area,
                           d_lon=15.2, d_lat=5.8)
    assert res["static"]["reach_lon_deg"] == pytest.approx(15.2 + 0.4 + 0.8)
    assert res["static"]["min_slack_deg"] == pytest.approx(0.2)
    assert res["empirical"]["D"] == {"lon": 15.2, "lat": 5.8}
    assert res["pass"] is True
    # and it FAILS if the escalated D is used against the un-escalated area
    assert advection.gate11(_cube(0.0, 0.0), k=1.0, d_lon=15.2, d_lat=5.8)["pass"] is False


def test_containment_is_a_fraction_normalised_over_the_whole_display_block():
    """Row-only normalisation multiplied the answer by the 175 display columns.

    Measured on the plan's code: frac_over_D 168.149252, frac_over_0.8D
    175.000000 -- exactly the display column count -- for a field where every
    sample exceeds. After the fix both saturate at 1.0, matching an independent
    explicit weighted mean to every printed digit.
    """
    import xarray as xr

    lon, lat = config.grid_axes()
    t = np.array(["2026-08-04T00:00", "2026-08-04T03:00"], dtype="datetime64[ns]")
    shape = (2, lat.size, lon.size)
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.full(shape, 60.0, "float32")),
         "v10": (("time", "latitude", "longitude"), np.full(shape, -18.0, "float32"))},
        coords={"time": t, "latitude": lat, "longitude": lon})
    out = advection.containment(ds, k=1.25)
    assert out["frac_over_D"] == pytest.approx(1.0)
    assert out["frac_over_0.8D"] == pytest.approx(1.0)
    assert out["display_block_shape"] == list(config.DISPLAY_BLOCK_SHAPE)

    calm = ds.copy()
    calm["u10"] = (("time", "latitude", "longitude"), np.full(shape, 1.0, "float32"))
    calm["v10"] = (("time", "latitude", "longitude"), np.zeros(shape, "float32"))
    assert advection.containment(calm, k=1.25)["frac_over_D"] == 0.0


def test_containment_survives_the_lattice_noise_real_cams_delivers():
    """Delivered coordinates carry ~1e-13 residuals; the block must not shrink.

    ci.size is now a divisor, so a dropped boundary row rescales the answer.
    Measured with a one-ulp nudge on lat[15]: (105, 175) instead of (106, 175)
    and frac_over_D off by -6.9e-4, 14% of the 0.5% threshold.
    """
    import xarray as xr

    lon, lat = config.grid_axes()
    lat = lat.copy()
    lat[15] = np.nextafter(72.0, 73.0)      # the display window's north edge
    lat[120] = np.nextafter(30.0, 29.0)     # and its south edge
    t = np.array(["2026-08-04T00:00"], dtype="datetime64[ns]")
    shape = (1, lat.size, lon.size)
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.full(shape, 60.0, "float32")),
         "v10": (("time", "latitude", "longitude"), np.full(shape, -18.0, "float32"))},
        coords={"time": t, "latitude": lat, "longitude": lon})
    out = advection.containment(ds, k=1.25)
    assert out["display_block_shape"] == list(config.DISPLAY_BLOCK_SHAPE)
    assert out["frac_over_D"] == pytest.approx(1.0)


def test_static_containment_reproduces_the_spec_margin_arithmetic():
    """§9 gate 11(a): 12.8/5.8 required against 13.0/6.0 delivered, 0.2 spare."""
    g = advection.static_containment()
    assert g["terms"]["blur_tap"] == 0.8
    for edge in ("west", "east"):
        assert g["edges"][edge]["required_deg"] == pytest.approx(12.8)
        assert g["edges"][edge]["delivered_margin_deg"] == pytest.approx(13.0)
    for edge in ("north", "south"):
        assert g["edges"][edge]["required_deg"] == pytest.approx(5.8)
        assert g["edges"][edge]["delivered_margin_deg"] == pytest.approx(6.0)
    assert g["min_slack_deg"] == pytest.approx(0.2, abs=1e-9)
    assert g["verdict"] == "PASS"


# ------------------------------------------------- MERGE: the two 11(a) shapes


def test_the_two_gate11a_shapes_agree():
    """Two fixes each added gate 11(a) with its own report shape, and both are
    kept: ``build.py`` reads ``static_containment()``, ``gate11()`` reads
    ``static_bound()``. They must never disagree about the verdict, the reach
    or the per-edge slack -- a duplicated gate that drifts is worse than one.
    """
    a = advection.static_bound()
    b = advection.static_containment()
    assert b["verdict"] == ("PASS" if a["pass"] else "FAIL")
    assert b["min_slack_deg"] == pytest.approx(a["min_slack_deg"], abs=1e-12)
    assert b["terms"]["blur_tap"] == pytest.approx(a["terms"]["blur_tap"])
    assert b["terms"]["curl_clamp"] == pytest.approx(a["terms"]["curl"])
    for edge in ("west", "east", "south", "north"):
        assert b["edges"][edge]["slack_deg"] == pytest.approx(
            a["edges"][edge]["slack_deg"], abs=1e-12), edge
        assert b["edges"][edge]["ok"] is a["edges"][edge]["ok"]
    assert b["edges"]["west"]["required_deg"] == pytest.approx(a["reach_lon_deg"])
    assert b["edges"]["north"]["required_deg"] == pytest.approx(a["reach_lat_deg"])


def test_the_two_gate11a_shapes_agree_when_the_area_is_shrunk(monkeypatch):
    """...including on the failing side, which is the half that matters."""
    monkeypatch.setattr(config, "AREA", (76.0, -36.0, 26.0, 56.0))
    a = advection.static_bound()
    b = advection.static_containment()
    assert a["pass"] is False and b["verdict"] == "FAIL"
    assert b["min_slack_deg"] == pytest.approx(a["min_slack_deg"], abs=1e-12)
