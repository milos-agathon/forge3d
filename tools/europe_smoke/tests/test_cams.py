# tools/europe_smoke/tests/test_cams.py
import datetime as dt

import numpy as np
import pytest
import xarray as xr

from tools.europe_smoke import cams, config

DIMS = ("forecast_period", "forecast_reference_time", "latitude", "longitude")

GREECE_NC = (config.REPO_ROOT / "examples" / ".cache" / "greece_smoke" / "cams"
             / "raw_nc" / "data_sfc.nc")


def _raw_like_ads(nper=4, nref=3, nlat=5, nlon=7, aod_name="omaod550", extra=None):
    """A dataset shaped exactly as ADS delivers (1.1): period-major dims,
    descending latitude, a 2-D valid_time coord."""
    period = np.array([0, 3, 6, 9], dtype="timedelta64[h]")[:nper].astype("timedelta64[ns]")
    ref = np.array([np.datetime64("2026-07-25T00:00") + np.timedelta64(12 * i, "h")
                    for i in range(nref)], dtype="datetime64[ns]")
    lat = 78.0 - 0.4 * np.arange(nlat)
    lon = -38.0 + 0.4 * np.arange(nlon)
    shape = (nper, nref, nlat, nlon)
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    ds = xr.Dataset(
        {
            aod_name: (DIMS, data,
                       {"units": "~", "long_name": "Organic Matter Aerosol Optical Depth at 550nm"}),
            "u10": (DIMS, data * 0 + 1,
                    {"units": "m s**-1", "long_name": "10 metre U wind component"}),
            "v10": (DIMS, data * 0 - 1,
                    {"units": "m s**-1", "long_name": "10 metre V wind component"}),
        },
        coords={"forecast_period": period, "forecast_reference_time": ref,
                "latitude": lat, "longitude": lon},
    )
    ds = ds.assign_coords(valid_time=(("forecast_reference_time", "forecast_period"),
                                      ref[:, None] + period[None, :]))
    for name, attrs in (extra or {}).items():
        ds[name] = (DIMS, data * 0 + 0.5, attrs)
    return ds


def _with_optionals(**kw):
    return _raw_like_ads(extra={
        "bcaod550": {"units": "~",
                     "long_name": "Black Carbon Aerosol Optical Depth at 550nm"},
        "aod550": {"units": "~",
                   "long_name": "Total Aerosol Optical Depth at 550nm"},
    }, **kw)


def _axis():
    from tools.europe_smoke.probe import Axis
    return Axis(
        d_an=dt.date(2026, 8, 4), d_prev=dt.date(2026, 8, 3), start=dt.date(2026, 7, 25),
        h_win=["00:00", "06:00", "12:00", "18:00"], h_new=["00:00"],
        t_now=dt.datetime(2026, 8, 4, 0, 0), run_date=dt.date(2026, 8, 4),
        run_hour="00:00", t_init=dt.datetime(2026, 8, 4, 0, 0),
    )


# ------------------------------------------------------------------ resolution


def test_resolve_var_finds_the_exact_short_name():
    ds = _raw_like_ads()
    assert cams.resolve_var(ds, "omaod550") == "omaod550"
    assert cams.resolve_var(ds, "u10") == "u10"


def test_resolve_var_falls_back_to_a_long_name_substring():
    ds = _raw_like_ads(aod_name="organic_matter_aod")
    assert cams.resolve_var(ds, "omaod550") == "organic_matter_aod"


def test_resolve_var_raises_and_lists_the_delivered_variables():
    ds = _raw_like_ads().drop_vars("omaod550")
    with pytest.raises(RuntimeError) as e:
        cams.resolve_var(ds, "omaod550")
    assert "u10" in str(e.value)   # the message must show what WAS delivered


def test_exact_aod550_is_not_shadowed_by_the_other_two_optical_depths():
    """'aod550' is a substring of 'omaod550' and 'bcaod550'. Tier 1 compares for
    equality, never containment, so the total-AOD name stays resolvable."""
    ds = _with_optionals()
    assert cams.resolve_var(ds, "aod550") == "aod550"
    assert cams.resolve_var(ds, "omaod550") == "omaod550"
    assert cams.resolve_var(ds, "bcaod550") == "bcaod550"


def test_exact_match_wins_over_a_competing_substring_match():
    ds = _raw_like_ads(extra={
        "organic_matter_aod": {"units": "~",
                               "long_name": "Organic Matter Aerosol Optical Depth at 550nm"},
    })
    assert cams.resolve_var(ds, "omaod550") == "omaod550"
    assert cams.candidates(ds, "omaod550") == ("exact", ["omaod550"])


def test_two_substring_candidates_are_a_hard_failure_naming_both():
    """No exact 'omaod550'; two plausible organic fields. Insertion order must
    NOT decide -- the old first-match-wins loop returned whichever came first."""
    ds = _raw_like_ads(aod_name="organic_matter_aod", extra={
        "omaod550_sfc": {"units": "~",
                         "long_name": "Organic Matter AOD at 550nm, surface"},
    })
    with pytest.raises(cams.CamsContractError) as e:
        cams.resolve_var(ds, "omaod550")
    msg = str(e.value)
    assert "organic_matter_aod" in msg and "omaod550_sfc" in msg
    assert "ambiguously" in msg
    assert "u10" in msg          # the delivered data_vars list is always printed


def test_case_variant_duplicates_are_ambiguous_not_silently_collapsed():
    """The old ``{name.lower(): name}`` dict dropped one of these on the floor."""
    ds = _raw_like_ads()
    ds["U10"] = ds["u10"]
    with pytest.raises(cams.CamsContractError, match="ambiguously"):
        cams.resolve_var(ds, "u10")


def test_a_single_renamed_organic_field_still_resolves_by_substring():
    ds = _raw_like_ads(aod_name="omaod550_sfc")
    assert cams.candidates(ds, "omaod550") == ("substring", ["omaod550_sfc"])
    assert cams.resolve_var(ds, "omaod550") == "omaod550_sfc"


def test_unknown_canonical_name_is_rejected():
    with pytest.raises(cams.CamsContractError, match="VAR_RESOLUTION"):
        cams.resolve_var(_raw_like_ads(), "co")


# ------------------------------------------------------------------ assertions


def test_assert_variables_returns_the_resolved_mapping_with_units_and_dims():
    resolved = cams.assert_variables(_raw_like_ads())
    assert set(resolved) == set(config.REQUIRED_VARS)
    assert resolved["omaod550"].units == "~"
    assert resolved["u10"].units == "m s**-1"
    assert set(resolved["omaod550"].dims) == config.RAW_DIMS


def test_assert_variables_accepts_the_flattened_layout_too():
    resolved = cams.assert_variables(cams.flatten_time(_raw_like_ads()))
    assert set(resolved["omaod550"].dims) == config.FLAT_DIMS


def test_an_aerosol_mass_field_must_not_satisfy_the_smoke_field():
    """``aermssom`` matches the 'organic' substring and would resolve. Only the
    unit class stops it being served as an optical depth."""
    ds = _raw_like_ads().drop_vars("omaod550")
    ds["aermssom"] = (DIMS, np.zeros((4, 3, 5, 7), "float32"),
                      {"units": "kg m**-2",
                       "long_name": "Vertically integrated mass of organic matter aerosol"})
    assert cams.resolve_var(ds, "omaod550") == "aermssom"     # resolution alone is not enough
    with pytest.raises(cams.CamsContractError) as e:
        cams.assert_variables(ds)
    assert "kg m**-2" in str(e.value) and "dimensionless" in str(e.value)


def test_assert_variables_rejects_the_wrong_dims():
    ds = _raw_like_ads()
    ds["omaod550"] = ds["omaod550"].isel(latitude=0, drop=True)
    with pytest.raises(cams.CamsContractError, match="has dims"):
        cams.assert_variables(ds)


def test_assert_variables_rejects_a_missing_units_attribute():
    ds = _raw_like_ads()
    ds["omaod550"].attrs.pop("units")
    with pytest.raises(cams.CamsContractError, match="no units attribute"):
        cams.assert_variables(ds)


def test_assert_variables_rejects_a_wind_field_in_the_wrong_unit():
    ds = _raw_like_ads()
    ds["u10"].attrs["units"] = "knots"
    with pytest.raises(cams.CamsContractError, match="not velocity"):
        cams.assert_variables(ds)


def test_assert_variables_accepts_the_cf_dimensionless_spellings():
    for spelling in ("~", "1", "", "dimensionless"):
        ds = _raw_like_ads()
        ds["omaod550"].attrs["units"] = spelling
        assert cams.assert_variables(ds)["omaod550"].units == spelling


def test_two_canonical_names_may_not_claim_the_same_delivered_variable():
    ds = _raw_like_ads().drop_vars(["u10", "v10"])
    ds["wind_u_component_v_component"] = (
        DIMS, np.zeros((4, 3, 5, 7), "float32"),
        {"units": "m s**-1", "long_name": "u_component and v_component"})
    with pytest.raises(cams.CamsContractError, match="both"):
        cams.assert_variables(ds)


# ------------------------------------------------------------------ canonicalise


def test_canonicalise_renames_to_the_canonical_set():
    out = cams.canonicalise(cams.flatten_time(_raw_like_ads(aod_name="organic_matter_aod")))
    assert "omaod550" in out.ds.data_vars
    assert "organic_matter_aod" not in out.ds.data_vars
    assert out.resolved["omaod550"].delivered == "organic_matter_aod"
    assert out.resolved["omaod550"].tier == "substring"


def test_canonicalise_hard_fails_on_a_missing_required_variable():
    ds = cams.flatten_time(_raw_like_ads()).drop_vars("omaod550")
    with pytest.raises(cams.CamsContractError) as e:
        cams.canonicalise(ds)
    assert "['u10', 'v10']" in str(e.value)   # delivered data_vars, verbatim


def test_canonicalise_records_a_shortfall_for_the_question_marked_names():
    """bcaod550/aod550 are documented-but-unverified ECMWF short names. Their
    absence is neither a crash nor a silent skip."""
    out = cams.canonicalise(cams.flatten_time(_raw_like_ads()))
    assert set(out.resolved) == set(config.REQUIRED_VARS)
    assert {s.canonical for s in out.shortfall} == {"bcaod550", "aod550"}
    assert all(s.kind == "unresolved" for s in out.shortfall)
    for s in out.shortfall:
        assert "omaod550" in s.detail    # the delivered list is in every record
    # and the dataset is still usable
    assert out.ds["omaod550"].sizes["time"] == 12


def test_canonicalise_resolves_the_optionals_when_they_are_delivered():
    out = cams.canonicalise(cams.flatten_time(_with_optionals()))
    assert set(out.resolved) == set(config.REQUIRED_VARS) | set(config.OPTIONAL_VARS)
    assert out.shortfall == ()


def test_canonicalise_records_a_rejected_optional_rather_than_dropping_it():
    ds = _with_optionals()
    ds["bcaod550"].attrs["units"] = "kg m**-2"
    out = cams.canonicalise(cams.flatten_time(ds))
    assert "bcaod550" not in out.resolved
    bad = next(s for s in out.shortfall if s.canonical == "bcaod550")
    assert bad.kind == "rejected" and "kg m**-2" in bad.detail


def test_strict_optional_turns_a_shortfall_into_a_hard_failure():
    with pytest.raises(cams.CamsContractError, match="strict_optional"):
        cams.canonicalise(cams.flatten_time(_raw_like_ads()), strict_optional=True)


def test_canonicalise_renames_short_coordinate_names():
    ds = cams.flatten_time(_raw_like_ads()).rename(latitude="lat", longitude="lon")
    out = cams.canonicalise(ds)
    assert "latitude" in out.ds.coords and "lat" not in out.ds.coords


# ------------------------------------------------- per-arm expectations (0.4)


def test_expected_vars_splits_the_segmented_analysis_arms():
    aerosol, wind, *_ = cams.analysis_requests(_axis())
    assert cams.expected_vars(aerosol) == (("omaod550",), ("bcaod550", "aod550"))
    assert cams.expected_vars(wind) == (("u10", "v10"), ())
    assert cams.expected_vars(cams.forecast_request(_axis())) == (
        ("omaod550", "u10", "v10"), ("bcaod550", "aod550"))


def test_a_wind_only_arm_canonicalises_without_demanding_the_smoke_field():
    """Analysis is segmented by variable, so the wind arm legitimately carries
    no omaod550. The default REQUIRED set would reject it."""
    wind_req = cams.analysis_requests(_axis())[1]
    required, optional = cams.expected_vars(wind_req)
    ds = cams.flatten_time(_raw_like_ads()).drop_vars("omaod550")
    out = cams.canonicalise(ds, required=required, optional=optional)
    assert set(out.resolved) == {"u10", "v10"}
    assert out.shortfall == ()


def test_an_aerosol_only_arm_reports_the_two_optionals_as_shortfall():
    aerosol_req = cams.analysis_requests(_axis())[0]
    required, optional = cams.expected_vars(aerosol_req)
    ds = cams.flatten_time(_raw_like_ads()).drop_vars(["u10", "v10"])
    out = cams.canonicalise(ds, required=required, optional=optional)
    assert set(out.resolved) == {"omaod550"}
    assert {s.canonical for s in out.shortfall} == {"bcaod550", "aod550"}


def test_expected_vars_rejects_an_unmapped_ads_name():
    with pytest.raises(cams.CamsContractError, match="no canonical mapping"):
        cams.expected_vars({"variable": ["dust_aerosol_optical_depth_550nm"]})


def test_a_repeated_request_variable_is_not_an_injectivity_failure():
    """A request listing the same ADS name twice is legal. Without the
    order-preserving de-duplication the injectivity check fires against the
    name itself: "'u10' and 'u10' both resolved to 'u10'"."""
    req = {"variable": ["10m_u_component_of_wind", "10m_u_component_of_wind",
                        "10m_v_component_of_wind"]}
    assert cams.expected_vars(req) == (("u10", "v10"), ())
    resolved = cams.assert_variables(_raw_like_ads(), required=("u10", "u10", "v10"))
    assert set(resolved) == {"u10", "v10"}


def test_an_optional_name_repeated_in_required_is_not_reported_twice():
    out = cams.canonicalise(cams.flatten_time(_with_optionals()),
                            required=("omaod550", "u10", "v10", "aod550"),
                            optional=("aod550", "aod550", "bcaod550"))
    assert out.resolved["aod550"].delivered == "aod550"
    assert {s.canonical for s in out.shortfall} == set()


# ---------------------------------------------------- claimed-candidate honesty


def test_a_candidate_taken_by_another_name_is_reported_as_claimed_not_missing():
    """One delivered field whose long_name answers to both 'organic' and
    'total aerosol'. omaod550 claims it first; aod550 must not be told that
    "no delivered variable matches" -- one did, it was taken."""
    ds = _raw_like_ads().drop_vars("omaod550")
    ds["aerosol_optical_depth"] = (
        DIMS, np.zeros((4, 3, 5, 7), "float32"),
        {"units": "~", "long_name": "organic matter and total aerosol optical depth"})
    out = cams.canonicalise(cams.flatten_time(ds))
    assert out.resolved["omaod550"].delivered == "aerosol_optical_depth"
    bad = next(s for s in out.shortfall if s.canonical == "aod550")
    assert bad.kind == "claimed"
    assert "aerosol_optical_depth" in bad.detail and "'omaod550'" in bad.detail
    assert "no delivered variable matches" not in bad.detail
    # bcaod550 genuinely matched nothing and keeps the honest kind
    assert next(s for s in out.shortfall
                if s.canonical == "bcaod550").kind == "unresolved"


# ------------------------------------------------------------------ config pins


def test_the_config_tables_cover_exactly_the_same_canonical_names():
    """_check_one indexes UNIT_CLASS by canonical name; a name present in
    VAR_RESOLUTION but absent from UNIT_CLASS would raise a bare KeyError
    instead of a CamsContractError."""
    canon = set(config.VAR_RESOLUTION)
    assert set(config.UNIT_CLASS) == canon
    assert set(config.REQUIRED_VARS) | set(config.OPTIONAL_VARS) == canon
    assert not set(config.REQUIRED_VARS) & set(config.OPTIONAL_VARS)
    assert set(config.ADS_TO_CANONICAL.values()) == canon
    assert set(config.ADS_TO_CANONICAL) == set(config.AEROSOL_VARS) | set(config.WIND_VARS)
    assert set(config.UNIT_CLASS.values()) <= set(config.UNIT_CLASS_MEMBERS)


# ------------------------------------------------------------------ real data


@pytest.mark.skipif(not GREECE_NC.exists(), reason="cached Greece delivery absent")
def test_the_cached_greece_delivery_satisfies_the_contract():
    raw = xr.open_dataset(GREECE_NC)
    resolved = cams.assert_variables(raw)
    assert {c: r.delivered for c, r in resolved.items()} == {
        "omaod550": "omaod550", "u10": "u10", "v10": "v10"}
    assert [r.tier for r in resolved.values()] == ["exact", "exact", "exact"]
    assert resolved["omaod550"].units == "~"
    assert resolved["u10"].units == "m s**-1"

    out = cams.canonicalise(cams.flatten_time(raw))
    assert out.ds.sizes["time"] == 40
    assert {s.canonical for s in out.shortfall} == {"bcaod550", "aod550"}


# ------------------------------------------------------------------ unchanged


def test_flatten_time_produces_one_ascending_gapless_axis():
    ds = cams.flatten_time(_raw_like_ads())
    assert ds.sizes["time"] == 12
    t = ds["time"].values
    assert (np.diff(t) > np.timedelta64(0)).all()
    assert set(np.unique(np.diff(t)).tolist()) == {np.timedelta64(3, "h").astype("timedelta64[ns]").tolist()}


def test_flatten_time_drops_valid_time_and_keeps_lat_descending():
    ds = cams.flatten_time(_raw_like_ads())
    assert "valid_time" not in ds.coords
    lat = ds["latitude"].values
    assert (np.diff(lat) < 0).all()


def test_assert_grid_accepts_the_real_lattice_with_float_error():
    lon, lat = config.grid_axes()
    lon = lon + 8.5e-14
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    cams.assert_grid(ds)


def test_assert_grid_rejects_an_off_lattice_axis():
    lon, lat = config.grid_axes()
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon + 0.13})
    with pytest.raises(AssertionError, match="lattice"):
        cams.assert_grid(ds)


def test_assert_grid_rejects_the_wrong_shape():
    lon, lat = config.grid_axes()
    ds = xr.Dataset(coords={"latitude": lat[:-1], "longitude": lon})
    with pytest.raises(AssertionError, match="grid"):
        cams.assert_grid(ds)


def test_analysis_requests_are_segmented_by_date_and_variable():
    reqs = cams.analysis_requests(_axis())
    assert len(reqs) == 4
    assert all(r["type"] == ["analysis"] for r in reqs)
    assert all(r["leadtime_hour"] == ["0"] for r in reqs)
    assert all(tuple(r["area"]) == config.AREA for r in reqs)
    newest = [r for r in reqs if r["date"] == ["2026-08-04/2026-08-04"]]
    assert len(newest) == 2 and all(r["time"] == ["00:00"] for r in newest)


def test_forecast_request_uses_three_hourly_leads_to_120():
    r = cams.forecast_request(_axis())
    assert r["type"] == ["forecast"]
    assert r["time"] == ["00:00"]
    assert r["leadtime_hour"][:3] == ["0", "3", "6"]
    assert r["leadtime_hour"][-1] == "120"
    assert len(r["leadtime_hour"]) == 41


def _fc_axis(day=4, run_hour="00:00"):
    from tools.europe_smoke.probe import Axis
    d_an = dt.date(2026, 8, day)
    hh = int(run_hour.split(":")[0])
    return Axis(
        d_an=d_an, d_prev=d_an - dt.timedelta(days=1),
        start=d_an - dt.timedelta(days=10),
        h_win=["00:00", "06:00", "12:00", "18:00"], h_new=["00:00"],
        t_now=dt.datetime.combine(d_an, dt.time(0, 0)), run_date=d_an,
        run_hour=run_hour, t_init=dt.datetime.combine(d_an, dt.time(hh, 0)),
    )


def _fc_ds(times):
    t = np.array(times, dtype="datetime64[ns]")
    return xr.Dataset({"omaod550": (("time",), np.zeros(t.size))},
                      coords={"time": t})


def test_requested_forecast_times_is_t_init_plus_every_lead():
    axis = _fc_axis()
    req = cams.forecast_request(axis)
    times = cams.requested_forecast_times(axis, req)
    assert len(times) == 41                            # spec §0.4 R2: 0,3,...,120
    assert times[0] == dt.datetime(2026, 8, 4, 0, 0)
    assert times[1] == dt.datetime(2026, 8, 4, 3, 0)
    assert times[-1] == dt.datetime(2026, 8, 9, 0, 0)   # +120 h


def test_assert_axis_accepts_a_matching_forecast_delivery():
    axis = _fc_axis()
    expected = cams.requested_forecast_times(axis, cams.forecast_request(axis))
    cams.assert_axis(_fc_ds(expected), np.array(expected, dtype="datetime64[ns]"))


def test_assert_axis_catches_a_stale_forecast_delivery():
    """Yesterday's forecast.zip: same request shape, wrong initialisation.

    Note this covers the HELPER only -- that the assertion is actually wired
    into fetch.run is test_run_stops_on_a_forecast_arm_from_an_older_run.
    """
    today, yesterday = _fc_axis(4), _fc_axis(3)
    expected = cams.requested_forecast_times(today, cams.forecast_request(today))
    stale = _fc_ds(cams.requested_forecast_times(
        yesterday, cams.forecast_request(yesterday)))
    with pytest.raises(AssertionError, match="delivered axis != requested axis"):
        cams.assert_axis(stale, np.array(expected, dtype="datetime64[ns]"))
