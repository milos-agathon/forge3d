import datetime as dt
import json

import numpy as np
import pytest
import xarray as xr

from tools.europe_smoke import (advection, basemap, build, config, derive,
                                population, probe, provenance)

AXIS = probe.Axis(
    d_an=dt.date(2026, 8, 4), d_prev=dt.date(2026, 8, 3), start=dt.date(2026, 7, 25),
    h_win=["00:00", "06:00", "12:00", "18:00"], h_new=["00:00"],
    t_now=dt.datetime(2026, 8, 4, 0, 0), run_date=dt.date(2026, 8, 4),
    run_hour="00:00", t_init=dt.datetime(2026, 8, 4, 0, 0),
)



def _ok(name, detail="verified"):
    """A passing Finding in the five-field shape the provenance fix introduced.

    MERGE NOTE: the original fixtures said `provenance.Finding(name, True,
    detail)`. Finding is now (name, status, level, detail, sha256) with `ok`
    and `blocking` derived from `level`, so the three-positional form raises
    TypeError. `_finding` is the module helper that fills `level` from the
    status through the one severity table every consumer reads.
    """
    return provenance._finding(name, provenance.VERIFIED, detail)


def _bad(name, detail="missing"):
    return provenance._finding(name, provenance.MISSING, detail)


# --------------------------------------------------------------- fixtures


def _cube(wind_ms=4.0, nlat=None, off_lattice=False):
    """A cube that genuinely advects, so fit_gain has something to find.

    A field of independent noise per step makes every gain score identically
    (the fit then returns k=0 and the containment gate becomes vacuous). Here
    each step is one smooth field displaced by the constant wind, so the
    correlation peaks at k=1 and gate 11(b) is actually exercised.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    from tools.europe_smoke import advection as adv

    lon, lat = config.grid_axes()
    if nlat is not None:
        lat = lat[:nlat]
    if off_lattice:
        lon = lon + 0.13
    times = [np.datetime64(t) for t in AXIS.requested_analysis_times()]
    times += [np.datetime64(AXIS.t_now) + np.timedelta64(h, "h")
              for h in range(3, 73, 3)]
    t = np.array(times, dtype="datetime64[ns]")
    rng = np.random.default_rng(3)
    base = gaussian_filter(rng.random((lat.size, lon.size)), 5.0) * 4.0
    hours = (t - t[0]).astype("timedelta64[m]").astype("float64") / 60.0
    frames = np.empty((t.size, lat.size, lon.size), dtype="float32")
    rr = np.arange(lat.size)[:, None]
    cc = np.arange(lon.size)[None, :]
    for n, h in enumerate(hours):
        dlon, dlat = adv.degree_offsets(
            np.full((lat.size, lon.size), wind_ms),
            np.full((lat.size, lon.size), -wind_ms * 0.3), lat, float(h))
        r = np.broadcast_to(rr - dlat / config.LATTICE, base.shape)
        c = np.broadcast_to(cc + dlon / config.LATTICE, base.shape)
        frames[n] = map_coordinates(base, [r, c], order=1, mode="grid-wrap")
    smooth = frames
    shape = (t.size, lat.size, lon.size)
    ds = xr.Dataset(
        {
            "omaod550": (("time", "latitude", "longitude"), smooth),
            "bcaod550": (("time", "latitude", "longitude"), smooth * 0.1),
            "aod550": (("time", "latitude", "longitude"), smooth * 2.0),
            "u10": (("time", "latitude", "longitude"),
                    np.full(shape, wind_ms, dtype="float32")),
            "v10": (("time", "latitude", "longitude"),
                    np.full(shape, -wind_ms * 0.3, dtype="float32")),
        },
        coords={"time": t, "latitude": lat, "longitude": lon,
                "arm": ("time", np.array(["an"] * 41 + ["fc"] * 24, dtype="<U2"))},
    )
    return ds


def _fake_population(monkeypatch):
    nlat, nlon = config.GRID_SHAPE
    rng = np.random.default_rng(11)
    grid = rng.integers(0, 700, size=(nlat, nlon)).astype("float64")
    grid[120, 173] = 22_693_914.0
    vals, keys = [], []
    for r in range(nlat):
        for c in range(nlon):
            if grid[r, c] > 0:
                vals.append(grid[r, c])
                keys.append(("AAA" if c % 2 else "BBB", r, c))
    arr = np.array(vals)
    rounded = population.largest_remainder(arr, total=int(round(arr.sum())))
    rows = [(k[0], k[1], k[2], int(n)) for k, n in zip(keys, rounded) if n > 0]
    stats = {"countries": ["AAA", "BBB"], "n_countries": 2,
             "total_pixels": 1, "unassigned_pixels_before_fill": 3,
             "unassigned_people_before_fill": 5.0,
             "unassigned_pixels_after_fill": 0}
    monkeypatch.setattr(derive.population, "aggregate",
                        lambda lon, lat, return_raw_sum=False:
                        (grid.copy(), float(grid.sum())) if return_raw_sum
                        else grid.copy())
    monkeypatch.setattr(derive.population, "country_cell_table",
                        lambda cache_dir, return_stats=False:
                        (list(rows), dict(stats)) if return_stats else list(rows))


class _Projector:
    """A 16:9 basemap widened past the display window, as §6.6 asks for."""
    frame_rect = (0.0, 0.0, 1.0, 1.0)

    def __init__(self, lon_span=(-55.874, 75.874), lat_span=(20.0, 78.0)):
        self.map_bounds_mercator = (
            build._merc_x(lon_span[0]), build._merc_y(lat_span[0]),
            build._merc_x(lon_span[1]), build._merc_y(lat_span[1]))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Everything network- or GPU-bound replaced; everything else runs for real."""
    from PIL import Image

    from tools.europe_smoke import fetch as fetch_mod

    _fake_population(monkeypatch)
    monkeypatch.setattr(provenance, "check_all",
                        lambda **kwargs: (True, [_ok("engine", "verified")]))
    monkeypatch.setattr(build.provenance, "check_all",
                        lambda **kwargs: (True, [_ok("engine", "verified")]))

    state = {"fetch": 0, "basemap": 0, "cube": _cube(), "aspect": (1920, 1080)}

    def fake_fetch_run(days=10, dry_run=False, today=None, axis=None,
                       write_report=True, report_path=None):
        state["fetch"] += 1
        nc = tmp_path / "cams_merged.nc"
        state["cube"].to_netcdf(nc)
        csv = tmp_path / "firms.csv"
        csv.write_text("latitude,longitude\n38.1,23.7\n", encoding="utf-8")
        bnd = tmp_path / "boundary.geojson"
        bnd.write_text("{}", encoding="utf-8")
        return {
            "cams_nc": str(nc), "firms_csv": str(csv), "boundary": str(bnd),
            "n_steps": int(state["cube"].sizes["time"]),
            "n_analysis_expected": 41, "n_forecast_expected": 24,
            "merge": {"n_merged": 65, "overlap": []},
            "requests": [{"type": ["analysis"]}] * 5,
            "seam": {"verdict": "PASS", "r_seam": 0.004, "p95": 0.03,
                     "n_reference": 63, "caveat": "power ~0.48"},
            "firms_total": 128_431, "firms_map_records": 79_940,
            "firms_window": ["2026-07-25", "2026-08-04"],
            "firms_sources": list(config.FIRMS_SOURCES),
        }

    def fake_render(width, height, cache_dir, dem_zoom=7, max_dem_size=4000,
                    force=False):
        state["basemap"] += 1
        if state.get("basemap_raises"):
            raise RuntimeError("DEM tile fetch failed")
        return Image.new("RGB", (1920, 1080), (14, 22, 34)), _Projector()

    monkeypatch.setattr(fetch_mod, "run", fake_fetch_run)
    monkeypatch.setattr(basemap, "render", fake_render)
    monkeypatch.setattr(build.basemap, "render", fake_render)
    return tmp_path, state


def _build(tmp_path, **kw):
    kw.setdefault("width", 1920)
    kw.setdefault("height", 1080)
    return build.build(axis=AXIS, build_dir=tmp_path,
                       report_path=tmp_path / "build_report.json", **kw)


# ------------------------------------------------------- one whole report


def test_one_command_writes_one_report_carrying_every_required_gate(harness):
    tmp, _ = harness
    data = _build(tmp)
    path = tmp / "build_report.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk == json.loads(json.dumps(data, default=str))
    assert set(on_disk["gates"]) == {"5", "6", "7", "8", "9", "10", "11", "16"}
    for n, g in on_disk["gates"].items():
        assert g.get("verdict") in {"PASS", "WARN", "N/A"}, (n, g)


def test_the_report_carries_everything_the_finding_names(harness):
    tmp, _ = harness
    d = _build(tmp)
    assert d["advection"]["fit"]["k"] <= config.MAX_ADVECTION_GAIN
    assert d["axis"]["t_now"] == "2026-08-04T00:00:00"
    assert d["axis"]["h_win"] == ["00:00", "06:00", "12:00", "18:00"]
    assert d["firms"]["window"] == ["2026-07-25", "2026-08-04"]
    assert d["firms"]["detections_total"] == 128_431
    assert d["firms"]["map_records_after_thinning"] == 79_940
    assert d["population"]["display_block_people"] > 0
    assert d["population"]["domain_people"] > d["population"]["display_block_people"]
    assert d["basemap"]["webp"]["86"]["bytes"] > 0
    assert d["basemap"]["webp"]["90"]["bytes"] > 0
    assert d["verdict"] == "PASS"


def test_the_report_names_the_printable_population_figure(harness):
    tmp, _ = harness
    d = _build(tmp)
    p = d["population"]["printable"]
    assert p["array"] == "population_display"
    assert p["people"] == d["population"]["display_block_people"]
    assert p["people"] != d["population"]["domain_people"]


def test_package_versions_are_recorded(harness):
    tmp, _ = harness
    d = _build(tmp)
    assert d["package_versions"]["numpy"]
    assert d["gates"]["16"]["package_versions"]["rasterio"]


# ------------------------------------------------------------ resumability


def test_second_build_reruns_nothing(harness):
    tmp, state = harness
    _build(tmp)
    assert (state["fetch"], state["basemap"]) == (1, 1)
    d = _build(tmp)
    assert (state["fetch"], state["basemap"]) == (1, 1)
    assert d["stages"]["fetch"]["skipped"] is True
    assert d["stages"]["derive"]["skipped"] is True
    assert d["stages"]["basemap"]["skipped"] is True


def test_a_failed_basemap_does_not_force_a_refetch(harness):
    """The whole point of staging: the expensive fragile step runs last."""
    tmp, state = harness
    _build(tmp)
    assert state["fetch"] == 1
    state["basemap_raises"] = True
    with pytest.raises(RuntimeError, match="DEM tile fetch failed"):
        _build(tmp, force=("basemap",))
    state["basemap_raises"] = False
    _build(tmp, force=("basemap",))
    assert state["fetch"] == 1, "fetch must not have rerun"
    assert state["basemap"] == 3


def test_a_resumed_build_reports_the_same_numbers_as_a_cold_one(harness):
    """JSON has no float keys, so an un-round-tripped stage would disagree."""
    tmp, _ = harness
    cold = _build(tmp)
    warm = _build(tmp)
    volatile = {"started_utc", "completed_utc", "seconds", "stages", "axis"}
    for key in set(cold) - volatile:
        assert cold[key] == warm[key], key
    assert cold["advection"]["fit"]["scores"] == warm["advection"]["fit"]["scores"]


def test_a_new_axis_reruns_fetch_but_not_derive(harness):
    tmp, state = harness
    _build(tmp)
    later = probe.Axis(
        d_an=dt.date(2026, 8, 5), d_prev=dt.date(2026, 8, 4),
        start=dt.date(2026, 7, 26), h_win=AXIS.h_win, h_new=AXIS.h_new,
        t_now=dt.datetime(2026, 8, 5, 0, 0), run_date=dt.date(2026, 8, 5),
        run_hour="00:00", t_init=dt.datetime(2026, 8, 5, 0, 0))
    d = build.build(axis=later, build_dir=tmp, width=1920, height=1080,
                    report_path=tmp / "build_report.json")
    assert state["fetch"] == 2, "a new analysis day must refetch"
    assert d["stages"]["derive"]["skipped"] is True, (
        "population does not move with the CAMS clock")
    assert d["stages"]["basemap"]["skipped"] is True


def test_a_deleted_cams_file_reruns_fetch(harness):
    tmp, state = harness
    _build(tmp)
    (tmp / "cams_merged.nc").unlink()
    _build(tmp)
    assert state["fetch"] == 2


def test_force_all_reruns_every_stage(harness):
    tmp, state = harness
    _build(tmp)
    _build(tmp, force=("all",))
    assert (state["fetch"], state["basemap"]) == (2, 2)


# ------------------------------------------------------------------ gates


def test_gate5_measures_the_lattice_rather_than_trusting_fetch(harness):
    tmp, _ = harness
    g = _build(tmp)["gates"]["5"]
    assert g["verdict"] == "PASS"
    assert g["checks"]["grid_shape"]["value"] == list(config.GRID_SHAPE)
    assert g["checks"]["lattice_residual"]["value"]["longitude"] <= config.LATTICE_TOL
    assert g["checks"]["analysis_axis"]["n_delivered"] == 41
    assert not g["checks"]["variables_resolved"]["missing"]


def test_gate5_fails_on_an_off_lattice_axis(harness):
    tmp, state = harness
    state["cube"] = _cube(off_lattice=True)
    d = _build(tmp)
    assert d["gates"]["5"]["verdict"] == "FAIL"
    assert d["verdict"] == "FAIL"
    assert any("gate 5" in f for f in d["failures"])


def test_gate5_fails_on_the_wrong_grid_shape(harness):
    tmp, state = harness
    state["cube"] = _cube(nlat=135)
    d = _build(tmp)
    assert d["gates"]["5"]["checks"]["grid_shape"]["ok"] is False
    assert d["verdict"] == "FAIL"


def test_gate5_fails_when_a_variable_never_resolved(harness):
    tmp, state = harness
    state["cube"] = _cube().drop_vars("bcaod550")
    d = _build(tmp)
    assert d["gates"]["5"]["checks"]["variables_resolved"]["missing"] == ["bcaod550"]
    assert d["gates"]["16"]["verdict"] == "FAIL"


def test_gate11a_is_exact_arithmetic_and_passes_on_the_shipped_area():
    g = advection.static_containment()
    assert g["verdict"] == "PASS"
    assert g["min_slack_deg"] == pytest.approx(0.2, abs=1e-9)
    for edge in ("west", "east", "south", "north"):
        assert g["edges"][edge]["ok"]


def test_gate11a_fails_if_the_area_is_shrunk(monkeypatch):
    monkeypatch.setattr(config, "AREA", (76.0, -36.0, 26.0, 56.0))
    g = advection.static_containment()
    assert g["verdict"] == "FAIL"
    assert g["edges"]["west"]["slack_deg"] < 0


def test_gate11b_reports_a_fraction_not_a_column_count(harness):
    """Row-only weight normalisation inflated this by the 175 display columns."""
    tmp, state = harness
    state["cube"] = _cube(wind_ms=60.0)
    b = _build(tmp)["gates"]["11"]["b_empirical"]
    assert 0.0 <= b["frac_over_D"] <= 1.0
    assert b["frac_over_0.8D"] == pytest.approx(1.0), (
        "every sample exceeds 0.8 D here, so the fraction must saturate at 1.0")


def test_gate11b_is_zero_when_nothing_exceeds(harness):
    tmp, state = harness
    state["cube"] = _cube(wind_ms=2.0)
    b = _build(tmp)["gates"]["11"]["b_empirical"]
    assert b["frac_over_D"] == 0.0 and b["frac_over_0.8D"] == 0.0
    assert b["pass"] is True


def test_gate11b_failure_fails_the_build_and_prints_the_escalation(harness, capsys):
    tmp, state = harness
    state["cube"] = _cube(wind_ms=40.0)
    d = _build(tmp)
    assert d["gates"]["11"]["b_empirical"]["pass"] is False
    assert d["gates"]["11"]["verdict"] == "FAIL"
    assert d["verdict"] == "FAIL"
    out = capsys.readouterr().out
    assert "WIDEN THE AREA" in out and "Never shrink D" in out
    assert str(build.ESCALATION_AREA) in out
    assert str(build.ESCALATION_GRID_SHAPE) in out


def test_the_escalation_area_is_the_spec_widening_not_a_shrink():
    north, west, south, east = build.ESCALATION_AREA
    n0, w0, s0, e0 = config.AREA
    assert north > n0 and east > e0 and west < w0 and south < s0
    nlon = round((east - west) / config.LATTICE) + 1
    nlat = round((north - south) / config.LATTICE) + 1
    assert (nlat, nlon) == build.ESCALATION_GRID_SHAPE == (142, 259)
    for edge in build.ESCALATION_AREA:
        assert config.on_lattice(edge), edge


def test_gate9_runs_on_the_delivered_dem_geometry_and_names_its_method(harness):
    tmp, _ = harness
    g = _build(tmp)["gates"]["9"]
    assert g["verdict"] == "PASS"
    assert g["corrected"]["rel_diff"] < g["tolerance"]
    assert g["control_has_teeth"] is True
    assert g["uncorrected"]["rel_diff"] > 10 * g["corrected"]["rel_diff"]
    assert "not the Atlas-vs-Scandes" in g["method"]
    assert g["probe"]["resolved"] is True


def test_gate9_declines_rather_than_false_failing_on_a_short_raster(harness,
                                                                   monkeypatch):
    """A band shorter than one ridge period cannot compare end to end."""
    tmp, _ = harness
    from PIL import Image
    monkeypatch.setattr(build.basemap, "render",
                        lambda *a, **k: (Image.new("RGB", (320, 180)), _Projector()))
    g = _build(tmp, force=("basemap",))["gates"]["9"]
    assert g["verdict"] == "N/A"
    assert g["probe"]["resolved"] is False
    assert "rows" in g["reason"]


def test_gate10_measures_the_rail_ceiling_on_a_widened_16_9_basemap(harness):
    """§6.6's own span gives 449.9 px, so a hardcoded 450 would fail itself."""
    tmp, _ = harness
    g = _build(tmp)["gates"]["10"]
    assert g["verdict"] == "PASS"
    assert g["frame_rect_contained"] is True
    assert g["covers_display_window"] is True
    assert g["rail_ceiling_px_at_1920"] == pytest.approx(449.9, abs=0.5)
    assert g["rail_ceiling_px_at_1920"] < build.SPEC_RAIL_REFERENCE_PX
    assert "measured ceiling" in g["note"]


def test_gate10_fails_when_the_basemap_is_flush_with_the_display_window(harness,
                                                                       monkeypatch):
    tmp, _ = harness
    from PIL import Image
    flush = _Projector(lon_span=(-25.0, 45.0))
    monkeypatch.setattr(build.basemap, "render",
                        lambda *a, **k: (Image.new("RGB", (1920, 1080)), flush))
    d = _build(tmp, force=("basemap",))
    assert d["gates"]["10"]["verdict"] == "FAIL"
    assert "flush" in d["gates"]["10"]["reason"]


def test_gate10_is_na_not_pass_at_the_data_aspect(harness):
    """4000x4241 is the domain aspect; §6.6's rails only exist when widened."""
    tmp, _ = harness
    g = _build(tmp, width=4000, height=4241)["gates"]["10"]
    assert g["verdict"] == "N/A"
    assert "data aspect" in g["reason"]


def test_gate10_fails_when_the_basemap_misses_the_display_window(harness, monkeypatch):
    tmp, _ = harness
    from PIL import Image
    narrow = _Projector(lon_span=(-20.0, 40.0))
    monkeypatch.setattr(build.basemap, "render",
                        lambda *a, **k: (Image.new("RGB", (1920, 1080)), narrow))
    d = _build(tmp, force=("basemap",))
    assert d["gates"]["10"]["verdict"] == "FAIL"
    assert d["verdict"] == "FAIL"


def test_gate16_fails_the_build_before_any_queue_time_is_spent(harness, monkeypatch):
    tmp, state = harness
    monkeypatch.setattr(build.provenance, "check_all",
                        lambda **kwargs: (False, [_bad("ghsl", "missing: x")]))
    d = _build(tmp)
    assert d["verdict"] == "FAIL"
    assert d["gates"]["16"]["verdict"] == "FAIL"
    assert state["fetch"] == 0, "provenance must fail before fetching"


def test_a_missing_natural_earth_does_not_block_the_bootstrap(harness, monkeypatch):
    """The build downloads NE itself; blocking on it makes it unable to start."""
    tmp, state = harness
    calls = {"n": 0}

    def staged_check(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:                      # stage 0, before derive
            return False, [_bad("natural_earth", "missing: ne.zip"),
                           _ok("engine", "verified")]
        return True, [_ok("natural_earth", "verified"),
                      _ok("engine", "verified")]

    monkeypatch.setattr(build.provenance, "check_all", staged_check)
    d = _build(tmp)
    assert state["fetch"] == 1, "a derivable artifact must not block the build"
    assert d["provenance_deferred"] == ["natural_earth"]
    assert d["gates"]["16"]["verdict"] == "PASS"
    assert d["verdict"] == "PASS"


def test_a_derivable_artifact_still_missing_at_the_end_fails_gate16(harness,
                                                                   monkeypatch):
    tmp, _ = harness
    monkeypatch.setattr(
        build.provenance, "check_all",
        lambda **kwargs: (False, [_bad("natural_earth", "missing")]))
    d = _build(tmp)
    assert d["gates"]["16"]["verdict"] == "FAIL"
    assert d["verdict"] == "FAIL"


def test_gate16_rejects_an_unpinned_artifact():
    finding = provenance.Finding(
        "natural_earth", provenance.UNPINNED, provenance.WARN,
        "on disk but manifest carries no hash", "a" * 64,
    )
    gate5 = {"checks": {"variables_resolved": {"ok": True, "missing": []}}}
    fit = {"k": 1.0, "k_unclamped": 1.0, "clamped": False}
    out = build.gate16([finding], fit, gate5, AXIS, {"window": "10 days"}, {})
    assert out["verdict"] == "FAIL"


def test_a_mismatched_derivable_artifact_blocks_before_fetch(harness, monkeypatch):
    tmp, state = harness
    bad = provenance._finding(
        "natural_earth", provenance.MISMATCH, "sha256 mismatch"
    )
    monkeypatch.setattr(
        build.provenance, "check_all", lambda **kwargs: (False, [bad])
    )
    out = _build(tmp)
    assert out["verdict"] == "FAIL"
    assert state["fetch"] == 0


def test_gate16_records_the_clamp_when_the_gain_fit_saturates(harness, monkeypatch):
    tmp, _ = harness
    monkeypatch.setattr(build.advection, "fit_gain",
                        lambda ds, **kw: advection.fit_gain_from_scores(
                            {0.0: 0.1, 1.5: 0.8, 3.0: 0.9},
                            kmax=config.MAX_ADVECTION_GAIN))
    d = _build(tmp)
    assert d["advection"]["fit"]["k"] == 1.5
    assert d["advection"]["fit"]["clamped"] is True
    assert d["gates"]["16"]["advection_gain"]["ok"] is True
    assert d["gates"]["16"]["verdict"] == "PASS"


# ---------------------------------------------------------------- skipping


def test_skip_basemap_marks_gates_9_and_10_not_run_rather_than_passed(harness):
    tmp, state = harness
    d = _build(tmp, skip_basemap=True)
    assert state["basemap"] == 0
    assert d["gates"]["9"]["verdict"] == "NOT RUN"
    assert d["gates"]["10"]["verdict"] == "NOT RUN"
    assert d["verdict"] == "INCOMPLETE"
    assert set(d["gates_not_run"]) == {"9", "10"}


def test_recorded_shortfalls_are_recorded_not_fatal(harness):
    tmp, _ = harness
    d = _build(tmp)
    # h_new carries one hour, not four (§0.4 records this, never fails on it)
    assert any("analysis hours" in s for s in d["axis"]["recorded_shortfalls"])
    assert d["verdict"] == "PASS"


def test_main_exit_code_tracks_the_verdict(monkeypatch):
    monkeypatch.setattr(build, "build", lambda **kw: {"verdict": "PASS"})
    assert build.main([]) == 0
    monkeypatch.setattr(build, "build", lambda **kw: {"verdict": "FAIL"})
    assert build.main([]) == 1
    monkeypatch.setattr(build, "build", lambda **kw: {"verdict": "INCOMPLETE"})
    assert build.main(["--skip-basemap"]) == 1


def test_main_parses_the_force_list_and_render_parameters(monkeypatch):
    seen = {}
    monkeypatch.setattr(build, "build",
                        lambda **kw: (seen.update(kw), {"verdict": "PASS"})[1])
    build.main(["--days", "7", "--force", "derive,basemap", "--dem-zoom", "6",
                "--webp-quality", "80,92", "--skip-basemap"])
    assert seen["days"] == 7
    assert seen["force"] == ("derive", "basemap")
    assert seen["dem_zoom"] == 6
    assert seen["webp_qualities"] == (80, 92)
    assert seen["skip_basemap"] is True


# ------------------------------------------------------ ADDED BY THE VERIFIER


def test_gate10_measures_the_canvas_not_the_map_when_frame_rect_insets():
    """The rail bound is in CANVAS pixels; the map may not fill the canvas.

    Every other gate-10 test uses frame_rect (0,0,1,1), so a map-relative
    formula reads identically and nothing catches it. Task 12 sets
    TERRAIN_PORTRAIT_WIDTH_FRAC / TERRAIN_FILL_FRAC precisely because the
    engine's frame_rect is not always full-bleed.
    """
    full = build.gate10(_Projector(), 1920, 1080)
    assert full["rail_ceiling_px_at_1920"] == pytest.approx(449.9353310866, abs=1e-6)

    class Inset(_Projector):
        frame_rect = (0.1, 0.0, 0.9, 1.0)

    inset = build.gate10(Inset(), 1920, 1080)
    # 0.1*1920 px of canvas gutter + 0.8 of the map's own 449.94 px margin
    assert inset["rail_ceiling_px_at_1920"] == pytest.approx(
        192.0 + 0.8 * 449.9353310866, abs=1e-6)
    assert inset["rail_ceiling_px_at_1920"] > full["rail_ceiling_px_at_1920"]


def test_the_report_lands_beside_the_artifacts_when_build_dir_moves(harness):
    """Every other test passes report_path, so the default was never exercised."""
    tmp, _ = harness
    d = build.build(axis=AXIS, build_dir=tmp, width=1920, height=1080)
    assert (tmp / "build_report.json").exists()
    assert json.loads((tmp / "build_report.json").read_text())["verdict"] == d["verdict"]


def test_gate9_does_not_claim_to_certify_the_delivered_pixels(harness):
    """install_hillshade_patch is defined once in the plan and called nowhere."""
    tmp, _ = harness
    g = _build(tmp)["gates"]["9"]
    assert "does NOT certify the pixels in basemap.png" in g["applies_to"]
    # the render is stubbed here, so the patch is certainly not installed;
    # None (engine never loaded) and False are both truthful, True is not
    assert g["engine_patch_installed"] is not True
