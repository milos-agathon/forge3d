# tools/europe_smoke/tests/test_fetch.py
#
# MERGE NOTE. Four fixes touch this file:
#   * the plan's two dry-run tests are kept verbatim;
#   * the plan's single `test_run_refuses_to_start_when_provenance_fails` is
#     replaced by three, because `Finding("ghsl", False, "missing")` no longer
#     type-checks against the five-field Finding;
#   * the request-digest cache block and the run()-level forecast-axis tests
#     come from the cache_zip fix, whose parametrised `_axis(day=...)` replaces
#     the plan's fixed one (day=4 reproduces it exactly);
#   * the write_report signature pin comes from the pipeline fix.
#
# `_grid_ds` gained u10/v10 and `units` attributes: run() now canonicalises each
# arm against the variables that arm asked for, so a fixture carrying only a
# unit-less omaod550 fails the §1.1 contract before it ever reaches the axis
# assertion the test is about. The teeth are unchanged -- deleting run()'s
# forecast `assert_axis` line still turns the stale test red.
import datetime as dt
import json
import stat
import zipfile

import pytest

from tools.europe_smoke import config, fetch, provenance

# fetch.run() opens with the full provenance gate (check_all over every
# manifest entry), so a dry run still needs every `class = "required"` artifact
# on disk. `required_artifacts` (conftest) skips these three with the manifest's
# own Finding when one is un-provisioned; a WRONG artifact is not absence and
# still fails here, which is what the two refusal tests below assert.


def test_dry_run_emits_every_request_without_touching_the_network(capsys,
                                                                  required_artifacts):
    plan = fetch.run(days=10, dry_run=True, today=dt.date(2026, 8, 4),
                     axis=_axis())
    # 4 analysis requests + 1 forecast request
    assert len(plan["requests"]) == 5
    assert all(tuple(r["area"]) == config.AREA for r in plan["requests"])
    assert plan["dry_run"] is True


def test_dry_run_reports_the_expected_step_count(required_artifacts):
    plan = fetch.run(days=10, dry_run=True, today=dt.date(2026, 8, 4), axis=_axis())
    # 41 analysis steps + forecast steps strictly after t_now, clipped to +72 h
    assert plan["n_analysis_expected"] == 41
    assert plan["n_forecast_expected"] == 24


def test_run_proceeds_with_a_pending_acquired_artifact(required_artifacts):
    plan = fetch.run(days=10, dry_run=True, today=dt.date(2026, 8, 4), axis=_axis())
    assert plan["dry_run"] is True   # natural_earth pending must NOT block


def test_run_refuses_to_start_when_provenance_fails(monkeypatch):
    monkeypatch.setattr(
        fetch.provenance, "check_all",
        lambda **k: (False, [fetch.provenance._finding(
            "ghsl", fetch.provenance.MISSING, "missing: D:/nope.tif")]))
    with pytest.raises(RuntimeError, match="provenance"):
        fetch.run(days=10, dry_run=True, today=dt.date(2026, 8, 4), axis=_axis())


def test_run_refuses_to_start_on_a_hash_mismatch(monkeypatch, tmp_path):
    p = tmp_path / "fake.tif"
    p.write_bytes(b"substituted")
    m = fetch.provenance.load_manifest()
    m["ghsl"].update(path=str(p), bytes=len(b"substituted"))
    monkeypatch.setattr(fetch.provenance, "load_manifest", lambda *a, **k: m)
    with pytest.raises(RuntimeError, match=r"refusing to fetch.*sha256"):
        fetch.run(days=10, dry_run=True, today=dt.date(2026, 8, 4), axis=_axis())


def test_write_report_defaults_to_true_so_standalone_use_is_unchanged():
    import inspect
    sig = inspect.signature(fetch.run)
    assert sig.parameters["write_report"].default is True
    assert sig.parameters["report_path"].default is None


# ==========================================================================
# §0.4: the requested axis moves every day. A filename like ``analysis_0.zip``
# is an ordinal, not a key, so tomorrow's build reads today's bytes.
# ==========================================================================


class FakeClient:
    """Stands in for cdsapi.Client; records what it was asked for."""

    def __init__(self, payload=b"PK-not-really", fail=False):
        self.calls = []
        self.payload = payload
        self.fail = fail

    def retrieve(self, dataset, request, target):
        from pathlib import Path
        self.calls.append((dataset, json.loads(fetch.canonical_request_json(request)),
                           str(target)))
        if self.fail:
            Path(target).write_bytes(self.payload[:4])   # a truncated download
            raise RuntimeError("ADS said no")
        Path(target).write_bytes(self.payload)


def test_canonical_json_ignores_key_order_but_not_list_order():
    a = {"variable": ["u10", "v10"], "date": ["2026-08-04/2026-08-04"]}
    b = {"date": ["2026-08-04/2026-08-04"], "variable": ["u10", "v10"]}
    c = {"variable": ["v10", "u10"], "date": ["2026-08-04/2026-08-04"]}
    assert fetch.canonical_request_json(a) == fetch.canonical_request_json(b)
    assert fetch.request_digest(a) == fetch.request_digest(b)
    assert fetch.request_digest(a) != fetch.request_digest(c)


def test_canonical_json_rejects_nan_and_canonicalises_negative_zero():
    assert fetch.canonical_request_json({"area": [-0.0]}) == '{"area":[0.0]}'
    with pytest.raises(ValueError, match="non-finite"):
        fetch.canonical_request_json({"area": [float("nan")]})


def test_cache_filename_carries_the_request_digest(tmp_path):
    axis = _axis()
    req = fetch.cams.analysis_requests(axis)[0]
    target = fetch.cache_target(req, tmp_path, "analysis_0")
    assert target.name == f"analysis_0_{fetch.request_digest(req)}.zip"
    assert len(fetch.request_digest(req)) == fetch.DIGEST_LEN


def test_tomorrows_axis_cannot_reuse_todays_cache_entry(tmp_path):
    """The finding, as an executable claim."""
    today = _axis()
    tomorrow = _axis(day=5)
    labels_today = [fetch.cache_target(r, tmp_path, f"analysis_{i}").name
                    for i, r in enumerate(fetch.cams.analysis_requests(today))]
    labels_tomorrow = [fetch.cache_target(r, tmp_path, f"analysis_{i}").name
                       for i, r in enumerate(fetch.cams.analysis_requests(tomorrow))]
    assert set(labels_today).isdisjoint(labels_tomorrow)
    # the forecast arm moves too -- it was the arm nothing asserted
    assert (fetch.cache_target(fetch.cams.forecast_request(today), tmp_path, "forecast")
            != fetch.cache_target(fetch.cams.forecast_request(tomorrow), tmp_path,
                                  "forecast"))


def test_a_same_day_hour_change_also_changes_the_key(tmp_path):
    # 00Z run in the morning, 00+12Z by the evening: same dates, new axis.
    morning = _axis()
    evening = _axis(h_new=["00:00", "12:00"])
    a = fetch.cache_target(fetch.cams.analysis_requests(morning)[2], tmp_path, "analysis_2")
    b = fetch.cache_target(fetch.cams.analysis_requests(evening)[2], tmp_path, "analysis_2")
    assert a != b


def test_retrieve_writes_a_sidecar_recording_the_request(tmp_path):
    req = fetch.cams.analysis_requests(_axis())[0]
    client = FakeClient()
    target = fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    side = fetch.read_sidecar(target)
    assert side["dataset"] == config.ADS_DATASET
    assert side["digest"] == fetch.request_digest(req)
    assert side["request"] == json.loads(fetch.canonical_request_json(req))
    assert side["sha256"] == provenance.sha256_file(target)
    assert side["bytes"] == target.stat().st_size
    assert dt.datetime.fromisoformat(side["retrieved_utc"]).tzinfo is not None


def test_retrieve_reuses_a_matching_cache_entry(tmp_path):
    req = fetch.cams.analysis_requests(_axis())[0]
    client = FakeClient()
    fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    assert len(client.calls) == 1


def test_retrieve_refetches_when_the_sidecar_is_missing(tmp_path):
    req = fetch.cams.analysis_requests(_axis())[0]
    client = FakeClient()
    target = fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    target.with_name(target.name + ".request.json").unlink()
    fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    assert len(client.calls) == 2


def test_retrieve_refetches_when_the_payload_drifted_from_its_sidecar(tmp_path):
    req = fetch.cams.analysis_requests(_axis())[0]
    client = FakeClient()
    target = fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    target.write_bytes(b"truncated")
    fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    assert len(client.calls) == 2
    assert fetch.read_sidecar(target)["sha256"] == provenance.sha256_file(target)


def test_retrieve_refuses_a_cache_entry_whose_sidecar_disagrees(tmp_path):
    """Digest collision, hand-edited sidecar, or a file copied into place."""
    req = fetch.cams.analysis_requests(_axis())[0]
    client = FakeClient()
    target = fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    side_path = target.with_name(target.name + ".request.json")
    side = json.loads(side_path.read_text())
    side["request"]["date"] = ["1999-01-01/1999-01-01"]
    side_path.write_text(json.dumps(side))
    with pytest.raises(RuntimeError, match="does not match"):
        fetch.retrieve(req, tmp_path, "analysis_0", client=client)


def test_a_failed_download_leaves_no_cache_entry(tmp_path):
    """A truncated download must not become tomorrow's cache hit."""
    req = fetch.cams.analysis_requests(_axis())[0]
    with pytest.raises(RuntimeError, match="ADS said no"):
        fetch.retrieve(req, tmp_path, "analysis_0", client=FakeClient(fail=True))
    target = fetch.cache_target(req, tmp_path, "analysis_0")
    assert not target.exists()
    assert fetch.read_sidecar(target) is None
    assert list(tmp_path.iterdir()) == []          # not even the .part file

    # and the next run really does fetch
    client = FakeClient()
    fetch.retrieve(req, tmp_path, "analysis_0", client=client)
    assert len(client.calls) == 1 and target.exists()


def test_open_rejects_a_malicious_cams_payload(tmp_path):
    z = tmp_path / "analysis_0_deadbeefdeadbeef.zip"
    with zipfile.ZipFile(z, "w") as zf:
        info = zipfile.ZipInfo("data.nc")
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        zf.writestr(info, "netcdf-ish")
        evil = zipfile.ZipInfo("x")
        evil.filename = "../../pwned.nc"
        evil.create_system = 3
        evil.external_attr = (stat.S_IFREG | 0o644) << 16
        zf.writestr(evil, "payload")
    with pytest.raises(provenance.UnsafeArchiveError, match=r"\.\."):
        fetch._open(z)
    outdir = tmp_path / "analysis_0_deadbeefdeadbeef"
    assert not outdir.exists() or list(outdir.rglob("*")) == []
    assert not (tmp_path / "pwned.nc").exists()


# ------------------------------------------ the forecast arm, at run() level
#
# Finding A's second half is not "assert_axis can spot a stale axis" -- it is
# that ``fetch.run`` never CALLED it on the forecast arm. Testing the helper in
# isolation would pass against the broken plan too, so drive run() itself.
# Measured: delete the forecast assert_axis line from run() and the first test
# below fails (run() proceeds into merge_arms with yesterday's axis) while
# every helper-level forecast test still passes.


class _ReachedMerge(Exception):
    """Sentinel: both axis assertions passed and run() moved on."""


def _grid_ds(times):
    import numpy as np
    import xarray as xr
    lon, lat = config.grid_axes()
    t = np.array(times, dtype="datetime64[ns]")
    data = np.zeros((t.size, lat.size, lon.size), dtype="float32")
    dims = ("time", "latitude", "longitude")
    return xr.Dataset(
        {"omaod550": (dims, data, {"units": "~",
                                   "long_name": "Organic Matter AOD at 550nm"}),
         "u10": (dims, data + 1.0, {"units": "m s**-1",
                                    "long_name": "10 metre U wind component"}),
         "v10": (dims, data - 1.0, {"units": "m s**-1",
                                    "long_name": "10 metre V wind component"})},
        coords={"time": t, "latitude": lat, "longitude": lon})


def _run_with_forecast(monkeypatch, tmp_path, axis, forecast_times):
    monkeypatch.setattr(fetch.provenance, "check_all",
                        lambda **k: (True, [fetch.provenance._finding(
                            "stub", fetch.provenance.VERIFIED, "test")]))
    monkeypatch.setattr(fetch, "retrieve",
                        lambda request, cache_dir, label, **kw: tmp_path / f"{label}.zip")
    monkeypatch.setattr(fetch, "record_delivery", lambda target, ds: {"stub": True})
    monkeypatch.setattr(fetch, "read_sidecar", lambda target: {"retrieved_utc": "stub"})
    # Serve the SEGMENTED delivery the real ADS produces (§0.4 R1a-d): one
    # payload per (variable group, day window). A single full dataset for every
    # label is exactly the unrealistic shape that masked the assembly defect
    # test_run_assembles_segmented_analysis_arms_without_holes pins.
    times = axis.requested_analysis_times()
    win = [t for t in times if t.date() <= axis.d_prev]
    new = [t for t in times if t.date() == axis.d_an]
    segments = {
        "analysis_0.zip": _segment_ds(win, ("omaod550",)),
        "analysis_1.zip": _segment_ds(win, ("u10", "v10")),
        "analysis_2.zip": _segment_ds(new, ("omaod550",)),
        "analysis_3.zip": _segment_ds(new, ("u10", "v10")),
        "forecast.zip": _grid_ds(forecast_times),
    }
    monkeypatch.setattr(fetch, "_open", lambda target: segments[target.name])

    def _stop(*a, **k):
        raise _ReachedMerge

    monkeypatch.setattr(fetch.cams, "merge_arms", _stop)
    return fetch.run(days=10, dry_run=False, today=dt.date(2026, 8, 4), axis=axis)


def test_run_stops_on_a_forecast_arm_from_an_older_run(monkeypatch, tmp_path):
    """Yesterday's forecast payload: right grid, right shape, wrong run."""
    today, yesterday = _axis(4), _axis(3)
    stale = fetch.cams.requested_forecast_times(
        yesterday, fetch.cams.forecast_request(yesterday))
    with pytest.raises(AssertionError, match="delivered axis != requested axis"):
        _run_with_forecast(monkeypatch, tmp_path, today, stale)


def test_run_accepts_a_forecast_arm_matching_this_runs_axis(monkeypatch, tmp_path):
    """Positive control: the test above must fail for the axis, not because
    this harness makes run() blow up on general principle."""
    today = _axis(4)
    good = fetch.cams.requested_forecast_times(
        today, fetch.cams.forecast_request(today))
    with pytest.raises(_ReachedMerge):
        _run_with_forecast(monkeypatch, tmp_path, today, good)


def _segment_ds(times, names):
    """One canonicalisable analysis segment carrying only ``names``."""
    import numpy as np
    import xarray as xr

    lon, lat = config.grid_axes()
    t = np.array(times, dtype="datetime64[ns]")
    data = np.zeros((t.size, lat.size, lon.size), dtype="float32")
    dims = ("time", "latitude", "longitude")
    meta = {
        "omaod550": ({"units": "~",
                      "long_name": "Organic Matter AOD at 550nm"}, 0.0),
        "u10": ({"units": "m s**-1",
                 "long_name": "10 metre U wind component"}, 1.0),
        "v10": ({"units": "m s**-1",
                 "long_name": "10 metre V wind component"}, -1.0),
    }
    return xr.Dataset(
        {n: (dims, data + meta[n][1], meta[n][0]) for n in names},
        coords={"time": t, "latitude": lat, "longitude": lon})


def test_run_assembles_segmented_analysis_arms_without_holes(monkeypatch, tmp_path):
    """The real ADS delivery is segmented by VARIABLE and by DAY WINDOW (§0.4
    R1a-d): four analysis payloads that partition (variable, time). The
    assembled analysis must carry every variable at every requested time.

    [measured on the 2026-08-05 delivery] ``xr.merge(compat="override",
    join="outer")`` takes each variable from the FIRST segment that carries it
    and reindexes it over the union time axis, so the newest-day segments are
    silently dropped to NaN and the lead-0 identity gate then fails with an
    all-non-finite comparison.
    """
    import numpy as np

    axis = _axis(4)
    monkeypatch.setattr(fetch.provenance, "check_all",
                        lambda **k: (True, [fetch.provenance._finding(
                            "stub", fetch.provenance.VERIFIED, "test")]))
    monkeypatch.setattr(fetch, "retrieve",
                        lambda request, cache_dir, label, **kw: tmp_path / f"{label}.zip")
    monkeypatch.setattr(fetch, "record_delivery", lambda target, ds: {"stub": True})
    monkeypatch.setattr(fetch, "read_sidecar", lambda target: {"retrieved_utc": "stub"})

    times = axis.requested_analysis_times()
    win = [t for t in times if t.date() <= axis.d_prev]
    new = [t for t in times if t.date() == axis.d_an]
    assert win and new, "the fixture must exercise both day windows"
    fc_times = fetch.cams.requested_forecast_times(
        axis, fetch.cams.forecast_request(axis))
    segments = {
        "analysis_0.zip": _segment_ds(win, ("omaod550",)),
        "analysis_1.zip": _segment_ds(win, ("u10", "v10")),
        "analysis_2.zip": _segment_ds(new, ("omaod550",)),
        "analysis_3.zip": _segment_ds(new, ("u10", "v10")),
        "forecast.zip": _segment_ds(fc_times, ("omaod550", "u10", "v10")),
    }
    monkeypatch.setattr(fetch, "_open", lambda target: segments[target.name])

    seen = {}

    def _capture(analysis, forecast, t_now, t_init):
        seen["analysis"] = analysis
        raise _ReachedMerge

    monkeypatch.setattr(fetch.cams, "merge_arms", _capture)
    with pytest.raises(_ReachedMerge):
        fetch.run(days=10, dry_run=False, today=dt.date(2026, 8, 4), axis=axis)

    analysis = seen["analysis"]
    assert int(analysis.sizes["time"]) == len(times)
    for var in ("omaod550", "u10", "v10"):
        assert np.isfinite(analysis[var].values).all(), (
            f"{var} has holes after assembling the segmented analysis arms")


def _axis(day=4, h_new=("00:00",)):
    """Replaces the plan's fixed _axis(); day=4 reproduces it exactly."""
    from tools.europe_smoke.probe import Axis
    d_an = dt.date(2026, 8, day)
    return Axis(
        d_an=d_an, d_prev=d_an - dt.timedelta(days=1),
        start=d_an - dt.timedelta(days=10),
        h_win=["00:00", "06:00", "12:00", "18:00"], h_new=list(h_new),
        t_now=dt.datetime.combine(d_an, dt.time(0, 0)), run_date=d_an,
        run_hour="00:00", t_init=dt.datetime.combine(d_an, dt.time(0, 0)),
    )
