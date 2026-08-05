# tools/europe_smoke/tests/test_probe.py
import datetime as dt

import pytest

from tools.europe_smoke import probe


class FakeConstraints:
    """Replays the shape ADS actually returned on 2026-08-04 (§0.4)."""

    def __init__(self):
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        var = payload.get("variable", [])
        aerosol = any("organic" in v or "aerosol" in v for v in var)
        date = payload.get("date")
        if date is None:
            # discovery call: what date ranges exist for analysis
            return {"date": ["2026-07-01/2026-08-04"] if aerosol else ["2026-07-01/2026-08-04"]}
        if date == ["2026-08-04/2026-08-04"]:
            return {"time": ["00:00"]}
        if date == ["2026-07-25/2026-08-03"]:
            return {"time": ["00:00", "06:00", "12:00", "18:00"]}
        # forecast run discovery
        return {"time": ["00:00", "12:00"], "variable": var}


def test_resolve_axis_picks_the_latest_analysis_day_and_its_hour_set():
    fake = FakeConstraints()
    axis = probe.resolve_axis(
        start_date=dt.date(2026, 7, 25),
        today=dt.date(2026, 8, 4),
        constraints=fake,
    )
    assert axis.d_an == dt.date(2026, 8, 4)
    assert axis.h_new == ["00:00"]
    assert axis.h_win == ["00:00", "06:00", "12:00", "18:00"]


def test_t_now_is_the_latest_delivered_analysis_valid_time_not_wall_clock():
    # §0.4: measured at 12:13 UTC on 4 Aug the latest analysis valid time was
    # 00:00Z -- a 12.2 h lag. NOW must trail, not lead.
    fake = FakeConstraints()
    axis = probe.resolve_axis(
        start_date=dt.date(2026, 7, 25), today=dt.date(2026, 8, 4), constraints=fake
    )
    assert axis.t_now == dt.datetime(2026, 8, 4, 0, 0)


def test_forecast_arm_is_clipped_strictly_after_t_now():
    fake = FakeConstraints()
    axis = probe.resolve_axis(
        start_date=dt.date(2026, 7, 25), today=dt.date(2026, 8, 4), constraints=fake
    )
    assert axis.t_init <= axis.t_now


def test_requested_axis_is_reproducible_from_the_probe_output():
    fake = FakeConstraints()
    axis = probe.resolve_axis(
        start_date=dt.date(2026, 7, 25), today=dt.date(2026, 8, 4), constraints=fake
    )
    times = axis.requested_analysis_times()
    # 10 full days x 4 hours + the newest day's single hour
    assert len(times) == 10 * 4 + 1
    assert times[0] == dt.datetime(2026, 7, 25, 0, 0)
    assert times[-1] == dt.datetime(2026, 8, 4, 0, 0)
    assert all(a < b for a, b in zip(times, times[1:]))


def test_probe_never_caches_across_days():
    assert probe.CACHE_CONSTRAINTS is False


@pytest.mark.network
def test_live_constraints_endpoint_is_reachable_and_unauthenticated():
    reply = probe.constraints({"variable": ["organic_matter_aerosol_optical_depth_550nm"],
                              "type": ["analysis"]})
    assert "date" in reply
