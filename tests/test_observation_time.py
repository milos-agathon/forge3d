from dataclasses import replace
from datetime import datetime, timezone

import pytest

from forge3d.observation_time import ObservationRecord, ObservationTrack


def record(name="a", observed="2023-01-01T00:00:00Z", start="2023-01-01T00:00:00Z", end="2024-01-01T00:00:00Z"):
    return ObservationRecord(name, observed, start, end, {"year": "2023"})


@pytest.mark.parametrize("policy,expected", [("hold", ["missing", "observed", "observed", "held"]),
                                            ("no-data", ["missing", "observed", "observed", "missing"])])
def test_before_at_after_acquisition_and_expiry(policy, expected):
    selected = ObservationTrack([record()], policy).select([
        "2022-12-31T23:59:59Z", "2023-01-01T00:00:00Z", "2023-01-01T00:00:01Z", "2024-01-01T00:00:00Z"])
    assert [s.state for s in selected] == expected
    assert selected[0].record is None and "No observation" in selected[0].caption
    assert selected[1].age_seconds == 0 and selected[2].age_seconds == 1
    if policy == "hold":
        assert "Held observation" in selected[-1].caption
        assert selected[-1].age_seconds == 365 * 24 * 60 * 60
        assert "Age 365 days" in selected[-1].caption


def test_timezones_normalize_and_naive_values_rejected():
    a = record(observed="2023-01-01T01:00:00+01:00")
    selected = ObservationTrack([a], "no-data").select(["2022-12-31T19:00:00-05:00"])[0]
    assert selected.state == "observed" and selected.age_seconds == 0
    assert selected.presentation_time == datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert selected.to_dict()["record"]["observed_at"] == "2023-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="timezone"):
        record(observed="2023-01-01")
    with pytest.raises(ValueError, match="timezone"):
        ObservationTrack([a], "hold").select([datetime(2023, 1, 1)])


def test_gaps_overlaps_latest_acquisition_and_ties():
    a = record(end="2023-02-01T00:00:00Z")
    b = record("b", "2023-03-01T00:00:00Z", "2023-03-01T00:00:00Z", "2023-04-01T00:00:00Z")
    gap = "2023-02-15T00:00:00Z"
    assert ObservationTrack([a, b], "no-data").select([gap])[0].record is None
    assert ObservationTrack([a, b], "hold").select([gap])[0].record == a
    overlap = replace(a, valid_until="2024-01-01T00:00:00Z")
    at = "2023-03-01T00:00:00Z"
    assert ObservationTrack([b, overlap], "hold").select([at])[0].record == b
    with pytest.raises(ValueError, match="Ambiguous"):
        ObservationTrack([b, replace(b, record_id="duplicate-time")], "hold").select([at])


def test_future_acquisition_or_validity_never_backfills_and_payload_snapshot_retained():
    future = record(observed="2023-02-01T00:00:00Z")
    assert ObservationTrack([future], "hold").select(["2023-01-15T00:00:00Z"])[0].record is None
    future_validity = record(start="2023-02-01T00:00:00Z")
    assert ObservationTrack([future_validity], "hold").select(["2023-01-15T00:00:00Z"])[0].record is None
    data = {"source": {"year": "2023"}}
    a = replace(record(), payload=data)
    data["source"]["year"] = "2024"
    table = ObservationTrack([a], "hold").selection_table(["2024-01-01T00:00:00Z"])
    assert table["selections"][0]["record"]["payload"]["source"]["year"] == "2023"
    assert table["selections"][0]["caption"].endswith("Age 365 days")
    assert table["selections"][0]["record"] == table["records"][0]


def test_declared_timezone_year_and_day_age_survive_selection_and_serialization():
    a = record(observed="2024-01-01T00:00:00+01:00", start="2024-01-01T00:00:00+01:00",
               end="2025-01-01T00:00:00+01:00")
    selected = ObservationTrack([a], "hold").select(["2025-01-01T00:00:00+01:00"])[0]
    assert a.observation_year == "2024"
    assert selected.caption == ("Time 2025-01-01T00:00:00+01:00 | Held observation "
                                "2024-01-01T00:00:00+01:00 | Age 366 days")
    data = selected.to_dict()
    assert data["presentation_time"] == "2024-12-31T23:00:00Z"
    assert data["declared_presentation_time"] == "2025-01-01T00:00:00+01:00"
    assert data["record"]["declared_observed_at"] == "2024-01-01T00:00:00+01:00"
    before, at, after = ObservationTrack([a], "no-data").select([
        "2023-12-31T22:59:59Z", "2023-12-31T23:00:00Z", "2023-12-31T23:00:01Z"])
    assert [s.state for s in (before, at, after)] == ["missing", "observed", "observed"]
    assert after.age_seconds == 1 and "Age 0 days" not in after.caption


def test_dst_fold_uses_utc_instants_for_ordering_and_age():
    from zoneinfo import ZoneInfo
    zone = ZoneInfo("Europe/Amsterdam")
    first = datetime(2024, 10, 27, 2, 30, tzinfo=zone, fold=0)
    second = first.replace(fold=1)
    a = record(observed=first, start=first, end=second)
    selected = ObservationTrack([a], "hold").select([second])[0]
    assert selected.state == "held" and selected.age_seconds == 3600


def test_invalid_contracts_and_empty_track():
    with pytest.raises(ValueError, match="positive duration"):
        record(end="2023-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="explicitly"):
        ObservationTrack([], "interpolate")
    with pytest.raises(ValueError, match="distinct"):
        ObservationTrack([record(), record()], "hold")
    assert ObservationTrack([], "no-data").select(["2023-01-01T00:00:00Z"])[0].state == "missing"
