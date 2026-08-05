# tools/europe_smoke/tests/test_firms.py
import datetime as dt

import pytest

from tools.europe_smoke import firms


@pytest.mark.parametrize("raw,expected", [
    ("16", dt.time(0, 16)),      # NOT 16:00
    ("211", dt.time(2, 11)),     # NOT 21:10
    ("0", dt.time(0, 0)),
    ("2359", dt.time(23, 59)),
    ("905", dt.time(9, 5)),
])
def test_acq_time_parses_stripped_hhmm(raw, expected):
    assert firms.parse_acq_time(raw) == expected


def test_acq_time_rejects_garbage():
    with pytest.raises(ValueError):
        firms.parse_acq_time("2461")


def test_day_spans_never_exceed_the_documented_five_day_limit():
    spans = firms.day_spans(dt.date(2026, 7, 25), dt.date(2026, 8, 4))
    assert all(n <= firms.MAX_DAY_RANGE for _, n in spans)
    assert sum(n for _, n in spans) == 11
    assert spans[0][0] == dt.date(2026, 7, 25)


def test_parse_csv_reads_the_fourteen_column_schema():
    text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
        "38.1,23.7,330.1,0.4,0.36,2026-08-01,16,N,VIIRS,n,2.0NRT,285.0,12.5,N\n"
    )
    rows = firms.parse_csv(text)
    assert len(rows) == 1
    r = rows[0]
    assert r["lat"] == pytest.approx(38.1) and r["lon"] == pytest.approx(23.7)
    assert r["when"] == dt.datetime(2026, 8, 1, 0, 16)
    assert r["frp"] == pytest.approx(12.5)
    assert r["confidence"] == "n"


def test_parse_csv_keeps_zero_and_missing_frp():
    text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
        "38.1,23.7,330.1,0.4,0.36,2026-08-01,16,N,VIIRS,n,2.0NRT,285.0,,N\n"
        "38.2,23.8,330.1,0.4,0.36,2026-08-01,16,N,VIIRS,n,2.0NRT,285.0,0,N\n"
    )
    rows = firms.parse_csv(text)
    assert len(rows) == 2, "missing/zero FRP must survive, not be silently dropped"
    assert rows[0]["frp"] == 0.0


def test_merc_y_is_in_mercator_degrees():
    # a 0.05 mercator-degree bin is 0.03536 deg of latitude at 45N
    d = firms.merc_y(45.0 + 0.03536) - firms.merc_y(45.0)
    assert d == pytest.approx(0.05, rel=0.01)


def test_thin_drops_low_confidence_and_respects_the_cap():
    # A merged record is a centroid of up to ~100 detections of mixed classes
    # (spec 4.3), so it has no scalar `confidence` and must not pretend to.
    # The filter is therefore verified by ABSENCE -- the low-confidence
    # detection's FRP must never reach the map set.
    when = dt.datetime(2026, 8, 1, 0, 16)
    rows = [
        # One detection per 0.05-mercator-degree bin: 0.1 deg of latitude is
        # ~0.13 mercator degrees, so these do not merge and the cap has
        # something to bite on.
        {"lat": 38.0 + i * 0.1, "lon": 23.0, "frp": 1.0, "confidence": "n",
         "when": when}
        for i in range(50)
    ] + [
        # 99 FRP: if the filter were removed this would sort FIRST, so it
        # cannot hide behind the cap.
        {"lat": 40.0, "lon": 25.0, "frp": 99.0, "confidence": "l",
         "when": when}
    ]

    # Uncapped: the low-confidence detection is absent from the merged set
    # itself, not merely truncated away by the cap.
    full = firms.thin(rows, cap=1000)
    assert len(full) == 50, "the 'l' detection must not survive as its own bin"
    assert sum(r["n"] for r in full) == 50, "it must not be folded into a bin"
    assert sum(r["frp"] for r in full) == pytest.approx(50.0), "its 99 FRP leaked"

    # Capped: the cap truncates, and the survivors are still only the
    # high-confidence ones (top-of-ranking is where a removed filter shows up).
    out = firms.thin(rows, cap=10)
    assert len(out) == 10
    assert max(r["frp"] for r in out) == pytest.approx(1.0)
    assert all(r["lon"] == pytest.approx(23.0) for r in out)


def test_thin_merges_a_bin_to_an_frp_weighted_centroid():
    when = dt.datetime(2026, 8, 1, 0, 16)
    rows = [
        {"lat": 38.0, "lon": 23.0, "frp": 1.0, "confidence": "n", "when": when},
        {"lat": 38.0, "lon": 23.0, "frp": 3.0, "confidence": "n", "when": when},
    ]
    out = firms.thin(rows, cap=100)
    assert len(out) == 1
    assert out[0]["frp"] == pytest.approx(4.0)
    assert out[0]["n"] == 2
