"""Build SIDERA's compact UT1 history from the official IERS C04 series.

The asset stores monthly Delta T (TT - UT1) knots from 2000-01-01 through
the final month in the downloaded C04 file, plus the final daily value.  The
runtime linearly interpolates those continuous Delta T values,
then recovers UT1 - UTC with the applicable leap-second offset.  After the
last observation it holds the final UT1 - UTC value constant.  That is a
declared compatibility forecast, not a bounded Earth-rotation prediction.
"""

from __future__ import annotations

import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/sidera/ut1_utc.bin"
SOURCE = "https://datacenter.iers.org/data/254/eopc04_20u24.1962-now.txt"
SOURCE_SHA256 = "7e39bb43bd1e1920517ed9316d3c5b14f898fe68afe9e0891bf18c4180b9d272"
START = date(2000, 1, 1)
TICKS_PER_SECOND = 10_000_000
LEAPS = (
    date(2006, 1, 1),
    date(2009, 1, 1),
    date(2012, 7, 1),
    date(2015, 7, 1),
    date(2017, 1, 1),
)


@dataclass(frozen=True)
class Row:
    day: date
    mjd: int
    dut1: float


def tai_minus_utc(day: date) -> int:
    return 32 + sum(day >= leap for leap in LEAPS)


def delta_t(row: Row) -> float:
    return 32.184 + tai_minus_utc(row.day) - row.dut1


def parse(source: bytes) -> list[Row]:
    rows = []
    for raw_line in source.decode("ascii").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split()
        day = date(int(fields[0]), int(fields[1]), int(fields[2]))
        if day >= START:
            rows.append(Row(day, int(float(fields[4])), float(fields[7])))
    if not rows or rows[0].day != START:
        raise ValueError("IERS C04 source does not begin SIDERA history at 2000-01-01")
    for previous, current in zip(rows, rows[1:]):
        if current.mjd != previous.mjd + 1:
            raise ValueError(f"IERS C04 source has a gap before {current.day}")
    return rows


def interpolate_delta_t(
    row: Row, knots: list[Row], final: Row, encoded_delta_t: list[int]
) -> float:
    index = (row.day.year - START.year) * 12 + row.day.month - 1
    begin = knots[index]
    end = knots[index + 1] if index + 1 < len(knots) else final
    fraction = (row.mjd - begin.mjd) / (end.mjd - begin.mjd)
    current = encoded_delta_t[index] / TICKS_PER_SECOND
    following = encoded_delta_t[index + 1] / TICKS_PER_SECOND
    return current + fraction * (following - current)


def main() -> None:
    source = urllib.request.urlopen(SOURCE).read()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise ValueError("IERS C04 source changed; review it before updating SIDERA UT1")
    rows = parse(source)
    knots = [row for row in rows if row.day.day == 1]
    final = rows[-1]
    if knots[-1].day.year != final.day.year or knots[-1].day.month != final.day.month:
        raise ValueError("final IERS C04 month has no first-day knot")

    encoded_delta_t = [round(delta_t(row) * TICKS_PER_SECOND) for row in (*knots, final)]
    worst = max(
        abs(
            (
                32.184
                + tai_minus_utc(row.day)
                - interpolate_delta_t(row, knots, final, encoded_delta_t)
            )
            - row.dut1
        )
        for row in rows
    )
    if worst >= 1.0:
        raise ValueError(f"monthly SIDERA UT1 interpolation residual is {worst} s")

    encoded = bytearray(
        struct.pack(
            "<4sHHii",
            b"U1C4",
            START.year,
            len(knots),
            final.mjd,
            round(final.dut1 * TICKS_PER_SECOND),
        )
    )
    for value in encoded_delta_t:
        encoded.extend(struct.pack("<i", value))
    OUT.write_bytes(encoded)

    print("source", SOURCE)
    print("source SHA-256", source_sha256)
    print("history", rows[0].day, "through", final.day, "daily rows", len(rows))
    print("monthly knots", len(knots), "worst daily UT1-UTC residual seconds", worst)
    print("post-history scenario: constant UT1-UTC seconds", final.dut1)
    print(OUT.relative_to(ROOT), len(encoded), hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    main()
