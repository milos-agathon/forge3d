"""Fit monthly cubic TT-UT1 segments to JPL Horizons' TDB-UT table.

Run only while refreshing the committed Sidera time asset. The production
library reads the generated coefficients offline and rejects other epochs.
"""

from __future__ import annotations

import hashlib
import bisect
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/sidera/delta_t.bin"
QUERY = {
    "format": "text",
    "COMMAND": "399",
    "OBJ_DATA": "NO",
    "MAKE_EPHEM": "YES",
    "EPHEM_TYPE": "VECTORS",
    "CENTER": "500@0",
    "START_TIME": "'2000-01-01'",
    "STOP_TIME": "'2051-01-01'",
    "STEP_SIZE": "'5 d'",
    "TIME_TYPE": "UT",
    "VEC_DELTA_T": "YES",
    "CSV_FORMAT": "YES",
}


def julian_day(year: int, month: int = 1, day: int = 1) -> float:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045 - 0.5


def main() -> None:
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(QUERY)
    response = urllib.request.urlopen(url).read()
    body = response.decode("ascii")
    match = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", body, re.S)
    if match is None:
        raise ValueError(body[-1000:])
    rows = [(float(parts[0]), float(parts[2])) for line in match.group(1).splitlines() if (parts := line.split(","))]
    print("Horizons rows", len(rows), "response SHA-256", hashlib.sha256(response).hexdigest())
    encoded = bytearray(struct.pack("<4sHH", b"DT50", 2000, 51 * 12))
    worst = 0.0
    segments = []
    for year in range(2000, 2051):
        for month in range(1, 13):
            start = julian_day(year, month)
            end = julian_day(year + 1, 1) if month == 12 else julian_day(year, month + 1)
            length = end - start
            subset = [(jd, dt) for jd, dt in rows if start <= jd < end]
            if len(subset) < 4:
                raise ValueError(f"insufficient Horizons samples in {year}-{month}: {len(subset)}")
            x = np.array([(jd - start) / length for jd, _ in subset], dtype=np.float64)
            y = np.array([dt for _, dt in subset], dtype=np.float64)
            coefficients = np.polynomial.polynomial.polyfit(x, y, 3)
            residual = float(np.max(np.abs(np.polynomial.polynomial.polyval(x, coefficients) - y)))
            worst = max(worst, residual)
            encoded.extend(struct.pack("<4d", *coefficients))
            segments.append((start, end, coefficients))
    # Seven-day holdout samples have a different cadence from the five-day
    # fit and test interpolation between the training points.
    holdout_query = dict(QUERY, STEP_SIZE="'7 d'")
    holdout_response = urllib.request.urlopen("https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(holdout_query)).read()
    holdout_match = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", holdout_response.decode("ascii"), re.S)
    if holdout_match is None:
        raise ValueError("missing Horizons delta-T holdout rows")
    starts = [segment[0] for segment in segments]
    worst_holdout = 0.0
    holdout_count = 0
    for line in holdout_match.group(1).splitlines():
        parts = line.split(",")
        jd, reference = float(parts[0]), float(parts[2])
        index = bisect.bisect_right(starts, jd) - 1
        if not (0 <= index < len(segments) and jd < segments[index][1]):
            continue
        start, end, coefficients = segments[index]
        x = (jd - start) / (end - start)
        predicted = float(np.polynomial.polynomial.polyval(x, coefficients))
        worst_holdout = max(worst_holdout, abs(predicted - reference))
        holdout_count += 1
    print("independent seven-day holdout", holdout_count, "maximum residual seconds", worst_holdout)
    if holdout_count < 1000 or worst_holdout >= 1.0:
        raise ValueError("SIDERA delta-T holdout did not prove the 1 s residual gate")
    OUT.write_bytes(encoded)
    print("maximum sampled residual seconds", worst)
    print("delta_t.bin", OUT.stat().st_size, hashlib.sha256(encoded).hexdigest())
    if worst >= 1.0:
        raise ValueError("ΔT fit exceeds Sidera's 1 s residual gate")


if __name__ == "__main__":
    main()
