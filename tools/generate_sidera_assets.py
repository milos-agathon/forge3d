"""Regenerate SIDERA's compact public-domain astronomy assets."""

from __future__ import annotations

import argparse
import ast
import calendar
import csv
import datetime as dt
import io
import json
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTRO = ROOT / "assets" / "astro"
VSOP_URL = "https://ftp.imcce.fr/pub/ephem/planets/vsop87/"
VSOP_FILES = ("mer", "ven", "ear", "mar", "jup", "sat")
VSOP_HEADER = re.compile(r"VARIABLE\s+(\d).*T\*\*(\d)\s+(\d+)\s+TERMS")
MOON_URL = "https://raw.githubusercontent.com/commenthol/astronomia/master/src/moonposition.js"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SITES = (
    ("amsterdam", 52.3676, 4.9041, 0.0),
    ("mauna_kea", 19.8207, -155.4681, 4.205),
    ("sydney", -33.8688, 151.2093, 0.058),
    ("quito", -0.1807, -78.4678, 2.85),
    ("longyearbyen", 78.2232, 15.6469, 0.028),
)
EPOCHS = (
    "2000-01-01 00:00",
    "2005-06-21 12:00",
    "2012-03-20 06:00",
    "2020-12-21 18:00",
    "2026-07-26 22:00",
    "2035-09-23 03:00",
    "2044-02-29 12:00",
    "2050-12-31 23:00",
)
BODIES = (
    ("sun", "10"),
    ("moon", "301"),
    ("mercury", "199"),
    ("venus", "299"),
    ("mars", "499"),
    ("jupiter", "599"),
    ("saturn", "699"),
)


def generate_vsop() -> None:
    output = ASTRO / "vsop87d.bin"
    sections: list[tuple[int, int, list[tuple[float, float, float]]]] = []
    bodies: list[tuple[str, list[int]]] = []
    for body in VSOP_FILES:
        text = urllib.request.urlopen(f"{VSOP_URL}VSOP87D.{body}", timeout=60).read().decode("ascii")
        body_sections: list[int] = []
        lines = iter(text.splitlines())
        for line in lines:
            match = VSOP_HEADER.search(line)
            if not match:
                continue
            variable, power, count = map(int, match.groups())
            terms = []
            for _ in range(count):
                values = next(lines).split()
                terms.append(tuple(map(float, values[-3:])))
            body_sections.append(len(sections))
            sections.append((variable - 1, power, terms))
        bodies.append((body, body_sections))

    with output.open("wb") as stream:
        stream.write(b"F3DVSOP1")
        stream.write(struct.pack("<II", len(bodies), len(sections)))
        for name, indices in bodies:
            stream.write(name.encode("ascii"))
            stream.write(struct.pack("<B", len(indices)))
            stream.write(struct.pack(f"<{len(indices)}I", *indices))
        for variable, power, terms in sections:
            stream.write(struct.pack("<BBHI", variable, power, 0, len(terms)))
            for term in terms:
                stream.write(struct.pack("<ddd", *term))
    print(f"{output}: {output.stat().st_size} bytes")


def generate_moon() -> None:
    source = urllib.request.urlopen(MOON_URL, timeout=60).read().decode("utf-8")
    longitude_radius = ast.literal_eval(
        "[" + re.search(r"const ta = \[(.*?)\n  \]", source, re.DOTALL).group(1) + "]"
    )
    latitude = ast.literal_eval(
        "[" + re.search(r"const tb = \[(.*?)\n  \]", source, re.DOTALL).group(1) + "]"
    )
    output = ASTRO / "moon_terms.bin"
    with output.open("wb") as stream:
        stream.write(b"F3DMOON1")
        stream.write(struct.pack("<II", len(longitude_radius), len(latitude)))
        for d, m, mp, f, longitude, radius in longitude_radius:
            stream.write(struct.pack("<bbbbii", d, m, mp, f, int(longitude), int(radius)))
        for d, m, mp, f, coefficient in latitude:
            stream.write(struct.pack("<bbbbi", d, m, mp, f, int(coefficient)))
    print(f"{output}: {output.stat().st_size} bytes")


def generate_horizons() -> None:
    output = ROOT / "tests" / "data" / "horizons_vectors.dat"
    rows = []
    for site, latitude, longitude, height_km in SITES:
        for body, command in BODIES:
            parameters = {
                "format": "json",
                "COMMAND": f"'{command}'",
                "OBJ_DATA": "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "OBSERVER",
                "CENTER": "'coord@399'",
                "COORD_TYPE": "GEODETIC",
                "SITE_COORD": f"'{longitude},{latitude},{height_km}'",
                "TLIST": " ".join(f"'{epoch}'" for epoch in EPOCHS),
                "TLIST_TYPE": "CAL",
                "QUANTITIES": "'4,10,13,20,24,30,49'",
                "ANG_FORMAT": "DEG",
                "APPARENT": "AIRLESS",
                "CSV_FORMAT": "YES",
                "REF_SYSTEM": "ICRF",
                "CAL_FORMAT": "CAL",
                "TIME_DIGITS": "SECONDS",
            }
            url = HORIZONS_URL + "?" + urllib.parse.urlencode(parameters)
            result = json.load(urllib.request.urlopen(url, timeout=60))["result"]
            block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
            body_rows = list(csv.reader(io.StringIO(block.strip())))
            if len(body_rows) != len(EPOCHS):
                raise RuntimeError(f"Horizons returned {len(body_rows)} rows for {site}/{body}")
            for epoch, values in zip(EPOCHS, body_rows):
                numeric = [value.strip() for value in values[3:]]
                rows.append(
                    (
                        site,
                        epoch.replace(" ", "T") + ":00Z",
                        body,
                        latitude,
                        longitude,
                        height_km * 1_000.0,
                        *numeric,
                    )
                )

    phase_parameters = {
        "format": "json",
        "COMMAND": "'301'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "TLIST": " ".join(f"'{epoch}'" for epoch in EPOCHS),
        "TLIST_TYPE": "CAL",
        "QUANTITIES": "'10,13,20,24'",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "CAL",
        "TIME_DIGITS": "SECONDS",
    }
    phase_url = HORIZONS_URL + "?" + urllib.parse.urlencode(phase_parameters)
    phase_result = json.load(urllib.request.urlopen(phase_url, timeout=60))["result"]
    phase_block = phase_result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    phase_rows = list(csv.reader(io.StringIO(phase_block.strip())))

    header = """# JPL Horizons topocentric observer vectors, generated 2026-07-26.
# Ephemeris: DE441; center: coord@399; frame: ICRF; apparent: AIRLESS.
# Quantities: 4,10,13,20,24,30,49; angular format: decimal degrees; CSV.
# Generator: tools/generate_sidera_assets.py --horizons
# 5 WGS84 sites x 8 UTC epochs = 40 epoch/site combinations x 7 bodies.
# columns: site utc body lat_deg lon_deg height_m az_deg alt_deg illum_percent
#          angular_diameter_arcsec distance_au range_rate_km_s phase_angle_deg
#          tdb_minus_ut_seconds ut1_minus_utc_seconds
# @moon_phase rows are geocentric: utc illum_percent diameter_arcsec distance_au phase_deg.
"""
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        for row in rows:
            stream.write(" ".join(map(str, row)) + "\n")
        for epoch, values in zip(EPOCHS, phase_rows):
            numeric = [value.strip() for value in values[3:]]
            stream.write(
                "@moon_phase "
                + epoch.replace(" ", "T")
                + ":00Z "
                + " ".join(numeric[0:1] + numeric[1:2] + numeric[2:3] + numeric[4:5])
                + "\n"
            )
    print(f"{output}: {len(rows)} vectors, {output.stat().st_size} bytes")


def generate_delta_t() -> None:
    parameters = {
        "format": "json",
        "COMMAND": "'10'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "START_TIME": "'2000-01-01'",
        "STOP_TIME": "'2051-01-01'",
        "STEP_SIZE": "'1 month'",
        "QUANTITIES": "'30,49'",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "JD",
    }
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(parameters)
    result = json.load(urllib.request.urlopen(url, timeout=60))["result"]
    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    dut1 = [float(row[4]) for row in csv.reader(io.StringIO(block.strip()))]
    dates = []
    year, month = 2000, 1
    while (year, month) <= (2051, 1):
        dates.append(dt.date(year, month, 1))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    if len(dates) != len(dut1):
        raise RuntimeError(f"Horizons returned {len(dut1)} ΔT rows for {len(dates)} dates")

    def decimal_year(date: dt.date) -> float:
        days = 366 if calendar.isleap(date.year) else 365
        return date.year + (date.timetuple().tm_yday - 1) / days

    def tai_minus_utc(date: dt.date) -> int:
        offset = 32
        for effective, value in (
            (dt.date(2006, 1, 1), 33),
            (dt.date(2009, 1, 1), 34),
            (dt.date(2012, 7, 1), 35),
            (dt.date(2015, 7, 1), 36),
            (dt.date(2017, 1, 1), 37),
        ):
            if date >= effective:
                offset = value
        return offset

    values = [tai_minus_utc(date) + 32.184 - delta for date, delta in zip(dates, dut1)]
    output = ASTRO / "delta_t_fit.dat"
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Piecewise-linear TT-UT1 fit to monthly JPL Horizons/IERS EOP nodes.\n")
        stream.write("# columns: start_year end_year origin_year c0_seconds c1_seconds_per_year\n")
        for date_a, date_b, value_a, value_b in zip(dates, dates[1:], values, values[1:]):
            start, end = decimal_year(date_a), decimal_year(date_b)
            slope = (value_b - value_a) / (end - start)
            stream.write(f"{start:.12f} {end:.12f} {start:.12f} {value_a:.9f} {slope:.9f}\n")
    print(f"{output}: {len(values) - 1} monthly segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsop", action="store_true")
    parser.add_argument("--moon", action="store_true")
    parser.add_argument("--horizons", action="store_true")
    parser.add_argument("--delta-t", action="store_true")
    options = parser.parse_args()
    if options.vsop:
        generate_vsop()
    if options.moon:
        generate_moon()
    if options.horizons:
        generate_horizons()
    if options.delta_t:
        generate_delta_t()
