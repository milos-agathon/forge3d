"""Refresh the independent Sidera topocentric Horizons oracle.

Each row is requested from NASA/JPL, never derived from forge3d. The output
is a frozen CSV with query settings and a digest of the 56 raw responses.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests/data/horizons_vectors.dat"
API = "https://ssd.jpl.nasa.gov/api/horizons.api"
SITES = [
    ("amsterdam", 52.37, 4.9),
    ("new_york", 40.7128, -74.0060),
    ("santiago", -33.4489, -70.6693),
    ("cape_town", -33.9249, 18.4241),
    ("sydney", -33.8688, 151.2093),
    ("tokyo", 35.6762, 139.6503),
    ("equator", 0.0, 0.0),
    ("tromso", 69.6492, 18.9553),
]
DATES = [
    ("2000-Jan-01 00:00", "2000-01-01T00:00:00Z"),
    ("2012-Apr-01 00:00", "2012-04-01T00:00:00Z"),
    ("2026-Sep-25 22:00", "2026-09-25T22:00:00Z"),
    ("2038-Mar-20 10:00", "2038-03-20T10:00:00Z"),
    ("2050-Dec-01 00:00", "2050-12-01T00:00:00Z"),
]
BODIES = [("Sun", 10), ("Moon", 301), ("Mercury", 199), ("Venus", 299), ("Mars", 499), ("Jupiter", 599), ("Saturn", 699)]


def main() -> None:
    rows = []
    digest = hashlib.sha256()
    for site, latitude, longitude in SITES:
        for body, command in BODIES:
            params = {
                "format": "text", "COMMAND": str(command), "OBJ_DATA": "NO",
                "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER",
                "CENTER": "coord@399", "COORD_TYPE": "GEODETIC",
                "SITE_COORD": f"{longitude},{latitude},0",
                "TLIST": " ".join(f"'{date}'" for date, _ in DATES),
                "TLIST_TYPE": "CAL", "TIME_TYPE": "UT", "QUANTITIES": "4,10,13,20",
                "APPARENT": "AIRLESS", "CSV_FORMAT": "YES", "ANG_FORMAT": "DEG",
                "TIME_DIGITS": "SECONDS", "EXTRA_PREC": "YES",
            }
            raw = urllib.request.urlopen(API + "?" + urllib.parse.urlencode(params)).read()
            digest.update(raw)
            response = raw.decode("ascii")
            match = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", response, re.S)
            if match is None:
                raise ValueError(f"Horizons failed for {site}/{body}: {response[-900:]}")
            records = list(csv.reader(io.StringIO(match.group(1))))
            if len(records) != len(DATES):
                raise ValueError(f"expected five epochs for {site}/{body}, got {len(records)}")
            for (_, utc), record in zip(DATES, records):
                rows.append((utc, site, latitude, longitude, body,
                    float(record[3]), float(record[4]), float(record[7]),
                    float(record[5]) / 100.0, float(record[6]) / 2.0))
        print(site, len(rows), flush=True)
    with OUT.open("w", encoding="ascii", newline="") as stream:
        stream.write("# NASA/JPL Horizons observer API; 40 UTC/site combinations, seven bodies each.\n")
        stream.write("# CENTER=coord@399; COORD_TYPE=GEODETIC; SITE_COORD=lon,lat,0 km; TIME_TYPE=UT; APPARENT=AIRLESS; QUANTITIES=4,10,13,20; EXTRA_PREC=YES.\n")
        stream.write("# Model and EOP selection are recorded by the Horizons responses; raw-response aggregate SHA-256 follows.\n")
        stream.write(f"# api={API}; raw_response_sha256={digest.hexdigest()}\n")
        writer = csv.writer(stream)
        writer.writerow(("utc", "site", "latitude_deg", "longitude_deg", "body", "azimuth_deg", "altitude_deg", "distance_au", "illuminated_fraction", "semidiameter_arcsec"))
        writer.writerows(rows)
    print("rows", len(rows), "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
