# tools/europe_smoke/firms.py
"""NASA FIRMS VIIRS acquisition, parsing and thinning (§1.2, §4.3).

Two things here are easy to get wrong and expensive to notice:
  * acq_time is an integer HHMM with leading zeros stripped. '16' is 00:16.
  * the documented DAY_RANGE limit is 1..5, not 10.
"""
from __future__ import annotations

import csv as csv_mod
import datetime as dt
import io
import math
import time
from collections import defaultdict
from urllib.request import Request, urlopen

from . import config

MAX_DAY_RANGE = config.FIRMS_MAX_DAY_RANGE
BIN_DEG = 0.05


def parse_acq_time(raw: str) -> dt.time:
    """HHMM with leading zeros stripped -> time. Never slice this string."""
    h, m = divmod(int(str(raw).strip()), 100)
    return dt.time(h, m)


def day_spans(start: dt.date, end: dt.date) -> list[tuple[dt.date, int]]:
    """Chunk [start, end] into spans of at most MAX_DAY_RANGE days."""
    spans, day = [], start
    while day <= end:
        n = min(MAX_DAY_RANGE, (end - day).days + 1)
        spans.append((day, n))
        day += dt.timedelta(days=n)
    return spans


def parse_csv(text: str) -> list[dict]:
    rows = []
    for row in csv_mod.DictReader(io.StringIO(text)):
        try:
            when = dt.datetime.combine(
                dt.date.fromisoformat(row["acq_date"]), parse_acq_time(row["acq_time"]))
            lat, lon = float(row["latitude"]), float(row["longitude"])
        except (KeyError, ValueError):
            continue
        raw_frp = (row.get("frp") or "").strip()
        rows.append({
            "lat": lat, "lon": lon, "when": when,
            "frp": float(raw_frp) if raw_frp else 0.0,
            "confidence": (row.get("confidence") or "n").strip().lower(),
            "satellite": (row.get("satellite") or "").strip(),
        })
    return rows


def merc_y(lat_deg: float) -> float:
    """Web-mercator y in DEGREES. Stated because an unqualified mercY is
    ambiguous across seven orders of magnitude (radians vs degrees vs metres)."""
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2)))


def _step_key(when: dt.datetime) -> dt.datetime:
    return when.replace(hour=(when.hour // 3) * 3, minute=0, second=0, microsecond=0)


def thin(records: list[dict], cap: int = config.FIRMS_MAP_CAP) -> list[dict]:
    """Drop low confidence, merge to FRP-weighted centroids, cap per step.

    Statistics are computed elsewhere from the FULL record set; this reduces
    only what the map draws.

    Merged records are {lat, lon, frp, n, when, step} and deliberately carry
    NO 'confidence'. A bin merges up to ~100 detections of mixed classes
    (spec 4.3), so a centroid has a *set* of confidences, not one; collapsing
    that set to a scalar would invent an attribute no detection has. The
    filter's provenance lives in the artifacts instead: fetch.py writes the
    verbatim firms.csv and reports firms_total (full set) alongside
    firms_map_records (this thinned set).
    """
    bins: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["confidence"] == "l":
            continue
        key = (_step_key(r["when"]),
               round(r["lon"] / BIN_DEG),
               round(merc_y(r["lat"]) / BIN_DEG))
        bins[key].append(r)

    merged = []
    for (step, _, _), group in bins.items():
        total = sum(g["frp"] for g in group)
        if total > 0:
            lat = sum(g["lat"] * g["frp"] for g in group) / total
            lon = sum(g["lon"] * g["frp"] for g in group) / total
        else:  # zero-FRP detections still count and still get drawn
            lat = sum(g["lat"] for g in group) / len(group)
            lon = sum(g["lon"] for g in group) / len(group)
        merged.append({"lat": lat, "lon": lon, "frp": total, "n": len(group),
                       "when": min(g["when"] for g in group), "step": step})

    per_step: dict[dt.datetime, list[dict]] = defaultdict(list)
    for m in merged:
        per_step[m["step"]].append(m)
    n_steps = max(len(per_step), 1)
    budget = max(1, cap // n_steps)
    out = []
    for step in sorted(per_step):
        group = sorted(per_step[step], key=lambda m: -m["frp"])
        out.extend(group[:budget])
    return out[:cap]


def fetch(start: dt.date, end: dt.date, key: str, area: tuple | None = None,
          sleep_s: float = 1.0) -> str:
    """Concatenated CSV across sources and day spans. 9 requests for 11 days."""
    north, west, south, east = config.AREA if area is None else area
    box = f"{west},{south},{east},{north}"
    header, body = None, []
    for day, span in day_spans(start, end):
        for source in config.FIRMS_SOURCES:
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{key}/{source}/{box}/{span}/{day.isoformat()}")
            req = Request(url, headers={"User-Agent": config.USER_AGENT})
            with urlopen(req, timeout=180) as resp:
                text = resp.read().decode("utf-8", "replace").strip()
            lines = text.splitlines()
            if not lines or "," not in lines[0]:
                print(f"firms: {source} {day}+{span}d unexpected reply {text[:120]!r}")
                continue
            header = header or lines[0]
            body.extend(ln for ln in lines[1:] if ln.strip())
            time.sleep(sleep_s)
    if header is None:
        raise RuntimeError("FIRMS returned no usable response")
    return "\n".join([header, *body]) + "\n"
