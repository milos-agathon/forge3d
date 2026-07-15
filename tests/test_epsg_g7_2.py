# tests/test_epsg_g7_2.py
# MENSURA win 2 (CI-gated): EPSG Guidance Note 7-2 conformance.
#
# CHOICE (stated per the spec): the authoritative per-method G7-2 worked
# example tests are RUST unit tests living next to each projection
# implementation (src/geo/projections/{tmerc,lcc,aea,stere,merc,geocentric}.rs),
# because four of the six methods (LCC, AEA, polar stereographic, Mercator A)
# are not addressable through bare EPSG codes from Python. This file verifies
# the Python-reachable subset end to end through the shipped wheel, plus an
# optional pyproj differential oracle at 1 mm.
# RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

import math

import pytest

import forge3d
from forge3d import gis


def dms(d, m, s):
    return math.copysign(abs(d) + m / 60.0 + s / 3600.0, d)


def test_g7_2_web_mercator_worked_example():
    # GN7-2 method 1024 (WGS 84 / Pseudo-Mercator):
    # 24°22'54.433"N, 100°20'00.000"W -> E = -11169055.58, N = 2800000.00
    t = gis.create_crs_transformer(4326, 3857)
    e, n = t.transform_point(-dms(100, 20, 0.0), dms(24, 22, 54.433))
    assert abs(e - (-11_169_055.58)) < 1e-2, f"easting residual {abs(e + 11_169_055.58)}"
    assert abs(n - 2_800_000.00) < 1e-2, f"northing residual {abs(n - 2_800_000.00)}"
    # Reverse worked example: N = 2810000.00 -> 24°27'48.889"N.
    inv = gis.create_crs_transformer(3857, 4326)
    lon, lat = inv.transform_point(-11_169_055.58, 2_810_000.00)
    assert abs(lat - dms(24, 27, 48.889)) < 1e-7
    assert abs(lon - (-dms(100, 20, 0.0))) < 1e-7


def test_g7_2_geocentric_worked_example():
    # GN7-2 method 9602 (WGS 84), published geocentric -> geographic:
    # X=3771793.968, Y=140253.342, Z=5124304.349
    # -> 53°48'33.820"N, 2°07'46.380"E, h = 73.0 m
    lon, lat, h = forge3d.ecef_to_wgs84(3_771_793.968, 140_253.342, 5_124_304.349)
    assert abs(lat - dms(53, 48, 33.820)) < 0.0005 / 3600.0
    assert abs(lon - dms(2, 7, 46.380)) < 0.0005 / 3600.0
    assert abs(h - 73.0) < 0.05
    # Forward closes on the published XYZ to well under 1 mm.
    x, y, z = forge3d.wgs84_to_ecef(lon, lat, h)
    residual = ((x - 3_771_793.968) ** 2 + (y - 140_253.342) ** 2 + (z - 5_124_304.349) ** 2) ** 0.5
    assert residual < 1e-6, f"forward residual {residual} m"


def test_utm_round_trip_precision():
    t = gis.create_crs_transformer(4326, 32631)
    inv = gis.create_crs_transformer(32631, 4326)
    worst = 0.0
    for lat in (-79.5, -30.0, 0.5, 45.0, 79.5):
        for lon in (0.1, 1.5, 3.0, 4.5, 5.9):
            e, n = t.transform_point(lon, lat)
            lon2, lat2 = inv.transform_point(e, n)
            worst = max(worst, abs(lon2 - lon), abs(lat2 - lat))
    assert worst < 1e-12, f"worst UTM roundtrip residual {worst} deg"


@pytest.mark.skipif(
    pytest.importorskip("pyproj", reason="pyproj differential oracle unavailable") is None,
    reason="pyproj unavailable",
)
def test_differential_oracle_against_pyproj_to_1mm():
    # Local differential oracle (not required in CI): forge3d's pure-Rust UTM
    # and Web Mercator vs PROJ, 1 mm over a spread of points.
    import numpy as np
    import pyproj

    rng = np.random.default_rng(42)
    lon = rng.uniform(0.05, 5.95, 500)  # UTM zone 31 span
    lat = rng.uniform(-79.9, 83.9, 500)

    t_f3d = gis.create_crs_transformer(4326, 32631)
    t_proj = pyproj.Transformer.from_crs(4326, 32631, always_xy=True)
    worst_utm = 0.0
    for lo, la in zip(lon, lat):
        e1, n1 = t_f3d.transform_point(lo, la)
        e2, n2 = t_proj.transform(lo, la)
        worst_utm = max(worst_utm, abs(e1 - e2), abs(n1 - n2))
    assert worst_utm < 1e-3, f"UTM differs from PROJ by {worst_utm} m"

    t_f3d = gis.create_crs_transformer(4326, 3857)
    t_proj = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    worst_merc = 0.0
    lat_m = rng.uniform(-85.0, 85.0, 500)
    lon_m = rng.uniform(-179.9, 179.9, 500)
    for lo, la in zip(lon_m, lat_m):
        e1, n1 = t_f3d.transform_point(lo, la)
        e2, n2 = t_proj.transform(lo, la)
        worst_merc = max(worst_merc, abs(e1 - e2), abs(n1 - n2))
    assert worst_merc < 1e-3, f"Web Mercator differs from PROJ by {worst_merc} m"

    # ECEF (EPSG 9602 / 4978) differential.
    t_proj = pyproj.Transformer.from_crs(4979, 4978, always_xy=True)
    worst_ecef = 0.0
    for lo, la in zip(lon_m[:100], lat_m[:100]):
        x1, y1, z1 = forge3d.wgs84_to_ecef(lo, la, 1234.5)
        x2, y2, z2 = t_proj.transform(lo, la, 1234.5)
        worst_ecef = max(worst_ecef, abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
    assert worst_ecef < 1e-3, f"ECEF differs from PROJ by {worst_ecef} m"
    print(
        f"pyproj differential oracle: UTM {worst_utm:.2e} m, "
        f"WebMercator {worst_merc:.2e} m, ECEF {worst_ecef:.2e} m"
    )
