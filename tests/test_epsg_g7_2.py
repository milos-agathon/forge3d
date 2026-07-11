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
    # GN7-2 method 1024; expected values retain f64 formula precision.
    t = gis.create_crs_transformer(4326, 3857)
    e, n = t.transform_point(-dms(100, 20, 0.0), dms(24, 22, 54.433))
    assert abs(e - (-11_169_055.576258447)) < 1e-3, f"easting residual {abs(e + 11_169_055.576258447)}"
    assert abs(n - 2_800_000.003136157) < 1e-3, f"northing residual {abs(n - 2_800_000.003136157)}"
    # Reverse worked example: N = 2810000.00 -> 24°27'48.889"N.
    inv = gis.create_crs_transformer(3857, 4326)
    lon, lat = inv.transform_point(-11_169_055.576258447, 2_810_000.00)
    assert abs(lat - 24.463580315801703) < 9e-9
    assert abs(lon + 100.33333333333333) < 9e-9


def test_g7_2_geocentric_worked_example():
    # GN7-2 method 9602 (WGS 84), published geocentric -> geographic:
    # X=3771793.968, Y=140253.342, Z=5124304.349
    # -> 53°48'33.820"N, 2°07'46.380"E, h = 73.0 m
    lon, lat, h = forge3d.ecef_to_wgs84(3_771_793.968, 140_253.342, 5_124_304.349)
    assert abs(lat - dms(53, 48, 33.820)) < 9e-9
    assert abs(lon - dms(2, 7, 46.380)) < 9e-9
    assert abs(h - 73.0) < 1e-3
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


@pytest.mark.parametrize(
    "definition,point,expected",
    [
        ({"method": "tmerc", "a": 6378137.0, "inv_f": 298.257223563, "lat0": 0.0, "lon0": 3.0, "k0": 0.9996, "false_easting": 500000.0, "false_northing": 0.0}, (3.5, 50.0), (535833.4591975091, 5538750.477579043)),
        ({"method": "lcc_2sp", "a": 6378137.0, "inv_f": 298.257223563, "lat0": 40.0, "lon0": -96.0, "lat1": 33.0, "lat2": 45.0, "false_easting": 0.0, "false_northing": 0.0}, (-95.0, 41.0), (83721.00872539113, 110933.69477422256)),
        ({"method": "aea", "a": 6378137.0, "inv_f": 298.257223563, "lat0": 23.0, "lon0": -96.0, "lat1": 29.5, "lat2": 45.5, "false_easting": 0.0, "false_northing": 0.0}, (-90.0, 35.0), (542742.2705834078, 1343939.731913535)),
        ({"method": "stere_a", "a": 6378137.0, "inv_f": 298.257223563, "lat0": 90.0, "lon0": 0.0, "k0": 0.994, "false_easting": 2000000.0, "false_northing": 2000000.0}, (44.0, 73.0), (3320416.747359853, 632668.4312721284)),
        ({"method": "merc_a", "a": 6378137.0, "inv_f": 298.257223563, "lon0": 0.0, "k0": 1.0, "false_easting": 0.0, "false_northing": 0.0}, (12.0, 45.0), (1335833.8895192828, 5591295.9185533915)),
    ],
)
def test_explicit_projection_definitions_reach_shipped_dispatcher(definition, point, expected):
    forward = gis.create_crs_transformer(4326, definition)
    inverse = gis.create_crs_transformer(definition, 4326)
    projected = forward.transform_point(*point)
    forward_error_mm = max(abs(projected[i] - expected[i]) for i in range(2)) * 1000.0
    recovered = inverse.transform_point(*expected)
    inverse_error_mm = max(abs(recovered[i] - point[i]) * 111_320_000.0 for i in range(2))
    print(f"{definition['method']}: forward={forward_error_mm:.6f} mm inverse={inverse_error_mm:.6f} mm")
    assert forward_error_mm <= 1.0
    assert inverse_error_mm <= 1.0


def test_web_mercator_method_code_reaches_shipped_dispatcher_without_ellipsoid():
    forward = gis.create_crs_transformer(4326, {"method": 1024})
    inverse = gis.create_crs_transformer({"method": "1024"}, 4326)
    projected = forward.transform_point(12.0, 45.0)
    assert inverse.transform_point(1335833.8895192828, 5621521.486192066) == pytest.approx((12.0, 45.0), abs=9e-9)


def test_geocentric_method_reaches_shipped_three_dimensional_dispatcher():
    forward = gis.create_crs_transformer(4326, 4978)
    inverse = gis.create_crs_transformer(4978, 4326)
    geographic = (2.129550001320768, 53.80939443996213, 72.99993067141622)
    ecef = forward.transform_point3(*geographic)
    assert inverse.transform_point3(*ecef) == pytest.approx(geographic, abs=1e-9)


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
