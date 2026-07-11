# tests/test_geodetic_conservation.py
# MENSURA win 1 (CI-gated): the conservation law.
# EPSG:4326 -> UTM zone -> EPSG:3857 -> ECEF -> EPSG:4326 over 10,000
# pseudo-random points (fixed seed, +/-80 deg latitude):
#   max angular residual < 1e-9 deg (~0.11 mm)
#   max ECEF residual    < 1e-4 m
# RELEVANT FILES: src/geo/projections/, src/gis/crs.rs, src/py_functions/geodesy.rs

import numpy as np
import pytest

import forge3d
from forge3d import gis


N_POINTS = 10_000
SEED = 20260710


def _random_points():
    rng = np.random.default_rng(SEED)
    lon = rng.uniform(-180.0, 180.0, N_POINTS)
    lat = rng.uniform(-80.0, 80.0, N_POINTS)
    h = rng.uniform(-500.0, 9000.0, N_POINTS)
    return lon, lat, h


def _utm_epsg(lon, lat):
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0.0 else 32700) + zone


def test_conservation_law_round_trip():
    lon, lat, h = _random_points()

    to_utm = {}
    from_utm = {}
    to_merc_from_utm = {}
    merc_to_wgs = gis.create_crs_transformer(3857, 4326)

    max_angular = 0.0
    max_ecef = 0.0
    for i in range(N_POINTS):
        epsg = _utm_epsg(lon[i], lat[i])
        if epsg not in to_utm:
            to_utm[epsg] = gis.create_crs_transformer(4326, epsg)
            to_merc_from_utm[epsg] = gis.create_crs_transformer(epsg, 3857)
            from_utm[epsg] = gis.create_crs_transformer(epsg, 4326)

        # 4326 -> UTM
        e, n = to_utm[epsg].transform_point(lon[i], lat[i])
        # UTM -> 3857
        mx, my = to_merc_from_utm[epsg].transform_point(e, n)
        # 3857 -> 4326
        lon2, lat2 = merc_to_wgs.transform_point(mx, my)
        # 4326 -> ECEF -> 4326
        ex, ey, ez = forge3d.wgs84_to_ecef(lon2, lat2, h[i])
        lon3, lat3, h3 = forge3d.ecef_to_wgs84(ex, ey, ez)

        max_angular = max(max_angular, abs(lon3 - lon[i]), abs(lat3 - lat[i]))

        # ECEF closure: re-projecting the recovered geodetic point must land
        # on the ECEF of the ORIGINAL point to < 1e-4 m.
        ox, oy, oz = forge3d.wgs84_to_ecef(lon[i], lat[i], h[i])
        bx, by, bz = forge3d.wgs84_to_ecef(lon3, lat3, h3)
        ecef_res = ((bx - ox) ** 2 + (by - oy) ** 2 + (bz - oz) ** 2) ** 0.5
        max_ecef = max(max_ecef, ecef_res)

    print(
        f"conservation law over {N_POINTS} points: "
        f"max angular residual = {max_angular:.3e} deg, "
        f"max ECEF residual = {max_ecef:.3e} m"
    )
    assert max_angular < 1e-9, f"max angular residual {max_angular} deg >= 1e-9"
    assert max_ecef < 1e-4, f"max ECEF residual {max_ecef} m >= 1e-4"


def test_ecef_round_trip_alone_is_tighter_than_f32_ever_could_be():
    # The old `as f32` truncation at src/tiles3d/bounds.rs:134 cost ~0.5 m at
    # ECEF magnitudes. The f64 path must close ~8 orders of magnitude tighter.
    lon, lat, h = 138.7274, 35.3606, 3776.24
    x, y, z = forge3d.wgs84_to_ecef(lon, lat, h)
    lon2, lat2, h2 = forge3d.ecef_to_wgs84(x, y, z)
    x2, y2, z2 = forge3d.wgs84_to_ecef(lon2, lat2, h2)
    residual = ((x2 - x) ** 2 + (y2 - y) ** 2 + (z2 - z) ** 2) ** 0.5
    assert residual < 1e-8, f"ECEF roundtrip residual {residual} m"
    # An f32-quantized X coordinate at this magnitude moves by ~0.25-0.5 m.
    assert abs(np.float64(np.float32(x)) - x) > 1e-2


def test_unsupported_crs_still_raises_not_passthrough():
    # A CRS outside the built-in engine raises (InvalidCrs at parse for codes
    # the TIFF backend does not know, BackendUnavailable at transform for
    # parseable-but-untransformable pairs) — never a silent passthrough.
    with pytest.raises(Exception) as excinfo:
        gis.create_crs_transformer(4326, 2154)
    message = str(excinfo.value)
    assert "InvalidCrs" in message or "BackendUnavailable" in message

    with pytest.raises(Exception) as excinfo:
        gis.create_crs_transformer(4326, 4269)  # parseable geographic CRS
    assert "BackendUnavailable" in str(excinfo.value)
