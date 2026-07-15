# tests/test_geodesic_karney.py
# MENSURA win 4 (CI-gated): Karney geodesics.
# Inverse vs the committed GeodTest subset: |Δs12| < 1e-8 m, |Δazi| < 1e-9°.
# Direct closes the same lines to < 1e-9° in position and azimuth.
# RELEVANT FILES: src/geo/geodesic.rs, tests/data/geodtest_subset.dat

from pathlib import Path

import forge3d


def _cases():
    path = Path(__file__).parent / "data" / "geodtest_subset.dat"
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = [float(v) for v in line.split()]
        cases.append(f)
    return cases


def _ang_diff(a, b):
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return abs(d)


def test_inverse_against_committed_geodtest_subset():
    cases = _cases()
    assert len(cases) == 50, "expected the 50 committed GeodTest cases"
    worst_s = 0.0
    worst_azi = 0.0
    for lat1, lon1, azi1, lat2, lon2, azi2, s12, _a12, _m12, _s12area in cases:
        r = forge3d.geodesic_inverse(lat1, lon1, lat2, lon2)
        ds = abs(r["s12"] - s12)
        da = max(_ang_diff(azi1, r["azi1"]), _ang_diff(azi2, r["azi2"]))
        worst_s = max(worst_s, ds)
        worst_azi = max(worst_azi, da)
        assert ds < 1e-8, f"|Δs12| = {ds} m for {lat1},{lon1} -> {lat2},{lon2}"
        assert da < 1e-9, f"|Δazi| = {da} deg for {lat1},{lon1} -> {lat2},{lon2}"
    print(
        f"GeodTest subset (50 cases): worst |ds12| = {worst_s:.3e} m, "
        f"worst |dazi| = {worst_azi:.3e} deg"
    )


def test_direct_against_committed_geodtest_subset():
    import math

    worst_pos = 0.0
    worst_azi = 0.0
    for lat1, lon1, azi1, lat2, lon2, azi2, s12, _a12, _m12, _s12area in _cases():
        r = forge3d.geodesic_direct(lat1, lon1, azi1, s12)
        dlat = abs(r["lat2"] - lat2)
        dlon = _ang_diff(lon2, r["lon2"]) * abs(math.cos(math.radians(lat2)))
        da = _ang_diff(azi2, r["azi2"])
        worst_pos = max(worst_pos, dlat, dlon)
        worst_azi = max(worst_azi, da)
        assert dlat < 1e-9 and dlon < 1e-9, f"direct position off by ({dlat}, {dlon}) deg"
        assert da < 1e-9, f"direct |Δazi2| = {da} deg"
    print(
        f"GeodTest direct: worst position = {worst_pos:.3e} deg, "
        f"worst |dazi2| = {worst_azi:.3e} deg"
    )


def test_measure_geometries_on_wgs84_returns_metres_never_degrees():
    from forge3d import gis

    square = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }
    result = gis.geometry_measure(square, crs=4326)
    # 1°x1° at the equator: ~1.2308e10 m², ~443.8 km perimeter. A planar
    # (square-degree) answer would be 1.0 / 4.0.
    assert abs(result["area"] - 1.2308e10) < 2e8, result["area"]
    assert abs(result["length"] - 443_770.0) < 2_000.0, result["length"]
    assert result["units"] == "metres_geodesic_wgs84"


def test_measure_geometries_rejects_non_wgs84_geographic_crs():
    import pytest
    from forge3d import gis

    square = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }
    with pytest.raises(Exception) as excinfo:
        gis.geometry_measure(square, crs=4269)
    assert "invalid_crs" in str(excinfo.value)


def test_measure_geometries_projected_stays_planar():
    from forge3d import gis

    square_m = {
        "type": "Polygon",
        "coordinates": [
            [[500000.0, 0.0], [501000.0, 0.0], [501000.0, 1000.0], [500000.0, 1000.0],
             [500000.0, 0.0]]
        ],
    }
    result = gis.geometry_measure(square_m, crs=32631)
    assert abs(result["area"] - 1_000_000.0) < 1e-6
    assert abs(result["length"] - 4_000.0) < 1e-9
    assert result["units"] == "source_crs_planar"


def test_dateline_polygon_regression_179_to_minus_179():
    from forge3d import gis

    ring = [[179.0, 0.0], [-179.0, 0.0], [-179.0, 1.0], [179.0, 1.0], [179.0, 0.0]]
    polygon = {"type": "Polygon", "coordinates": [ring]}
    result = gis.geometry_measure(polygon, crs=4326)
    # 2°x1° patch (~2.46e10 m²), NOT the 358°-wide world-spanning complement.
    assert abs(result["area"] - 2.0 * 1.2308e10) < 4e8, result["area"]

    centroid = gis.geometry_centroid(polygon)
    lon, lat = centroid["geometry"]["coordinates"]
    assert abs(lon) > 179.0, f"centroid lon must sit at the dateline, got {lon}"
    assert abs(lat - 0.5) < 1e-9


def test_transform_bounds_densify_matches_dense_reference():
    from forge3d import gis

    # UTM zone 31N over a wide mid-latitude extent: the northern edge's
    # extremum lies mid-edge, so corner sampling understates the bounds.
    bounds = (-1.5, 40.0, 7.5, 60.0)
    corners_only = gis.transform_bounds(4326, 32631, bounds)
    densified = gis.transform_bounds(4326, 32631, bounds, densify=64)
    reference = gis.transform_bounds(4326, 32631, bounds, densify=1000)

    extent = max(reference[2] - reference[0], reference[3] - reference[1])
    tol = 1e-6 * extent
    for i in range(4):
        assert abs(densified[i] - reference[i]) < tol, (
            f"densified bound {i} off by {abs(densified[i] - reference[i])} m "
            f"(tolerance {tol})"
        )
    # And the corner-only result genuinely differs (the old behaviour was wrong).
    assert any(abs(corners_only[i] - reference[i]) > tol for i in range(4))
