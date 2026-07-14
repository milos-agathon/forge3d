# tests/test_projection_oracle.py
# MENSURA M-02 evidence closure: the full external PROJ/pyproj differential
# oracle over a fixed-seed 10,000-point-per-method corpus.
#
# This gate is DEV-ONLY (pyproj is never a runtime/wheel dependency of forge3d)
# and exercises the SHIPPED public GIS dispatcher end to end
# (`gis.create_crs_transformer` + `forge3d.wgs84_to_ecef`/`ecef_to_wgs84`),
# never a private duplicated formula. Every shipped numerical path is covered:
# generic transverse Mercator, WGS84 UTM (north, southern hemisphere, and a
# high-zone edge), Web Mercator, Mercator A, Lambert Conformal Conic 2SP,
# Albers Equal Area, Polar Stereographic A (north and south), and geodetic<->ECEF.
#
# Thresholds: forward residual < 1 mm (projected-metre Euclidean), inverse
# residual < 1 mm (measured in TRUE metres via an authoritative WGS84 geodesic
# distance, not a degrees-to-metres approximation), and ECEF residual < 1 mm
# (3D Euclidean metres). Exactly 10,000 points are evaluated per method, and any
# non-finite output fails the gate.
# RELEVANT FILES: src/geo/projections/mod.rs, src/gis/crs.rs

import math

import pytest

import forge3d
from forge3d import gis

np = pytest.importorskip("numpy")
pyproj = pytest.importorskip("pyproj", reason="pyproj differential oracle is dev-only")

# Documented fixed seed for the 10,000-point-per-method corpus.
SEED = 20260713
N_POINTS = 10_000
FORWARD_TOL_MM = 1.0
INVERSE_TOL_MM = 1.0
ECEF_TOL_MM = 1.0

_TMERC_DEF = {
    "method": "tmerc",
    "a": 6378137.0,
    "inv_f": 298.257223563,
    "lat0": 0.0,
    "lon0": 3.0,
    "k0": 0.9996,
    "false_easting": 500000.0,
    "false_northing": 0.0,
}
_TMERC_PROJ4 = (
    "+proj=tmerc +lat_0=0 +lon_0=3 +k=0.9996 +x_0=500000 +y_0=0 "
    "+a=6378137 +rf=298.257223563 +units=m +no_defs"
)

# (method label, forge3d CRS spec, pyproj CRS spec, lon range, lat range)
PROJECTED_METHODS = [
    ("tmerc_generic", _TMERC_DEF, _TMERC_PROJ4, (-9.0, 15.0), (-80.0, 84.0)),
    ("utm31n_epsg32631", 32631, 32631, (0.1, 5.9), (-79.0, 83.0)),
    # Southern hemisphere + easternmost UTM zone (zone 60 edge).
    ("utm60s_epsg32760", 32760, 32760, (174.1, 179.9), (-79.0, -0.5)),
    ("web_mercator_epsg3857", 3857, 3857, (-179.9, 179.9), (-85.0, 85.0)),
    ("mercator_a_epsg3395", 3395, 3395, (-179.9, 179.9), (-80.0, 80.0)),
    ("lcc_2sp_epsg2154", 2154, 2154, (-4.5, 8.0), (42.0, 51.0)),
    ("albers_equal_area_epsg5070", 5070, 5070, (-119.0, -75.0), (25.0, 49.0)),
    ("polar_stereo_north_epsg5041", 5041, 5041, (-180.0, 180.0), (60.0, 89.0)),
    ("polar_stereo_south_epsg5042", 5042, 5042, (-180.0, 180.0), (-89.0, -60.0)),
]

# Authoritative WGS84 geodesic for measuring inverse residuals in true metres.
_GEOD = pyproj.Geod(ellps="WGS84")


def _sample(rng, lon_range, lat_range):
    lon = rng.uniform(lon_range[0], lon_range[1], N_POINTS)
    lat = rng.uniform(lat_range[0], lat_range[1], N_POINTS)
    return lon, lat


def _run_projected_method(label, f3d_crs, proj_crs, lon, lat):
    """Return (worst_forward_mm, worst_forward_input, worst_inverse_mm,
    worst_inverse_input, evaluated) for one projection method."""
    f_fwd = gis.create_crs_transformer(4326, f3d_crs)
    f_inv = gis.create_crs_transformer(f3d_crs, 4326)
    p_fwd = pyproj.Transformer.from_crs(4326, proj_crs, always_xy=True)
    p_inv = pyproj.Transformer.from_crs(proj_crs, 4326, always_xy=True)

    # pyproj forward (vectorized) gives the authoritative projected points, which
    # also serve as the inputs we invert with BOTH backends.
    xp, yp = p_fwd.transform(lon, lat)
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    assert np.all(np.isfinite(xp)) and np.all(np.isfinite(yp)), (
        f"{label}: pyproj forward produced a non-finite point"
    )

    # forge3d forward (looped: transform_point is single-point) and forge3d
    # inverse of the pyproj-forward point (both through the shipped dispatcher).
    xf = np.empty(N_POINTS)
    yf = np.empty(N_POINTS)
    lon_rec = np.empty(N_POINTS)
    lat_rec = np.empty(N_POINTS)
    evaluated = 0
    for i in range(N_POINTS):
        ef, nf = f_fwd.transform_point(float(lon[i]), float(lat[i]))
        xf[i], yf[i] = ef, nf
        loni, lati = f_inv.transform_point(float(xp[i]), float(yp[i]))
        lon_rec[i], lat_rec[i] = loni, lati
        evaluated += 1

    assert np.all(np.isfinite(xf)) and np.all(np.isfinite(yf)), (
        f"{label}: forge3d forward produced a non-finite point"
    )
    assert np.all(np.isfinite(lon_rec)) and np.all(np.isfinite(lat_rec)), (
        f"{label}: forge3d inverse produced a non-finite point"
    )

    # Forward residual: projected-metre Euclidean distance between backends.
    fwd_m = np.hypot(xf - xp, yf - yp)
    fwd_i = int(np.argmax(fwd_m))
    worst_fwd_mm = float(fwd_m[fwd_i]) * 1000.0
    worst_fwd_in = (float(lon[fwd_i]), float(lat[fwd_i]))

    # Inverse residual: TRUE metres via the WGS84 geodesic between forge3d's and
    # pyproj's inverse of the SAME projected point (never deg-to-m by a constant).
    lon_pp, lat_pp = p_inv.transform(xp, yp)
    lon_pp = np.asarray(lon_pp, dtype=float)
    lat_pp = np.asarray(lat_pp, dtype=float)
    _, _, inv_dist_m = _GEOD.inv(lon_rec, lat_rec, lon_pp, lat_pp)
    inv_dist_m = np.abs(np.asarray(inv_dist_m, dtype=float))
    assert np.all(np.isfinite(inv_dist_m)), (
        f"{label}: inverse residual is non-finite"
    )
    inv_i = int(np.argmax(inv_dist_m))
    worst_inv_mm = float(inv_dist_m[inv_i]) * 1000.0
    worst_inv_in = (float(xp[inv_i]), float(yp[inv_i]))

    return worst_fwd_mm, worst_fwd_in, worst_inv_mm, worst_inv_in, evaluated


def _run_ecef(lon, lat, h):
    """Geodetic<->ECEF, forward and inverse residuals as 3D Euclidean metres."""
    p_fwd = pyproj.Transformer.from_crs(4979, 4978, always_xy=True)
    Xp, Yp, Zp = p_fwd.transform(lon, lat, h)
    Xp = np.asarray(Xp, dtype=float)
    Yp = np.asarray(Yp, dtype=float)
    Zp = np.asarray(Zp, dtype=float)

    worst_fwd_m = 0.0
    worst_fwd_in = None
    worst_inv_m = 0.0
    worst_inv_in = None
    evaluated = 0
    for i in range(N_POINTS):
        xf, yf, zf = forge3d.wgs84_to_ecef(float(lon[i]), float(lat[i]), float(h[i]))
        assert math.isfinite(xf) and math.isfinite(yf) and math.isfinite(zf), (
            "ecef: forge3d forward produced a non-finite point"
        )
        fwd = math.sqrt((xf - Xp[i]) ** 2 + (yf - Yp[i]) ** 2 + (zf - Zp[i]) ** 2)
        if fwd > worst_fwd_m:
            worst_fwd_m = fwd
            worst_fwd_in = (float(lon[i]), float(lat[i]), float(h[i]))

        # forge3d inverse of the pyproj-forward ECEF point, re-projected to ECEF
        # by pyproj so the residual is a true 3D metre distance.
        lonf, latf, hf = forge3d.ecef_to_wgs84(float(Xp[i]), float(Yp[i]), float(Zp[i]))
        assert math.isfinite(lonf) and math.isfinite(latf) and math.isfinite(hf), (
            "ecef: forge3d inverse produced a non-finite point"
        )
        xr, yr, zr = p_fwd.transform(lonf, latf, hf)
        inv = math.sqrt((xr - Xp[i]) ** 2 + (yr - Yp[i]) ** 2 + (zr - Zp[i]) ** 2)
        if inv > worst_inv_m:
            worst_inv_m = inv
            worst_inv_in = (float(Xp[i]), float(Yp[i]), float(Zp[i]))
        evaluated += 1

    return (
        worst_fwd_m * 1000.0,
        worst_fwd_in,
        worst_inv_m * 1000.0,
        worst_inv_in,
        evaluated,
    )


def test_full_projection_oracle_10k_points_per_method_below_1mm():
    rng = np.random.default_rng(SEED)
    report = []

    for label, f3d_crs, proj_crs, lon_range, lat_range in PROJECTED_METHODS:
        lon, lat = _sample(rng, lon_range, lat_range)
        fwd_mm, fwd_in, inv_mm, inv_in, evaluated = _run_projected_method(
            label, f3d_crs, proj_crs, lon, lat
        )
        assert evaluated == N_POINTS, (
            f"{label}: evaluated {evaluated} points, expected {N_POINTS}"
        )
        report.append((label, fwd_mm, fwd_in, inv_mm, inv_in))
        assert fwd_mm < FORWARD_TOL_MM, (
            f"{label}: worst forward residual {fwd_mm:.6e} mm at lon/lat {fwd_in}"
        )
        assert inv_mm < INVERSE_TOL_MM, (
            f"{label}: worst inverse residual {inv_mm:.6e} mm at x/y {inv_in}"
        )

    # geodetic <-> ECEF
    lon, lat = _sample(rng, (-179.9, 179.9), (-89.0, 89.0))
    h = rng.uniform(-500.0, 9000.0, N_POINTS)
    e_fwd_mm, e_fwd_in, e_inv_mm, e_inv_in, evaluated = _run_ecef(lon, lat, h)
    assert evaluated == N_POINTS, (
        f"ecef: evaluated {evaluated} points, expected {N_POINTS}"
    )
    report.append(("geodetic_ecef", e_fwd_mm, e_fwd_in, e_inv_mm, e_inv_in))
    assert e_fwd_mm < ECEF_TOL_MM, (
        f"ecef: worst forward residual {e_fwd_mm:.6e} mm at lon/lat/h {e_fwd_in}"
    )
    assert e_inv_mm < ECEF_TOL_MM, (
        f"ecef: worst inverse residual {e_inv_mm:.6e} mm at ecef {e_inv_in}"
    )

    print(
        f"\nMENSURA projection oracle (seed={SEED}, {N_POINTS} pts/method):\n"
        f"{'method':<30} {'fwd (mm)':>12} {'inv (mm)':>12}"
    )
    for label, fwd_mm, fwd_in, inv_mm, inv_in in report:
        print(f"{label:<30} {fwd_mm:>12.6e} {inv_mm:>12.6e}")
        print(f"    worst forward input : {fwd_in}")
        print(f"    worst inverse input : {inv_in}")
