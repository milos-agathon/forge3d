# tests/test_geoid_egm96.py
# MENSURA win 3 (CI-gated): EGM96 geoid.
# N(lat, lon) at degree/order 120 matches the committed NGA-published test
# values to < 0.5 m, and a DEM tagged orthometric converts to ellipsoidal
# heights differing from the raw values by exactly N (per-pixel, 1e-6 m).
# RELEVANT FILES: src/geo/geoid.rs, assets/geoid/egm96_n120.bin,
#                 tests/data/egm96_test_values.txt

from pathlib import Path

import numpy as np

import forge3d


def _reference_points():
    path = Path(__file__).parent / "data" / "egm96_test_values.txt"
    points = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lat, lon, n_ref, source = line.split()
        points.append((float(lat), float(lon), float(n_ref), source))
    return points


def test_egm96_degree_120_matches_nga_published_values():
    points = _reference_points()
    assert len(points) == 20, "expected the 20 committed NGA reference points"
    worst = 0.0
    worst_at = None
    for lat, lon, n_ref, source in points:
        n = forge3d.geoid_undulation(lat, lon)
        err = abs(n - n_ref)
        if err > worst:
            worst, worst_at = err, (lat, lon, source)
        assert err < 0.5, (
            f"EGM96 residual {err:.3f} m at ({lat}, {lon}) [{source}]: "
            f"got {n:.3f}, want {n_ref:.3f}"
        )
    print(
        "EGM96 degree-120 worst residual vs published degree-360 values: "
        f"{worst:.4f} m at {worst_at}"
    )


def test_known_undulation_signs_and_magnitudes():
    # Sanity anchors: strongly negative over the Indian Ocean low, strongly
    # positive over the North Atlantic/Iceland high.
    assert forge3d.geoid_undulation(5.0, 78.0) < -80.0
    assert forge3d.geoid_undulation(64.0, -22.0) > 50.0


def test_dem_orthometric_to_ellipsoidal_differs_by_exactly_n():
    rng = np.random.default_rng(7)
    rows, cols = 12, 16
    dem = rng.uniform(-100.0, 3000.0, (rows, cols))
    bounds = (13.0, 52.0, 13.4, 52.3)  # (left, bottom, right, top), EPSG:4326
    out = forge3d.dem_orthometric_to_ellipsoidal(dem, bounds)
    assert out.shape == (rows, cols)
    assert out.dtype == np.float64

    left, bottom, right, top = bounds
    worst = 0.0
    for r in range(rows):
        lat = top - (r + 0.5) * (top - bottom) / rows
        for c in range(cols):
            lon = left + (c + 0.5) * (right - left) / cols
            n = forge3d.geoid_undulation(lat, lon)
            expected = dem[r, c] + n
            worst = max(worst, abs(out[r, c] - expected))
    assert worst < 1e-6, f"per-pixel residual {worst} m exceeds 1e-6"


def test_scalar_height_conversions_are_exact_inverses():
    lat, lon, h = 46.8743190, 102.4487290, 812.5
    n = forge3d.geoid_undulation(lat, lon)
    ell = forge3d.orthometric_to_ellipsoidal(h, lat, lon)
    assert abs(ell - (h + n)) < 1e-12
    back = forge3d.ellipsoidal_to_orthometric(ell, lat, lon)
    assert abs(back - h) < 1e-12
