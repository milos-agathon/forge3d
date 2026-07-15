# tests/test_height_system_safety.py
# MENSURA win 6 (part): the height-system boundary is explicit and typed.
# DEM ingestion carries a height-system tag; the orthometric/ellipsoidal
# bridge goes through the geoid; wgs84_to_ecef only accepts a typed
# ellipsoidal height at the Rust level (source-asserted).
# RELEVANT FILES: src/gis/terrarium.rs, src/gis/domain.rs, src/geo/geoid.rs,
#                 src/tiles3d/bounds.rs

from pathlib import Path

import numpy as np

import forge3d
from forge3d import gis

REPO_SRC = Path(__file__).parent.parent / "src"


def test_terrarium_decode_carries_orthometric_height_system_tag():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 128  # 128*256 - 32768 = 0 m
    result = gis.decode_terrarium_dem(rgb)
    assert result["height_system"] == "orthometric_egm96"


def test_prepare_dem_carries_height_system_tag():
    dem = np.linspace(0.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    result = gis.prepare_dem(dem)
    assert "height_system" in result
    assert result["height_system"] == "unspecified"


def test_orthometric_and_ellipsoidal_differ_by_exactly_n():
    lat, lon = -14.6212170, 305.0211140
    n = forge3d.geoid_undulation(lat, lon)
    for h in (-431.0, 0.0, 8848.86):
        ell = forge3d.orthometric_to_ellipsoidal(h, lat, lon)
        assert abs((ell - h) - n) < 1e-12
        assert abs(forge3d.ellipsoidal_to_orthometric(ell, lat, lon) - h) < 1e-12


def test_wgs84_to_ecef_signature_is_typed_ellipsoidal_only():
    # Source-level lock: the Rust conversion consumed by 3D Tiles bounding
    # volumes must accept ONLY a typed ellipsoidal height — no bare f64/f32
    # height parameter (that was the pre-MENSURA silent orthometric mix-in).
    bounds_rs = (REPO_SRC / "tiles3d" / "bounds.rs").read_text(encoding="utf-8")
    assert "height: Height<Ellipsoidal>" in bounds_rs, (
        "tiles3d wgs84_to_ecef must take Height<Ellipsoidal>"
    )
    assert "height: f64" not in bounds_rs and "height: f32" not in bounds_rs, (
        "tiles3d wgs84_to_ecef must not accept an untyped height"
    )
    # And the f64 conversion is the geo engine's, not a local reimplementation.
    assert "wgs84_geodetic_to_ecef" in bounds_rs


def test_compile_fail_doctests_are_present_in_units_rs():
    # cargo test --doc proves these actually fail to compile; this test locks
    # that at least the five required proofs stay present in the source.
    units_rs = (REPO_SRC / "geo" / "units.rs").read_text(encoding="utf-8")
    count = units_rs.count("```compile_fail")
    assert count >= 5, f"expected >= 5 compile_fail doctests, found {count}"
    for marker in (
        "let _ = l + a;",           # metre + degree
        "let _ = e - o;",           # ellipsoidal - orthometric
        "let _ = a - b;",           # epoch mismatch and CRS mismatch
        "as f32",                    # raw world f64 -> f32 cast
    ):
        assert marker in units_rs, f"missing compile_fail proof body: {marker}"


def test_dem_conversion_is_per_pixel_exact():
    dem = np.full((3, 3), 100.0)
    bounds = (102.0, 46.5, 103.0, 47.25)
    out = forge3d.dem_orthometric_to_ellipsoidal(dem, bounds)
    # Different pixels get different N; every pixel differs from the raw DEM
    # by its own local undulation.
    lat0 = bounds[3] - 0.5 * (bounds[3] - bounds[1]) / 3
    lon0 = bounds[0] + 0.5 * (bounds[2] - bounds[0]) / 3
    n00 = forge3d.geoid_undulation(lat0, lon0)
    assert abs(out[0, 0] - (100.0 + n00)) < 1e-9
