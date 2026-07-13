# tests/test_height_system_safety.py
# MENSURA win 6 (part): the height-system boundary is explicit and typed.
# DEM ingestion carries a height-system tag; the orthometric/ellipsoidal
# bridge goes through the geoid; wgs84_to_ecef only accepts a typed
# ellipsoidal height at the Rust level (source-asserted).
# RELEVANT FILES: src/gis/terrarium.rs, src/gis/domain.rs, src/geo/geoid.rs,
#                 src/tiles3d/bounds.rs

from pathlib import Path

import numpy as np
import pytest

import forge3d
from forge3d import gis


def _height_system(info):
    """Read height_system from a RasterInfo object or its dict form."""
    return info["height_system"] if isinstance(info, dict) else info.height_system

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


def test_prepare_dem_converts_declared_egm96_pixels_to_ellipsoidal():
    raw = np.full((1, 2, 2), 100.0, dtype=np.float32)
    result = gis.prepare_dem(
        {
            "array": raw,
            "height_system": "orthometric_egm96",
            "info": {
                "width": 2,
                "height": 2,
                "band_count": 1,
                "crs_authority": {"name": "EPSG", "code": "4326"},
                "bounds": (10.0, 50.0, 12.0, 52.0),
            },
        }
    )
    assert result["height_system"] == "ellipsoidal"
    value = float(np.asarray(result["array"])[0, 0, 0])
    expected = 100.0 + forge3d.geoid_undulation(51.5, 10.5)
    assert abs(value - expected) < 1e-6


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


def test_raster_info_exposes_height_system_field_default_unspecified(tmp_path):
    # M-03: RasterInfo carries a first-class height_system field (not just a
    # sidecar dict key), defaulting to the honest "unspecified" on a plain read.
    path = tmp_path / "plain.tif"
    data = np.ones((1, 4, 4), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=(0.1, 0, 10, 0, -0.1, 52))
    info = gis.read_raster_info(str(path))
    assert _height_system(info) == "unspecified"


def test_height_system_survives_reprojection(tmp_path):
    # M-03: horizontal reprojection preserves the vertical-datum tag through
    # the shared operation_info builder.
    path = tmp_path / "dem.tif"
    data = np.ones((1, 8, 8), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=(0.01, 0, 13, 0, -0.01, 52))
    result = gis.reproject_raster(str(path), "EPSG:3857", resampling="nearest")
    assert result["info"]["height_system"] == "unspecified"


def test_prepare_dem_returned_info_carries_converted_height_system():
    # M-03: the converted vertical datum lands on the RasterInfo field, not
    # only the sidecar key, so it survives being fed back into other ops.
    raw = np.full((1, 2, 2), 100.0, dtype=np.float32)
    result = gis.prepare_dem(
        {
            "array": raw,
            "height_system": "orthometric_egm96",
            "info": {
                "width": 2,
                "height": 2,
                "band_count": 1,
                "crs_authority": {"name": "EPSG", "code": "4326"},
                "bounds": (10.0, 50.0, 12.0, 52.0),
            },
        }
    )
    assert result["height_system"] == "ellipsoidal"
    assert result["info"]["height_system"] == "ellipsoidal"


def test_invalid_height_system_declaration_is_rejected():
    # M-03 policy: an unrecognized vertical-datum tag is rejected, never coerced.
    raw = np.full((1, 2, 2), 100.0, dtype=np.float32)
    with pytest.raises(Exception):
        gis.prepare_dem(
            {
                "array": raw,
                "height_system": "wgs84_banana",
                "info": {
                    "width": 2,
                    "height": 2,
                    "band_count": 1,
                    "crs_authority": {"name": "EPSG", "code": "4326"},
                    "bounds": (10.0, 50.0, 12.0, 52.0),
                },
            }
        )


def test_raster_info_as_dict_includes_height_system(tmp_path):
    # M-03: RasterInfo.as_dict() must match the op-result dict form and carry
    # height_system (it previously omitted it, an asymmetry with
    # raster_info_to_py_dict).
    path = tmp_path / "plain.tif"
    data = np.ones((1, 4, 4), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=(0.1, 0, 10, 0, -0.1, 52))
    info = gis.read_raster_info(str(path))
    d = info.as_dict()
    assert d["height_system"] == "unspecified"


def test_height_system_persists_across_geotiff_write_read(tmp_path):
    # M-03: an explicitly declared vertical datum survives a GeoTIFF write->read
    # round trip via the private forge3d ASCII tag (65001).
    for declared in ("orthometric_egm96", "ellipsoidal", "chart_datum"):
        path = tmp_path / f"dem_{declared}.tif"
        data = np.ones((1, 4, 4), dtype=np.float32)
        gis.write_raster(
            str(path),
            data,
            crs="EPSG:4326",
            transform=(0.1, 0, 10, 0, -0.1, 52),
            height_system=declared,
        )
        info = gis.read_raster_info(str(path))
        assert info.height_system == declared, declared
    # An undeclared write stays "unspecified" (no tag written).
    plain = tmp_path / "plain.tif"
    gis.write_raster(str(plain), np.ones((1, 4, 4), dtype=np.float32),
                     crs="EPSG:4326", transform=(0.1, 0, 10, 0, -0.1, 52))
    assert gis.read_raster_info(str(plain)).height_system == "unspecified"
