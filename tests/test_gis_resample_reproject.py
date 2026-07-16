"""G-002b raster resampling and reprojection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS G-002b tests require the compiled _forge3d extension",
)


def _write(path: Path, data=None, **kwargs):
    if data is None:
        data = np.arange(4, dtype=np.float32).reshape(2, 2)
    return gis.write_raster(path, data, **kwargs)


def test_nearest_and_shape_target(tmp_path: Path):
    path = tmp_path / "nearest.tif"
    _write(
        path,
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        nodata=0,
    )

    result = gis.resample_raster(path, (4, 4), method="nearest")

    assert result["resampling"] == "nearest"
    assert result["array"].shape == (1, 4, 4)
    assert isinstance(result["info"], dict)
    assert result["array"][0, 0, 0] == 1
    assert result["array"][0, -1, -1] == 4
    assert result["info"]["width"] == 4
    assert result["info"]["height"] == 4
    assert result["info"]["transform"] == pytest.approx((0.5, 0.0, 0.0, 0.0, -0.5, 2.0))
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert result["info"]["nodata_per_band"] == [0.0]
    assert result["info"]["dtype_per_band"] == ["uint8"]


def test_resample_accepts_array_source():
    data = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)

    result = gis.resample_raster(data, (1, 1), method="bilinear")

    assert result["array"].shape == (1, 1, 1)
    assert isinstance(result["info"], dict)
    assert result["array"][0, 0, 0] == pytest.approx(15.0)
    assert result["info"]["is_georeferenced"] is False


def test_array_resolution_target_requires_transform():
    data = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="missing_transform"):
        gis.resample_raster(data, {"resolution": 1.0}, method="nearest")


def test_resample_target_forms_are_explicit(tmp_path: Path):
    path = tmp_path / "target_forms.tif"
    info = _write(
        path,
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
    )

    by_shape = gis.resample_raster(path, {"shape": (4, 4)}, method="nearest")
    by_info_resolution = gis.resample_raster(info, {"resolution": 1.0}, method="nearest")
    by_float_tuple = gis.resample_raster(path, (1.0, 1.0), method="nearest")

    assert by_shape["info"]["width"] == 4
    assert by_shape["info"]["height"] == 4
    assert by_info_resolution["info"]["resolution"] == pytest.approx((1.0, 1.0))
    assert by_float_tuple["info"]["resolution"] == pytest.approx((1.0, 1.0))

    with pytest.raises(ValueError, match="InvalidArgument"):
        gis.resample_raster(
            path,
            {"shape": (4, 4), "resolution": 1.0},
            method="nearest",
        )


def test_bilinear_resampling(tmp_path: Path):
    path = tmp_path / "bilinear.tif"
    _write(
        path,
        np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )

    result = gis.resample_raster(path, (1, 1), method="bilinear")

    assert result["array"].shape == (1, 1, 1)
    assert result["array"][0, 0, 0] == pytest.approx(15.0)
    assert result["resampling"] == "bilinear"


def test_nodata_aware_resampling_policy(tmp_path: Path):
    path = tmp_path / "nodata_resample.tif"
    _write(
        path,
        np.array([[10.0, -9999.0], [30.0, 50.0]], dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        nodata=-9999.0,
    )

    nearest = gis.resample_raster(path, (4, 4), method="nearest")
    bilinear = gis.resample_raster(path, (1, 1), method="bilinear")

    assert nearest["array"][0, 0, 3] == pytest.approx(-9999.0)
    assert bilinear["array"][0, 0, 0] == pytest.approx(30.0)
    assert bilinear["info"]["nodata_per_band"] == [-9999.0]


def test_resolution_target(tmp_path: Path):
    path = tmp_path / "resolution.tif"
    _write(
        path,
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
    )

    result = gis.resample_raster(path, {"resolution": (1.0, 1.0)}, method="nearest")

    assert result["info"]["width"] == 4
    assert result["info"]["height"] == 4
    assert result["info"]["resolution"] == pytest.approx((1.0, 1.0))


def test_resampling_requires_explicit_supported_method(tmp_path: Path):
    path = tmp_path / "method.tif"
    _write(path)

    with pytest.raises(ValueError, match="resampling_required"):
        gis.resample_raster(path, (4, 4))

    with pytest.raises(ValueError, match="unsupported_resampling_method"):
        gis.resample_raster(path, (4, 4), method="cubic")


def test_valid_epsg_to_epsg_reprojection(tmp_path: Path):
    path = tmp_path / "reproject.tif"
    _write(
        path,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, -1.0, 0.0, -1.0, 1.0),
        nodata=-9999.0,
    )

    result = gis.reproject_raster(path, "EPSG:3857", resampling="nearest")
    info = result["info"]

    assert result["resampling"] == "nearest"
    assert result["array"].shape == (1, 2, 2)
    assert isinstance(info, dict)
    assert info["width"] == 2
    assert info["height"] == 2
    assert info["crs_authority"] == {"name": "EPSG", "code": "3857"}
    assert info["bounds"][0] < 0.0 < info["bounds"][2]
    assert info["bounds"][1] < 0.0 < info["bounds"][3]
    assert info["nodata_per_band"] == [-9999.0]
    assert result["resampling"] == "nearest"


def test_calculate_default_transform_returns_grid_and_crs_metadata(tmp_path: Path):
    path = tmp_path / "default_transform.tif"
    info = _write(
        path,
        np.ones((2, 2), dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, -1.0, 0.0, -1.0, 1.0),
    )

    result = gis.calculate_default_transform(info, "EPSG:3857")

    assert result["width"] == 2
    assert result["height"] == 2
    assert result["dst_crs"]["authority"] == {"name": "EPSG", "code": "3857"}
    assert result["bounds"][0] < 0.0 < result["bounds"][2]
    assert result["bounds"][1] < 0.0 < result["bounds"][3]
    assert result["transform"][0] > 0.0
    assert result["transform"][4] < 0.0


def test_warped_vrt_info_returns_virtual_grid_without_materializing(tmp_path: Path):
    path = tmp_path / "virtual_source.tif"
    info = _write(
        path,
        np.ones((2, 2), dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, -1.0, 0.0, -1.0, 1.0),
    )

    result = gis.warped_vrt_info(info, "EPSG:3857", resampling="bilinear")

    assert result["driver"] == "VRT"
    assert result["is_virtual"] is True
    assert result["materialized"] is False
    assert result["resampling"] == "bilinear"
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "3857"}
    assert result["info"]["width"] == 2
    assert result["info"]["height"] == 2
    with pytest.raises(ValueError, match="resampling_required"):
        gis.warped_vrt_info(info, "EPSG:3857")


def test_reprojection_failures_are_stable(tmp_path: Path):
    missing_crs = tmp_path / "missing_crs.tif"
    invalid_dst = tmp_path / "invalid_dst.tif"
    _write(
        missing_crs,
        transform=(1.0, 0.0, -1.0, 0.0, -1.0, 1.0),
    )
    _write(
        invalid_dst,
        crs="EPSG:4326",
        transform=(1.0, 0.0, -1.0, 0.0, -1.0, 1.0),
    )

    with pytest.raises(ValueError, match="MissingCrs"):
        gis.reproject_raster(missing_crs, "EPSG:3857", resampling="nearest")

    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.reproject_raster(invalid_dst, "EPSG:9999", resampling="nearest")

    with pytest.raises(ValueError, match="resampling_required"):
        gis.reproject_raster(invalid_dst, "EPSG:3857")

    with pytest.raises(ValueError, match="unsupported_resampling_method"):
        gis.reproject_raster(invalid_dst, "EPSG:3857", resampling="lanczos")


def test_crs_transformer_creation_and_use():
    transform = gis.CrsTransform.from_crs("EPSG:4326", "EPSG:3857")
    x, y = transform.transform_point(0.0, 0.0)
    bounds = transform.transform_bounds((-1.0, -1.0, 1.0, 1.0))

    assert transform.src_crs == "EPSG:4326"
    assert transform.dst_crs == "EPSG:3857"
    assert transform.axis_order_policy == "always_xy"
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert bounds[0] < 0.0 < bounds[2]
    assert bounds[1] < 0.0 < bounds[3]


def test_crs_transformer_rejects_invalid_or_unavailable():
    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.CrsTransform.from_crs("not-a-crs", "EPSG:3857")

    # MENSURA: WGS84 UTM zones and a curated set of projected CRSs (3857, 3395,
    # 2154, 5070, 5041, 5042) are handled by the built-in pure-Rust engine now.
    # A parseable-but-untransformable pair reports BackendUnavailable; codes
    # outside the parse whitelist fail earlier with InvalidCrs.
    with pytest.raises(RuntimeError, match="BackendUnavailable"):
        gis.CrsTransform.from_crs("EPSG:4326", "EPSG:4269")
    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.CrsTransform.from_crs("EPSG:4326", "EPSG:27700")  # OSGB36: not curated
    utm = gis.CrsTransform.from_crs("EPSG:4326", "EPSG:32631")
    e, n = utm.transform_point(3.0, 0.0)
    assert e == pytest.approx(500000.0, abs=1e-6)
    assert n == pytest.approx(0.0, abs=1e-6)


def test_curated_projected_crs_are_reachable_natively():
    # MENSURA M-02: LCC-2SP (2154), Mercator-A (3395), Albers (5070), and
    # Polar-Stereographic-A (5041/5042) are reachable through the pure-Rust
    # engine and round-trip through WGS84 to sub-millidegree.
    cases = [
        ("EPSG:3395", 10.0, 45.0),
        ("EPSG:2154", 2.5, 47.0),
        ("EPSG:5070", -96.0, 38.0),
        ("EPSG:5041", 30.0, 85.0),
        ("EPSG:5042", 30.0, -85.0),
    ]
    for code, lon, lat in cases:
        fwd = gis.CrsTransform.from_crs("EPSG:4326", code)
        inv = gis.CrsTransform.from_crs(code, "EPSG:4326")
        e, n = fwd.transform_point(lon, lat)
        rlon, rlat = inv.transform_point(e, n)
        assert rlon == pytest.approx(lon, abs=1e-6), code
        assert rlat == pytest.approx(lat, abs=1e-6), code


def test_reproject_unsupported_crs_pair_reports_backend_unavailable(tmp_path: Path):
    path = tmp_path / "unsupported_pair.tif"
    _write(
        path,
        np.ones((2, 2), dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )

    # MENSURA: EPSG:32631 reprojects natively now. A parseable geographic CRS
    # with no transform path reaches the per-pixel policy and raises
    # TransformFailed (M-05 contract; see test_reproject_error_policy); codes
    # outside the parse whitelist fail earlier at parse with InvalidCrs.
    with pytest.raises(RuntimeError, match="TransformFailed"):
        gis.reproject_raster(path, "EPSG:4269", resampling="nearest")
    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.reproject_raster(path, "EPSG:27700", resampling="nearest")
    result = gis.reproject_raster(path, "EPSG:32631", resampling="nearest")
    assert result["info"]["crs_authority"]["code"] == "32631"


def test_reprojection_uses_nodata_for_out_of_source_pixels(tmp_path: Path):
    path = tmp_path / "reproject_nodata.tif"
    _write(
        path,
        np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, -1.0, 0.0, -1.0, 1.0),
        nodata=-9999.0,
    )

    result = gis.reproject_raster(path, "EPSG:3857", resampling="bilinear")

    assert result["info"]["nodata_per_band"] == [-9999.0]
    assert np.any(np.isclose(result["array"], -9999.0))


def test_reprojection_uses_nodata_for_rotated_source_footprint(tmp_path: Path):
    path = tmp_path / "rotated_reproject_nodata.tif"
    _write(
        path,
        np.arange(16, dtype=np.float32).reshape(4, 4),
        crs="EPSG:4326",
        transform=(1.0, 1.0, -2.0, 0.0, -1.0, 2.0),
        nodata=-9999.0,
    )

    result = gis.reproject_raster(path, "EPSG:3857", resampling="nearest")

    assert np.any(np.isclose(result["array"], -9999.0))
    assert result["info"]["nodata_per_band"] == [-9999.0]
