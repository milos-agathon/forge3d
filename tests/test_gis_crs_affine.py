"""G-002b CRS, affine, bounds, nodata, and windowing tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS G-002b tests require the compiled _forge3d extension",
)


WGS84_WKT = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984"],'
    'CS[ellipsoidal,2],AXIS["longitude",east],AXIS["latitude",north],'
    'ANGLEUNIT["degree",0.0174532925199433]]'
)


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _basic(path: Path, **kwargs):
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    return gis.write_raster(path, data, **kwargs)


def test_inspect_valid_epsg_crs():
    result = gis.inspect_crs("EPSG:4326")

    assert result["missing"] is False
    assert result["authority"] == {"name": "EPSG", "code": "4326"}
    assert "authority" not in result["authority"]
    assert result["axis_order_policy"] == "traditional_gis_xy"
    assert result["source_kind"] == "crs"


def test_python_gis_module_has_no_backend_gis_libraries():
    import ast
    import inspect

    source = inspect.getsource(gis)
    tree = ast.parse(source)
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    for banned in ("rasterio", "geopandas", "shapely", "rioxarray", "xarray", "terra"):
        assert banned not in imports
    assert "_looks_like_dataset_path" not in source


def test_gis_stub_exposes_runtime_api():
    stub = Path(gis.__file__).with_suffix(".pyi").read_text(encoding="utf-8")

    for name in (
        "AffineTransform",
        "CrsTransform",
        "parse_crs",
        "inspect_crs",
        "raster_crs",
        "create_crs_transformer",
        "transform_bounds",
        "web_mercator_bounds",
        "raster_transform",
        "transform_from_origin",
        "transform_from_bounds",
        "array_bounds",
        "raster_bounds",
        "raster_resolution",
        "validate_transform",
        "pixel_convention",
        "rowcol",
        "xy",
        "index",
        "bounds",
        "assign_crs",
        "window_from_bounds",
        "resample_raster",
        "assert_grid_compatible",
        "align_raster_grid",
        "align_raster_to",
        "reproject_vector",
        "validate_geometry",
        "repair_geometry",
        "geometry_measure",
        "geometry_centroid",
        "representative_point",
        "interpolate_line",
        "union_geometries",
        "dissolve_vector",
        "buffer_geometry",
        "clip_vector",
        "intersect_vectors",
        "simplify_geometry",
        "load_boundary",
        "reproject_raster",
        "calculate_default_transform",
        "read_raster_window",
        "window_transform",
    ):
        assert hasattr(gis, name)
        assert name in stub


def test_parse_crs_accepts_epsg_int_string_wkt_and_dict():
    from_int = gis.parse_crs(4326)
    from_string = gis.parse_crs("EPSG:4326")
    from_wkt = gis.parse_crs(WGS84_WKT)
    from_dict = gis.parse_crs({"name": "EPSG", "code": "3857"})

    assert from_int["authority"] == {"name": "EPSG", "code": "4326"}
    assert from_string == from_int
    assert from_wkt["wkt"] == WGS84_WKT
    assert from_dict["authority"] == {"name": "EPSG", "code": "3857"}


def test_inspect_valid_wkt_crs():
    result = gis.inspect_crs(WGS84_WKT)

    assert result["missing"] is False
    assert result["wkt"] == WGS84_WKT
    assert result["authority"] is None


def test_inspect_missing_crs(tmp_path: Path):
    path = tmp_path / "missing_crs.tif"
    _basic(path, transform=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0))

    result = gis.inspect_crs(path)
    from_string = gis.inspect_crs(str(path))

    assert result["missing"] is True
    assert "missing_crs" in _codes(result["warnings"])
    assert from_string["missing"] is True
    assert "missing_crs" in _codes(from_string["warnings"])


def test_inspect_missing_path_reports_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="path not found"):
        gis.inspect_crs(tmp_path / "does-not-exist.tif")


def test_inspect_missing_string_path_reports_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="path not found"):
        gis.inspect_crs(str(tmp_path / "does-not-exist.tif"))

    with pytest.raises(FileNotFoundError, match="path not found"):
        gis.inspect_crs("missing.tif")


def test_inspect_invalid_crs_string_and_dict():
    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.inspect_crs("not-a-crs")

    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.inspect_crs({"authority": "EPSG", "code": "4326"})


def test_raster_crs_reads_path_and_info(tmp_path: Path):
    path = tmp_path / "crs.tif"
    info = _basic(path, crs="EPSG:4326")

    assert gis.raster_crs(info)["authority"] == {"name": "EPSG", "code": "4326"}
    assert gis.raster_crs(path)["authority"] == {"name": "EPSG", "code": "4326"}


def test_create_crs_transformer_alias_matches_class():
    via_function = gis.create_crs_transformer("EPSG:4326", "EPSG:3857")
    via_class = gis.CrsTransform.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    via_ints = gis.CrsTransform.from_crs(4326, 3857, always_xy=True)
    via_dicts = gis.CrsTransform.from_crs(
        {"name": "EPSG", "code": "4326"},
        {"name": "EPSG", "code": "3857"},
        always_xy=True,
    )
    function_from_dicts = gis.create_crs_transformer(
        {"name": "EPSG", "code": "4326"},
        {"name": "EPSG", "code": "3857"},
        always_xy=True,
    )

    assert via_function.src_crs == via_class.src_crs == "EPSG:4326"
    assert via_function.dst_crs == via_class.dst_crs == "EPSG:3857"
    assert via_function.axis_order_policy == "always_xy"
    assert via_function.src_authority == {"name": "EPSG", "code": "4326"}
    assert via_function.dst_authority == {"name": "EPSG", "code": "3857"}
    assert via_ints.src_authority == via_dicts.src_authority == function_from_dicts.src_authority
    assert via_ints.dst_authority == via_dicts.dst_authority == function_from_dicts.dst_authority
    assert via_function.transform_point(0.0, 0.0) == pytest.approx(
        via_class.transform_point(0.0, 0.0)
    )


def test_crs_assignment_overwrite_and_no_reprojection(tmp_path: Path):
    path = tmp_path / "assign.tif"
    original = np.arange(6, dtype=np.uint16).reshape(2, 3)
    gis.write_raster(
        path,
        original,
        crs="EPSG:4326",
        transform=(1.0, 0.0, 10.0, 0.0, -1.0, 20.0),
        nodata=65535,
    )
    before = gis.read_raster_info(path)

    with pytest.raises(ValueError, match="CrsAlreadyExists"):
        gis.assign_crs(path, "EPSG:3857")

    info = gis.assign_crs(path, "EPSG:3857", overwrite=True)
    reread = gis.read_raster_info(path)
    sampled = gis.resample_raster(path, (2, 3), method="nearest")["array"][0]

    assert info.crs_authority == {"name": "EPSG", "code": "3857"}
    assert "assignment_not_reprojection" in _codes(info.warnings)
    assert reread.crs_authority == {"name": "EPSG", "code": "3857"}
    for attr in (
        "width",
        "height",
        "band_count",
        "dtype_per_band",
        "transform",
        "bounds",
        "resolution",
        "nodata_per_band",
    ):
        assert getattr(info, attr) == getattr(before, attr)
        assert getattr(reread, attr) == getattr(before, attr)
    np.testing.assert_array_equal(sampled, original)


def test_assign_crs_metadata_object_is_non_mutating(tmp_path: Path):
    path = tmp_path / "assign_info.tif"
    info = gis.write_raster(
        path,
        np.ones((2, 2), dtype=np.uint8),
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        nodata=0,
    )

    assigned = gis.assign_crs(info, "EPSG:4326")
    reread = gis.read_raster_info(path)

    assert assigned.crs_authority == {"name": "EPSG", "code": "4326"}
    assert "assignment_not_reprojection" in _codes(assigned.warnings)
    assert reread.crs_authority is None
    for attr in (
        "width",
        "height",
        "band_count",
        "dtype_per_band",
        "transform",
        "bounds",
        "resolution",
        "nodata_per_band",
    ):
        assert getattr(assigned, attr) == getattr(info, attr)
        assert getattr(reread, attr) == getattr(info, attr)


def test_transform_order_bounds_and_resolution(tmp_path: Path):
    path = tmp_path / "affine.tif"
    info = _basic(
        path,
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -3.0, 50.0),
    )

    assert info.transform == pytest.approx((2.0, 0.0, 10.0, 0.0, -3.0, 50.0))
    assert gis.raster_transform(info) == pytest.approx((2.0, 0.0, 10.0, 0.0, -3.0, 50.0))
    assert gis.raster_bounds(info) == pytest.approx((10.0, 38.0, 20.0, 50.0))
    assert gis.bounds(info) == pytest.approx(gis.raster_bounds(info))
    assert gis.raster_resolution(info) == pytest.approx((2.0, 3.0))
    assert info.resolution == pytest.approx((2.0, 3.0))


def test_affine_helpers_use_documented_order_and_pixel_center():
    transform = gis.AffineTransform(2.0, 0.0, 10.0, 0.0, -3.0, 50.0)

    assert transform.coefficients == pytest.approx((2.0, 0.0, 10.0, 0.0, -3.0, 50.0))
    assert transform.resolution == pytest.approx((2.0, 3.0))
    assert not hasattr(transform, "resolution_py")
    stub = Path(gis.__file__).with_suffix(".pyi").read_text(encoding="utf-8")
    assert "def resolution(self) -> tuple[float, float]" in stub
    assert gis.transform_from_origin(10.0, 50.0, 2.0, 3.0) == pytest.approx(
        transform.coefficients
    )
    assert gis.transform_from_bounds((10.0, 38.0, 20.0, 50.0), 5, 4) == pytest.approx(
        transform.coefficients
    )
    assert gis.array_bounds(4, 5, transform) == pytest.approx((10.0, 38.0, 20.0, 50.0))

    validation = gis.validate_transform(transform)
    convention = gis.pixel_convention(transform)
    x, y = gis.xy(transform, 1, 2)

    assert validation["valid"] is True
    assert validation["rotated_or_sheared"] is False
    assert validation["resolution"] == pytest.approx((2.0, 3.0))
    assert convention["default_offset"] == "center"
    assert "pixel_convention_explicit" in _codes(convention["warnings"])
    assert (x, y) == pytest.approx((15.0, 45.5))
    assert gis.index(transform, x, y) == (1, 2)
    assert gis.rowcol(transform, x, y) == (1, 2)


def test_transform_bounds_public_api_matches_web_mercator():
    bounds = (-1.0, -1.0, 1.0, 1.0)

    assert gis.transform_bounds("EPSG:4326", "EPSG:3857", bounds) == pytest.approx(
        gis.web_mercator_bounds(bounds, "EPSG:4326")
    )
    # MENSURA: densify is functional. densify=None samples corners only;
    # densify=0 is rejected; for this equator-centred web-mercator extent the
    # corner extrema are exact, so densified bounds agree with corner bounds.
    assert gis.transform_bounds("EPSG:4326", "EPSG:3857", bounds, densify=None) == pytest.approx(
        gis.transform_bounds("EPSG:4326", "EPSG:3857", bounds, densify=8)
    )
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.transform_bounds("EPSG:4326", "EPSG:3857", bounds, densify=0)


def test_missing_and_invalid_transform_diagnostics(tmp_path: Path):
    path = tmp_path / "missing_transform.tif"
    gis.write_raster(path, np.ones((2, 2), dtype=np.uint8), crs="EPSG:4326")

    with pytest.raises(ValueError, match="MissingTransform"):
        gis.bounds(path)

    with pytest.raises(ValueError, match="InvalidTransform"):
        gis.write_raster(
            tmp_path / "nan_transform.tif",
            np.ones((2, 2), dtype=np.uint8),
            transform=(math.nan, 0.0, 0.0, 0.0, -1.0, 2.0),
        )

    with pytest.raises(ValueError, match="InvalidTransform"):
        gis.write_raster(
            tmp_path / "zero_pixel.tif",
            np.ones((2, 2), dtype=np.uint8),
            transform=(0.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )


def test_rotated_sheared_transform_warning(tmp_path: Path):
    info = gis.write_raster(
        tmp_path / "rotated.tif",
        np.ones((2, 2), dtype=np.uint8),
        crs="EPSG:4326",
        transform=(1.0, 0.25, 0.0, 0.0, -1.0, 2.0),
    )

    assert "rotated_or_sheared_transform" in _codes(info.warnings)


def test_window_from_bounds_inside_and_partial(tmp_path: Path):
    path = tmp_path / "window.tif"
    info = _basic(
        path,
        crs="EPSG:4326",
        transform=(10.0, 0.0, 100.0, 0.0, -10.0, 200.0),
    )

    inside = gis.window_from_bounds(info, (110.0, 170.0, 140.0, 190.0))
    clipped = gis.window_from_bounds(path, (80.0, 170.0, 120.0, 210.0))
    boundless = gis.window_from_bounds(
        path,
        (80.0, 170.0, 120.0, 210.0),
        boundless=True,
    )

    assert inside["window"] == (1, 1, 3, 2)
    assert inside["output_shape"] == (2, 3)
    assert inside["output_transform"] == pytest.approx(
        (10.0, 0.0, 110.0, 0.0, -10.0, 190.0)
    )
    assert clipped["window"] == (0, 0, 2, 3)
    assert boundless["window"] == (-2, -1, 4, 4)


def test_window_rejects_invalid_bounds_and_crs_mismatch(tmp_path: Path):
    path = tmp_path / "bad_window.tif"
    _basic(
        path,
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
    )

    with pytest.raises(ValueError, match="InvalidBounds"):
        gis.window_from_bounds(path, (5.0, 0.0, 4.0, 1.0))

    with pytest.raises(ValueError, match="do not intersect raster extent"):
        gis.window_from_bounds(path, (10.0, 10.0, 11.0, 11.0))

    with pytest.raises(ValueError, match="CrsMismatch"):
        gis.window_from_bounds(
            path,
            {"bounds": (0.0, 0.0, 1.0, 1.0), "crs": "EPSG:3857"},
        )


def test_nodata_contracts_and_preservation(tmp_path: Path):
    scalar = gis.write_raster(
        tmp_path / "scalar_nodata.tif",
        np.ones((2, 2), dtype=np.uint8),
        nodata=0,
    )
    per_band = gis.write_raster(
        tmp_path / "per_band_nodata.tif",
        np.ones((2, 2, 2), dtype=np.int16),
        nodata=[None, -2],
    )

    with pytest.raises(ValueError, match="InvalidNodata"):
        gis.write_raster(tmp_path / "bad_nodata.tif", np.ones((2, 2), dtype=np.uint8), nodata=-1)

    resampled = gis.resample_raster(scalar.path, (4, 4), method="nearest")

    assert scalar.nodata_per_band == [0.0]
    assert per_band.nodata_per_band == [None, -2.0]
    assert "per_band_nodata_mismatch" in _codes(per_band.warnings)
    assert resampled["info"]["nodata_per_band"] == [0.0]


def test_resample_regenerates_georeference_warnings(tmp_path: Path):
    path = tmp_path / "plain.tif"
    gis.write_raster(path, np.ones((2, 2), dtype=np.uint8))

    result = gis.resample_raster(path, (4, 4), method="nearest")
    codes = _codes(result["info"]["warnings"])

    assert {"missing_crs", "missing_transform", "not_georeferenced"} <= codes
