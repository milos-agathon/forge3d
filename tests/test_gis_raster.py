"""G-002a1 GIS raster read/write contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE
from forge3d._native import get_native_module


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS raster tests require the compiled _forge3d extension",
)


def _real_rasterio():
    try:
        import rasterio
        from rasterio.transform import Affine
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"real rasterio is not available: {exc}")

    if getattr(rasterio, "__forge3d_stub__", False):
        pytest.skip("real rasterio is not available")
    return rasterio, Affine


def _warning_codes(info) -> set[str]:
    return {warning["code"] for warning in info.warnings}


def _assert_epsg_4326(info):
    assert info.crs_authority == {"name": "EPSG", "code": "4326"}
    if info.crs_wkt is None:
        assert "metadata_unavailable" in _warning_codes(info)


WGS84_WKT = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984"],'
    'CS[ellipsoidal,2],AXIS["longitude",east],AXIS["latitude",north],'
    'ANGLEUNIT["degree",0.0174532925199433]]'
)


def test_read_raster_info_matches_rasterio_geotiff_contract(tmp_path: Path):
    rasterio, Affine = _real_rasterio()
    path = tmp_path / "reference.tif"
    transform = Affine(2.0, 0.0, 10.0, 0.0, -3.0, 50.0)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    info = gis.read_raster_info(path)

    assert Path(info.path) == path
    assert info.driver == "GTiff"
    assert info.width == 3
    assert info.height == 2
    assert info.band_count == 1
    assert info.dtype_per_band == ["float32"]
    _assert_epsg_4326(info)
    assert info.transform == pytest.approx((2.0, 0.0, 10.0, 0.0, -3.0, 50.0))
    assert info.bounds == pytest.approx((10.0, 44.0, 16.0, 50.0))
    assert info.resolution == pytest.approx((2.0, 3.0))
    assert info.nodata_per_band == [-9999.0]
    assert info.is_georeferenced is True


def test_read_raster_info_preserves_arbitrary_epsg_metadata(tmp_path: Path):
    rasterio, Affine = _real_rasterio()
    path = tmp_path / "utm33-metadata.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=Affine(2.0, 0.0, 500_000.0, 0.0, -2.0, 5_000_000.0),
    ) as dst:
        dst.write(np.ones((2, 2), dtype=np.uint8), 1)

    info = gis.read_raster_info(path)

    assert info.crs_authority == {"name": "EPSG", "code": "32633"}
    assert info.transform == pytest.approx(
        (2.0, 0.0, 500_000.0, 0.0, -2.0, 5_000_000.0)
    )
    assert info.is_georeferenced is True


def test_write_raster_single_band_returns_authoritative_info(tmp_path: Path):
    path = tmp_path / "single.tif"
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    info = gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(1.0, 0.0, 100.0, 0.0, -2.0, 200.0),
        nodata=-9999.0,
    )
    reread = gis.read_raster_info(path)

    assert Path(info.path) == path
    assert info.driver == "GTiff"
    assert info.width == 3
    assert info.height == 2
    assert info.band_count == 1
    assert info.dtype_per_band == ["float32"]
    assert info.nodata_per_band == [-9999.0]
    _assert_epsg_4326(info)
    assert info.transform == pytest.approx((1.0, 0.0, 100.0, 0.0, -2.0, 200.0))
    assert reread.as_dict() == info.as_dict()


def test_public_gis_wrapper_surface():
    native = get_native_module()

    assert gis.RasterInfo is native.RasterInfo
    assert gis.VectorInfo is native.VectorInfo
    assert set(gis.__all__) == {
        "RasterInfo",
        "VectorInfo",
        "AffineTransform",
        "CrsTransform",
        "RasterReadResult",
        "read_raster_info",
        "read_raster",
        "read_vector",
        "reproject_vector",
        "geometry_type",
        "vector_schema",
        "feature_count",
        "vector_crs",
        "vector_bounds",
        "validate_geometry",
        "repair_geometry",
        "geometry_measure",
        "measure_geometries",
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
        "rasterize_vectors",
        "geometry_mask",
        "mask_raster",
        "normalize_raster",
        "classify_raster",
        "write_raster",
        "parse_crs",
        "inspect_crs",
        "raster_crs",
        "assign_crs",
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
        "apply_nodata",
        "read_raster_mask",
        "resample_raster",
        "assert_grid_compatible",
        "align_raster_grid",
        "align_raster_to",
        "reproject_raster",
        "calculate_default_transform",
        "window_from_bounds",
        "read_raster_window",
        "window_transform",
        "bounds",
        "fetch_remote_geodata",
        "cache_geodata",
        "fetch_vector",
        "read_cog",
        "slippy_tile_index",
        "query_osm_features",
        "parse_osm_features",
        "load_context_vectors",
        "prepare_osm_scene",
        "prepare_dem",
        "prepare_terrain_derivatives",
        "read_gridded_dataset",
        "subset_grid",
        "decode_terrarium_dem",
        "build_terrarium_dem",
        "prepare_landcover_raster",
        "prepare_population_raster",
        "load_building_footprints",
        "extract_building_heights",
        "estimate_local_utm",
    }
    assert callable(gis.read_raster_info)
    assert callable(gis.read_raster)
    assert callable(gis.read_vector)
    assert callable(gis.reproject_vector)
    assert callable(gis.geometry_type)
    assert callable(gis.vector_schema)
    assert callable(gis.feature_count)
    assert callable(gis.vector_crs)
    assert callable(gis.vector_bounds)
    assert callable(gis.validate_geometry)
    assert callable(gis.repair_geometry)
    assert callable(gis.geometry_measure)
    assert callable(gis.geometry_centroid)
    assert callable(gis.representative_point)
    assert callable(gis.interpolate_line)
    assert callable(gis.union_geometries)
    assert callable(gis.dissolve_vector)
    assert callable(gis.buffer_geometry)
    assert callable(gis.clip_vector)
    assert callable(gis.intersect_vectors)
    assert callable(gis.simplify_geometry)
    assert callable(gis.load_boundary)
    assert callable(gis.rasterize_vectors)
    assert callable(gis.geometry_mask)
    assert callable(gis.mask_raster)
    assert callable(gis.normalize_raster)
    assert callable(gis.classify_raster)
    assert callable(gis.write_raster)
    assert callable(gis.parse_crs)
    assert callable(gis.inspect_crs)
    assert callable(gis.raster_crs)
    assert callable(gis.assign_crs)
    assert callable(gis.create_crs_transformer)
    assert callable(gis.transform_bounds)
    assert callable(gis.web_mercator_bounds)
    assert callable(gis.raster_transform)
    assert callable(gis.transform_from_origin)
    assert callable(gis.transform_from_bounds)
    assert callable(gis.array_bounds)
    assert callable(gis.raster_bounds)
    assert callable(gis.raster_resolution)
    assert callable(gis.fetch_remote_geodata)
    assert callable(gis.cache_geodata)
    assert callable(gis.fetch_vector)
    assert callable(gis.read_cog)
    assert callable(gis.slippy_tile_index)
    assert callable(gis.query_osm_features)
    assert callable(gis.parse_osm_features)
    assert callable(gis.load_context_vectors)
    assert callable(gis.prepare_osm_scene)
    assert callable(gis.prepare_dem)
    assert callable(gis.prepare_terrain_derivatives)
    assert callable(gis.read_gridded_dataset)
    assert callable(gis.subset_grid)
    assert callable(gis.decode_terrarium_dem)
    assert callable(gis.build_terrarium_dem)
    assert callable(gis.prepare_landcover_raster)
    assert callable(gis.prepare_population_raster)
    assert callable(gis.load_building_footprints)
    assert callable(gis.extract_building_heights)
    assert callable(gis.estimate_local_utm)
    assert callable(gis.validate_transform)
    assert callable(gis.pixel_convention)
    assert callable(gis.rowcol)
    assert callable(gis.xy)
    assert callable(gis.index)
    assert callable(gis.apply_nodata)
    assert callable(gis.read_raster_mask)
    assert callable(gis.resample_raster)
    assert callable(gis.assert_grid_compatible)
    assert callable(gis.align_raster_grid)
    assert callable(gis.align_raster_to)
    assert callable(gis.reproject_raster)
    assert callable(gis.calculate_default_transform)
    assert callable(gis.window_from_bounds)
    assert callable(gis.read_raster_window)
    assert callable(gis.window_transform)
    assert callable(gis.bounds)
    assert hasattr(gis.CrsTransform, "from_crs")


def test_public_gis_all_matches_stub_exports():
    stub_exports = set()
    for line in Path(gis.__file__).with_suffix(".pyi").read_text(encoding="utf-8").splitlines():
        if line.startswith(("class ", "def ")):
            stub_exports.add(line.split()[1].split("(")[0].rstrip(":"))

    assert set(gis.__all__) == stub_exports


def test_apply_nodata_scalar_per_band_mask_nan_and_empty_valid():
    data = np.array(
        [
            [[1.0, -9999.0], [np.nan, 4.0]],
            [[10.0, 11.0], [12.0, 13.0]],
        ],
        dtype=np.float32,
    )
    explicit_mask = np.array(
        [
            [[True, True], [True, False]],
            [[True, False], [True, True]],
        ],
        dtype=bool,
    )

    result = gis.apply_nodata(data, [-9999.0, None], mask=explicit_mask)

    assert result["mask_polarity"] == "true_valid"
    assert result["nodata_per_band"] == [-9999.0, None]
    assert result["valid_count"] == 4
    assert result["mask"].shape == data.shape
    assert result["mask"][0, 0, 1] == np.False_
    assert result["mask"][0, 1, 0] == np.False_
    assert result["mask"][0, 1, 1] == np.False_
    assert result["mask"][1, 0, 1] == np.False_

    empty = gis.apply_nodata(np.array([[0, 0]], dtype=np.uint8), 0)
    assert empty["valid_count"] == 0
    assert "empty_raster" in {warning["code"] for warning in empty["warnings"]}


def test_read_raster_mask_reports_polarity_flags_and_nodata(tmp_path: Path):
    path = tmp_path / "mask.tif"
    gis.write_raster(path, np.array([[0, 2], [3, 0]], dtype=np.uint8), nodata=0)

    result = gis.read_raster_mask(path)

    assert result["mask_polarity"] == "true_valid"
    assert result["mask_flags"] == ["nodata"]
    assert result["nodata_per_band"] == [0.0]
    assert result["mask"].shape == (1, 2, 2)
    np.testing.assert_array_equal(
        result["mask"][0],
        np.array([[False, True], [True, False]], dtype=bool),
    )


def test_write_raster_pixels_are_readable_by_rasterio(tmp_path: Path):
    rasterio, _Affine = _real_rasterio()
    path = tmp_path / "payload.tif"
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)

    gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(1.0, 0.0, 5.0, 0.0, -1.0, 9.0),
        nodata=65535,
    )

    with rasterio.open(path) as src:
        assert src.driver == "GTiff"
        assert src.width == 4
        assert src.height == 3
        assert src.count == 1
        assert src.dtypes == ("uint16",)
        assert src.nodata == 65535
        assert src.crs.to_authority() == ("EPSG", "4326")
        np.testing.assert_array_equal(src.read(1), data)


def test_write_raster_multi_band_bhw(tmp_path: Path):
    path = tmp_path / "rgb.tif"
    data = np.arange(3 * 2 * 5, dtype=np.uint8).reshape(3, 2, 5)

    info = gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
        nodata=0,
    )

    assert info.width == 5
    assert info.height == 2
    assert info.band_count == 3
    assert info.dtype_per_band == ["uint8", "uint8", "uint8"]
    assert info.nodata_per_band == [0.0, 0.0, 0.0]


def test_write_raster_two_band_bhw(tmp_path: Path):
    path = tmp_path / "two_band.tif"
    data = np.arange(2 * 3 * 5, dtype=np.int16).reshape(2, 3, 5)

    info = gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        nodata=[-1, -2],
    )

    assert info.width == 5
    assert info.height == 3
    assert info.band_count == 2
    assert info.dtype_per_band == ["int16", "int16"]
    assert info.nodata_per_band == [-1.0, -2.0]
    assert "per_band_nodata_mismatch" in _warning_codes(info)


def test_write_raster_accepts_wkt_and_dict_crs(tmp_path: Path):
    wkt_info = gis.write_raster(
        tmp_path / "wkt.tif",
        np.ones((2, 2), dtype=np.float32),
        crs=WGS84_WKT,
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )
    dict_info = gis.write_raster(
        tmp_path / "dict.tif",
        np.ones((2, 2), dtype=np.float32),
        crs={"name": "EPSG", "code": 4326},
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )

    assert wkt_info.crs_wkt == WGS84_WKT
    assert wkt_info.crs_authority is None
    assert wkt_info.is_georeferenced is True
    _assert_epsg_4326(dict_info)
    assert dict_info.is_georeferenced is True


def test_write_raster_rejects_mixed_crs_authority_and_wkt(tmp_path: Path):
    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.write_raster(
            tmp_path / "mixed_crs.tif",
            np.ones((2, 2), dtype=np.uint8),
            crs={"name": "EPSG", "code": "4326", "wkt": WGS84_WKT},
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )


@pytest.mark.parametrize(
    "crs",
    [
        "EPSG:0",
        "EPSG:9999",
        "not-a-crs",
        'GEOGCRS["broken"',
        'GEOGCRS["broken"]',
        {"name": "EPSG"},
        {"name": "EPSG", "code": "9999"},
        {"authority": "EPSG", "code": "4326"},
        {"epsg": "4326"},
        {"wkt": WGS84_WKT},
        {"crs_wkt": WGS84_WKT},
        {"wkt": "not-wkt"},
        {"name": "EPSG", "code": "4326", "wkt": "not-wkt"},
    ],
)
def test_write_raster_rejects_invalid_crs(tmp_path: Path, crs):
    with pytest.raises(ValueError, match="InvalidCrs"):
        gis.write_raster(
            tmp_path / "bad_crs.tif",
            np.ones((2, 2), dtype=np.uint8),
            crs=crs,
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )


def test_write_raster_non_georeferenced_warnings(tmp_path: Path):
    info = gis.write_raster(tmp_path / "plain.tif", np.ones((2, 2), dtype=np.uint16))

    assert info.crs_wkt is None
    assert info.transform is None
    assert info.bounds is None
    assert info.resolution is None
    assert info.is_georeferenced is False
    assert {"missing_crs", "missing_transform", "not_georeferenced"} <= _warning_codes(
        info
    )


def test_read_raster_info_missing_crs_warning(tmp_path: Path):
    path = tmp_path / "missing_crs.tif"
    gis.write_raster(
        path,
        np.ones((2, 2), dtype=np.uint8),
        transform=(1.0, 0.0, 10.0, 0.0, -1.0, 20.0),
    )

    info = gis.read_raster_info(path)

    assert info.crs_wkt is None
    assert info.crs_authority is None
    assert info.transform == pytest.approx((1.0, 0.0, 10.0, 0.0, -1.0, 20.0))
    assert info.is_georeferenced is True
    assert "missing_crs" in _warning_codes(info)
    assert "not_georeferenced" not in _warning_codes(info)


def test_read_raster_info_missing_transform_warning(tmp_path: Path):
    path = tmp_path / "missing_transform.tif"
    gis.write_raster(path, np.ones((2, 2), dtype=np.uint8), crs="EPSG:4326")

    info = gis.read_raster_info(path)

    _assert_epsg_4326(info)
    assert info.transform is None
    assert info.bounds is None
    assert info.resolution is None
    assert {"missing_transform", "not_georeferenced"} <= _warning_codes(info)


def test_read_raster_info_rejects_non_tiff_driver(tmp_path: Path):
    path = tmp_path / "not_a_tiff.txt"
    path.write_text("not a raster", encoding="utf-8")

    with pytest.raises(ValueError, match="UnsupportedDriver"):
        gis.read_raster_info(path)


def test_read_raster_info_rejects_malformed_wkt(tmp_path: Path):
    path = tmp_path / "malformed_wkt.tif"
    gis.write_raster(path, np.ones((2, 2), dtype=np.uint8), crs=WGS84_WKT)

    original = WGS84_WKT.encode("ascii")
    payload = path.read_bytes()
    assert original in payload
    path.write_bytes(payload.replace(original, original[:-1] + b" ", 1))

    with pytest.raises(ValueError, match="InvalidCrs: CRS WKT has unbalanced brackets"):
        gis.read_raster_info(path)


def test_write_raster_positive_y_transform_round_trips_as_gis_metadata(tmp_path: Path):
    path = tmp_path / "positive_y.tif"
    transform = (1.0, 0.0, 10.0, 0.0, 1.0, 20.0)

    gis.write_raster(
        path,
        np.ones((2, 2), dtype=np.uint8),
        crs="EPSG:4326",
        transform=transform,
    )

    info = gis.read_raster_info(path)
    assert tuple(info.transform) == pytest.approx(transform)


@pytest.mark.parametrize("shape", [(2, 3, 3), (2, 3, 4)])
def test_write_raster_accepts_valid_bhw_with_width_3_or_4(tmp_path: Path, shape):
    info = gis.write_raster(tmp_path / f"bhw_{shape[-1]}.tif", np.zeros(shape, dtype=np.uint8))

    assert info.band_count == shape[0]
    assert info.height == shape[1]
    assert info.width == shape[2]


def test_write_raster_3d_arrays_are_band_first_not_hwc(tmp_path: Path):
    info = gis.write_raster(tmp_path / "band_first.tif", np.zeros((5, 6, 3), dtype=np.uint8))

    assert info.band_count == 5
    assert info.height == 6
    assert info.width == 3


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 2), dtype=np.bool_),
        np.zeros((2, 2), dtype=np.complex64),
        np.array([["a", "b"], ["c", "d"]], dtype=object),
    ],
)
def test_write_raster_rejects_unsupported_dtype(tmp_path: Path, array):
    with pytest.raises((TypeError, ValueError), match="UnsupportedDType"):
        gis.write_raster(tmp_path / "bad_dtype.tif", array)


def test_write_raster_rejects_existing_path_without_overwrite(tmp_path: Path):
    path = tmp_path / "exists.tif"
    gis.write_raster(path, np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(FileExistsError, match="AlreadyExists"):
        gis.write_raster(path, np.ones((2, 2), dtype=np.uint8))


def test_write_raster_rejects_missing_parent(tmp_path: Path):
    path = tmp_path / "missing" / "out.tif"

    with pytest.raises(FileNotFoundError, match="NotFound"):
        gis.write_raster(path, np.ones((2, 2), dtype=np.uint8))


def test_write_raster_rejects_unknown_creation_option(tmp_path: Path):
    with pytest.raises(ValueError, match="UnsupportedCreationOption"):
        gis.write_raster(
            tmp_path / "option.tif",
            np.ones((2, 2), dtype=np.uint8),
            creation_options={"photometric": "minisblack"},
        )


def test_write_raster_supported_creation_options_are_reported(tmp_path: Path):
    path = tmp_path / "compressed.tif"

    info = gis.write_raster(
        path,
        np.ones((2, 2), dtype=np.uint8),
        creation_options={"compress": "LZW", "tiled": False, "bigtiff": False},
    )

    assert info.compression == "LZW"
    assert gis.read_raster_info(path).compression == "LZW"


@pytest.mark.parametrize(
    "creation_options",
    [
        {"compress": "JPEG"},
        {"tiled": True},
        {"blockxsize": 16},
        {"blockxsize": None},
        {"blockysize": 16},
        {"blockysize": None},
        {"predictor": 2},
        {"predictor": None},
        {"bigtiff": True},
    ],
)
def test_write_raster_rejects_unsupported_creation_option_values(
    tmp_path: Path, creation_options
):
    with pytest.raises(ValueError, match="InvalidArgument"):
        gis.write_raster(
            tmp_path / "unsupported_option_value.tif",
            np.ones((2, 2), dtype=np.uint8),
            creation_options=creation_options,
        )


def test_write_raster_rejects_unsupported_driver(tmp_path: Path):
    with pytest.raises(ValueError, match="UnsupportedDriver"):
        gis.write_raster(
            tmp_path / "driver.jp2",
            np.ones((2, 2), dtype=np.uint8),
            driver="JP2OpenJPEG",
        )


def test_write_raster_overwrite_replaces_existing_file(tmp_path: Path):
    path = tmp_path / "overwrite.tif"
    first = gis.write_raster(path, np.zeros((2, 2), dtype=np.uint8), nodata=0)
    second = gis.write_raster(
        path,
        np.ones((3, 4), dtype=np.uint16),
        nodata=1,
        overwrite=True,
    )

    assert first.width == 2
    assert second.width == 4
    assert second.height == 3
    assert second.dtype_per_band == ["uint16"]
    assert gis.read_raster_info(path).as_dict() == second.as_dict()


def test_write_raster_validates_nodata_against_dtype(tmp_path: Path):
    with pytest.raises(ValueError, match="InvalidNodata"):
        gis.write_raster(tmp_path / "bad_negative.tif", np.ones((2, 2), dtype=np.uint8), nodata=-1)

    with pytest.raises(ValueError, match="InvalidNodata"):
        gis.write_raster(tmp_path / "bad_nan.tif", np.ones((2, 2), dtype=np.uint8), nodata=np.nan)

    with pytest.raises(ValueError, match="InvalidNodata"):
        gis.write_raster(
            tmp_path / "bad_float32_inf.tif",
            np.ones((2, 2), dtype=np.float32),
            nodata=np.inf,
        )

    with pytest.raises(ValueError, match="InvalidNodata"):
        gis.write_raster(
            tmp_path / "bad_float32_range.tif",
            np.ones((2, 2), dtype=np.float32),
            nodata=float(np.finfo(np.float32).max) * 2.0,
        )

    info = gis.write_raster(
        tmp_path / "float_nan.tif",
        np.ones((2, 2), dtype=np.float32),
        nodata=np.nan,
    )
    assert np.isnan(info.nodata_per_band[0])

    mixed = gis.write_raster(
        tmp_path / "mixed_nodata.tif",
        np.ones((2, 2, 5), dtype=np.int16),
        nodata=[None, -2],
    )
    assert mixed.nodata_per_band == [None, -2.0]
    assert "per_band_nodata_mismatch" in _warning_codes(mixed)


def test_write_raster_like_path_and_like_info_contracts(tmp_path: Path):
    like_path = tmp_path / "like.tif"
    like = gis.write_raster(
        like_path,
        np.ones((2, 3), dtype=np.float32),
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
        nodata=-1.0,
        creation_options={"compress": "LZW"},
    )

    copied = gis.write_raster(
        tmp_path / "copied.tif",
        np.zeros((2, 3), dtype=np.float32),
        like_path=like_path,
    )
    assert copied.transform == pytest.approx(like.transform)
    assert copied.crs_authority == like.crs_authority
    assert copied.nodata_per_band == like.nodata_per_band
    assert copied.compression == "LZW"

    copied_from_info = gis.write_raster(
        tmp_path / "copied_info.tif",
        np.zeros((2, 3), dtype=np.float32),
        like_info=like,
    )
    assert copied_from_info.transform == pytest.approx(like.transform)
    assert copied_from_info.crs_authority == like.crs_authority
    assert copied_from_info.nodata_per_band == like.nodata_per_band
    assert copied_from_info.compression == "LZW"

    no_crs_like_path = tmp_path / "like_no_crs.tif"
    no_crs_like = gis.write_raster(
        no_crs_like_path,
        np.ones((2, 3), dtype=np.float32),
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
    )
    explicit_crs_with_no_crs_like = gis.write_raster(
        tmp_path / "explicit_crs_with_no_crs_like.tif",
        np.zeros((2, 3), dtype=np.float32),
        like_info=no_crs_like,
        crs="EPSG:4326",
    )
    _assert_epsg_4326(explicit_crs_with_no_crs_like)
    assert explicit_crs_with_no_crs_like.transform == pytest.approx(no_crs_like.transform)

    wkt_like = gis.write_raster(
        tmp_path / "wkt_like.tif",
        np.ones((2, 3), dtype=np.float32),
        crs=WGS84_WKT,
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
    )
    explicit_epsg_with_wkt_like = gis.write_raster(
        tmp_path / "explicit_epsg_with_wkt_like.tif",
        np.zeros((2, 3), dtype=np.float32),
        like_info=wkt_like,
        crs="EPSG:4326",
    )
    _assert_epsg_4326(explicit_epsg_with_wkt_like)
    assert explicit_epsg_with_wkt_like.transform == pytest.approx(wkt_like.transform)

    with pytest.raises(ValueError, match="InvalidArgument"):
        gis.write_raster(
            tmp_path / "both_like.tif",
            np.zeros((2, 3), dtype=np.float32),
            like_path=like_path,
            like_info=like,
        )

    with pytest.raises(ValueError, match="ShapeMismatch"):
        gis.write_raster(
            tmp_path / "shape_conflict.tif",
            np.zeros((3, 3), dtype=np.float32),
            like_path=like_path,
        )

    with pytest.raises(ValueError, match="ShapeMismatch"):
        gis.write_raster(
            tmp_path / "explicit_transform_shape_conflict.tif",
            np.zeros((3, 3), dtype=np.float32),
            like_info=like,
            transform=like.transform,
        )

    with pytest.raises(ValueError, match="InvalidArgument"):
        gis.write_raster(
            tmp_path / "transform_conflict.tif",
            np.zeros((2, 3), dtype=np.float32),
            like_info=like,
            transform=(1.0, 0.0, 10.0, 0.0, -2.0, 20.0),
        )

    with pytest.raises(ValueError, match="InvalidArgument"):
        gis.write_raster(
            tmp_path / "crs_conflict.tif",
            np.zeros((2, 3), dtype=np.float32),
            like_info=like,
            crs="EPSG:3857",
        )
