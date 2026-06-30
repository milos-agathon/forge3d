"""G-002a1 public read_raster contract tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE
from forge3d._native import get_native_module


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="read_raster tests require the compiled _forge3d extension",
)


def _warning_codes(warnings: list[dict[str, object]]) -> set[str]:
    return {str(warning["code"]) for warning in warnings}


def _real_rasterio():
    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"real rasterio is not available: {exc}")
    if getattr(rasterio, "__forge3d_stub__", False):
        pytest.skip("real rasterio is not available")
    return rasterio, Window


def _write_int8_tiff(path: Path) -> None:
    data = b"\x01\xfe"
    entries = [
        (256, 4, 1, 2),  # ImageWidth
        (257, 4, 1, 1),  # ImageLength
        (258, 3, 1, 8),  # BitsPerSample
        (259, 3, 1, 1),  # Compression
        (262, 3, 1, 1),  # PhotometricInterpretation
        (273, 4, 1, 8),  # StripOffsets
        (277, 3, 1, 1),  # SamplesPerPixel
        (278, 4, 1, 1),  # RowsPerStrip
        (279, 4, 1, len(data)),  # StripByteCounts
        (339, 3, 1, 2),  # SampleFormat = signed integer
    ]
    ifd_offset = 16
    header = b"II*\x00" + struct.pack("<I", ifd_offset) + data + b"\x00" * 6
    body = struct.pack("<H", len(entries))
    for tag, typ, count, value in entries:
        body += struct.pack("<HHII", tag, typ, count, value)
    body += struct.pack("<I", 0)
    path.write_bytes(header + body)


def test_read_raster_schema_full_read_and_public_surface(tmp_path: Path):
    native = get_native_module()
    path = tmp_path / "rgb.tif"
    data = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -3.0, 50.0),
        nodata=[99, 100, None],
    )

    result = gis.read_raster(path)

    assert hasattr(native, "read_raster")
    assert "read_raster" in gis.__all__
    assert set(result) == {
        "array",
        "info",
        "bands",
        "window",
        "window_transform",
        "mask",
        "mask_polarity",
        "nodata_per_band",
        "warnings",
    }
    np.testing.assert_array_equal(result["array"], data)
    assert result["array"].shape == (3, 2, 4)
    assert result["bands"] == (1, 2, 3)
    assert result["window"] is None
    assert result["window_transform"] is None
    assert result["mask"] is None
    assert result["mask_polarity"] is None
    assert result["nodata_per_band"] == [99.0, 100.0, None]
    assert result["warnings"] == result["info"]["warnings"]
    assert result["info"]["band_count"] == 3
    assert result["info"]["dtype_per_band"] == ["uint16", "uint16", "uint16"]
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert result["info"]["transform"] == pytest.approx((2.0, 0.0, 10.0, 0.0, -3.0, 50.0))
    assert result["info"]["bounds"] == pytest.approx((10.0, 44.0, 18.0, 50.0))


def test_read_raster_band_selection_is_one_based_and_ordered(tmp_path: Path):
    path = tmp_path / "bands.tif"
    data = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(3, 2, 3)
    gis.write_raster(path, data, nodata=[None, 0, 255])

    all_bands = gis.read_raster(path, bands=None)
    single = gis.read_raster(path, bands=1)
    reordered = gis.read_raster(path, bands=[3, 1])

    np.testing.assert_array_equal(all_bands["array"], data)
    np.testing.assert_array_equal(single["array"], data[[0]])
    np.testing.assert_array_equal(reordered["array"], data[[2, 0]])
    assert single["array"].shape == (1, 2, 3)
    assert single["bands"] == (1,)
    assert reordered["bands"] == (3, 1)
    assert reordered["info"]["band_count"] == 2
    assert reordered["info"]["dtype_per_band"] == ["uint8", "uint8"]
    assert reordered["nodata_per_band"] == [255.0, None]


@pytest.mark.parametrize("bands", [[], [1, 1], [0], [4], [-1], [1.5], ["1"]])
def test_read_raster_rejects_invalid_band_selection(tmp_path: Path, bands):
    path = tmp_path / "bad_bands.tif"
    gis.write_raster(path, np.ones((3, 2, 2), dtype=np.uint8))

    with pytest.raises((TypeError, ValueError), match="InvalidArgument"):
        gis.read_raster(path, bands=bands)


def test_read_raster_window_tuple_dict_metadata_and_bounds(tmp_path: Path):
    path = tmp_path / "window.tif"
    data = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    transform = (2.0, 0.0, 10.0, 0.0, -3.0, 50.0)
    gis.write_raster(path, data, crs="EPSG:4326", transform=transform)

    tuple_result = gis.read_raster(path, bands=[1, 3], window=(1, 1, 3, 2))
    dict_result = gis.read_raster(
        path,
        bands=[1, 3],
        window={"col_off": 1, "row_off": 1, "width": 3, "height": 2},
    )

    expected = data[[0, 2], 1:3, 1:4]
    np.testing.assert_array_equal(tuple_result["array"], expected)
    np.testing.assert_array_equal(dict_result["array"], expected)
    assert tuple_result["window"] == (1, 1, 3, 2)
    assert tuple_result["window_transform"] == pytest.approx((2.0, 0.0, 12.0, 0.0, -3.0, 47.0))
    assert tuple_result["info"]["width"] == 3
    assert tuple_result["info"]["height"] == 2
    assert tuple_result["info"]["band_count"] == 2
    assert tuple_result["info"]["transform"] == pytest.approx(tuple_result["window_transform"])
    assert tuple_result["info"]["bounds"] == pytest.approx((12.0, 41.0, 18.0, 47.0))
    assert tuple_result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}


@pytest.mark.parametrize(
    "window",
    [
        (0, 0, 0, 1),
        (0, 0, 1),
        (-1, 0, 1, 1),
        (4, 0, 2, 1),
        {"col_off": 0, "row_off": 0, "width": 5},
    ],
)
def test_read_raster_rejects_invalid_window(tmp_path: Path, window):
    path = tmp_path / "bad_window.tif"
    gis.write_raster(path, np.ones((2, 4), dtype=np.uint8))

    with pytest.raises((OverflowError, TypeError, ValueError), match="InvalidArgument"):
        gis.read_raster(path, window=window)


def test_read_raster_masked_true_valid_nodata_and_nan(tmp_path: Path):
    path = tmp_path / "masked.tif"
    data = np.array(
        [
            [[0.0, np.nan, 2.0], [3.0, 4.0, 5.0]],
            [[0.0, -9999.0, 2.0], [3.0, 4.0, -9999.0]],
        ],
        dtype=np.float32,
    )
    gis.write_raster(path, data, nodata=[np.nan, -9999.0])

    plain = gis.read_raster(path, masked=False)
    masked = gis.read_raster(path, masked=True)

    assert plain["mask"] is None
    assert plain["mask_polarity"] is None
    assert masked["mask_polarity"] == "true_valid"
    assert masked["mask"].dtype == np.bool_
    np.testing.assert_array_equal(
        masked["mask"],
        np.array(
            [
                [[True, False, True], [True, True, True]],
                [[True, False, True], [True, True, False]],
            ],
            dtype=bool,
        ),
    )
    assert masked["mask"][0, 0, 0]
    assert masked["mask"][1, 0, 0]


def test_read_raster_masked_zero_nodata_and_empty_raster_warning(tmp_path: Path):
    path = tmp_path / "empty.tif"
    gis.write_raster(path, np.zeros((2, 3), dtype=np.uint8), nodata=0)

    result = gis.read_raster(path, masked=True)

    assert result["mask_polarity"] == "true_valid"
    assert not result["mask"].any()
    assert "empty_raster" in _warning_codes(result["warnings"])


def test_read_raster_non_georeferenced_reports_warnings(tmp_path: Path):
    path = tmp_path / "plain.tif"
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    gis.write_raster(path, data)

    result = gis.read_raster(path)

    np.testing.assert_array_equal(result["array"], data.reshape(1, 2, 3))
    assert result["info"]["crs_wkt"] is None
    assert result["info"]["crs_authority"] is None
    assert result["info"]["transform"] is None
    assert {"missing_crs", "missing_transform", "not_georeferenced"} <= _warning_codes(
        result["warnings"]
    )


def test_read_raster_errors_are_stable(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="NotFound"):
        gis.read_raster(tmp_path / "missing.tif")

    text = tmp_path / "not_a_raster.txt"
    text.write_text("not a raster", encoding="utf-8")
    with pytest.raises(ValueError, match="UnsupportedDriver"):
        gis.read_raster(text)

    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not a tiff")
    with pytest.raises(RuntimeError, match="InvalidRaster"):
        gis.read_raster(corrupt)

    unsupported = tmp_path / "int8.tif"
    _write_int8_tiff(unsupported)
    with pytest.raises(TypeError, match="UnsupportedDType"):
        gis.read_raster(unsupported)


def test_read_raster_matches_rasterio_reference_when_available(tmp_path: Path):
    rasterio, Window = _real_rasterio()
    path = tmp_path / "reference.tif"
    data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(2.0, 0.0, 10.0, 0.0, -3.0, 50.0),
    )

    result = gis.read_raster(path, bands=[1, 3], window=(1, 1, 3, 2))

    with rasterio.open(path) as src:
        expected = src.read([1, 3], window=Window(1, 1, 3, 2))
    np.testing.assert_array_equal(result["array"], expected)
