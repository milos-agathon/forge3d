"""G-002c C6 thematic raster contract tests."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE, get_native_module


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS thematic tests require the compiled _forge3d extension",
)


def _memory_info(array: np.ndarray) -> dict:
    shape = array.shape
    bands, height, width = (1, shape[0], shape[1]) if array.ndim == 2 else shape
    return {
        "path": "",
        "driver": "memory",
        "width": width,
        "height": height,
        "band_count": bands,
        "dtype_per_band": [array.dtype.name] * bands,
        "crs_wkt": None,
        "crs_authority": None,
        "transform": None,
        "bounds": None,
        "resolution": None,
        "nodata_per_band": [None] * bands,
        "warnings": [],
    }


def _expected_classes(values, bins, *, right=False, valid=None):
    values = np.asarray(values)
    valid = np.ones(values.shape, dtype=bool) if valid is None else valid
    out = np.zeros(values.shape, dtype=np.uint16)
    out[valid] = np.digitize(values[valid], bins, right=right).astype(np.uint16) + 1
    return out


def test_public_surface_exposes_thematic_functions():
    native = get_native_module()

    assert callable(gis.normalize_raster)
    assert callable(gis.classify_raster)
    assert callable(native.normalize_raster)
    assert callable(native.classify_raster)
    assert "normalize_raster" in gis.__all__
    assert "classify_raster" in gis.__all__


def test_normalize_minmax_numpy_array_stats_and_dtype():
    source = np.array([[1, 3], [5, 9]], dtype=np.uint16)

    result = gis.normalize_raster(source)

    expected = np.array([[[0.0, 0.25], [0.5, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(result["array"], expected)
    assert result["array"].dtype == np.float32
    assert result["method"] == "minmax"
    assert result["class_table"] is None
    assert result["valid_count"] == 4
    assert result["nodata_count"] == 0
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(9.0)
    assert result["mean"] == pytest.approx(4.5)
    assert result["std"] == pytest.approx(np.std(source.astype(np.float64)))
    assert result["warnings"] == []
    assert result["info"]["driver"] == "memory"
    assert result["info"]["dtype_per_band"] == ["float32"]


def test_normalize_clip_clamps_before_stats_and_scaling():
    source = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)

    result = gis.normalize_raster(source, clip=(2.0, 8.0))

    expected = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(result["array"], expected)
    assert result["min"] == pytest.approx(2.0)
    assert result["max"] == pytest.approx(8.0)


def test_normalize_excludes_nodata_valid_mask_and_nan_cells():
    source = np.array([[1.0, -9999.0], [np.nan, 7.0]], dtype=np.float32)
    valid_mask = np.array([[True, True], [True, False]], dtype=bool)

    result = gis.normalize_raster(source, nodata=-9999.0, valid_mask=valid_mask)

    assert result["valid_count"] == 1
    assert result["nodata_count"] == 3
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(1.0)
    assert result["array"][0, 0, 0] == pytest.approx(0.0)
    assert np.isnan(result["array"][0, 0, 1])
    assert np.isnan(result["array"][0, 1, 0])
    assert np.isnan(result["array"][0, 1, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodata": 0},
        {"valid_mask": np.zeros((2, 2), dtype=bool)},
    ],
)
def test_normalize_all_invalid_raises_empty_raster(kwargs):
    source = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="empty_raster"):
        gis.normalize_raster(source, **kwargs)


def test_normalize_unsupported_method_and_bad_clip():
    source = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported_option"):
        gis.normalize_raster(source, method="zscore")
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.normalize_raster(source, clip=(2.0, 2.0))
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.normalize_raster(source, clip=(0.0, np.inf))


def test_valid_mask_shape_mismatch_raises_shape_mismatch():
    source = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shape_mismatch"):
        gis.normalize_raster(source, valid_mask=np.ones((3, 3), dtype=bool))


def test_valid_mask_one_band_3d_does_not_broadcast_to_multiband():
    source = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shape_mismatch"):
        gis.normalize_raster(source, valid_mask=np.ones((1, 2, 2), dtype=bool))


def test_nodata_list_length_mismatch_raises_invalid_nodata():
    source = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="invalid_nodata"):
        gis.normalize_raster(source, nodata=[0.0])


def test_bad_source_diagnostics_include_expected_tokens():
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.normalize_raster({"info": _memory_info(np.ones((2, 2), dtype=np.float32))})
    with pytest.raises(TypeError, match="unsupported_dtype"):
        gis.normalize_raster(np.ones((2, 2), dtype=bool))


def test_classify_explicit_bins_boundary_semantics_and_counts():
    source = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    bins = [1.0, 2.0]

    left_closed = gis.classify_raster(source, bins=bins, right=False)
    right_closed = gis.classify_raster(source, bins=bins, right=True)

    np.testing.assert_array_equal(
        left_closed["array"],
        _expected_classes(source.reshape(1, 1, 4), bins, right=False),
    )
    np.testing.assert_array_equal(
        right_closed["array"],
        _expected_classes(source.reshape(1, 1, 4), bins, right=True),
    )
    assert left_closed["method"] == "explicit_bins"
    assert left_closed["array"].dtype == np.uint16
    assert [row["count"] for row in left_closed["class_table"]] == [0, 1, 1, 2]
    assert sum(row["count"] for row in left_closed["class_table"]) == source.size


def test_classify_accepts_numpy_bins_array():
    source = np.array([[0, 5, 10]], dtype=np.uint8)

    result = gis.classify_raster(source, bins=np.array([5], dtype=np.int32))

    np.testing.assert_array_equal(result["array"], np.array([[[1, 2, 2]]], dtype=np.uint16))


def test_classify_labels_and_class_table_shape():
    source = np.array([[0, 5, 10, 15]], dtype=np.uint8)

    result = gis.classify_raster(
        source,
        bins=[5, 10],
        labels=["low", "mid", "high"],
        right=True,
    )

    table = result["class_table"]
    assert table[0] == {
        "class_id": 0,
        "label": "nodata",
        "left": None,
        "right": None,
        "right_inclusive": False,
        "count": 0,
        "nodata": True,
    }
    assert [row["label"] for row in table[1:]] == ["low", "mid", "high"]
    assert [row["class_id"] for row in table] == [0, 1, 2, 3]
    assert [row["right_inclusive"] for row in table[1:]] == [True, True, True]
    assert table[1]["left"] is None
    assert table[1]["right"] == pytest.approx(5.0)
    assert table[2]["left"] == pytest.approx(5.0)
    assert table[2]["right"] == pytest.approx(10.0)
    assert table[3]["left"] == pytest.approx(10.0)
    assert table[3]["right"] is None


def test_classify_invalid_nodata_cells_use_zero_and_stats_exclude_them():
    source = np.array([[1.0, -1.0], [np.nan, 9.0]], dtype=np.float32)
    valid_mask = np.array([[True, False], [True, True]], dtype=bool)

    result = gis.classify_raster(
        source,
        bins=[5.0],
        nodata=-1.0,
        valid_mask=valid_mask,
        dtype="uint8",
    )

    np.testing.assert_array_equal(
        result["array"],
        np.array([[[1, 0], [0, 2]]], dtype=np.uint8),
    )
    assert result["valid_count"] == 2
    assert result["nodata_count"] == 2
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(9.0)
    assert [row["count"] for row in result["class_table"]] == [2, 1, 1]


@pytest.mark.parametrize(
    "bins",
    [None, [], [1.0, np.inf], [2.0, 1.0], [1.0, 1.0], np.array([[1.0, 2.0]])],
)
def test_classify_bad_bins_raise_invalid_argument(bins):
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.classify_raster(np.ones((2, 2), dtype=np.float32), bins=bins)


def test_classify_label_count_mismatch():
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.classify_raster(
            np.ones((2, 2), dtype=np.float32),
            bins=[1.0, 2.0],
            labels=["low", "high"],
        )


@pytest.mark.parametrize("dtype", ["float32", "int8", "bool", "not-a-dtype"])
def test_classify_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="unsupported_dtype"):
        gis.classify_raster(np.ones((2, 2), dtype=np.float32), bins=[1.0], dtype=dtype)


def test_classify_output_dtype_too_small():
    bins = np.arange(255, dtype=np.float64)

    with pytest.raises(TypeError, match="unsupported_dtype"):
        gis.classify_raster(np.array([[1.0]], dtype=np.float32), bins=bins, dtype="uint8")


def test_source_as_read_raster_style_dict():
    source = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    result = gis.normalize_raster({"array": source, "info": _memory_info(source)})

    np.testing.assert_allclose(
        result["array"],
        np.array([[[0.0, 1 / 3], [2 / 3, 1.0]]], dtype=np.float32),
    )
    assert result["info"]["width"] == 2
    assert result["info"]["height"] == 2


def test_no_runtime_python_gis_backend_library_usage():
    repo = Path(__file__).resolve().parents[1]
    files = [repo / "python/forge3d/gis.py", repo / "src/gis/thematic.rs"]
    banned = ("rasterio", "geopandas", "shapely", "rioxarray", "xarray", "terra")

    for path in files:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
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
            for name in banned:
                assert name not in imports
        else:
            for name in banned:
                assert name not in source
