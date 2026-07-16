"""G-002c C5 rasterization and explicit mask contract tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS rasterize/mask tests require the compiled _forge3d extension",
)


def _target_info(
    *,
    width: int = 4,
    height: int = 4,
    transform: tuple[float, float, float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        4.0,
    ),
    dtype: str = "uint8",
    crs: str | None = "4326",
) -> dict:
    info = {
        "path": "",
        "driver": "memory",
        "width": width,
        "height": height,
        "band_count": 1,
        "dtype_per_band": [dtype],
        "crs_wkt": None,
        "crs_authority": None if crs is None else {"name": "EPSG", "code": crs},
        "transform": transform,
        "bounds": gis.array_bounds(height, width, transform),
        "resolution": (abs(transform[0]), abs(transform[4])),
        "nodata_per_band": [None],
        "warnings": [],
    }
    return info


def _square() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]]
        ],
    }


def _tiny_boundary_square() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[0.9, 0.9], [1.1, 0.9], [1.1, 1.1], [0.9, 1.1], [0.9, 0.9]]
        ],
    }


def _fc(features: list[dict], *, crs: str | None = "4326") -> dict:
    out = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        out["crs"] = {"type": "name", "properties": {"name": f"EPSG:{crs}"}}
    return out


def _feature(geometry: dict, properties: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": geometry,
    }


def _write_geojson(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _codes(warnings) -> set[str]:
    return {warning["code"] for warning in warnings}


def test_rasterize_vectors_constant_burn_on_explicit_grid(tmp_path: Path):
    path = _write_geojson(tmp_path / "poly.geojson", _fc([_feature(_square())]))

    result = gis.rasterize_vectors(path, _target_info(), value=7)

    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[1:3, 1:3] = 7
    np.testing.assert_array_equal(result["array"], expected)
    assert result["target_shape"] == (4, 4)
    assert result["target_transform"] == pytest.approx((1.0, 0.0, 0.0, 0.0, -1.0, 4.0))
    assert result["target_bounds"] == pytest.approx((0.0, 0.0, 4.0, 4.0))
    assert result["dtype"] == "uint8"
    assert result["fill"] == 0
    assert result["burned_pixels"] == 4
    assert result["all_touched"] is False
    assert result["warnings"] == []
    assert result["info"]["width"] == 4
    assert result["info"]["height"] == 4


def test_rasterize_vectors_attribute_burn_fill_and_dtype():
    source = _fc([_feature(_square(), {"burn": -2})])

    result = gis.rasterize_vectors(
        source,
        _target_info(dtype="int16"),
        attribute="burn",
        dtype="int16",
        fill=-9,
    )

    expected = np.full((4, 4), -9, dtype=np.int16)
    expected[1:3, 1:3] = -2
    np.testing.assert_array_equal(result["array"], expected)
    assert result["dtype"] == "int16"
    assert result["fill"] == -9
    assert result["burned_pixels"] == 4


def test_rasterize_vectors_per_feature_burn_values_and_merge_alg():
    left = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
    }
    right = {
        "type": "Polygon",
        "coordinates": [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]],
    }
    source = _fc([_feature(left), _feature(right)])

    replaced = gis.rasterize_vectors(
        source, _target_info(), burn_values=[2, 5], merge_alg="replace"
    )
    added = gis.rasterize_vectors(
        source, _target_info(), burn_values=[2, 5], merge_alg="add"
    )

    assert replaced["array"][2, 1] == 5
    assert added["array"][2, 1] == 7
    assert replaced["merge_alg"] == "replace"
    assert added["merge_alg"] == "add"


def test_rasterize_vectors_burn_value_contract_errors():
    source = _fc([_feature(_square()), _feature(_square())])
    with pytest.raises(ValueError, match="burn_values length"):
        gis.rasterize_vectors(source, _target_info(), burn_values=[1, 2, 3])
    with pytest.raises(ValueError, match="mutually exclusive"):
        gis.rasterize_vectors(
            source, _target_info(), burn_values=[1, 2], attribute="burn"
        )
    with pytest.raises(ValueError, match="merge_alg"):
        gis.rasterize_vectors(source, _target_info(), merge_alg="max")


def test_rasterize_vectors_all_touched_cell_touch_semantics():
    source = _fc([_feature(_tiny_boundary_square())])
    target = _target_info(width=3, height=3, transform=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))

    center = gis.rasterize_vectors(source, target, value=1, all_touched=False)
    touched = gis.rasterize_vectors(source, target, value=1, all_touched=True)

    np.testing.assert_array_equal(center["array"], np.zeros((3, 3), dtype=np.uint8))
    expected = np.zeros((3, 3), dtype=np.uint8)
    expected[1:3, 0:2] = 1
    np.testing.assert_array_equal(touched["array"], expected)
    assert touched["burned_pixels"] == 4


def test_geometry_mask_true_inside_and_invert():
    source = _fc([_feature(_square())])

    inside = gis.geometry_mask(source, _target_info(), mask_polarity="true_inside")
    inverted = gis.geometry_mask(
        source,
        _target_info(),
        invert=True,
        mask_polarity="true_inside",
    )

    expected = np.zeros((4, 4), dtype=bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(inside["mask"], expected)
    assert inside["mask_polarity"] == "true_inside"
    assert inside["true_count"] == 4
    assert inside["false_count"] == 12
    assert inside["crop_window"] is None
    np.testing.assert_array_equal(inverted["mask"], ~expected)
    assert inverted["mask_polarity"] == "true_outside"
    assert inverted["true_count"] == 12
    assert inverted["false_count"] == 4


def test_mask_raster_requires_explicit_valid_polarity_and_applies_fill():
    source = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    mask = np.array(
        [[True, False, True], [False, True, False], [True, False, True]],
        dtype=bool,
    )

    with pytest.raises(ValueError, match="mask_polarity_explicit"):
        gis.mask_raster(source, mask)
    with pytest.raises(ValueError, match="mask_polarity_explicit"):
        gis.mask_raster(source, mask, mask_polarity="inside")

    result = gis.mask_raster(source, mask, mask_polarity="true_valid", fill=0)

    expected = np.array(
        [[[1, 0, 3], [0, 5, 0], [7, 0, 9]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result["array"], expected)
    assert result["mask_polarity"] == "true_valid"
    assert result["fill"] == 0.0
    assert result["nodata"] == [None]
    assert result["true_count"] == 5
    assert result["false_count"] == 4
    assert result["valid_count"] == 5
    assert result["crop_window"] is None


def test_mask_raster_crop_updates_window_transform_and_bounds():
    source = {
        "array": np.arange(16, dtype=np.uint8).reshape(1, 4, 4),
        "info": _target_info(),
    }
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    result = gis.mask_raster(source, mask, mask_polarity="true_valid", crop=True)

    np.testing.assert_array_equal(
        result["array"],
        np.array([[[5, 6], [9, 10]]], dtype=np.uint8),
    )
    assert result["crop_window"] == (1, 1, 2, 2)
    assert result["info"]["width"] == 2
    assert result["info"]["height"] == 2
    assert result["info"]["transform"] == pytest.approx((1.0, 0.0, 1.0, 0.0, -1.0, 3.0))
    assert result["info"]["bounds"] == pytest.approx((1.0, 1.0, 3.0, 3.0))


def test_mask_raster_nodata_fill_invalid_nodata_and_empty_crop():
    source = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    result = gis.mask_raster(source, mask, mask_polarity="true_valid", nodata=255)
    expected = np.full((1, 3, 3), 255, dtype=np.uint8)
    expected[0, 1, 1] = 5
    np.testing.assert_array_equal(result["array"], expected)
    assert result["fill"] is None
    assert result["nodata"] == [255.0]
    assert result["nodata_per_band"] == [255.0]
    assert result["true_count"] == 1
    assert result["false_count"] == 8

    with pytest.raises(ValueError, match="invalid_nodata"):
        gis.mask_raster(source, mask, mask_polarity="true_valid", nodata=-1)
    empty = gis.mask_raster(source, np.zeros((3, 3), dtype=bool), mask_polarity="true_valid", crop=True)
    assert empty["array"].shape == (1, 0, 0)
    assert empty["crop_window"] == (0, 0, 0, 0)
    assert "empty_raster" in _codes(empty["warnings"])


def test_mask_raster_shape_mismatch():
    source = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape_mismatch"):
        gis.mask_raster(source, np.zeros((2, 2), dtype=bool), mask_polarity="true_valid")
    with pytest.raises(ValueError, match="shape_mismatch"):
        gis.mask_raster(source, np.zeros((3, 3, 3), dtype=bool), mask_polarity="true_valid")


def test_rasterize_vectors_crs_mismatch_and_missing_crs_errors():
    with pytest.raises(ValueError, match="crs_mismatch"):
        gis.rasterize_vectors(_fc([_feature(_square())], crs="3857"), _target_info())

    with pytest.raises(ValueError, match="missing_crs"):
        gis.rasterize_vectors(_fc([_feature(_square())], crs=None), _target_info())

    with pytest.raises(ValueError, match="missing_crs"):
        gis.rasterize_vectors(_fc([_feature(_square())]), _target_info(crs=None))


def test_rasterize_vectors_empty_unsupported_and_dtype_errors():
    with pytest.raises(ValueError, match="empty_feature_set"):
        gis.rasterize_vectors(_fc([]), _target_info())

    with pytest.raises(ValueError, match="empty_geometry"):
        gis.rasterize_vectors(_fc([_feature({"type": "Polygon", "coordinates": []})]), _target_info())

    with pytest.raises(ValueError, match="unsupported_geometry_type"):
        gis.rasterize_vectors(
            _fc([_feature({"type": "Point", "coordinates": [1.0, 1.0]})]),
            _target_info(),
        )

    with pytest.raises(TypeError, match="unsupported_dtype"):
        gis.rasterize_vectors(_fc([_feature(_square())]), _target_info(), dtype="bool")


def test_no_runtime_python_gis_backend_imports_or_rust_backend_mentions():
    repo = Path(__file__).resolve().parents[1]
    files = [repo / "python/forge3d/gis.py", repo / "src/gis/rasterize.rs"]
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


def test_optional_rasterio_reference_rasterize_matches_small_polygon():
    rio_features = pytest.importorskip("rasterio.features")
    rio_transform = pytest.importorskip("rasterio.transform")
    source = _fc([_feature(_square())])

    result = gis.rasterize_vectors(source, _target_info(), value=1)
    expected = rio_features.rasterize(
        [(_square(), 1)],
        out_shape=(4, 4),
        transform=rio_transform.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(result["array"], expected)
