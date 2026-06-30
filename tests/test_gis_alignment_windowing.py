"""G-002b raster alignment and Web Mercator bounds tests."""

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


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _write(path: Path, *, crs="EPSG:4326", transform=None, shape=(3, 3), nodata=None):
    if transform is None:
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, float(shape[0]))
    data = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    return gis.write_raster(path, data, crs=crs, transform=transform, nodata=nodata)


def test_identical_grid_alignment_passes(tmp_path: Path):
    source = tmp_path / "source.tif"
    target = tmp_path / "target.tif"
    source_info = _write(source, nodata=-9999.0)
    target_info = _write(target, nodata=-9999.0)

    result = gis.align_raster_to(source_info, target_info, resampling="nearest")

    assert result["diagnostics"] == []
    assert isinstance(result["info"], dict)
    assert result["info"]["width"] == 3
    assert result["info"]["height"] == 3
    assert result["array"].shape == (1, 3, 3)


def test_assert_grid_compatible_reports_matching_and_mismatches(tmp_path: Path):
    source = tmp_path / "source_assert.tif"
    target = tmp_path / "target_assert.tif"
    left = _write(source, nodata=-9999.0)
    right = _write(target, nodata=-9999.0)

    matching = gis.assert_grid_compatible(left, right)
    assert matching["compatible"] is True
    assert matching["diagnostics"] == []

    changed = _write(
        tmp_path / "target_assert_changed.tif",
        crs="EPSG:3857",
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
        shape=(2, 4),
        nodata=-1.0,
    )
    mismatch = gis.assert_grid_compatible(left, changed)
    codes = _codes(mismatch["diagnostics"])

    assert mismatch["compatible"] is False
    assert {
        "crs_mismatch",
        "shape_mismatch",
        "transform_mismatch",
        "resolution_mismatch",
        "nodata_mismatch",
    } <= codes


def test_align_raster_grid_alias_matches_older_name(tmp_path: Path):
    source = tmp_path / "source_alias.tif"
    target = tmp_path / "target_alias.tif"
    _write(source)
    target_info = _write(target, transform=(0.5, 0.0, 0.0, 0.0, -0.5, 3.0), shape=(6, 6))

    canonical = gis.align_raster_grid(source, target_info, resampling="nearest")
    alias = gis.align_raster_to(source, target_info, resampling="nearest")

    assert canonical["info"] == alias["info"]
    np.testing.assert_array_equal(canonical["array"], alias["array"])


def test_alignment_crs_mismatch_fails_without_reprojection(tmp_path: Path):
    source = tmp_path / "source_crs.tif"
    target = tmp_path / "target_crs.tif"
    _write(source, crs="EPSG:4326")
    target_info = _write(target, crs="EPSG:3857")

    with pytest.raises(ValueError, match="CrsMismatch"):
        gis.align_raster_to(source, target_info, resampling="nearest")


def test_alignment_reports_grid_mismatches(tmp_path: Path):
    source = tmp_path / "source_grid.tif"
    target = tmp_path / "target_grid.tif"
    _write(source, nodata=-1.0)
    target_info = _write(
        target,
        transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
        shape=(2, 4),
        nodata=-2.0,
    )

    result = gis.align_raster_to(source, target_info, resampling="bilinear")
    codes = _codes(result["diagnostics"])

    assert {
        "shape_mismatch",
        "transform_mismatch",
        "resolution_mismatch",
        "bounds_mismatch",
        "nodata_mismatch",
    } <= codes
    assert result["array"].shape == (1, 2, 4)
    assert result["info"]["transform"] == pytest.approx(target_info.transform)


def test_alignment_uses_nodata_for_nodata_and_outside_pixels(tmp_path: Path):
    source = tmp_path / "source_nodata.tif"
    target = tmp_path / "target_nodata.tif"
    gis.write_raster(
        source,
        np.array([[10.0, -9999.0], [30.0, 50.0]], dtype=np.float32),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        nodata=-9999.0,
    )
    target_info = gis.write_raster(
        target,
        np.zeros((1, 2), dtype=np.float32),
        crs="EPSG:4326",
        transform=(2.0, 0.0, 0.0, 0.0, -2.0, 2.0),
        nodata=-9999.0,
    )

    result = gis.align_raster_to(source, target_info, resampling="bilinear")

    assert result["array"][0, 0, 0] == pytest.approx(30.0)
    assert result["array"][0, 0, 1] == pytest.approx(-9999.0)
    assert result["info"]["nodata_per_band"] == [-9999.0]


def test_alignment_requires_explicit_resampling(tmp_path: Path):
    source = tmp_path / "source_method.tif"
    target = tmp_path / "target_method.tif"
    _write(source)
    target_info = _write(target)

    with pytest.raises(ValueError, match="resampling_required"):
        gis.align_raster_grid(source, target_info)


def test_categorical_alignment_rejects_non_nearest_when_marked(tmp_path: Path):
    source = tmp_path / "source_categorical.tif"
    target = tmp_path / "target_categorical.tif"
    _write(source)
    target_info = _write(target)

    with pytest.raises(ValueError, match="categorical_resampling_requires_nearest"):
        gis.align_raster_grid(
            {"path": source, "categorical": True},
            target_info,
            resampling="bilinear",
        )


def test_window_transform_and_read_raster_window(tmp_path: Path):
    path = tmp_path / "window_read.tif"
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    info = gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(10.0, 0.0, 100.0, 0.0, -10.0, 200.0),
        nodata=0,
    )
    window = gis.window_from_bounds(info, (110.0, 170.0, 130.0, 190.0))

    assert gis.window_transform(info, window["window"]) == pytest.approx(
        (10.0, 0.0, 110.0, 0.0, -10.0, 190.0)
    )

    unmasked = gis.read_raster_window(path, window["window"], masked=False)
    result = gis.read_raster_window(path, window["window"], masked=True)

    assert unmasked["mask"] is None
    assert unmasked["mask_polarity"] is None
    assert result["array"].shape == (1, 2, 2)
    assert isinstance(result["info"], dict)
    np.testing.assert_array_equal(result["array"][0], data[1:3, 1:3])
    assert result["mask"].shape == (1, 2, 2)
    assert result["mask_polarity"] == "true_valid"
    assert result["info"]["width"] == 2
    assert result["info"]["height"] == 2
    assert result["info"]["transform"] == pytest.approx(window["output_transform"])
    assert result["window_transform"] == pytest.approx(window["output_transform"])


def test_read_raster_window_accepts_bounds_and_boundless(tmp_path: Path):
    path = tmp_path / "window_bounds.tif"
    gis.write_raster(
        path,
        np.arange(9, dtype=np.uint8).reshape(3, 3),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        nodata=0,
    )

    clipped = gis.read_raster_window(path, (-1.0, 1.0, 2.0, 4.0), masked=True)
    boundless = gis.read_raster_window(
        path,
        (-1.0, 1.0, 2.0, 4.0),
        boundless=True,
        masked=True,
    )

    assert clipped["array"].shape == (1, 2, 2)
    assert boundless["array"].shape == (1, 3, 3)


def test_web_mercator_bounds_from_wgs84():
    bounds = gis.web_mercator_bounds((-1.0, -1.0, 1.0, 1.0), "EPSG:4326")

    assert bounds[0] < 0.0 < bounds[2]
    assert bounds[1] < 0.0 < bounds[3]
    assert bounds[0] == pytest.approx(-111319.49, rel=1e-5)
    assert bounds[2] == pytest.approx(111319.49, rel=1e-5)


def test_web_mercator_rejects_invalid_latitude_and_antimeridian():
    with pytest.raises(ValueError, match="invalid_latitude_range"):
        gis.web_mercator_bounds((-1.0, -90.0, 1.0, 1.0), "EPSG:4326")

    with pytest.raises(ValueError, match="antimeridian_bounds_unsupported"):
        gis.web_mercator_bounds((170.0, -1.0, -170.0, 1.0), "EPSG:4326")


def test_web_mercator_rejects_invalid_3857_bounds_order():
    with pytest.raises(ValueError, match="InvalidBounds"):
        gis.web_mercator_bounds((10.0, 0.0, 5.0, 1.0), "EPSG:3857")


def test_web_mercator_output_order():
    left, bottom, right, top = gis.web_mercator_bounds((0.0, 0.0, 1.0, 1.0), "EPSG:4326")

    assert left < right
    assert bottom < top
