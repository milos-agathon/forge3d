# tests/test_reproject_error_policy.py
# MENSURA win 6 (CI-gated): the silent lie in warp.rs is dead.
# reproject_raster raises TransformFailed{count, first_pixel} by default when
# any pixel's transform fails; on_transform_error="nodata" is an explicit
# opt-in that records a diagnostic. An unsupported CRS pair raises instead of
# returning an all-nodata array with a success status.
# RELEVANT FILES: src/gis/warp.rs, src/gis/mod.rs

import numpy as np
import pytest

from forge3d import gis


@pytest.fixture
def merc_raster_beyond_web_mercator_latitude(tmp_path):
    # EPSG:3857 raster whose top rows sit above the Web Mercator latitude
    # limit (85.051°): reprojecting to EPSG:4326 makes the per-pixel
    # dst->src (4326 -> 3857) transform fail for those rows.
    path = tmp_path / "high_lat_merc.tif"
    data = np.arange(16 * 16, dtype=np.float32).reshape(1, 16, 16) + 1.0
    top = 21_000_000.0  # ~ lat 86.6 deg
    pixel = 200_000.0
    transform = (pixel, 0.0, 0.0, 0.0, -pixel, top)
    gis.write_raster(str(path), data, crs="EPSG:3857", transform=transform, nodata=-9999.0)
    return path


def test_default_policy_raises_transform_failed(merc_raster_beyond_web_mercator_latitude):
    with pytest.raises(Exception) as excinfo:
        gis.reproject_raster(
            str(merc_raster_beyond_web_mercator_latitude),
            "EPSG:4326",
            resampling="nearest",
        )
    message = str(excinfo.value)
    assert "transform_failed" in message
    assert "pixel" in message and "row" in message, message


def test_nodata_policy_fills_and_reports_diagnostic(merc_raster_beyond_web_mercator_latitude):
    result = gis.reproject_raster(
        str(merc_raster_beyond_web_mercator_latitude),
        "EPSG:4326",
        resampling="nearest",
        on_transform_error="nodata",
    )
    codes = [d["code"] for d in result["diagnostics"]]
    assert "transform_failures_filled_nodata" in codes
    # The failing rows are filled with the declared nodata value.
    arr = np.asarray(result["array"])
    assert np.any(arr == -9999.0)
    # And valid rows still carry real data.
    assert np.any(arr > 0.0)


def test_unsupported_crs_pair_raises_not_nodata_success(tmp_path):
    # Regression for the pre-MENSURA behaviour: an unsupported CRS pair must
    # RAISE, never return an all-nodata array with success status.
    path = tmp_path / "plain.tif"
    data = np.ones((1, 8, 8), dtype=np.float32)
    transform = (0.01, 0.0, 13.0, 0.0, -0.01, 52.0)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=transform)
    with pytest.raises(Exception) as excinfo:
        gis.reproject_raster(str(path), "EPSG:27700", resampling="nearest")
    message = str(excinfo.value)
    assert "InvalidCrs" in message or "BackendUnavailable" in message
    # A parseable-but-untransformable pair reaches the per-pixel policy.
    with pytest.raises(Exception) as excinfo:
        gis.reproject_raster(str(path), "EPSG:4269", resampling="nearest")
    message = str(excinfo.value)
    assert "TransformFailed" in message
    assert "64 pixel(s)" in message
    assert "row 0, col 0" in message


@pytest.fixture
def singular_source_transform_raster(tmp_path):
    # A source transform whose LINEAR PART is singular (det = a*e - b*d = 0):
    # (2,1,0,2,1,0) maps every pixel onto the line x==y, so the source-affine
    # INVERSE fails for every destination pixel even though the CRS transform
    # (4326 -> 3857) succeeds. Its bounding box is non-degenerate, so it reaches
    # the per-pixel reproject loop. Before the fix, `inverse_apply(...).ok()`
    # dropped each failure to a nodata fill and reported an all-zero SUCCESS.
    path = tmp_path / "singular.tif"
    data = np.arange(64, dtype=np.float32).reshape(1, 8, 8) + 1.0
    transform = (2.0, 1.0, 0.0, 2.0, 1.0, 0.0)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=transform)
    return path


def test_singular_source_transform_raises_not_silent_nodata(singular_source_transform_raster):
    # M-05: a non-invertible SOURCE transform must be counted like any other
    # transform failure, never silently filled to an all-nodata SUCCESS.
    from forge3d._forge3d import TransformFailed

    with pytest.raises(TransformFailed) as excinfo:
        gis.reproject_raster(
            str(singular_source_transform_raster), "EPSG:3857", resampling="nearest"
        )
    assert excinfo.value.count == 64, excinfo.value.count
    assert tuple(excinfo.value.first_pixel) == (0, 0)
    assert excinfo.value.policy == "raise"


def test_singular_source_transform_nodata_policy_reports_diagnostic(
    singular_source_transform_raster,
):
    # With the explicit nodata opt-in the caller accepts the fill, but the
    # failure is still surfaced as a diagnostic (never silent).
    result = gis.reproject_raster(
        str(singular_source_transform_raster),
        "EPSG:3857",
        resampling="nearest",
        on_transform_error="nodata",
    )
    codes = [d["code"] for d in result["diagnostics"]]
    assert "transform_failures_filled_nodata" in codes


def test_align_rejects_singular_source_transform(
    singular_source_transform_raster, tmp_path
):
    # M-05 (resample/align path, not reprojection): aligning a raster whose
    # SOURCE transform is singular must RAISE (InvalidTransform), never silently
    # fill an all-nodata aligned raster with success. align_raster_to has no
    # per-pixel raise/nodata policy, so it rejects the non-invertible source up
    # front (require_invertible) instead of dropping every pixel to nodata.
    target = tmp_path / "target.tif"
    tdata = np.zeros((1, 8, 8), dtype=np.float32)
    gis.write_raster(
        str(target),
        tdata,
        crs="EPSG:4326",
        transform=(0.01, 0.0, 13.0, 0.0, -0.01, 52.0),
    )
    with pytest.raises(Exception) as excinfo:
        gis.align_raster_grid(
            str(singular_source_transform_raster), str(target), resampling="nearest"
        )
    msg = str(excinfo.value).lower()
    assert "singular" in msg or "invertible" in msg or "invalid_transform" in msg, msg


def test_invalid_policy_string_is_rejected():
    with pytest.raises(Exception) as excinfo:
        gis.reproject_raster("nonexistent.tif", "EPSG:4326", resampling="nearest",
                             on_transform_error="ignore")
    assert "unsupported_option" in str(excinfo.value)


def test_supported_reprojection_still_succeeds(tmp_path):
    path = tmp_path / "ok.tif"
    data = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    transform = (0.01, 0.0, 13.0, 0.0, -0.01, 52.0)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=transform)
    result = gis.reproject_raster(str(path), "EPSG:3857", resampling="bilinear")
    assert result["info"]["crs_authority"]["code"] == "3857"
    assert not result["diagnostics"]


def test_transform_failed_carries_structured_payload(tmp_path):
    # MENSURA M-05: the raise path exposes a STABLE structured payload as
    # exception attributes, so callers never parse the display text.
    from forge3d._forge3d import TransformFailed

    path = tmp_path / "plain.tif"
    data = np.ones((1, 8, 8), dtype=np.float32)
    transform = (0.01, 0.0, 13.0, 0.0, -0.01, 52.0)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=transform)
    with pytest.raises(TransformFailed) as excinfo:
        gis.reproject_raster(str(path), "EPSG:4269", resampling="nearest")
    exc = excinfo.value
    assert exc.count == 64
    assert tuple(exc.first_pixel) == (0, 0)
    assert exc.src_crs == "EPSG:4326"
    assert exc.dst_crs == "EPSG:4269"
    assert exc.policy == "raise"


def test_transform_failed_count_is_per_pixel_not_per_band(tmp_path):
    # MENSURA M-05 band-count fix: a failed pixel in a MULTIBAND raster is
    # counted ONCE, not once per band. A 3-band 8x8 all-failing reprojection
    # must report 64 failed pixels, never 3*64 = 192.
    from forge3d._forge3d import TransformFailed

    path = tmp_path / "rgb.tif"
    data = np.ones((3, 8, 8), dtype=np.float32)
    transform = (0.01, 0.0, 13.0, 0.0, -0.01, 52.0)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=transform)
    with pytest.raises(TransformFailed) as excinfo:
        gis.reproject_raster(str(path), "EPSG:4269", resampling="nearest")
    assert excinfo.value.count == 64, f"expected 64 per-pixel failures, got {excinfo.value.count}"


def test_transform_failed_is_surfaced_at_package_level():
    # M-05/M-07: TransformFailed is re-exported at the forge3d package level
    # (like MemoryBudgetExceeded/DegradedCapability), not only on the private
    # _forge3d module, so `except forge3d.TransformFailed` works and it is in
    # the public __all__.
    import forge3d
    from forge3d._forge3d import TransformFailed as _native_tf

    assert hasattr(forge3d, "TransformFailed")
    assert forge3d.TransformFailed is _native_tf
    assert "TransformFailed" in forge3d.__all__
