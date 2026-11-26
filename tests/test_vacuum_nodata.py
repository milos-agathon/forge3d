"""Marker identity through terrain ingest, GIS derivatives and native GPU readback."""

from hashlib import sha256

import numpy as np
import pytest

import forge3d as f3d
from forge3d import gis
from forge3d.map_scene import _load_native_heightmap
from forge3d.terrain_params import make_terrain_params_config
from _terrain_runtime import _write_test_hdr

MARKERS = (-9999.0, -32768.0, 3.4e38, np.nan)


def grid(marker, flat=False):
    y, x = np.indices((32, 32), dtype=np.float32)
    values = np.full_like(x, 0.5) if flat else (x + 2 * y) / 128
    values[10:19, 8:17] = marker
    values[22, 23] = marker
    return values


def ingest(values, marker):
    return _load_native_heightmap(
        f3d.TerrainSource(data=values, metadata={"nodata": marker})
    )


def test_marker_derivatives():
    from forge3d._raster_sampling import gradient

    expected_hash = None
    for marker in MARKERS:
        raw = grid(marker)
        heights = ingest(raw, marker)
        valid = np.isfinite(heights)
        assert valid.sum() == 32 * 32 - 82
        gy, gx = gradient(heights)
        np.testing.assert_array_equal(gx[valid], np.full(valid.sum(), 1 / 128))
        np.testing.assert_array_equal(gy[valid], np.full(valid.sum(), 2 / 128))
        normal = np.stack((-gx, -gy, np.ones_like(gx)), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        analytic = np.array([-1 / 128, -2 / 128, 1], np.float32)
        analytic /= np.linalg.norm(analytic)
        np.testing.assert_array_equal(
            normal[valid], np.tile(analytic, (valid.sum(), 1))
        )
        from forge3d._detail_normals import detail_to_normal_map

        encoded = ((analytic * 0.5 + 0.5) * 255).astype(np.uint8)
        np.testing.assert_array_equal(
            detail_to_normal_map(heights)[valid], np.tile(encoded, (valid.sum(), 1))
        )
        result = gis.prepare_terrain_derivatives(
            {
                "array": raw,
                "info": {
                    "width": 32,
                    "height": 32,
                    "transform": (1, 0, 0, 0, -1, 32),
                    "nodata_per_band": [marker],
                },
            }
        )["derivatives"]
        slope = result["slope"]["array"][0]
        np.testing.assert_array_equal(np.isfinite(slope), valid)
        np.testing.assert_array_equal(result["hillshade"]["mask"][0], valid)
        expected = np.float32(np.degrees(np.arctan(np.hypot(1 / 128, 2 / 128))))
        np.testing.assert_array_equal(slope[valid], np.full(valid.sum(), expected))
        digest = sha256(
            slope[valid].tobytes()
            + result["hillshade"]["array"][0][valid].tobytes()
            + normal[valid].tobytes()
        ).hexdigest()
        print(
            "analysis",
            marker,
            digest,
            "valid",
            valid.sum(),
            "max slope error",
            abs(slope[valid] - expected).max(),
            "max normal error",
            abs(normal[valid] - analytic).max(),
        )
        if expected_hash is not None:
            assert digest == expected_hash
        expected_hash = digest


def renderer_assets(directory):
    hdr = directory / "environment.hdr"
    _write_test_hdr(hdr)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    color = f3d.Colormap1D.from_stops(
        stops=[(0, "#809040"), (1, "#809040")], domain=(0, 1)
    )
    config = make_terrain_params_config(
        size_px=(96, 96),
        terrain_span=2.8,
        render_scale=1,
        msaa_samples=1,
        z_scale=1,
        exposure=1,
        domain=(0, 1),
        cam_radius=4,
        cam_theta_deg=80,
        camera_mode="mesh:zup:dem",
        clip=(0.1, 100),
        aa_seed=77,
        terrain_shading_model="lambert_physical",
        ibl_enabled=False,
        ibl_intensity=0,
        albedo_mode="colormap",
        overlays=[f3d.OverlayLayer.from_colormap1d(color, strength=1)],
        hue_variation_strength=0,
        colormap_strength=1,
        material_slope_bias=0,
    )
    config.shadows.enabled = False
    params = f3d.TerrainRenderParams(config)
    return session, renderer, material, ibl, params


def draw(assets, heights, *, normals=False):
    _, renderer, material, ibl, params = assets
    frame, aov = renderer.render_with_aov(material, ibl, params, heights)
    outputs = frame.to_numpy(), aov.albedo(), aov.depth()
    return outputs + (aov.normal(),) if normals else outputs


def test_marker_frames(tmp_path):
    from PIL import Image

    assets = renderer_assets(tmp_path)
    print("marker frame device", assets[0].info(), flush=True)
    complete, complete_albedo, _ = draw(assets, np.full((32, 32), 0.5, np.float32))
    expected = None
    sloped = None
    for i, marker in enumerate(MARKERS):
        frame, albedo, depth = draw(assets, ingest(grid(marker, flat=True), marker))
        Image.fromarray(frame).save(tmp_path / f"marker-{i}.png")
        valid = np.any(albedo > 0, axis=-1)
        assert 0 < valid.sum() < valid.size
        dense = np.any(complete_albedo > 0, axis=-1)
        assert np.all(dense[valid]) and dense.sum() > valid.sum()
        np.testing.assert_array_equal(frame[valid], complete[valid])
        colors, counts = np.unique(frame[valid], axis=0, return_counts=True)
        rim = int(valid.sum() - counts.max())
        digest = sha256(frame[valid].tobytes()).hexdigest()
        print("render", marker, digest, "valid", valid.sum(), "rim", rim)
        assert rim == 0, (colors, counts)
        if expected is not None:
            for actual, reference in zip((frame, valid, depth), expected):
                np.testing.assert_array_equal(actual, reference)
        expected = frame, valid, depth
        relief = draw(assets, ingest(grid(marker), marker), normals=True)
        actual = relief[3][np.any(relief[1] > 0, axis=-1)]
        analytic = np.array([-31 / (128 * 2.8), 1.0, -62 / (128 * 2.8)])
        analytic /= np.linalg.norm(analytic)
        error = np.max(np.abs(actual - analytic), axis=0)
        # The renderer stores the normal AOV as RGBA16Float.
        rounding = np.abs(np.spacing(analytic.astype(np.float16))).astype(float)
        print("normal AOV", marker, error, "valid", len(actual))
        assert np.all(error <= rounding)
        assert np.any(relief[2] != depth)
        print("relief", marker, sha256(b"".join(a.tobytes() for a in relief)).hexdigest())
        if sloped is not None:
            for actual, reference in zip(relief, sloped):
                np.testing.assert_array_equal(actual, reference)
        sloped = relief


def test_single_nan_bilinear_support():
    grid = np.ones((3, 3), np.float32)
    grid[1, 1] = np.nan
    image = gis.resample_raster(grid, (9, 9), method="bilinear")["array"][0]
    coords = (np.arange(9) + 0.5) / 3 - 0.5
    weight = np.maximum(0.0, 1.0 - np.abs(coords - 1.0))
    expected = 1.0 - np.outer(weight, weight) >= 0.5
    np.testing.assert_array_equal(np.isfinite(image), expected)
    np.testing.assert_array_equal(image[expected], np.ones(expected.sum()))
    for marker in MARKERS:
        raw = np.ones_like(grid)
        raw[1, 1] = marker
        masked = np.ma.array(raw, mask=~np.isfinite(grid))
        result = gis.resample_raster(masked, (9, 9), method="bilinear")["array"][0]
        np.testing.assert_array_equal(result, image)
    print("single NaN invalidates", (~expected).sum(), "of", expected.size)


def test_legacy_height_and_empty_atlas(tmp_path):
    from PIL import Image
    from forge3d._native import get_native_module

    native = get_native_module()
    scene = native.Scene(64, 64)
    scene.set_height_from_r32f(np.full((32, 32), 0.5, np.float32))
    scene.enable_altitude_overlay(1.0)
    solid = np.asarray(scene.render_rgba()).copy()
    frames = []
    for marker in MARKERS:
        scene.set_height_from_r32f(ingest(grid(marker, flat=True), marker))
        frames.append(np.asarray(scene.render_rgba()).copy())
    assert np.any(solid != frames[0])
    for frame in frames[1:]:
        np.testing.assert_array_equal(frame, frames[0])

    spike = native.TerrainSpike(64, 64)
    spike.enable_height_mosaic(8, 2, 2, None, False)
    path = tmp_path / "empty-atlas.png"
    spike.render_png(str(path))
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    assert np.unique(rgba.reshape(-1, 4), axis=0).shape[0] == 1


def test_file_mask_and_continuous_kernels(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from forge3d.alignment import _warp_band, reproject_dem_to_target
    from forge3d._detail_normals import gaussian_blur
    from forge3d.io import load_dem

    valid = np.ones((8, 8), bool)
    valid[2:5, 2:5] = False
    for i, marker in enumerate(MARKERS):
        path = tmp_path / f"mask-{i}.tif"
        raw = np.full((8, 8), 2.0, np.float32)
        raw[~valid] = marker
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype="float32",
            width=8,
            height=8,
            count=1,
            crs="EPSG:3857",
            transform=from_origin(0, 8, 1, 1),
        ) as dataset:
            dataset.write(raw, 1)
            dataset.write_mask(valid)
        data = load_dem(path, fill_nodata_values=False)
        assert data.domain == (2.0, 2.0)
        assert data.stats["count"] == valid.sum()
        for method in ("bilinear", "cubic"):
            result = reproject_dem_to_target(path, "EPSG:3857", resampling=method)
            np.testing.assert_array_equal(result["validity_mask"], valid)
            np.testing.assert_array_equal(
                result["array"][valid], np.full(valid.sum(), 2.0)
            )
        # Shift the sample grid by a quarter cell. The cubic convolution and
        # bilinear filters then give different support at the very same edge.
        for method, offsets, weights in (
            ("bilinear", (0, 1), (0.75, 0.25)),
            ("cubic", (-1, 0, 1, 2), (-9 / 128, 111 / 128, 29 / 128, -3 / 128)),
        ):
            dst = np.empty_like(raw)
            covered = _warp_band(
                raw, valid, dst, src_transform=from_origin(0, 8, 1, 1),
                dst_transform=from_origin(0.25, 7.75, 1, 1),
                src_crs="EPSG:3857", dst_crs="EPSG:3857", method=method, nodata=np.nan,
            )
            for y in range(1, 6):
                for x in range(1, 6):
                    support = sum(
                        wx * wy for dx, wx in zip(offsets, weights)
                        for dy, wy in zip(offsets, weights) if valid[y + dy, x + dx]
                    )
                    assert covered[y, x] == (support >= 0.5)
                    if support >= 0.5:
                        assert dst[y, x] == 2.0
                    else:
                        assert np.isnan(dst[y, x])
        blurred = gaussian_blur(np.where(valid, raw, np.nan), 1.0)
        np.testing.assert_array_equal(
            blurred[np.isfinite(blurred)], np.full(np.isfinite(blurred).sum(), 2.0)
        )


def test_stored_marker_and_explicit_mask_are_both_applied(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from forge3d.alignment import reproject_dem_to_target, resample_raster_to_grid
    from forge3d.preparation import PreparationGrid, PreparationRecipe, PreparationSource, prepare_scene

    transform = from_origin(0, 4, 1, 1)
    grid = dict(crs="EPSG:3857", transform=transform, width=4, height=4)
    expected = np.ones((4, 4), bool)
    expected[1, 1] = False
    for i, marker in enumerate(MARKERS):
        raw = np.ones((4, 4), np.float32)
        raw[1, 1] = marker
        path = tmp_path / f"explicit-valid-{i}.tif"
        with rasterio.open(path, "w", driver="GTiff", count=1, dtype=raw.dtype,
                           nodata=float(np.float32(marker)), **grid) as ds:
            ds.write(raw, 1)
            ds.write_mask(np.ones(raw.shape, bool))
        source = PreparationSource("dem", path, "elevation", units="m", vertical_reference="NAP")
        target = PreparationGrid("EPSG:3857", 4, 4, transform.to_gdal())
        prepared = prepare_scene(PreparationRecipe(target, (source,), "NAP"))
        np.testing.assert_array_equal(prepared.layers["dem"].valid_mask, expected)
        for method in ("nearest", "bilinear", "cubic"):
            dem = reproject_dem_to_target(path, grid["crs"], resampling=method)
            aligned = resample_raster_to_grid(path, grid, resampling=method)
            for result in (dem, aligned):
                np.testing.assert_array_equal(result["validity_mask"], expected)
                assert np.all(result["array"].reshape(4, 4)[expected] == 1)


@pytest.mark.parametrize("explicit", [False, True])
def test_load_dem_preserves_validity_before_narrowing(tmp_path, explicit):
    import rasterio
    from rasterio.transform import from_origin
    from forge3d.io import load_dem

    path = tmp_path / "float64.tif"
    raw = np.full((4, 4), 16777217, np.float64)
    raw[1, 1] = 16777216
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype=raw.dtype,
                       nodata=16777216, width=4, height=4, crs="EPSG:3857",
                       transform=from_origin(0, 4, 1, 1)) as ds:
        ds.write(raw, 1)
        if explicit:
            ds.write_mask(raw != 16777216)
    for dtype in (np.float32, np.float64):
        dem = load_dem(path, dtype=dtype, fill_nodata_values=False)
        assert dem.stats["count"] == 15
        assert dem.domain == (float(dtype(16777217)),) * 2
        valid = np.isfinite(dem.data) & (dem.data != dem.nodata_value)
        np.testing.assert_array_equal(valid, raw != 16777216)
        filled = load_dem(path, dtype=dtype)
        assert filled.stats["count"] == 16
        assert np.all(filled.data == dtype(16777217))

    from forge3d.alignment import reproject_dem_to_target, resample_raster_to_grid
    from forge3d._detail_normals import load_dem as load_detail_heights
    from forge3d.preparation import PreparationGrid, PreparationRecipe, PreparationSource, prepare_scene

    expected = raw != 16777216
    detail = load_detail_heights(path)
    np.testing.assert_array_equal(detail, np.where(expected, raw.astype(np.float32), np.nan))
    for target_info in (None, gis.read_raster_info(path)):
        native_dem = gis.prepare_dem(path, target_info=target_info)
        assert native_dem["valid_count"] == 15
        assert np.isnan(native_dem["nodata_summary"][0])
        np.testing.assert_array_equal(native_dem["mask"][0], expected)
        np.testing.assert_array_equal(native_dem["array"][0], detail)
    target = PreparationGrid("EPSG:3857", 4, 4, from_origin(0, 4, 1, 1).to_gdal())
    source = PreparationSource("dem", path, "elevation", units="m", vertical_reference="NAP")
    prepared = prepare_scene(PreparationRecipe(target, (source,), "NAP"))
    np.testing.assert_array_equal(prepared.layers["dem"].valid_mask, expected)
    grid = dict(crs="EPSG:3857", transform=from_origin(0, 4, 1, 1), width=4, height=4)
    aligned = resample_raster_to_grid(path, grid, output_path=tmp_path / "aligned.tif")
    np.testing.assert_array_equal(aligned["array"], raw)
    np.testing.assert_array_equal(aligned["validity_mask"], expected)
    for method in ("nearest", "bilinear", "cubic"):
        aligned = resample_raster_to_grid(path, grid, resampling=method)
        np.testing.assert_array_equal(aligned["validity_mask"], expected)
        np.testing.assert_array_equal(aligned["array"][expected], raw[expected])
        out = tmp_path / f"{method}.tif"
        warped = reproject_dem_to_target(path, grid["crs"], resampling=method, output_path=out)
        np.testing.assert_array_equal(warped["validity_mask"], expected)
        assert np.isnan(warped["profile"]["nodata"]) and np.isnan(warped["metadata"]["nodata"])
        with rasterio.open(out) as ds:
            assert np.isnan(ds.nodata)
            np.testing.assert_array_equal(ds.read_masks(1) != 0, expected)
            np.testing.assert_array_equal(ds.read(1), warped["array"])
        reopened = load_dem(out, fill_nodata_values=False)
        np.testing.assert_array_equal(np.isfinite(reopened.data), expected)
        np.testing.assert_array_equal(reopened.data[expected], raw[expected].astype(np.float32))


@pytest.mark.parametrize("internal", [False, True], ids=["sidecar", "internal"])
@pytest.mark.parametrize("marker", MARKERS)
def test_native_file_mask_reaches_resampling(tmp_path, internal, marker):
    import rasterio
    from rasterio.transform import from_origin

    raw = np.full((4, 4), 2.0, dtype=np.float64)
    raw[1, 1] = marker
    valid = np.ones(raw.shape, bool)
    valid[1, 1] = False
    path = tmp_path / "mask-only.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=internal), rasterio.open(
        path, "w", driver="GTiff", count=1, dtype=raw.dtype, width=4, height=4,
        crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
    ) as ds:
        ds.write(raw, 1)
        ds.write_mask(valid)
    for method in ("nearest", "bilinear"):
        at_source = gis.resample_raster(path, raw.shape, method=method)["array"][0]
        expected = np.where(valid, 2.0, np.nan)
        np.testing.assert_array_equal(at_source, expected)
        aligned = gis.align_raster_to(path, gis.read_raster_info(path), resampling=method)
        reprojected = gis.reproject_raster(path, "EPSG:3857", resampling=method)
        for result in (aligned, reprojected):
            np.testing.assert_array_equal(result["array"][0], expected)
        # Fractional source coordinates exercise reconstruction around the missing cell.
        shifted = gis.resample_raster(path, (7, 7), method=method)["array"]
        reference = gis.resample_raster(np.where(valid, 2.0, np.nan), (7, 7), method=method)
        np.testing.assert_array_equal(shifted, reference["array"])


def test_native_tiled_mask_keeps_row_padding_and_edge_tiles_outside_raster(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    height, width = 19, 23
    raw = np.arange(height * width, dtype=np.int16).reshape(height, width)
    y, x = np.indices(raw.shape)
    valid = (x + 2 * y) % 7 != 0
    path = tmp_path / "tiled.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
        path, "w", driver="GTiff", count=1, dtype=raw.dtype, width=width, height=height,
        crs="EPSG:3857", transform=from_origin(0, height, 1, 1),
        tiled=True, blockxsize=16, blockysize=16,
    ) as ds:
        ds.write(raw, 1)
        ds.write_mask(valid)
    result = gis.resample_raster(path, raw.shape, method="nearest")["array"][0]
    np.testing.assert_array_equal(result, np.where(valid, raw, np.nan))


@pytest.mark.parametrize("internal", [True, False], ids=["internal", "sidecar"])
def test_file_terrain_analysis_is_independent_of_hidden_values(tmp_path, internal):
    import rasterio
    from rasterio.transform import from_origin
    from forge3d.raster_style import _single

    y, x = np.indices((5, 5))
    plane = 2.0 + x + 2 * y
    valid = np.ones(plane.shape, bool)
    valid[2, 2] = False
    analytic = np.float32(np.degrees(np.arctan(np.hypot(1.0, 2.0))))
    first_slope = first_hillshade = None
    for i, hidden in enumerate(MARKERS):
        path = tmp_path / f"plane-{i}.tif"
        stored = np.where(valid, plane, hidden)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=internal), rasterio.open(
            path, "w", driver="GTiff", width=5, height=5, count=1, dtype="float64",
            crs="EPSG:3857", transform=from_origin(10, 15, 1, 1),
        ) as ds:
            ds.write(stored, 1)
            ds.write_mask(valid)
        mask_result = gis.read_raster_mask(path)
        assert mask_result["mask_flags"] == ["per_dataset"]
        np.testing.assert_array_equal(mask_result["mask"][0], valid)
        decoded = gis.read_raster(path, masked=True)
        np.testing.assert_array_equal(decoded["array"][0], stored)
        np.testing.assert_array_equal(decoded["mask"][0], valid)
        np.testing.assert_array_equal(_single(path), np.where(valid, plane, np.nan))

        prepared = gis.prepare_dem(path)
        assert prepared["valid_count"] == 24
        for source in (path, gis.read_raster_info(path), decoded, prepared):
            derivatives = gis.prepare_terrain_derivatives(source)["derivatives"]
            slope, shade = derivatives["slope"], derivatives["hillshade"]
            np.testing.assert_array_equal(slope["mask"][0], valid)
            np.testing.assert_array_equal(shade["mask"][0], valid)
            np.testing.assert_array_equal(slope["array"][0], np.where(valid, analytic, np.nan))
            slope_bytes = slope["array"][0][valid].tobytes()
            shade_bytes = shade["array"][0][valid].tobytes()
            if first_slope is None:
                first_slope, first_hillshade = slope_bytes, shade_bytes
            assert (slope_bytes, shade_bytes) == (first_slope, first_hillshade)

        # A caller mask can remove another cell without making the file hole valid.
        caller = np.ones_like(valid)
        caller[0, 0] = False
        for source in (path, decoded):
            for result in (gis.normalize_raster(source, valid_mask=caller),
                           gis.classify_raster(source, bins=[5], valid_mask=caller)):
                assert result["valid_count"] == 23
                assert result["min"] == plane[valid & caller].min()
                assert result["max"] == plane[valid & caller].max()


@pytest.mark.parametrize("layout", ["shared", "pixel", "band", "partial"])
def test_public_file_masks_follow_band_and_window_selection(tmp_path, layout):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "bands.tif"
    raw = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
    raw[1, 2, 2] = 250
    explicit = np.ones(raw.shape, bool)
    explicit[:, 1, 2] = False
    profile = dict(driver="GTiff", width=5, height=4, count=3, dtype="uint8",
                   crs="EPSG:3857", transform=from_origin(10, 14, 1, 1), photometric="RGB")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(path, "w", nodata=250, **profile) as ds:
        ds.write(raw)
        if layout == "shared":
            ds.write_mask(explicit[0])
    if layout != "shared":
        for b in range(3):
            explicit[b, 2, b + 1] = False
        assigned = (2,) if layout == "partial" else (1, 2, 3)
        stored = explicit.astype(np.uint8) * 255
        if layout == "partial":
            stored[[0, 2]] = 0
            explicit[[0, 2]] = True
        with rasterio.open(str(path) + ".msk", "w", **profile,
                           interleave="pixel" if layout == "pixel" else "band",
                           compress="deflate") as ds:
            ds.write(stored)
            ds.update_tags(**{f"INTERNAL_MASK_FLAGS_{b}": "0" for b in assigned})
    with rasterio.open(path) as ds:
        np.testing.assert_array_equal(ds.read_masks() != 0, explicit)
    expected = explicit & (raw != 250)
    flags = ["per_band" if layout != "shared" else "per_dataset", "nodata"]
    for band in (1, 2, 3):
        result = gis.read_raster_mask(path, band)
        assert result["mask_flags"] == flags
        np.testing.assert_array_equal(result["mask"], expected[band-1:band])
    window = (1, 1, 3, 2)
    for masked in (True, False):
        read = gis.read_raster(path, bands=[3, 1, 2], window=window, masked=masked)
        assert read["array"].dtype == raw.dtype
        np.testing.assert_array_equal(read["array"], raw[[2, 0, 1], 1:3, 1:4])
        if masked:
            np.testing.assert_array_equal(read["mask"], expected[[2, 0, 1], 1:3, 1:4])
        else:
            assert read["mask"] is None
    padded = gis.read_raster_window(path, (-1, -1, 7, 6), boundless=True, masked=True)
    np.testing.assert_array_equal(padded["mask"], np.pad(expected, ((0, 0), (1, 1), (1, 1))))
