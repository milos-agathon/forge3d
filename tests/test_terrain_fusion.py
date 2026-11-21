"""IUNCTURA analytics, attribution, validity and preparation round trip."""
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.warp import transform as project
from rasterio.windows import Window, transform as window_transform

import forge3d as f3d
from forge3d.preparation import PreparationError, PreparationGrid, PreparationSource


def grid(w=12, h=12, dx=2, dy=3, x=100_000, y=440_000, crs="EPSG:28992"):
    return PreparationGrid(crs, w, h, (x, dx, 0, y, 0, -dy))


def raster(tmp_path, sid, values, geometry, *, valid=None, nodata=None,
           registration="Area", surface="bare_earth", reference="NAP", units="m"):
    path = tmp_path / f"{sid}.tif"
    values = np.asarray(values)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
        path, "w", driver="GTiff", width=geometry.width, height=geometry.height, count=1,
        dtype=values.dtype, crs=geometry.crs, transform=geometry.transform, nodata=nodata,
    ) as ds:
        ds.write(values, 1)
        if valid is not None:
            ds.write_mask(valid)
        ds.update_tags(AREA_OR_POINT=registration, surface_kind=surface, vertical_reference=reference)
        ds.set_band_unit(1, units)
    return PreparationSource(sid, path, "elevation", units=units, vertical_reference=reference)


def fuse(sources, target, **overrides):
    options = {
        "surface_kinds": {source.source_id: "bare_earth" for source in sources},
        "vertical_reference": "NAP", "transition_width_m": 5,
    }
    return f3d.fuse_terrain(sources, target, **{**options, **overrides})


def test_planes_on_different_grids_and_smoothstep_endpoints(tmp_path):
    target = grid()
    regional = grid(w=10, h=10, dx=4, dy=6, x=99_992, y=440_012)

    def plane(geometry):
        r, c = np.indices((geometry.height, geometry.width))
        x, y = geometry.xy(c, r)
        return 100 + (x - 100_000) / 4 + (y - 440_000) / 2

    fine_mask = np.zeros((12, 12), dtype=bool)
    fine_mask[2:10, 3:10] = True
    fine_values = plane(target) + 8
    fine = raster(tmp_path, "fine", fine_values, target, valid=fine_mask)
    coarse = raster(tmp_path, "coarse", plane(regional), regional)
    result = fuse((fine, coarse), target)

    # Distances at cols 3..6 are 1, 3, 5, 7 metres from the fine western
    # edge. At row 5 the other three edges are at least 7 metres away.
    expected_fine_weights = [0, 13 / 125, 81 / 125, 1, 1]
    u = np.finfo(np.float64).eps / 2
    np.testing.assert_allclose(result.weights[0, 5, 2:7], expected_fine_weights, atol=4 * u, rtol=0)
    np.testing.assert_allclose(result.height[5, 2:7],
                               plane(target)[5, 2:7] + 8 * np.asarray(expected_fine_weights),
                               rtol=8 * u, atol=0)
    np.testing.assert_array_equal(result.height[~fine_mask], plane(target)[~fine_mask])
    np.testing.assert_array_equal(result.height[result.weights[0] == 1], fine_values[result.weights[0] == 1])
    assert result.overlap_statistics["fine"]["median_offset_m"] == 8
    assert result.overlap_statistics["fine"]["median_absolute_deviation_m"] == 0
    # Independently, 1 - W(1-t) = W(t) = t²(3-2t), so at both ends
    # the secant slope is bounded by 3t and tends to zero.
    for t in (1 / 4, 1 / 16, 1 / 64):
        wider = fuse((fine, coarse), target, transition_width_m=1 / t)
        narrower = fuse((fine, coarse), target, transition_width_m=1 / (1 - t))
        assert wider.weights[0, 5, 3] / t <= 3 * t
        assert (1 - narrower.weights[0, 5, 3]) / t <= 3 * t


def test_blend_steps_report_total_separately_from_weight_change(tmp_path):
    target = grid(w=9, h=1, dy=12)
    terrain = 100 + np.arange(9, dtype=np.float64)[None, :] / 2
    fine_mask = np.zeros(terrain.shape, dtype=bool)
    fine_mask[:, 3:8] = True
    fine = raster(tmp_path, "fine", terrain + 8, target, valid=fine_mask)
    coarse = raster(tmp_path, "coarse", terrain, target)
    result = fuse((fine, coarse), target)

    # The step from the first to the second fine cell spans weights 13/125
    # and 81/125. Both planes rise by 0.5 m over that step.
    blend_bound = 8 * (81 - 13) / 125
    total_step = .5 + blend_bound
    steps = result.overlap_statistics["fine"]["blend_steps"]
    assert steps["pair_count"] == 6
    # Use a gamma bound for roundoff in weight evaluation, mixture, and step.
    u = np.finfo(np.float64).eps / 2
    allowance = (16 * u / (1 - 16 * u)) * (terrain.max() + 8)
    np.testing.assert_allclose([steps["max_total_step_m"], steps["max_blend_step_m"],
                                steps["max_blend_step_bound_m"]],
                               [total_step, blend_bound, blend_bound], atol=allowance, rtol=0)
    assert steps["max_total_step_m"] > steps["max_blend_step_bound_m"]
    np.testing.assert_array_equal(result.height[~fine_mask].view(np.uint64), terrain[~fine_mask].view(np.uint64))
    np.testing.assert_array_equal(result.height[:, 5].view(np.uint64), (terrain[:, 5] + 8).view(np.uint64))

    reopened = f3d.TerrainFusion.load(result.save(tmp_path / "saved"))
    assert reopened.overlap_statistics["fine"]["blend_steps"] == steps


def test_blend_step_reporting_accepts_finite_heights_with_a_wide_step(tmp_path):
    target = grid(w=2, h=1)
    maximum = np.finfo(np.float64).max
    terrain = np.array([[-maximum, maximum]])
    fine = raster(tmp_path, "fine", terrain, target, valid=np.array([[False, True]]))
    coarse = raster(tmp_path, "coarse", terrain, target)
    result = fuse((fine, coarse), target)

    np.testing.assert_array_equal(result.height, terrain)
    steps = result.overlap_statistics["fine"]["blend_steps"]
    assert steps == {"pair_count": 1, "max_total_step_m": 2 * int(maximum),
                     "max_blend_step_m": 0, "max_blend_step_bound_m": 0}
    reopened = f3d.TerrainFusion.load(result.save(tmp_path / "saved"))
    assert reopened.overlap_statistics["fine"]["blend_steps"] == steps


def test_blend_step_check_rejects_an_excess_after_accounting_for_terrain():
    from forge3d.terrain_fusion import _blend_step_statistics

    low = np.array([[100., 101.]])
    high = low + 8
    weight = np.array([[0., .5]])
    overlap = np.ones(low.shape, dtype=bool)
    # At unchanged weight the end height is 101. The changed weight allows
    # 4 m more. The extra metre here is a blend error, beyond terrain rise.
    fused = np.array([[100., 106.]])
    with pytest.raises(PreparationError, match="fusion_blend_step_bound"):
        _blend_step_statistics(high, low, fused, weight, overlap, "fine")


def footprint_distance(mask, row, col, dx, dy):
    """Oracle: enumerate each exposed edge and project onto its segment."""
    x, y = (col + 0.5) * dx, (row + 0.5) * dy
    distances = []
    for r, c in zip(*np.where(mask)):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                continue
            if dr:
                point = (min(max(x, c * dx), (c + 1) * dx), (r + (dr == 1)) * dy)
            else:
                point = ((c + (dc == 1)) * dx, min(max(y, r * dy), (r + 1) * dy))
            distances.append(math.dist((x, y), point))
    return min(distances)


def test_holes_and_priority_composition_match_segment_oracle(tmp_path):
    target = grid(w=6, h=5)
    masks = np.array([
        [[0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 0],
         [1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]],
        [[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]],
    ], dtype=bool)
    levels = np.array([50, 30, 10], dtype=np.float64)
    sources = tuple(raster(tmp_path, str(i), np.full(mask.shape, value), target, valid=mask)
                    for i, (mask, value) in enumerate(zip(masks, levels)))
    result = fuse(sources, target, transition_width_m=3)
    expected = np.zeros_like(result.weights)
    for r, c in np.ndindex(result.height.shape):
        sources_at_cell = np.flatnonzero(masks[:, r, c])
        remaining = 1.0
        for i in sources_at_cell[:-1]:
            x = min(footprint_distance(masks[i], r, c, 2, 3) / 3, 1)
            contribution = x**3 + 3 * x**2 * (1 - x)
            expected[i, r, c] = remaining * contribution
            remaining *= 1 - contribution
        if sources_at_cell.size:
            expected[sources_at_cell[-1], r, c] = remaining

    u = np.finfo(np.float64).eps / 2
    np.testing.assert_allclose(result.weights, expected, atol=12 * u, rtol=0)
    valid = np.any(masks, axis=0)
    np.testing.assert_array_equal(result.valid_mask, valid)
    np.testing.assert_allclose(result.weights.sum(axis=0), valid.astype(float), atol=9 * u, rtol=0)
    expected_height = np.where(valid, np.sum(expected * levels[:, None, None], axis=0), np.nan)
    np.testing.assert_allclose(result.height, expected_height, rtol=16 * u, atol=0)
    assert result.dominant_source_id[2, 2] != "0"
    assert result.dominant_source_id[4, 0] is None
    assert result.dominant_source_id[1, 0] == "2"
    # Their validity is identical at this overlap edge, but changing the
    # priority changes both the blend and the first returned weight plane.
    reordered = fuse(sources[::-1], target, transition_width_m=3)
    assert reordered.source_ids == ("2", "1", "0")
    assert reordered.height[0, 1] != result.height[0, 1]


def test_projected_feet_and_all_valid_edges(tmp_path):
    target = grid(w=4, h=4, dx=10, dy=20, x=1_000_000, y=200_000, crs="EPSG:2263")
    fine = raster(tmp_path, "fine", np.full((4, 4), 5, dtype=np.float64), target)
    coarse = raster(tmp_path, "coarse", np.ones((4, 4)), target)
    result = fuse((fine, coarse), target, transition_width_m=10 * 1200 / 3937)
    # The EPSG conversion and the independently evaluated rational differ
    # by a rounding step before smoothstep. Allow roundoff in the conversion.
    u = np.finfo(np.float64).eps / 2
    np.testing.assert_allclose(result.weights[:, 1, 0], (0.5, 0.5), atol=4 * u, rtol=0)
    assert result.dominant_source_id[1, 0] == "fine"
    assert math.isclose(result.height[1, 0], 3, rel_tol=4 * u)
    assert result.height[1, 1] == 5


def test_cropped_target_keeps_transition_context(tmp_path):
    whole = grid(w=16, h=16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:14, 2:14] = True
    fine = raster(tmp_path, "fine", np.full(mask.shape, 20.0), whole, valid=mask)
    low = raster(tmp_path, "coarse", np.full(mask.shape, 10.0), whole)
    reference = fuse((fine, low), whole)
    for start in (3, 6):
        cropped = grid(w=4, h=4, x=100_000 + 2 * start, y=440_000 - 3 * start)
        actual = fuse((fine, low), cropped)
        region = (slice(start, start + 4), slice(start, start + 4))
        # At start=3 the closest high-source edge lies outside the target.
        # At start=6 all requested cells are farther than the transition width.
        np.testing.assert_array_equal(actual.height, reference.height[region])
        np.testing.assert_array_equal(actual.weights, reference.weights[(slice(None), *region)])


def cross_crs_sources(tmp_path, target, resampling):
    x, dx, _, y, _, negative_dy = target.geotransform
    dy = -negative_dy
    projected_x, projected_y = project(target.crs, "EPSG:3857", [x + 5 * dx], [y - 4 * dy])
    source_grid = grid(w=60, h=52, dx=1.46 * dx / 2, dy=2.18 * dy / 3,
                       x=projected_x[0] - .27 * dx / 2, y=projected_y[0] + .41 * dy / 3, crs="EPSG:3857")
    row, col = np.indices((source_grid.height, source_grid.width))
    values = 100 + .125 * col + .25 * row + .125 * ((col + 2 * row) % 4)
    mask = np.ones(values.shape, dtype=bool)
    mask[20:26, 43:48] = False
    mask[:7, :7] = False
    fine = replace(raster(tmp_path, "fine", values, source_grid, valid=mask), resampling=resampling)
    target_row, target_col = np.indices((target.height, target.width))
    coarse = raster(tmp_path, "coarse", 50.0 + target_row * .25 + target_col * .5, target)
    return fine, coarse


@pytest.mark.parametrize("resampling", [None, "cubic"])
@pytest.mark.parametrize(("x", "y", "dx", "dy"), [
    (2_559_982, 1_140_000, 2, 3),
    (2_560_000, 1_140_000, .2, .3),
    (2_559_982.05, 1_140_000.05, .2, .3),
], ids=["crosses-block-boundary", "decimal-spacing", "shifted-decimal-origin"])
def test_cross_crs_crop_matches_whole_grid(tmp_path, resampling, x, y, dx, dy):
    whole = grid(w=44, h=36, x=x, y=y, dx=dx, dy=dy, crs="EPSG:2056")
    fine, coarse = cross_crs_sources(tmp_path, whole, resampling)
    band = 4.5 * dx
    unblended = fuse((fine,), whole, transition_width_m=band)
    blended = fuse((fine, coarse), whole, transition_width_m=band)
    standalone_source = replace(fine, resampling=f"fusion_{resampling or 'bilinear'}")
    prepared = f3d.prepare_scene(f3d.PreparationRecipe(whole, (standalone_source,), "NAP")).layers[fine.source_id]
    np.testing.assert_array_equal(prepared.valid_mask, unblended.valid_mask)
    np.testing.assert_array_equal(prepared.data.view(np.uint64), unblended.height.view(np.uint64))
    selected = blended.weights[0] == 1
    np.testing.assert_array_equal(blended.height[selected].view(np.uint64),
                                  prepared.data[selected].view(np.uint64))
    for left, top in ((6, 8), (16, 13)):
        window = Window(left, top, 5, 4)
        cropped = PreparationGrid(whole.crs, 5, 4, window_transform(window, whole.transform).to_gdal())
        region = window.toslices()
        for sources, reference in (((fine,), unblended), ((fine, coarse), blended)):
            actual = fuse(sources, cropped, transition_width_m=band)
            np.testing.assert_array_equal(actual.height.view(np.uint64), reference.height[region].view(np.uint64))
            np.testing.assert_array_equal(actual.valid_mask, reference.valid_mask[region])
            np.testing.assert_array_equal(actual.weights, reference.weights[(slice(None), *region)])
        cropped_preparation = f3d.prepare_scene(
            f3d.PreparationRecipe(cropped, (standalone_source,), "NAP"),
        ).layers[fine.source_id]
        np.testing.assert_array_equal(cropped_preparation.data.view(np.uint64),
                                      prepared.data[region].view(np.uint64))
        np.testing.assert_array_equal(cropped_preparation.valid_mask, prepared.valid_mask[region])


@pytest.mark.parametrize("resampling", ["bilinear", "cubic"])
@pytest.mark.parametrize("projected", [False, True], ids=["same-crs-downsample", "projected-origin"])
def test_crop_uses_the_parent_resampling_context(tmp_path, resampling, projected):
    fine_grid = grid(w=64, h=64, x=2_560_950, y=1_141_000, dx=.5, dy=.5, crs="EPSG:2056")
    row, col = np.indices((64, 64))
    terrain = 500 + .125 * (row + 2 * col) + ((row + 3 * col) % 5)
    fine = replace(raster(tmp_path, "fine", terrain, fine_grid), resampling=resampling)
    if projected:
        from pyproj import Transformer

        anchor = fine_grid.transform * (56, 8)
        x, y = Transformer.from_crs(fine_grid.crs, "EPSG:3857", always_xy=True).transform(*anchor)
        whole = grid(w=44, h=36, x=x, y=y, dx=.2, dy=.3, crs="EPSG:3857")
        window = Window(6, 8, 5, 4)
        width_m = .9
    else:
        x, y = fine_grid.transform * (1, 1)
        whole = grid(w=44, h=36, x=x, y=y, dx=2, dy=3, crs=fine_grid.crs)
        window = Window(6, 4, 5, 4)
        width_m = 3
    coarse = raster(tmp_path, "coarse", np.full((whole.height, whole.width), 400.), whole)
    cropped = PreparationGrid(whole.crs, 5, 4, window_transform(window, whole.transform).to_gdal())
    region = window.toslices()
    full = fuse((fine, coarse), whole, transition_width_m=width_m)
    crop = fuse((fine, coarse), cropped, transition_width_m=width_m)
    np.testing.assert_array_equal(crop.weights, full.weights[(slice(None), *region)])
    np.testing.assert_array_equal(crop.weights[0], 1)
    np.testing.assert_array_equal(crop.valid_mask, full.valid_mask[region])
    np.testing.assert_array_equal(crop.height.view(np.uint64), full.height[region].view(np.uint64))


def test_saved_grid_window_reuses_parent_origin_and_integer_offsets(tmp_path):
    whole = grid(w=44, h=36, x=2_560_000.000000005, y=1_140_000.050000005, dx=.2, dy=.3, crs="EPSG:2056")
    fine, coarse = cross_crs_sources(tmp_path, whole, "bilinear")
    crop = whole.window(2, 3, 25, 20).window(4, 5, 5, 4)
    assert crop == whole.window(6, 8, 5, 4)
    with pytest.raises(PreparationError, match="invalid_grid_lattice"):
        replace(crop, sampling_lattice=(whole.geotransform[0], whole.geotransform[3], 7, 8))
    rows, cols = np.indices((4, 5))
    np.testing.assert_array_equal(crop.xy(cols, rows), whole.xy(cols + 6, rows + 8))
    fused = fuse((fine, coarse), whole, transition_width_m=.9)
    window_fusion = fuse((fine, coarse), crop, transition_width_m=.9)
    reopened = f3d.TerrainFusion.load(window_fusion.save(tmp_path / "fused"))
    region = (slice(8, 12), slice(6, 11))
    assert reopened.grid == crop
    np.testing.assert_array_equal(reopened.height.view(np.uint64), fused.height[region].view(np.uint64))
    np.testing.assert_array_equal(reopened.weights, fused.weights[(slice(None), *region)])

    standalone = replace(fine, resampling="fusion_bilinear")
    recipe_path = tmp_path / "window-recipe.json"
    f3d.PreparationRecipe(crop, (standalone,), "NAP").save(recipe_path)
    recipe = f3d.PreparationRecipe.load(recipe_path)
    assert recipe.target_grid == crop
    prepared = f3d.prepare_scene(recipe).layers[fine.source_id]
    full_source = fuse((fine,), whole, transition_width_m=.9)
    np.testing.assert_array_equal(prepared.data.view(np.uint64), full_source.height[region].view(np.uint64))


@pytest.mark.parametrize("resampling", ["bilinear", "cubic"])
@pytest.mark.parametrize("inside", [True, False], ids=["inside", "outside"])
def test_interpolated_grid_keeps_centres_on_the_declared_side_of_source_edge(tmp_path, resampling, inside):
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    # Start on a GDAL block boundary so the unmodified parent affine also
    # defines the independent reference block.
    x, y = 2_560_512.5, 1_140_992.0
    east = x + 2.5
    fine_grid = grid(w=64, h=64, x=east - 32, y=y + 4, dx=.5, dy=.5, crs="EPSG:2056")
    rows, cols = np.indices((fine_grid.height, fine_grid.width))
    terrain = 500 + .0625 * (rows**2 + 2 * cols) + ((rows + cols) % 5)
    fine = replace(raster(tmp_path, "fine", terrain, fine_grid), resampling=resampling)
    x = np.nextafter(x, -np.inf if inside else np.inf)
    parent = grid(w=5, h=4, x=x, y=y, dx=1, dy=1, crs=fine_grid.crs)
    target = parent.window(1, 0, 4, 4)
    coarse = raster(tmp_path, "regional", np.full((4, 5), 400.), parent)
    assert target.xy(1, 1)[0] == np.nextafter(east, -np.inf if inside else np.inf)

    with rasterio.open(fine.path) as ds, WarpedVRT(
        ds, crs=parent.crs, transform=parent.transform, width=512, height=128,
        resampling=Resampling[resampling], nodata=np.nan, XSCALE=.5, YSCALE=.5,
    ) as reference:
        expected = reference.read(1, window=Window(2, 1, 1, 1))[0, 0]
    assert np.isfinite(expected) == inside

    full = fuse((fine, coarse), parent, transition_width_m=.5)
    crop = fuse((fine, coarse), target, transition_width_m=.5)
    standalone_source = replace(fine, resampling=f"fusion_{resampling}")
    standalone = f3d.prepare_scene(
        f3d.PreparationRecipe(target, (standalone_source,), "NAP"),
    ).layers[fine.source_id]
    assert standalone.valid_mask[1, 1] == inside
    assert crop.source_index[1, 1] == (0 if inside else 1)
    np.testing.assert_array_equal(crop.weights[:, 1, 1], [1, 0] if inside else [0, 1])
    if inside:
        assert standalone.data[1, 1] == expected
        assert crop.height[1, 1] == expected
    else:
        assert np.isnan(standalone.data[1, 1])
        assert crop.height[1, 1] == 400
    np.testing.assert_array_equal(crop.height.view(np.uint64), full.height[:, 1:].view(np.uint64))
    np.testing.assert_array_equal(crop.valid_mask, full.valid_mask[:, 1:])
    np.testing.assert_array_equal(crop.weights, full.weights[:, :, 1:])


@pytest.mark.parametrize("resampling", ["bilinear", "cubic"])
@pytest.mark.parametrize(("edge", "parent_row", "inside"), [
    ("north", 2, False),
    ("north", 2, True),
    ("north", 0, False),
    ("south", 0, True),
    ("south", -4, True),
], ids=[
    "north-reported-outside", "north-reported-inside", "north-block-rounding",
    "south-block-rounding", "south-negative-parent-row",
])
def test_fractional_grid_preserves_coverage_at_north_and_south_edges(tmp_path, resampling, edge, parent_row, inside):
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    boundary = 1_141_000.0
    north = boundary if edge == "north" else boundary + 32
    fine_grid = grid(w=64, h=64, x=2_560_968, y=north, dx=.5, dy=.5, crs="EPSG:2056")
    fine = replace(raster(tmp_path, "fine", np.full((64, 64), 512.), fine_grid), resampling=resampling)
    dx, dy = .2, .3
    outside = np.inf if edge == "north" else -np.inf
    direction = -outside if inside else outside
    y = np.nextafter(boundary + (parent_row + .5) * dy, direction)
    parent = grid(w=5, h=5, x=2_560_994.5, y=y, dx=dx, dy=dy, crs=fine_grid.crs)
    top = min(parent_row, 1)
    target = parent.window(1, top, 4, 5 - top)
    row = parent_row - top
    assert target.xy(1, row)[1] == np.nextafter(boundary, direction)
    full_top = min(top, 0)
    full_grid = parent.window(0, full_top, 5, 5 - full_top) if full_top < 0 else parent
    regional = raster(tmp_path, "regional", np.full((full_grid.height, 5), 400.), full_grid)

    # Refer to the declared parent affine directly, with the same physical kernel scale.
    transform, reference_row = parent.transform, parent_row
    if parent_row < 0:
        transform = transform * Affine.scale(1, -1)
        reference_row = -parent_row - 1
    with rasterio.open(fine.path) as ds, WarpedVRT(
        ds, crs=parent.crs, transform=transform, width=512, height=128,
        resampling=Resampling[resampling], nodata=np.nan, XSCALE=.5 / dx, YSCALE=.5 / dy,
    ) as reference:
        expected = reference.read(1, window=Window(2, reference_row, 1, 1))[0, 0]
    assert np.isfinite(expected) == inside

    full = fuse((fine, regional), full_grid, transition_width_m=dx / 2)
    result = fuse((fine, regional), target, transition_width_m=dx / 2)
    standalone_source = replace(fine, resampling=f"fusion_{resampling}")
    layer = f3d.prepare_scene(
        f3d.PreparationRecipe(target, (standalone_source,), "NAP"),
    ).layers[fine.source_id]
    assert layer.valid_mask[row, 1] == inside
    assert result.source_index[row, 1] == (0 if inside else 1)
    np.testing.assert_array_equal(result.weights[:, row, 1], [1, 0] if inside else [0, 1])
    assert result.height[row, 1] == (expected if inside else 400.)
    if inside:
        assert layer.data[row, 1] == expected
    np.testing.assert_array_equal(result.height.view(np.uint64), full.height[top - full_top:, 1:].view(np.uint64))
    np.testing.assert_array_equal(result.weights, full.weights[:, top - full_top:, 1:])


def test_aligned_window_copies_source_bits_and_masks_at_same_resolution(tmp_path):
    source_grid = grid(w=2, h=2, dx=.2, dy=.3)
    values = np.array([[-0.0, 10.0], [100.0, 200.0]])
    valid = np.array([[True, False], [True, True]])
    fine = raster(tmp_path, "fine", values, source_grid, valid=valid)
    target = source_grid.window(-1, 0, 4, 2)
    result = fuse((fine,), target)
    expected = np.pad(np.where(valid, values, np.nan), ((0, 0), (1, 1)), constant_values=np.nan)
    np.testing.assert_array_equal(result.height.view(np.uint64), expected.view(np.uint64))
    crop = fuse((fine,), target.window(1, 0, 2, 2))
    np.testing.assert_array_equal(crop.height.view(np.uint64), result.height[:, 1:3].view(np.uint64))
    np.testing.assert_array_equal(crop.valid_mask, valid)


def test_cross_crs_nearest_matches_raw_source_and_preparation(tmp_path):
    target = grid(w=44, h=36, x=2_560_000, y=1_140_000, crs="EPSG:2056")
    fine, coarse = cross_crs_sources(tmp_path, target, "nearest")
    row, col = np.indices((target.height, target.width))
    x, y = target.xy(col, row)
    with rasterio.open(fine.path) as dataset:
        sx, sy = project(target.crs, dataset.crs, x.ravel(), y.ravel())
        affine = dataset.transform
        source_col = np.floor((np.array(sx).reshape(x.shape) - affine.c) / affine.a).astype(int)
        source_row = np.floor((np.array(sy).reshape(y.shape) - affine.f) / affine.e).astype(int)
        in_bounds = ((0 <= source_row) & (source_row < dataset.height)
                     & (0 <= source_col) & (source_col < dataset.width))
        ids = (source_row[in_bounds], source_col[in_bounds])
        expected_valid = np.zeros(x.shape, dtype=bool)
        expected_valid[in_bounds] = dataset.read_masks(1)[ids] != 0
        expected = np.full(x.shape, np.nan)
        expected[in_bounds] = dataset.read(1)[ids]

    single = fuse((fine,), target, transition_width_m=9)
    np.testing.assert_array_equal(single.valid_mask, expected_valid)
    np.testing.assert_array_equal(single.height[expected_valid].view(np.uint64),
                                  expected[expected_valid].view(np.uint64))
    result = fuse((fine, coarse), target, transition_width_m=9)
    selected = result.weights[0] == 1
    assert selected[9, 25] and expected[9, 25] == 107.5
    prepared = f3d.prepare_scene(f3d.PreparationRecipe(target, (fine,), "NAP")).layers[fine.source_id]
    assert np.all(prepared.valid_mask[selected])
    np.testing.assert_array_equal(result.height[selected].view(np.uint64), expected[selected].view(np.uint64))
    np.testing.assert_array_equal(result.height[selected].view(np.uint64), prepared.data[selected].view(np.uint64))
    recipe = f3d.PreparationRecipe(target, (replace(fine, resampling="fusion_nearest"),), "NAP")
    recipe_path = tmp_path / "recipe.json"
    recipe.save(recipe_path)
    reopened = f3d.prepare_scene(f3d.PreparationRecipe.load(recipe_path)).layers[fine.source_id]
    np.testing.assert_array_equal(reopened.valid_mask, expected_valid)
    np.testing.assert_array_equal(reopened.data.view(np.uint64), single.height.view(np.uint64))
    assert reopened.metadata["operations"][1]["resampling"] == "fusion_nearest"


def test_cross_crs_nearest_keeps_valid_cells_at_projection_boundary(tmp_path):
    orthographic = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
    source = grid(w=10, h=10, dx=1_000_000, dy=1_000_000, x=-5_000_000, y=5_000_000, crs=orthographic)
    target = grid(w=4, h=1, dx=4_000_000, dy=1_000_000, x=0, y=500_000, crs="EPSG:3857")
    # Only the first centre is inside the source grid. The last two also lie
    # beyond the orthographic projection's visible hemisphere.
    fine = replace(raster(tmp_path, "fine", np.full((10, 10), 103.0), source), resampling="nearest")
    result = fuse((fine,), target)
    np.testing.assert_array_equal(result.height, [[103.0, np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(result.valid_mask, [[True, False, False, False]])


@pytest.mark.parametrize("resampling", ["bilinear", "cubic"])
def test_cross_crs_interpolation_retains_eligible_cells_beside_mask_hole(tmp_path, resampling):
    target = grid(w=44, h=36, x=2_560_000, y=1_140_000, crs="EPSG:2056")
    fine, _ = cross_crs_sources(tmp_path, target, resampling)
    regional = raster(tmp_path, "regional", np.full((36, 44), 50.), target)
    with rasterio.open(fine.path) as ds:
        declared = np.ones(ds.shape, dtype=bool)
        declared[35:38, 36:40] = False
        source_valid = (ds.read_masks(1) != 0) & declared
        rr, cc = np.indices((target.height, target.width))
        x, y = target.xy(cc, rr)
        sx, sy = project(target.crs, ds.crs, x.ravel(), y.ravel())
        affine = ds.transform
        rows = np.floor((np.asarray(sy).reshape(y.shape) - affine.f) / affine.e).astype(int)
        cols = np.floor((np.asarray(sx).reshape(x.shape) - affine.c) / affine.a).astype(int)
        covered = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        eligible = np.zeros_like(covered)
        eligible[covered] = source_valid[rows[covered], cols[covered]]
    assert eligible.sum() == 760
    assert (rows[21, 23], cols[21, 23]) == (34, 37) and eligible[21, 23]
    assert (rows[22, 23], cols[22, 23]) == (36, 37) and not eligible[22, 23]

    mask = tmp_path / "hole.npy"
    np.save(mask, declared, allow_pickle=False)
    fine = replace(fine, mask_path=mask)
    full = fuse((fine, regional), target, transition_width_m=1)
    standalone = replace(fine, resampling=f"fusion_{resampling}")
    layer = f3d.prepare_scene(f3d.PreparationRecipe(target, (standalone,), "NAP")).layers[fine.source_id]
    np.testing.assert_array_equal(layer.valid_mask, eligible)
    np.testing.assert_array_equal(full.weights[0], eligible)
    np.testing.assert_array_equal(full.source_index, np.where(eligible, 0, 1))
    np.testing.assert_array_equal(full.height[eligible].view(np.uint64), layer.data[eligible].view(np.uint64))
    np.testing.assert_array_equal(full.height[~eligible], 50.)
    # The independent GDAL control gives 113.17962 m, distinct from the raw 113.25 m.
    assert round(float(full.height[21, 23]), 5) == 113.17962

    crop = fuse((fine, regional), target.window(21, 19, 5, 5), transition_width_m=1)
    region = np.s_[19:24, 21:26]
    np.testing.assert_array_equal(crop.height.view(np.uint64), full.height[region].view(np.uint64))
    np.testing.assert_array_equal(crop.weights, full.weights[(slice(None), *region)])


def test_unresolved_interpolation_reports_the_source_and_sample(tmp_path, monkeypatch):
    from forge3d import terrain_fusion

    source = grid(w=2, h=2)
    fine = raster(tmp_path, "fine", np.full((2, 2), 110.), source)
    target = grid(w=2, h=2, dx=1, dy=1)
    regional = raster(tmp_path, "regional", np.full((2, 2), 50.), target)
    attempts = []

    def read_nodata(dataset, transform, width, height, window, *, exact, warp_options):
        attempts.append(exact)
        return np.full((window.height, window.width), np.nan)

    monkeypatch.setattr(terrain_fusion, "_read_warp_window", read_nodata)
    with pytest.raises(PreparationError, match="parent row 0, column 0") as error:
        fuse((fine, regional), target, transition_width_m=.5)
    assert error.value.diagnostic.code == "fusion_interpolation_failed"
    assert error.value.diagnostic.layer_id == fine.source_id
    assert attempts == [False, True]


def test_cross_crs_nearest_respects_the_source_cell_edge(tmp_path):
    source = grid(w=2, h=1, dx=1, dy=1, x=-1, y=.5, crs="EPSG:4326")
    target = grid(w=2, h=1, dx=1, dy=1, x=-.500001, y=.5, crs="EPSG:3857")
    fine = replace(raster(tmp_path, "fine", np.array([[-0.0, 44.0]]), source), resampling="nearest")
    # The first centre is just west of the boundary at 0 degrees. It still
    # belongs to column zero, even below GDAL's nearest-index rounding offset.
    x, y = target.xy(0, 0)
    lon, _ = project(target.crs, source.crs, [x], [y])
    assert -1 < lon[0] < 0
    np.testing.assert_array_equal(fuse((fine,), target).height.view(np.uint64),
                                  np.array([[-0.0, 44.0]]).view(np.uint64))
    single_cell = replace(target, width=1)
    assert np.signbit(fuse((fine,), single_cell).height[0, 0])


def test_rotated_source_and_height_units_share_target_sample_positions(tmp_path):
    target = grid(w=4, h=3)
    rotated = PreparationGrid(target.crs, 3, 4, (100_000, 0, 2, 439_991, 3, 0))
    in_feet = np.arange(12, dtype=np.float64).reshape(3, 4) + 100
    fine = raster(tmp_path, "fine", in_feet[::-1].T, rotated, units="ft")
    low = raster(tmp_path, "coarse", np.ones((3, 4)), target)
    result = fuse((replace(fine, resampling="nearest"), low), target, transition_width_m=1)
    np.testing.assert_array_equal(result.height, in_feet * 0.3048)
    np.testing.assert_array_equal(result.weights[0], 1)
    assert result.record["sources"][0]["operations"][2]["resampling"] == "nearest"


def test_calibration_is_explicit_and_uses_only_valid_selection(tmp_path):
    target = grid(w=4, h=4)
    low_mask = np.ones((4, 4), dtype=bool)
    low_mask[0, 0] = False
    low = raster(tmp_path, "coarse", np.full((4, 4), 100.0), target, valid=low_mask)
    fine_values = np.full((4, 4), 112.0)
    fine_values[3, 3] = 130
    high = raster(tmp_path, "fine", fine_values, target)
    calibration = np.zeros((4, 4), dtype=bool)
    calibration[0, :3] = True

    uncorrected = fuse((high, low), target)
    calibrated = fuse((high, low), target, calibration_masks={"fine": calibration})
    assert uncorrected.height[0, 0] == 112
    np.testing.assert_array_equal(calibrated.height[:3], 100)
    stats = calibrated.overlap_statistics["fine"]
    assert stats["sample_count"] == 15
    assert stats["calibration"]["sample_count"] == 2
    assert stats["calibration"]["median_offset_m"] == 12
    assert stats["calibration"]["median_absolute_deviation_m"] == 0
    assert stats["applied_offset_m"] == -12
    assert stats["calibration"]["mask_sha256"] == hashlib.sha256(calibration.tobytes()).hexdigest()
    assert uncorrected.overlap_statistics["fine"]["applied_offset_m"] == 0
    assert calibrated.height[3, 3] > 100
    with pytest.raises(PreparationError, match="empty_calibration_overlap"):
        fuse((high, low), target, calibration_masks={"fine": ~low_mask})
    with pytest.raises(PreparationError, match="invalid_calibration_mask"):
        fuse((high, low), target, calibration_masks={"fine": low_mask.astype(int)})
    with pytest.raises(PreparationError, match="invalid_calibration_mask"):
        fuse((high, low), target, calibration_masks={"coarse": low_mask})


def test_bias_estimation_filters_stable_low_slope_cells_without_applying(tmp_path):
    target = grid(w=7, h=7)
    low_values = np.full((7, 7), 100.0)
    high_values = low_values + 1.5
    high_values[4, 5] = 130
    stable = np.zeros((7, 7), dtype=bool)
    stable[2, 2] = stable[4, 4] = stable[5, 4] = True
    fine = raster(tmp_path, "fine", high_values, target)
    coarse = raster(tmp_path, "coarse", low_values, target)

    uncorrected = fuse((fine, coarse), target)
    estimated = fuse((fine, coarse), target, bias_estimation_masks={"fine": stable}, max_bias_slope=0)
    stats = estimated.overlap_statistics["fine"]
    estimate = stats["bias_estimate"]
    # The cell beside the outlier is steep. The other chosen cells and their
    # cardinal neighbours are flat in both DEMs.
    assert estimate["sample_count"] == 2
    assert estimate["median_offset_m"] == 1.5
    assert estimate["median_absolute_deviation_m"] == 0
    assert stats["applied_offset_m"] == 0
    assert estimated.height[2, 2] == 101.5
    np.testing.assert_array_equal(estimated.height, uncorrected.height)
    assert uncorrected.overlap_statistics["fine"]["bias_estimate"] is None
    reopened = f3d.TerrainFusion.load(estimated.save(tmp_path / "estimated"))
    assert reopened.overlap_statistics["fine"]["bias_estimate"] == estimate

    calibrated = fuse((fine, coarse), target, calibration_masks={"fine": stable}, max_bias_slope=0)
    assert calibrated.overlap_statistics["fine"]["bias_estimate"] == estimate
    assert calibrated.overlap_statistics["fine"]["applied_offset_m"] == -1.5
    assert calibrated.height[2, 2] == 100


def test_surface_type_override_preserves_mixed_surface_in_saved_terrain(tmp_path):
    target = grid(w=8, h=8)
    inset = np.zeros((8, 8), dtype=bool)
    inset[2:6, 2:6] = True
    surface = raster(tmp_path, "surface", np.full((8, 8), 120.0), target, valid=inset, surface="DSM")
    ground = raster(tmp_path, "ground", np.full((8, 8), 100.0), target)
    kinds = {"surface": "DSM", "ground": "bare_earth"}
    with pytest.raises(PreparationError, match="invalid_surface_kind"):
        fuse((surface, ground), target, surface_kinds=kinds)
    with pytest.raises(PreparationError, match="invalid_surface_kind"):
        fuse((surface, ground), target, allow_mixed_surfaces=True)

    result = fuse((surface, ground), target, surface_kinds=kinds, allow_mixed_surfaces=True, transition_width_m=2)
    assert result.record["surface_kind"] == "mixed"
    assert result.record["allow_mixed_surfaces"] is True
    assert [entry["surface_kind"] for entry in result.record["sources"]] == ["DSM", "bare_earth"]
    assert result.height[0, 0] == 100
    np.testing.assert_array_equal(result.weights[:, 3, 2], [.5, .5])
    assert result.source_index[3, 2] == 0
    assert result.height[3, 2] == 110
    assert result.height[3, 3] == 120
    saved = result.save(tmp_path / "mixed")
    with rasterio.open(saved / "terrain.tif") as ds:
        assert ds.tags()["surface_kind"] == "mixed"
    reopened = f3d.TerrainFusion.load(saved)
    assert reopened.record["surface_kind"] == "mixed"
    prepared = f3d.prepare_scene(f3d.PreparationRecipe.load(saved / "recipe.json"))
    assert prepared.layers["fused_terrain"].metadata["surface_kind"] == "mixed"

    standalone = replace(surface, resampling="fusion_nearest", metadata={"surface_kind": "DSM"})
    reference = f3d.prepare_scene(f3d.PreparationRecipe(target, (standalone,), "NAP")).layers["surface"]
    selected = result.weights[0] == 1
    np.testing.assert_array_equal(result.height[selected].view(np.uint64), reference.data[selected].view(np.uint64))
    assert fuse((surface,), target, surface_kinds={"surface": "DSM"}).record["surface_kind"] == "DSM"


@pytest.mark.parametrize("source_reference", ["unknown", "LN02"])
def test_unresolved_vertical_override_converts_units_and_preserves_label(tmp_path, source_reference):
    target = grid(w=4, h=4)
    surface = raster(tmp_path, "surface", np.full((4, 4), 330.0), target, reference=source_reference, units="ft")
    ground = raster(tmp_path, "ground", np.full((4, 4), 100.0), target)
    strict_error = "missing_vertical_reference" if source_reference == "unknown" else "unsupported_vertical_transform"
    with pytest.raises(PreparationError, match=strict_error):
        fuse((surface, ground), target)
    with pytest.raises(PreparationError, match="missing_vertical_reference"):
        fuse((surface, ground), target, vertical_reference="unresolved")
    with pytest.raises(PreparationError, match="invalid_vertical_reference_policy"):
        fuse((surface, ground), target, allow_unresolved_vertical_reference=True)
    with pytest.raises(PreparationError, match="conflicting_vertical_reference"):
        fuse((replace(surface, vertical_reference="ellipsoidal"), ground), target, vertical_reference="unresolved",
             allow_unresolved_vertical_reference=True)

    result = fuse((surface, ground), target, vertical_reference="unresolved",
                  allow_unresolved_vertical_reference=True, transition_width_m=.5)
    assert result.vertical_reference == "unresolved"
    assert result.record["vertical_reference_policy"] == "unresolved"
    assert result.record["sources"][0]["source"]["vertical_reference"] == source_reference
    assert result.overlap_statistics["surface"]["applied_offset_m"] == 0
    np.testing.assert_array_equal(result.height, 330 * .3048)
    saved = result.save(tmp_path / "unresolved")
    reopened = f3d.TerrainFusion.load(saved)
    assert reopened.vertical_reference == "unresolved"
    np.testing.assert_array_equal(reopened.height, result.height)
    prepared = f3d.prepare_scene(f3d.PreparationRecipe.load(saved / "recipe.json"))
    assert prepared.layers["fused_terrain"].metadata["vertical_reference"] == "unresolved"
    reference = f3d.prepare_scene(f3d.PreparationRecipe(
        target, (replace(surface, resampling="fusion_nearest"),), "unresolved",
    )).layers["surface"]
    np.testing.assert_array_equal(result.height.view(np.uint64), reference.data.view(np.uint64))


def test_registration_and_nodata_representation_are_invariant_after_warp(tmp_path):
    source_grid = grid(w=4, h=4, dx=4, dy=6)
    target = grid(w=8, h=8)
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    holes = (values == 0) | (values == 5)
    point = raster(tmp_path, "point", np.where(holes, -9999, values), source_grid,
                   nodata=-9999, registration="Point")
    area = raster(tmp_path, "area", np.where(holes, np.nan, values), source_grid,
                  registration="Area")
    with rasterio.open(point.path) as ds:
        assert ds.tags()["AREA_OR_POINT"] == "Point"
        assert ds.transform == source_grid.transform
        assert ds.xy(1, 2) == source_grid.xy(2, 1)
    bottom = raster(tmp_path, "coarse", np.full((8, 8), 30.0), target)
    first = fuse((point, bottom), target)
    second = fuse((area, bottom), target)
    np.testing.assert_array_equal(first.height, second.height)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.record["sources"][0]["reader_metadata"]["source_pixel_registration"] == "Point"
    assert first.weights[0, 2, 2] == 0


def test_file_nodata_excludes_raw_sample_even_if_sidecar_mask_allows_it(tmp_path):
    target = grid(w=5, h=5)
    values = np.full((5, 5), 103, dtype=np.int16)
    values[2, 2] = -9999
    fine = raster(tmp_path, "fine", values, target, nodata=-9999)
    sidecar_mask = np.ones(values.shape, dtype=bool)
    sidecar_mask[0, 0] = False
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False), rasterio.open(fine.path, "r+") as ds:
        ds.write_mask(sidecar_mask)
        ds.scales = (0.5,)
        ds.offsets = (50,)
    with rasterio.open(fine.path) as ds:
        assert ds.read_masks(1)[2, 2] == 255
    assert (tmp_path / "fine.tif.msk").exists()
    coarse = raster(tmp_path, "coarse", np.full(values.shape, 100.0), target)

    actual = fuse((fine, coarse), target, transition_width_m=.5)
    explicit = fuse((replace(fine, nodata=-9999), coarse), target, transition_width_m=.5)
    assert actual.height[2, 2] == 100
    np.testing.assert_array_equal(actual.weights[:, 2, 2], [0, 1])
    assert actual.dominant_source_id[2, 2] == "coarse"
    assert actual.height[0, 0] == 100
    assert actual.height[1, 1] == 101.5
    np.testing.assert_array_equal(actual.height, explicit.height)
    np.testing.assert_array_equal(actual.weights, explicit.weights)
    source = replace(fine, resampling="fusion_bilinear")
    standalone = f3d.prepare_scene(f3d.PreparationRecipe(target, (source,), "NAP")).layers[fine.source_id]
    assert not standalone.valid_mask[2, 2] and np.isnan(standalone.data[2, 2])
    assert standalone.valid_mask[1, 1] and standalone.data[1, 1] == 101.5


def test_save_reopen_and_prepare_existing_scene(tmp_path):
    target = grid(w=4, h=4)
    values = np.arange(16, dtype=np.float64).reshape(4, 4) + 100 + 2**-30
    path = tmp_path / "local.npy"
    mask = tmp_path / "bool.npy"
    valid = values != values[0, 0]
    np.save(path, values, allow_pickle=False)
    np.save(mask, valid, allow_pickle=False)
    source = PreparationSource("precise", path.name, "elevation", grid=target,
                               units="m", vertical_reference="NAP", mask_path=mask.name)
    result = fuse((source,), target, base_dir=tmp_path)
    identity = result.record["sources"][0]
    assert identity["identity"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["mask_identity"]["sha256"] == hashlib.sha256(mask.read_bytes()).hexdigest()
    output = result.save(tmp_path / "saved")
    assert identity["identity"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["mask_identity"]["sha256"] == hashlib.sha256(mask.read_bytes()).hexdigest()
    path.unlink()
    mask.unlink()
    restored = f3d.TerrainFusion.load(output)
    assert restored.grid.same_as(target)
    assert restored.vertical_reference == "NAP"
    assert restored.height.dtype == np.float64
    np.testing.assert_array_equal(restored.height[valid], values[valid])
    np.testing.assert_array_equal(restored.valid_mask, valid)
    np.testing.assert_array_equal(restored.weights, result.weights)
    expected_index = np.where(valid, 0, -1).astype(np.int64)
    np.testing.assert_array_equal(restored.source_index, expected_index)
    np.testing.assert_array_equal(np.load(output / "source_index.npy", allow_pickle=False), expected_index)
    np.testing.assert_array_equal(restored.dominant_source_id, result.dominant_source_id)
    prepared = f3d.prepare_scene(f3d.PreparationRecipe.load(output / "recipe.json"))
    dem = prepared.layers["fused_terrain"]
    np.testing.assert_array_equal(dem.data, result.height)
    np.testing.assert_array_equal(dem.valid_mask, valid)
    assert dem.metadata["vertical_reference"] == "NAP"
    assert dem.metadata["fusion_receipt_sha256"] == hashlib.sha256((output / "fusion.json").read_bytes()).hexdigest()
    prepared.save(tmp_path / "prepared")
    scene = prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(width=64, height=64),
                              terrain_void_fill=100)
    assert scene.recipe.terrain.metadata["preparation"]["vertical_reference"] == "NAP"
    scene.compile_plan()

    with pytest.raises(PreparationError, match="output_exists"):
        result.save(output)
    index_path = output / "source_index.npy"
    index_bytes = index_path.read_bytes()
    index_path.write_bytes(index_bytes + b"corruption")
    with pytest.raises(PreparationError, match="Checksum mismatch for source_index.npy"):
        f3d.TerrainFusion.load(output)
    index_path.write_bytes(index_bytes)
    with (output / "weights.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(PreparationError, match="Checksum mismatch for weights.npy"):
        f3d.TerrainFusion.load(output)


def test_reopen_rejects_receipt_with_changed_source_order(tmp_path):
    target = grid(w=4, h=4)
    fine = raster(tmp_path, "fine", np.full((4, 4), 20.0), target)
    coarse = raster(tmp_path, "coarse", np.full((4, 4), 10.0), target)
    output = fuse((fine, coarse), target, transition_width_m=.5).save(tmp_path / "saved")
    assert f3d.TerrainFusion.load(output).dominant_source_id[2, 2] == "fine"

    receipt_path = output / "fusion.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["source_ids"].reverse()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PreparationError, match=r"invalid_fusion_assets.*Checksum mismatch for fusion\.json"):
        f3d.TerrainFusion.load(output)


def test_all_invalid_and_disjoint_coverage(tmp_path):
    target = grid(w=4, h=4)
    missing = raster(tmp_path, "void", np.full((4, 4), -9999.0), target, nodata=-9999)
    disjoint = raster(tmp_path, "disjoint", np.ones((4, 4)), grid(w=4, h=4, x=100_100))
    covered = raster(tmp_path, "covered", np.ones((4, 4)), target)
    covered_result = fuse((missing, covered, disjoint), target)
    np.testing.assert_array_equal(covered_result.height, 1)
    np.testing.assert_array_equal(covered_result.source_index, 1)
    result = fuse((missing, disjoint), target)
    np.testing.assert_array_equal(result.source_index, -1)
    assert not result.valid_mask.any() and np.isnan(result.height).all()
    assert not result.weights.any()
    assert result.dominant_source_id.tolist() == [[None] * 4 for _ in range(4)]
    loaded = f3d.TerrainFusion.load(result.save(tmp_path / "void_only"))
    np.testing.assert_array_equal(loaded.valid_mask, result.valid_mask)
    np.testing.assert_array_equal(loaded.height, result.height)


def test_save_is_atomic_when_recipe_publication_fails(tmp_path, monkeypatch):
    target = grid(w=4, h=4)
    source = raster(tmp_path, "fine", np.ones((4, 4)), target)
    result = fuse((source,), target)
    destination = tmp_path / "saved"

    def fail(*args):
        raise OSError("simulated recipe write error")

    with monkeypatch.context() as patch:
        patch.setattr(f3d.PreparationRecipe, "save", fail)
        with pytest.raises(PreparationError, match="fusion_write_failed.*simulated recipe"):
            result.save(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".saved.*"))
    assert result.save(destination) == destination


@pytest.mark.parametrize("width", [0, -1, np.nan, np.inf])
def test_invalid_width_is_refused(tmp_path, width):
    source = PreparationSource("fine", tmp_path / "unread.npy", "elevation",
                               grid=grid(), units="m", vertical_reference="NAP")
    with pytest.raises(PreparationError, match="invalid_transition_width"):
        fuse((source,), grid(), transition_width_m=width)


def test_invalid_surface_datum_grid_and_source_data_are_refused(tmp_path):
    target = grid(w=4, h=4)
    fine = raster(tmp_path, "fine", np.ones((4, 4)), target)
    dsm = raster(tmp_path, "dsm", np.ones((4, 4)), target, surface="DSM")
    ln02 = raster(tmp_path, "ln02", np.ones((4, 4)), target, reference="LN02")
    with pytest.raises(PreparationError, match="invalid_fusion_sources"):
        fuse((fine, fine), target)
    with pytest.raises(PreparationError, match="invalid_surface_kind"):
        fuse((fine,), target, surface_kinds={"fine": "DSM"})
    with pytest.raises(PreparationError, match="invalid_surface_kind"):
        fuse((dsm, fine), target)
    with pytest.raises(PreparationError, match="missing_vertical_reference"):
        fuse((fine,), target, vertical_reference="unknown")
    with pytest.raises(PreparationError, match="unsupported_vertical_transform"):
        fuse((ln02, fine), target, calibration_masks={"ln02": np.ones((4, 4), dtype=bool)})
    with pytest.raises(PreparationError, match="unsupported_fusion_grid.*axis-aligned"):
        fuse((fine,), replace(target, geotransform=(100_000, 2, 1, 440_000, 1, -3)))
    with pytest.raises(PreparationError, match="unsupported_fusion_crs"):
        fuse((fine,), grid(crs="EPSG:4326", dx=0.1, dy=0.1, x=5, y=52))
    with pytest.raises(PreparationError, match="invalid_grid"):
        replace(target, geotransform=(100_000, 2, 2, 440_000, 2, 2))
    bad = tmp_path / "bad.npy"
    np.save(bad, np.ones((2, 4, 4)))
    with pytest.raises(PreparationError, match="invalid_band_count"):
        fuse((replace(fine, path=bad, grid=target),), target)
    np.save(bad, np.ones((4, 4)))
    with pytest.raises(PreparationError, match="invalid_mask"):
        fuse((replace(fine, mask_path=bad),), target)
    np.save(bad, np.full((4, 4), 2**53 + 1, dtype=np.int64))
    with pytest.raises(PreparationError, match="fusion_height_precision"):
        fuse((replace(fine, path=bad, grid=target),), target)
    encoded = raster(tmp_path, "encoded", np.full((4, 4), 2**53 + 1, dtype=np.int64), target)
    with rasterio.open(encoded.path, "r+") as dataset:
        dataset.scales = (0.5,)
    with pytest.raises(PreparationError, match="fusion_height_precision"):
        fuse((encoded,), target)


@pytest.mark.parametrize(("band", "tag", "value", "error"), [
    (1, "vertical_reference", "NAP", "conflicting_vertical_reference"),
    (0, "HEIGHT_SYSTEM", "NAP", "conflicting_vertical_reference"),
    (1, "surface_kind", "DSM", "invalid_surface_kind"),
    (0, "SURFACE_KIND", "DSM", "invalid_surface_kind"),
])
def test_fusion_checks_all_geotiff_declarations_before_calibration(tmp_path, band, tag, value, error):
    target = grid(w=4, h=4, x=2_560_000, y=1_140_000, crs="EPSG:2056")
    fine = raster(tmp_path, "fine", np.full((4, 4), 101.5), target, reference="LN02")
    coarse = raster(tmp_path, "coarse", np.full((4, 4), 100.0), target, reference="LN02")
    with rasterio.open(fine.path, "r+") as ds:
        ds.update_tags(band, **{tag: value})
    with pytest.raises(PreparationError, match=error):
        fuse((fine, coarse), target, vertical_reference="LN02",
             calibration_masks={"fine": np.ones((4, 4), dtype=bool)})
