"""Registration controls use world coordinates independent of the renderer."""
from dataclasses import replace

import numpy as np
import pytest

from forge3d.io import load_dem
from forge3d.preparation import PreparationGrid, PreparationRecipe, PreparationSource, prepare_scene


def write_tiff(path, data, grid):
    import rasterio

    bands = data[None, ...] if data.ndim == 2 else np.moveaxis(data, -1, 0)
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height,
                       count=len(bands), dtype=data.dtype, crs=grid.crs, transform=grid.transform) as dst:
        dst.write(bands)
        dst.update_tags(AREA_OR_POINT=grid.registration.title())


def terrain_fixture(registration):
    grid = PreparationGrid("EPSG:32631", 9, 9, (500000, 8, 0, 5200000, 0, -8), registration)
    ridge = np.zeros((9, 9), dtype=bool)
    ridge[:, [2, 6]] = True
    ridge[[2, 5], :] = True
    heights = ridge.astype(np.float32) * 2
    rgba = np.ones((9, 9, 4), dtype=np.float32)
    rgba[..., :3] = (0.0, 0.0, 0.0)
    rgba[ridge, :3] = (1.0, 0.0, 0.0)
    return grid, heights, rgba


def prepare_dem(path, target, *, conversion=None):
    return prepare_scene(PreparationRecipe(target, (
        PreparationSource("dem", path, "elevation", units="m",
                          vertical_reference="NAP", registration_conversion=conversion),
    ), "NAP"))


@pytest.mark.parametrize("registration", ["area", "point"])
def test_prepare_export_registration_round_trip(tmp_path, registration):
    import rasterio

    grid, heights, _ = terrain_fixture(registration)
    source = tmp_path / "dem.tif"
    write_tiff(source, heights, grid)
    dem = load_dem(source)
    assert dem.registration == registration
    assert dem.transform == grid.transform
    prepared = prepare_dem(source, grid)
    assert prepared.layers["dem"].grid == grid
    out = prepared.viewer_assets(tmp_path / "export")
    with rasterio.open(out["terrain_path"]) as dataset:
        assert dataset.tags()["AREA_OR_POINT"].lower() == registration
        assert dataset.transform == grid.transform
        assert dataset.crs == rasterio.crs.CRS.from_string(grid.crs)
        np.testing.assert_array_equal(dataset.read(1), heights)
    restored = load_dem(out["terrain_path"])
    assert (restored.registration, restored.transform, restored.crs) == (dem.registration, dem.transform, dem.crs)
    np.testing.assert_array_equal(restored.data, dem.data)


def test_mixed_registration_requires_explicit_conversion(tmp_path):
    from forge3d.alignment import resample_raster_to_grid
    from forge3d.preparation import PreparationError

    point, heights, _ = terrain_fixture("point")
    area = replace(point, registration="area")
    path = tmp_path / "point.tif"
    write_tiff(path, heights, point)
    with pytest.raises(PreparationError, match="raster registration mismatch"):
        prepare_dem(path, area)
    with pytest.raises(ValueError, match="raster registration mismatch"):
        resample_raster_to_grid(path, area.to_dict())
    converted = prepare_dem(path, area, conversion="preserve_sample_locations")
    layer = converted.layers["dem"]
    assert layer.grid == area
    assert layer.metadata["operations"][0]["native_origin_shift_pixels"] == [-0.5, -0.5]
    np.testing.assert_array_equal(layer.data, heights)


def test_misdeclared_point_tiepoint_shifts_by_half_a_cell(tmp_path):
    point, heights, _ = terrain_fixture("point")
    # A mislabeled file retains the native point tiepoint but encodes it as Area.
    gt = point.geotransform
    wrong = replace(point, registration="area",
                    geotransform=(gt[0] + 4, gt[1], gt[2], gt[3] - 4, gt[4], gt[5]))
    path = tmp_path / "misdeclared.tif"
    write_tiff(path, heights, wrong)
    dem = load_dem(path)
    for row, col in [(2, 4), (4, 6)]:
        found = dem.transform * (col + 0.5, row + 0.5)
        expected = point.xy(col, row)
        assert tuple(np.subtract(found, expected)) == (4.0, -4.0)

def ridge_residual(profile, centres, pitch, *, minimum=False):
    """Measure the peak centre in a control window halfway to its neighbours."""
    residuals = []
    positions = np.arange(len(profile), dtype=np.float64) + 0.5
    for centre in centres:
        region = np.flatnonzero(np.abs(positions - centre) < pitch / 2)
        candidates = profile[region]
        extreme = np.nanmin(candidates) if minimum else np.nanmax(candidates)
        assert np.isfinite(extreme) and np.nanmax(candidates) > np.nanmin(candidates)
        estimated = positions[region[candidates == extreme]].mean()
        residuals.append(float(estimated - centre))
    return residuals


def ridge_vector_mesh(grid):
    vertices = []
    triangles = []
    # One physical output pixel wide in the registered orthographic camera.
    for col in (2, 6):
        x, north = grid.xy(col, 1)
        _, south = grid.xy(col, 7)
        start = len(vertices)
        vertices.extend([(x - 0.5, north, 3), (x + 0.5, north, 3),
                         (x + 0.5, south, 3), (x - 0.5, south, 3)])
        triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
    for row in (2, 5):
        west, y = grid.xy(1, row)
        east, _ = grid.xy(7, row)
        start = len(vertices)
        vertices.extend([(west, y + 0.5, 3), (east, y + 0.5, 3),
                         (east, y - 0.5, 3), (west, y - 0.5, 3)])
        triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
    return np.array(vertices), np.array(triangles, dtype=np.uint32)


@pytest.mark.offscreen
@pytest.mark.parametrize("registration", ["area", "point"])
def test_traced_checkerboard_coincidence(tmp_path, registration):
    from PIL import Image
    from forge3d.path_tracing import render_registered_terrain

    grid, heights, rgba = terrain_fixture(registration)
    source = tmp_path / "dem.tif"
    write_tiff(source, heights, grid)
    dem = load_dem(source)
    # AOVs use unjittered primary rays. Zero incoming light gives equal
    # beauty samples, with two frames sufficient to define their variance.
    kwargs = dict(pixel_size=1, max_frames=2, min_frames=2, seed=7, albedo_sampling="bilinear",
                  sun_intensity=0.0, env_intensity=0.0)
    result = render_registered_terrain(dem.data, grid, 63, 63, drape=(rgba, grid), **kwargs)
    with_vectors = render_registered_terrain(
        dem.data, grid, 63, 63, drape=(rgba, grid), world_mesh=ridge_vector_mesh(grid), **kwargs,
    )
    Image.fromarray(result["rgba"]).save(tmp_path / "traced.png")
    for name, frame in (("drape", result), ("vectors", with_vectors)):
        color = np.rint(np.clip(frame["albedo"], 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(color).save(tmp_path / f"traced-{name}-albedo.png")

    output = result["grid"]
    assert output.registration == "area"
    from forge3d.export import render_to_geotiff
    import rasterio
    exported = tmp_path / "traced.tif"
    render_to_geotiff(result, exported, layer="albedo")
    with rasterio.open(exported) as dataset:
        assert dataset.transform == output.transform
        assert dataset.tags()["AREA_OR_POINT"] == "Area"
        assert dataset.crs.to_string() == grid.crs
        np.testing.assert_array_equal(np.moveaxis(dataset.read(), 0, -1), result["albedo"])
    centres = [(grid.xy(col, 4)[0] - output.geotransform[0]) for col in (2, 6)]
    middle = 31
    north = output.geotransform[3]
    row_centres = [north - grid.xy(4, j)[1] for j in (2, 5)]
    for axis, positions, depth, drape, vectors in (
        ("east", centres, result["depth"][middle], result["albedo"][middle, :, 0],
         with_vectors["albedo"][middle, :, 2]),
        ("south", row_centres, result["depth"][:, middle], result["albedo"][:, middle, 0],
         with_vectors["albedo"][:, middle, 2]),
    ):
        pitch = positions[1] - positions[0]
        measured = [ridge_residual(depth, positions, pitch, minimum=True),
                    ridge_residual(drape, positions, pitch),
                    ridge_residual(vectors, positions, pitch)]
        print(f"traced {registration} {axis} residual pixels (terrain, drape, vector) {measured}")
        assert np.max(np.abs(measured)) <= 1.0
        assert np.max(np.ptp(measured, axis=0)) <= 1.0

    if registration == "point":
        # Preserve the raw point tiepoint while declaring Area and holding the output grid fixed.
        wrong_transform = grid.transform * grid.transform.translation(0.5, 0.5)
        wrong_grid = replace(grid, geotransform=wrong_transform.to_gdal(), registration="area")
        wrong_path = tmp_path / "mislabeled.tif"
        write_tiff(wrong_path, heights, wrong_grid)
        wrong = load_dem(wrong_path)
        misregistered = render_registered_terrain(
            wrong.data, wrong_grid, 63, 63, centre=grid.xy(4, 4), **kwargs,
        )
        np.testing.assert_array_equal(misregistered["grid"].geotransform, output.geotransform)
        Image.fromarray(misregistered["rgba"]).save(tmp_path / "traced-mislabeled.png")
        shift = grid.geotransform[1] * 0.5
        shifted = ridge_residual(misregistered["depth"][middle], np.asarray(centres) + shift, 32, minimum=True)
        print(f"traced mislabeled point: measured eastward shifts {[shift + r for r in shifted]} pixels")
        assert shifted == [0.0, 0.0]
        shifted_y = ridge_residual(misregistered["depth"][:, middle], np.asarray(row_centres) + shift, 24, minimum=True)
        print(f"traced mislabeled point: measured southward shifts {[shift + r for r in shifted_y]} pixels")
        assert shifted_y == [0.0, 0.0]

@pytest.mark.offscreen
@pytest.mark.parametrize("registration", ["area", "point"])
def test_mapscene_checkerboard_coincidence(tmp_path, registration):
    import forge3d as f3d
    import rasterio
    from forge3d.map_scene import _render_native_offscreen_result

    grid, heights, rgba = terrain_fixture(registration)
    dem_path = tmp_path / "dem.tif"
    drape_path = tmp_path / "drape.tif"
    write_tiff(dem_path, heights, grid)
    write_tiff(drape_path, (rgba * 255).astype(np.uint8), grid)
    lines = [[grid.xy(col, 1), grid.xy(col, 7)] for col in (2, 6)]
    lines += [[grid.xy(1, row), grid.xy(7, row)] for row in (2, 5)]
    features = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": line},
                 "properties": {}} for line in lines]
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=dem_path, crs=grid.crs),
        layers=[
            f3d.RasterOverlay("drape", path=drape_path, crs=grid.crs),
            f3d.VectorOverlay("lines", features=features, crs=grid.crs,
                              style={"line": {"paint": {"line-color": "#0000ff"}}},
                              width_px=1.0, metadata={"coordinate_space": "world"}),
            f3d.LabelLayer(
                "labels", occlusion="none",
                labels=[{"id": "control", "kind": "point", "text": "P",
                         "geometry": {"type": "Point", "coordinates": grid.xy(1, 1)}}],
                glyph_atlas={"glyphs": ["P"]},
                metadata={"coordinate_space": "world"},
            ),
        ],
        lighting=f3d.LightingPreset(settings={"unlit_albedo": True}),
        output=f3d.OutputSpec(width=65, height=65, format="geotiff", aovs=("normal", "albedo")),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=7),
    )
    compiled = scene.compile_plan()
    assert not compiled.validation_report.render_blocked(scene.render_policy)
    label, = compiled.label_plans["labels"].accepted
    anchor = next(candidate.anchor for candidate in label.candidates if candidate.candidate_type == "center")
    np.testing.assert_allclose(anchor[:2], [65 / 8] * 2, rtol=0, atol=np.finfo(np.float32).eps * 65)
    result = _render_native_offscreen_result(scene.recipe, compiled)
    assert result is not None and result.aov_frame is not None
    output_path = tmp_path / "mapscene.tif"
    scene._finalize_native_result(str(output_path), compiled, result)
    assert scene.last_render_backend == "gpu_terrain"
    assert result.metadata["raster_overlay_backend"] == "native_terrain_uv"
    with rasterio.open(output_path) as published:
        assert published.tags()["AREA_OR_POINT"] == "Area"
        assert published.transform * (0, 0) == grid.xy(0, 0)
        assert published.transform * (65, 65) == grid.xy(8, 8)
        np.testing.assert_array_equal(np.moveaxis(published.read(), 0, -1), result.rgba)
    normals = np.asarray(result.aov_frame.normal())
    albedo = np.asarray(result.aov_frame.albedo())
    centre_pixel = 32
    axes = [
        ("east", (2, 6), normals[centre_pixel, :, 0], albedo[centre_pixel, :, 0], result.rgba[centre_pixel, :, 2]),
        ("south", (2, 5), normals[:, centre_pixel, 2], albedo[:, centre_pixel, 0], result.rgba[:, centre_pixel, 2]),
    ]
    positions = np.arange(65) + 0.5
    for axis, indices, normal, drape, vector in axes:
        centres = np.asarray(indices) * 65 / 8
        pitch = centres[1] - centres[0]
        # Fullscreen's triangles cover the viewport. Opposite slopes mark
        # the ridge in the DEM sampled by the fragment shader.
        terrain = []
        for centre in centres:
            region = np.abs(positions - centre) < pitch / 2
            weights = np.abs(normal[region])
            assert weights.max() > weights.min()
            terrain.append(float(np.average(positions[region], weights=weights) - centre))
        measured = [terrain, ridge_residual(drape, centres, pitch), ridge_residual(vector, centres, pitch)]
        print(f"MapScene {registration} {axis} residual pixels (terrain, drape, vector) {measured}")
        assert np.max(np.abs(measured)) <= 1.0
        assert np.max(np.ptp(measured, axis=0)) <= 1.0


@pytest.mark.parametrize("registration", ["area", "point"])
def test_native_gis_reads_and_writes_same_registration(tmp_path, registration):
    from forge3d import gis

    grid, heights, _ = terrain_fixture(registration)
    path = tmp_path / "native.tif"
    coefficients = tuple(grid.transform)[:6]
    info = gis.write_raster(path, heights, crs=grid.crs, transform=coefficients,
                            registration=registration)
    assert info.registration == registration
    assert info.as_dict()["registration"] == registration
    assert gis.read_raster_info(path).transform == coefficients
    read = load_dem(path)
    assert read.transform == grid.transform and read.registration == registration
    np.testing.assert_array_equal(read.data, heights)


def test_thematic_overlay_rejects_mixed_registration(tmp_path):
    import forge3d as f3d
    from forge3d.map_scene import _mapscene_thematic_surface
    from forge3d.raster_style import CategoricalRasterStyle

    grid, heights, _ = terrain_fixture("point")
    path = tmp_path / "area-classes.tif"
    write_tiff(path, heights.astype(np.uint8), replace(grid, registration="area"))
    overlay = f3d.RasterOverlay("classes", path=path, crs=grid.crs,
                                style=CategoricalRasterStyle({0: (0, 0, 0, 255), 2: (255, 0, 0, 255)}))
    scene = f3d.MapScene(terrain=f3d.TerrainSource(data=heights, crs=grid.crs, metadata=grid.to_dict()),
                         layers=[overlay], lighting=f3d.LightingPreset(), output=f3d.OutputSpec(65, 65))
    with pytest.raises(ValueError, match="raster registration mismatch"):
        _mapscene_thematic_surface(scene.recipe, heights.shape)
    overlay.metadata = {"registration_conversion": "preserve_sample_locations"}
    aligned = _mapscene_thematic_surface(scene.recipe, heights.shape)
    np.testing.assert_array_equal(aligned.rgba[..., 0] > 0, heights > 0)


@pytest.mark.offscreen
@pytest.mark.parametrize("registration", ["area", "point"])
def test_thematic_boundary_uses_terrain_sample_lattice(tmp_path, registration):
    import forge3d as f3d
    import rasterio

    grid, heights, _ = terrain_fixture(registration)
    classes = np.zeros(heights.shape, dtype=np.uint8)
    classes[:, 2:] = 1
    path = tmp_path / "classes.tif"
    write_tiff(path, classes, grid)
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(data=heights, crs=grid.crs, metadata=grid.to_dict()),
        layers=[f3d.RasterOverlay(
            "classes", path=path, crs=grid.crs,
            style=f3d.CategoricalRasterStyle({0: (0, 0, 0, 255), 1: (255, 0, 0, 255)}),
        )],
        lighting=f3d.LightingPreset(settings={"unlit_albedo": True}),
        output=f3d.OutputSpec(131, 65, format="geotiff"),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=7),
    )
    output = tmp_path / "thematic.tif"
    scene.render(str(output))
    assert scene.last_render_backend == "gpu_terrain"
    with rasterio.open(output) as published:
        assert published.tags()["AREA_OR_POINT"] == "Area"
        red = published.read(1)[32]
        boundary = int(np.flatnonzero(np.diff(red.astype(np.int16)) > 0)[0]) + 1
        expected = (grid.xy(1, 4)[0] + grid.xy(2, 4)[0]) / 2
        east = (published.transform * (boundary, 32.5))[0]
        residual = (east - expected) / published.res[0]
        print(f"MapScene thematic {registration} class boundary residual {residual:.6f} pixels")
        assert abs(residual) <= 1.0


def test_native_prepare_dem_uses_both_affines_and_rejects_mixed_registration(tmp_path):
    from forge3d import gis

    grid, heights, _ = terrain_fixture("point")
    source = tmp_path / "source.tif"
    write_tiff(source, heights, grid)
    point = gis.read_raster_info(source)
    with pytest.raises(RuntimeError, match="raster registration mismatch"):
        gis.prepare_dem(source, point.as_dict() | {"registration": "area"})

    affine = grid.transform * grid.transform.translation(1, 0)
    target = point.as_dict() | {"transform": tuple(affine)[:6]}
    result = gis.prepare_dem(source, target)
    target_path = tmp_path / "target.tif"
    write_tiff(target_path, heights, replace(grid, geotransform=affine.to_gdal()))
    aligned = gis.align_raster_to(source, target_path, resampling="nearest")
    assert result["info"]["transform"] == tuple(affine)[:6]
    assert result["info"]["registration"] == "point"
    np.testing.assert_array_equal(result["array"][..., :-1], aligned["array"][..., :-1])
    assert np.flatnonzero(result["array"][0, 4] > 0).tolist() == [1, 5]
    assert not result["mask"][..., -1].any()


def test_decoded_raster_keeps_registration_through_native_gis(tmp_path):
    from forge3d import gis

    grid, heights, _ = terrain_fixture("point")
    path = tmp_path / "point.tif"
    write_tiff(path, heights, grid)
    source = gis.read_raster(path, masked=True)
    for operation in (gis.prepare_dem, gis.normalize_raster,
                      lambda raster: gis.classify_raster(raster, bins=[1])):
        result = operation(source)
        assert result["info"]["registration"] == "point"
        assert result["info"]["transform"] == tuple(grid.transform)[:6]
        with pytest.raises(ValueError, match="invalid raster registration"):
            operation(source | {"info": source["info"] | {"registration": "invalid"}})


@pytest.mark.parametrize(("target_crs", "bit_depth", "message"), [
    ("EPSG:3857", 8, "terrain grid CRS mismatch"),
    ("EPSG:32631", 16, "native 8-bit RGBA"),
])
def test_mapscene_geotiff_validation_and_render_share_eligibility(tmp_path, target_crs, bit_depth, message):
    import forge3d as f3d

    grid, heights, _ = terrain_fixture("area")
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(data=heights, crs=grid.crs, metadata=grid.to_dict()),
        lighting=f3d.LightingPreset(), target_crs=target_crs,
        output=f3d.OutputSpec(4, 4, format="geotiff", bit_depth=bit_depth),
    )
    report = scene.validate()
    assert report.render_blocked()
    assert any(d.code == "geotiff_output_unavailable" and message in d.message for d in report.diagnostics)
    path = tmp_path / "rejected.tif"
    with pytest.raises(ValueError, match=message):
        scene.render(str(path))
    assert not path.exists()


@pytest.mark.offscreen
def test_decoded_point_thematic_matches_file_and_bundle(tmp_path):
    import forge3d as f3d
    import rasterio
    from forge3d import gis

    grid, heights, _ = terrain_fixture("point")
    dem, classes_path = tmp_path / "dem.tif", tmp_path / "classes.tif"
    write_tiff(dem, heights, grid)
    shifted = replace(grid, geotransform=(grid.transform * grid.transform.translation(1, 0)).to_gdal())
    values = np.zeros(heights.shape, dtype=np.uint8)
    values[:, 2] = 1
    write_tiff(classes_path, values, shifted)
    style = f3d.CategoricalRasterStyle({0: (0, 0, 0, 255), 1: (255, 0, 0, 255)})

    def scene(layer):
        return f3d.MapScene(
            terrain=f3d.TerrainSource(path=dem, crs=grid.crs), layers=[layer],
            lighting=f3d.LightingPreset(settings={"unlit_albedo": True}),
            output=f3d.OutputSpec(65, 65, format="geotiff"),
        )

    file_scene = scene(f3d.RasterOverlay("classes", path=classes_path, crs=grid.crs, style=style))
    decoded = gis.read_raster(classes_path, masked=True)
    decoded_scene = scene(f3d.RasterOverlay("classes", data=decoded, crs=grid.crs, style=style))
    bundle = tmp_path / "scene.forge3d"
    assert not decoded_scene.save_bundle(bundle).render_blocked()
    restored = f3d.MapScene.load_bundle(bundle)
    stored = restored.recipe.layers[0].data["info"]
    assert stored["registration"] == "point"
    assert tuple(stored["transform"]) == tuple(shifted.transform)[:6]
    frames = []
    for label, recipe in [("file", file_scene), ("decoded", decoded_scene), ("restored", restored)]:
        path = tmp_path / f"{label}.tif"
        recipe.render(str(path))
        assert recipe.last_render_backend == "gpu_terrain"
        with rasterio.open(path) as result:
            frames.append(result.read())
    assert frames[0][0, 32, 24] > 0
    assert frames[0][0, 32, 16] == 0
    np.testing.assert_array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(frames[0], frames[2])


@pytest.mark.offscreen
@pytest.mark.parametrize("unlit", [False, True])
def test_small_registered_geotiff_samples_destination_centres(tmp_path, unlit):
    import forge3d as f3d
    import rasterio

    grid, heights, rgba = terrain_fixture("area")
    rgba[..., 0] = np.arange(grid.width) % 2
    dem, drape = tmp_path / "dem.tif", tmp_path / "drape.tif"
    write_tiff(dem, heights, grid)
    write_tiff(drape, (rgba * 255).astype(np.uint8), grid)
    frames = {}
    for size in (64, 4):
        path = tmp_path / f"output-{size}.tif"
        scene = f3d.MapScene(
            terrain=f3d.TerrainSource(path=dem, crs=grid.crs),
            layers=[f3d.RasterOverlay("drape", path=drape, crs=grid.crs)],
            lighting=f3d.LightingPreset(settings={"unlit_albedo": unlit}),
            output=f3d.OutputSpec(size, size, format="geotiff", aovs=("albedo",)),
        )
        scene.render(str(path))
        assert scene.last_render_backend == "gpu_terrain"
        with rasterio.open(path) as result:
            assert result.transform * (0, 0) == grid.xy(0, 0)
            assert result.transform * (size, size) == grid.xy(8, 8)
            frames[size] = result.read(1)
    assert np.all(frames[4] > 0)
    np.testing.assert_array_equal(frames[4], frames[64][8::16, 8::16])
