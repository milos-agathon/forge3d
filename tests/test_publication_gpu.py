"""C06 reuse and colour proof on real native frames. Opt-in physical GPU lane."""
import os

import numpy as np
from PIL import Image
import pytest

import forge3d as f3d
from forge3d import gis
from forge3d.map_plate import BBox
from forge3d.publication import PublicationExtent, PublicationRecipe, PublicationSource

pytestmark = [pytest.mark.usefixtures("pro_license"),
              pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
                                 reason="native publication reuse requires a physical GPU")]
TRANSFORM = (100, 0, 600000, 0, -100, 5820000)
PALETTE = [(240, 230, 200, 255), (220, 130, 70, 255), (110, 30, 50, 255)]
BOUNDS = BBox(600000, 5818000, 603000, 5820000, "EPSG:32631")
PAGE = PublicationExtent("city", "City", BBox(600000, 5818000, 601000, 5820000, "EPSG:32631"))


def test_flyover_uses_executed_frames_and_tracks_moving_furniture(recipe, tmp_path):
    from dataclasses import replace
    import hashlib
    import json
    from forge3d.map_scene import OrbitCamera
    from forge3d.publication import PublicationLayout
    recipe.camera_mode = "flat-perspective"
    bounds = BBox(600000, 5819000, 601000, 5820000, "EPSG:32631")
    frames = [replace(PAGE, page_id=f"frame-{i}", bounds=bounds,
        camera=OrbitCamera(distance=500, elevation_deg=30, azimuth_deg=angle, fov_deg=35))
        for i, angle in enumerate((270, 260, 250))]
    output = tmp_path / "flyover"
    result = recipe.render_flyover(frames, output, layout=PublicationLayout(recipe.config), base_seed=7)
    manifest = json.loads((output / result["chronos_manifest"]).read_text())
    assert len(result["frames"]) == len(frames)
    locator_pixels = []
    for page, frame, record in zip(result["frames"], frames, manifest["frames"], strict=True):
        image = np.asarray(Image.open(page["output"]["path"]).convert("RGBA"))
        x0,y0,x1,y1 = page["map_rect"]
        native = np.asarray(Image.open(output / "native" / record["image"]).convert("RGBA"))
        assert np.array_equal(image[y0:y1,x0:x1], native)
        assert hashlib.sha256(native.tobytes()).hexdigest() == record["pixel_hash"]
        assert page["flyover"]["camera_hash"] == record["camera_hash"]
        assert page["scale"]["omitted"]
        assert page["main_render"]["max_legend_deviation_levels"] <= 1
        # Independent pinhole intersection; validate every rendered map pixel
        # and the locator corners against the fixture's registered source cells.
        camera = frame.camera
        theta,phi = np.deg2rad([camera.elevation_deg,camera.azimuth_deg])
        eye = np.asarray(camera.target)+camera.distance*np.array([
            np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        forward = np.asarray(camera.target)-eye; forward /= np.linalg.norm(forward)
        side = np.cross(forward,[0,0,1]); side /= np.linalg.norm(side)
        up = np.cross(side,forward)
        width,height = page["main_viewport"]
        def ground(x,y):
            ray = (forward+((np.asarray(x)*2/width-1)*np.tan(np.deg2rad(camera.fov_deg/2))*width/height)[...,None]*side
                   +((1-np.asarray(y)*2/height)*np.tan(np.deg2rad(camera.fov_deg/2)))[...,None]*up)
            return eye+((-0.5-eye[2])/ray[...,2])[...,None]*ray
        yy,xx = np.mgrid[:height,:width]
        world = ground(xx+.5,yy+.5)
        cols = np.floor((world[...,0]+500)/100).astype(int)
        rows = np.floor((500-world[...,1])/100).astype(int)
        classes = np.searchsorted([100,300],rows*30+cols,side="right")+1
        classes[(rows==0)&(cols==0)] = 0
        palette = np.array([e["rgba"] for e in page["legend"]["entries"]])
        difference = np.abs(native.astype(int)-palette[classes]).max(axis=-1)
        # A raster pixel covers an area. At a class boundary, native float32
        # projection may choose the neighbouring cell while this independent
        # float64 centre ray chooses the other. Require every such pixel to
        # match a class intersecting its own footprint, still within ONE level.
        # No pixel is excluded from the palette or geometry check.
        covered = difference <= 1
        for dx,dy in ((0,0),(1,0),(1,1),(0,1)):
            corner = ground(xx+dx,yy+dy)
            cc = np.floor((corner[...,0]+500)/100).astype(int)
            rr = np.floor((500-corner[...,1])/100).astype(int)
            corner_classes = np.searchsorted([100,300],rr*30+cc,side="right")+1
            corner_classes[(rr==0)&(cc==0)] = 0
            covered |= np.abs(native.astype(int)-palette[corner_classes]).max(axis=-1) <= 1
        assert covered.all()
        corners = ground(np.array([0,width,width,0]),np.array([0,0,height,height]))[:,:2]+[600500,5819500]
        lx,ly,rx,by = page["locator_rect"]
        for east,north in corners:
            x = lx+int((east-600000)/3000*(rx-lx))
            y = ly+int((5820000-north)/2000*(by-ly))
            assert np.any(np.all(image[y-1:y+2,x-1:x+2]==[18,33,43,255],axis=-1))
        locator_pixels.append(image[ly:by,lx:rx].copy())
    assert not np.array_equal(locator_pixels[0],locator_pixels[-1])
    for name, endpoint in (("title",result["frames"][0]),("end",result["frames"][-1])):
        assert (output / "bookends" / f"{name}.png").read_bytes() == __import__("pathlib").Path(endpoint["output"]["path"]).read_bytes()
    with pytest.raises(ValueError, match="empty output directory"):
        recipe.render_flyover(frames, output, layout=PublicationLayout(recipe.config), base_seed=7)


def test_flat_perspective_colour_camera_cache_and_ground_coverage(recipe, tmp_path):
    from dataclasses import replace
    from forge3d.map_scene import OrbitCamera
    recipe.camera_mode = "flat-perspective"
    page = replace(PAGE, bounds=BBox(600000, 5819000, 601000, 5820000, "EPSG:32631"),
                   camera=OrbitCamera(distance=500, elevation_deg=30, azimuth_deg=270, fov_deg=35))
    cache = tmp_path / "cache"
    first = recipe.render_series([page], tmp_path / "first", cache=cache)[0]
    assert first["scale"]["omitted"] is True
    assert first["main_render"]["max_legend_deviation_levels"] <= 1
    assert first["footprint"]["kind"] == "flat_terrain_perspective_ground_intersection"
    # Independent pinhole rays in geographic XYZ. This never calls the recipe's
    # native-projection/homography helper or consumes its recorded footprint.
    camera = page.camera
    theta, phi = np.deg2rad([camera.elevation_deg, camera.azimuth_deg])
    eye = np.asarray(camera.target) + camera.distance * np.array(
        [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    forward = np.asarray(camera.target) - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, [0, 0, 1]); side /= np.linalg.norm(side)
    up = np.cross(side, forward)
    width, height = first["main_viewport"]
    def ground(x, y):
        ray = (forward + ((np.asarray(x)*2/width-1)*np.tan(np.deg2rad(camera.fov_deg/2))*width/height)[..., None]*side
               + ((1-np.asarray(y)*2/height)*np.tan(np.deg2rad(camera.fov_deg/2)))[..., None]*up)
        # Native flat fixture domain is [0,1], centred at z=-0.5.
        return eye + ((-0.5-eye[2])/ray[..., 2])[..., None]*ray
    corners = ground(np.array([0,width,width,0]), np.array([0,0,height,height]))
    expected = corners[:, :2] + [600500, 5819500]
    ground_pixel = np.linalg.norm(ground(np.array([1,width-1]), np.array([1,height-1])) -
                                 ground(np.array([0,width]), np.array([0,height])), axis=1).max()
    assert np.max(np.linalg.norm(np.asarray(first["footprint"]["coordinates"])-expected, axis=1)) < ground_pixel
    image = np.asarray(Image.open(first["output"]["path"]).convert("RGBA"))
    x0,y0,x1,y1 = first["map_rect"]
    yy,xx = np.mgrid[:height,:width]
    world = ground(xx+.5, yy+.5)
    cols = np.floor((world[...,0]+500)/100).astype(int)
    rows = np.floor((500-world[...,1])/100).astype(int)
    values = rows*30+cols
    expected_classes = np.searchsorted([100,300],values,side="right")+1
    expected_classes[(rows==0)&(cols==0)] = 0
    palette = np.array([entry["rgba"] for entry in first["legend"]["entries"]])
    assert np.abs(image[y0:y1,x0:x1].astype(int)-palette[expected_classes]).max() <= 1
    # Check the actually DRAWN locator against independently ray-intersected
    # corner coordinates through Rasterio's inverse registration.
    from rasterio.transform import from_bounds,rowcol
    lx,ly,rx,by = first["locator_rect"]
    transform = from_bounds(600000,5818000,603000,5820000,rx-lx,by-ly)
    ys,xs = rowcol(transform,expected[:,0],expected[:,1])
    for x,y in zip(xs,ys):
        assert np.any(np.all(image[ly+y-1:ly+y+2,lx+x-1:lx+x+2]==[18,33,43,255],axis=-1))
    same = recipe.render_series([page], tmp_path / "same", cache=cache)[0]
    assert _hits(same) == (True, True)
    moved = recipe.render_series([replace(page, camera=replace(page.camera, azimuth_deg=260))],
                                tmp_path / "moved", cache=cache)[0]
    assert _hits(moved) == (False, True)
    assert first["footprint"]["coordinates"] != moved["footprint"]["coordinates"]
    assert first["output"]["pixel_sha256"] != moved["output"]["pixel_sha256"]
    with pytest.raises(ValueError, match="corner rays"):
        recipe.render_series([replace(page, camera=replace(page.camera, distance=2000))], tmp_path / "uncovered")


@pytest.mark.parametrize("case", ["nonflat", "nonfinite", "offline", "effects", "clipmap", "yup", "other", "horizon"])
def test_unlit_perspective_rejects_unsupported_cases(recipe, tmp_path, case):
    from forge3d.map_scene import OrbitCamera
    values, valid, grid, _, description = recipe._read()
    classes, style, _, _ = recipe._classify(values, valid)
    scene = recipe._scene(classes, style, grid, (100, 100), description)
    scene.recipe.camera = OrbitCamera(distance=500, elevation_deg=30, azimuth_deg=270, fov_deg=35)
    scene.recipe.terrain.metadata = {**scene.recipe.terrain.metadata, "width": 30, "height": 20}
    scene.recipe.lighting.settings = {"unlit_albedo": True, "cli_params": {"camera_mode": "mesh:zup"}}
    if case == "nonflat": scene.recipe.terrain.data[0, 0] = 1
    if case == "nonfinite": scene.recipe.terrain.data[0, 0] = np.nan
    if case == "offline": scene.recipe.output.samples = 2
    if case == "effects": scene.recipe.lighting.settings["screen_space"] = {"gtao": {"enabled": True}}
    if case == "clipmap": scene.recipe.lighting.settings["cli_params"]["camera_mode"] = "clipmap:zup"
    if case == "yup": scene.recipe.lighting.settings["cli_params"]["camera_mode"] = "mesh"
    if case == "other": scene.recipe.lighting.settings["cli_params"]["camera_mode"] = "mesh:zup:extra"
    if case == "horizon": scene.recipe.camera.elevation_deg = 90
    with pytest.raises((ValueError, RuntimeError)):
        scene.render(str(tmp_path / "unsupported.png"))
    assert not (tmp_path / "unsupported.png").exists()


@pytest.fixture
def recipe(tmp_path):
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert probe["status"] == "ok" and not probe.get("software_fallback", False), probe
    values = np.arange(600, dtype=np.float32).reshape(20, 30)
    values[0, 0] = -9999
    path = tmp_path / "source.tif"
    gis.write_raster(path, values, crs="EPSG:32631", nodata=-9999, transform=TRANSFORM)
    source = PublicationSource(path, "Fixture", "2024", "v2", "https://example.test", "people", "missing")
    return PublicationRecipe(source, [100, 300], PALETTE, BOUNDS)


def _hits(receipt):
    return receipt["main_render"]["metadata"]["cache_hit"], receipt["locator_render"]["metadata"]["cache_hit"]


def test_native_subject_series_keeps_insets_linked_to_each_dataset(recipe, tmp_path):
    from dataclasses import replace
    from forge3d.publication import PublicationInset, PublicationLayout, PublicationSubject

    # Synthetic, explicitly labelled subjects exercise the native publication
    # route; these are not demographic observations or the CBS time evidence.
    people = np.arange(600, dtype=np.float32).reshape(20, 30)
    people[0, 0] = -9999
    houses = np.where(people == -9999, people, people / 2)
    path = tmp_path / "households.tif"
    gis.write_raster(path, houses, crs=BOUNDS.crs, nodata=-9999, transform=TRANSFORM)
    population = replace(recipe.source, name="Synthetic population fixture")
    households = replace(recipe.source, path=path, name="Synthetic households fixture", units="households")
    detail = BBox(600000, 5819000, 601000, 5820000, BOUNDS.crs)
    recipe.layout = PublicationLayout(
        replace(recipe.config, width=2000, margin=(100, 460, 170, 450)),
        [PublicationInset("detail", "Linked detail", detail, (20, 120, 420, 550))],
    )
    subjects = [PublicationSubject("population", "Synthetic population", population, [100, 300], PALETTE),
                PublicationSubject("households", "Synthetic households", households, [50, 250], PALETTE)]
    receipts = recipe.render_subjects(subjects, [PAGE], tmp_path / "subjects")
    for receipt, values, breaks in zip(receipts, [people, houses], [[100, 300], [50, 250]], strict=True):
        assert receipt["main_render"]["backend"] == "gpu_terrain"
        inset = receipt["insets"][0]
        assert inset["render"]["backend"] == "gpu_terrain"
        assert inset["classification_id"] == receipt["dependencies"]["classification"]["id"]
        _assert_cells_show_legend(receipt, values[:, :10], breaks)
        image = np.asarray(Image.open(receipt["output"]["path"]).convert("RGBA")).astype(int)
        x0, y0, x1, y1 = inset["map_rect"]
        ys = y0 + ((np.arange(10) + .5) * (y1-y0) / 10).astype(int)
        xs = x0 + ((np.arange(10) + .5) * (x1-x0) / 10).astype(int)
        classes = np.searchsorted(breaks, values[:10, :10], side="right") + 1
        classes[0, 0] = 0
        palette = np.array([entry["rgba"] for entry in receipt["legend"]["entries"]])
        assert np.abs(image[np.ix_(ys, xs)] - palette[classes]).max() <= 1
    assert receipts[0]["source"]["sha256"] != receipts[1]["source"]["sha256"]
    assert receipts[0]["insets"][0]["id"] != receipts[1]["insets"][0]["id"]
    assert receipts[0]["main_viewport"] == receipts[1]["main_viewport"]
    assert receipts[1]["legend"]["units"] == "households"


def _assert_cells_show_legend(receipt, values, breaks):
    """Saved page pixels vs NumPy classification, independent of the recipe."""
    image = np.asarray(Image.open(receipt["output"]["path"]).convert("RGBA")).astype(int)
    x0, y0, x1, y1 = receipt["map_rect"]
    rows, cols = values.shape
    ys = y0 + ((np.arange(rows) + .5) * (y1 - y0) / rows).astype(int)
    xs = x0 + ((np.arange(cols) + .5) * (x1 - x0) / cols).astype(int)
    classes = np.where(values != -9999, np.searchsorted(breaks, values, side="right") + 1, 0)
    palette = {entry["class_id"]: entry["rgba"] for entry in receipt["legend"]["entries"]}
    expected = np.array([palette[k] for k in range(len(palette))])[classes]
    bx0, by0, bx1, by1 = receipt["scale"]["bar_rect"]
    keep = ~((ys[:, None] >= by0) & (ys[:, None] < by1) & (xs[None] >= bx0) & (xs[None] < bx1))
    assert np.abs(image[np.ix_(ys, xs)] - expected)[keep].max() <= 1


def test_unchanged_inputs_reuse_frames_and_each_revision_misses(recipe, tmp_path):
    cache = tmp_path / "cache"
    values = np.arange(600, dtype=np.float32).reshape(20, 30)
    values[0, 0] = -9999
    def run(name, active=recipe, page=PAGE):
        return active.render_series([page], tmp_path / name, cache=cache)[0]

    first = run("first")
    assert _hits(first) == (False, False)
    assert first["main_render"]["metadata"]["frame_color_source"] == "native_albedo_srgb"
    _assert_cells_show_legend(first, values[:20, :10], recipe.breaks)

    same = run("same")
    assert _hits(same) == (True, True)
    assert same["output"]["pixel_sha256"] == first["output"]["pixel_sha256"]
    assert same["main_render"]["max_legend_deviation_levels"] == first["main_render"]["max_legend_deviation_levels"]

    # Same path, size and mtime: only the bytes identify the change.
    source = recipe.source.path
    before = source.stat()
    edited = values[::-1].copy()
    gis.write_raster(source, edited, crs="EPSG:32631", nodata=-9999, transform=TRANSFORM, overwrite=True)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (source.stat().st_size, source.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    changed = run("edited")
    assert _hits(changed) == (False, False)
    assert changed["output"]["pixel_sha256"] != first["output"]["pixel_sha256"]
    _assert_cells_show_legend(changed, edited[:20, :10], recipe.breaks)

    recipe.breaks = [50, 250]
    revised = run("breaks")
    assert _hits(revised) == (False, False)
    _assert_cells_show_legend(revised, edited[:20, :10], recipe.breaks)

    moved = PublicationExtent("city", "City", BBox(600100, 5818000, 601100, 5820000, "EPSG:32631"))
    assert _hits(run("extent", page=moved)) == (False, True)  # The locator's scene is unchanged.


def test_lit_and_unlit_frames_never_share_a_cache_entry(recipe, tmp_path):
    cache = tmp_path / "cache"
    recipe.render_series([PAGE], tmp_path / "first", cache=cache)
    values, valid, grid, _, description = recipe._read()
    classes, style, _, _ = recipe._classify(values, valid)
    lit = recipe._scene(classes, style, grid, (300, 200), description)
    unlit = recipe._scene(classes, style, grid, (300, 200), description)
    lit.recipe.lighting.settings = {"cli_params": {"camera_mode": "screen"}}
    unlit.render(str(tmp_path / "unlit.png"), cache=cache)
    lit.render(str(tmp_path / "lit.png"), cache=cache)
    assert not lit.last_render_metadata["cache_hit"]
    assert np.mean(np.asarray(Image.open(tmp_path / "lit.png"))[..., :3]) < np.mean(
        np.asarray(Image.open(tmp_path / "unlit.png"))[..., :3])


def test_unlit_frame_holds_only_opaque_legend_colours(recipe, tmp_path):
    values, valid, grid, _, description = recipe._read()
    classes, style, _, _ = recipe._classify(values, valid)
    scene = recipe._scene(classes, style, grid, (300, 200), description)
    scene.render(str(tmp_path / "frame.png"))
    assert scene.last_render_backend == "gpu_terrain"
    frame = np.asarray(Image.open(tmp_path / "frame.png").convert("RGBA"))
    assert (frame[..., 3] == 255).all()
    assert {tuple(colour) for colour in np.unique(frame.reshape(-1, 4), axis=0)} <= {
        tuple(rgba) for rgba in style.palette.values()}


def test_corrupt_cached_frame_is_never_published(recipe, tmp_path):
    cache = tmp_path / "cache"
    first = recipe.render_series([PAGE], tmp_path / "first", cache=cache)[0]
    files = [path for path in cache.rglob("*") if path.is_file() and path.stat().st_size > 1024]
    assert files
    for path in files:
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(data)
    try:
        again = recipe.render_series([PAGE], tmp_path / "again", cache=cache)[0]
    except (RuntimeError, ValueError, OSError):
        return  # Refusing corrupt storage is valid; publishing from it is not.
    assert again["output"]["pixel_sha256"] == first["output"]["pixel_sha256"]


@pytest.mark.parametrize("width", [60, 300])
def test_native_short_map_scale_matches_independent_geodesic(recipe, tmp_path, width):
    from dataclasses import replace
    from pyproj import Geod, Transformer
    cell = width / 10
    bounds = BBox(600000, 5820000 - width, 600000 + width, 5820000, "EPSG:32631")
    gis.write_raster(recipe.source.path, np.arange(100, dtype=np.float32).reshape(10, 10),
                     crs=bounds.crs, transform=(cell, 0, bounds.west, 0, -cell, bounds.north), overwrite=True)
    recipe.locator_bounds = bounds
    page = replace(PAGE, bounds=bounds)
    result = recipe.render_series([page], tmp_path / "short")[0]
    assert result["main_render"]["backend"] == "gpu_terrain"
    transform = Transformer.from_crs(bounds.crs, 4326, always_xy=True)
    lon, lat = transform.transform([bounds.west, bounds.east], [bounds.center_lat] * 2)
    mpp = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])[2] / result["main_viewport"][0]
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA"))
    x0, y0, x1, y1 = result["scale"]["bar_rect"]
    border = np.flatnonzero(np.all(image[y0 + 12, x0:x1, :3] == 0, axis=1))
    number, unit = result["scale"]["bar_label"].split()
    assert unit == "m"
    assert abs(float(number) - (border[-1] - border[0] + 1) * mpp) <= mpp
