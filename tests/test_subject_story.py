from dataclasses import replace

import numpy as np
import pytest

import forge3d as f3d
from forge3d.camera_rigs import TerrainClearance, TerrainOrbitRig
from forge3d.map_scene import _build_mapscene_terrain_params, _load_native_heightmap
from forge3d.raster_style import RasterHeightSurfaceStyle
from forge3d import subject_story


def _north_up_grid(bounds):
    west, _, east, north = bounds
    step = (east - west) / 511
    return {"geotransform": (west - step / 2, step, 0, north + step / 2, 0, -step)}


@pytest.fixture
def story(tmp_path):
    path = tmp_path / "terrain.npy"
    np.save(path, np.zeros((512, 512), dtype=np.float32))
    bounds = (2640000, 1160000, 2650000, 1170000)
    return {
        "terrain": f3d.TerrainSource(path=path, crs="EPSG:2056", metadata=_north_up_grid(bounds)),
        "features": [
            f3d.SubjectFeature("summit-a", "Aiguille", 2644000, 1164000),
            f3d.SubjectFeature("summit-b", "Ridge", 2643500, 1164500),
        ],
        "subject_id": "summit-a",
        "bounds_wsen": bounds,
        "feature_crs": "EPSG:2056",
        "orbit": lambda target: TerrainOrbitRig(target, 1, 2000, 0, 10),
        "fps": 4,
        "hold_seconds": 0.5,
        "output": f3d.OutputSpec(width=640, height=480),
        "lighting": f3d.LightingPreset(),
        "framing": (0.05, 0.05, 0.95, 0.95),
        "reserved_text": (0.03, 0.04, 0.75, 0.25),
        "reveal_annotation": f3d.StoryAnnotation("Above {subject}", (0.06, 0.14), 24),
    }


def test_subject_binding_aspect_and_frame_selection(story):
    first = f3d.compile_subject_shot(**story)
    revision = {**story, "subject_id": "summit-b", "output": f3d.OutputSpec(width=480, height=480)}
    second = f3d.compile_subject_shot(**revision)
    repeat = f3d.compile_subject_shot(**revision)
    assert first.key_frames == second.key_frames == {"establish": 0, "reveal": 4, "hold": 5}
    assert second.frame_checks == repeat.frame_checks
    assert first.frame_checks[0]["target_xyz_m"] == (-1000, -1000, -0.5)
    assert second.frame_checks[0]["target_xyz_m"] == (-1500, -500, -0.5)
    assert second.frame_checks[0]["features"]["summit-b"]["viewport_xy"] == pytest.approx((0.5, 0.5))
    assert second.frame_checks[4]["eye_xyz_m"] == second.frame_checks[5]["eye_xyz_m"]
    scene = second.frame_scene(4)
    assert scene.recipe.layers[0].plan.accepted[0].text == "Above Ridge"
    assert scene.recipe.map_furniture.keepouts[0].bounds == pytest.approx((14.4, 19.2, 360, 120))
    terrain = np.load(story["terrain"].path)
    params = _build_mapscene_terrain_params(scene.recipe, terrain, (480, 480))
    assert tuple(params.cam_target) == second.frame_checks[4]["target_xyz_m"]
    assert params.cam_radius == pytest.approx(scene.recipe.camera.distance)
    assert params.camera_mode == "mesh:zup:north"
    with pytest.raises(IndexError, match="no subject shot frame"):
        second.frame_scene(-1)
    np.save(story["terrain"].path, terrain + 1)
    with pytest.raises(ValueError, match="changed since compilation"):
        second.frame_scene(4)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"subject_id": "absent"}, "unknown feature ID"),
        ({"feature_crs": "EPSG:4326"}, "share the same projected metre CRS"),
        ({"feature_crs": "EPSG:4326", "terrain": f3d.TerrainSource(crs="EPSG:4326")},
         "share the same projected metre CRS"),
        ({"hold_seconds": 0.3}, "positive whole number of frames"),
        ({"framing": (0.1, 0.1, 0.4, 0.4)}, "outside framing"),
        ({"reserved_text": (0.03, 0.04, 0.75, 0.55)}, "overlaps reserved text space"),
        ({"reserved_text": (0.03, 0.04, 0.12, 0.25)}, "glyphs do not fit"),
    ],
)
def test_explains_failed_constraints(story, changes, message):
    with pytest.raises(ValueError, match=message):
        f3d.compile_subject_shot(**{**story, **changes})


def test_south_first_heightmap_is_rejected_before_planning(story):
    x0, step, _, _, _, _ = story["terrain"].metadata["geotransform"]
    south = story["bounds_wsen"][1]
    south_first = np.zeros((512, 512), dtype=np.float32)
    south_first[201:210, 201:210] = 1500
    np.save(story["terrain"].path, south_first)
    wrong_grid = replace(story["terrain"], metadata={
        "geotransform": (x0, step, 0, south - step / 2, 0, step),
    })
    with pytest.raises(ValueError, match="requires a north-up geotransform"):
        f3d.compile_subject_shot(**{**story, "terrain": wrong_grid})
    np.save(story["terrain"].path, south_first[::-1])
    shot = f3d.compile_subject_shot(**story)
    assert shot.frame_checks[0]["features"]["summit-a"]["mesh_xyz_m"][2] == pytest.approx(750)


@pytest.mark.parametrize("metadata", [None, {"geotransform": (0, 10, 0, 0, 0, 10)}])
def test_terrain_registration_requires_a_matching_transform(story, metadata):
    terrain = replace(story["terrain"], metadata=metadata)
    with pytest.raises(ValueError, match="geotransform"):
        f3d.compile_subject_shot(**{**story, "terrain": terrain})


def test_styled_hill_rejects_an_orbit_that_clears_raw_heights(story):
    heights = np.zeros((512, 512), dtype=np.float32)
    column = round((4000 + 2000 / np.sqrt(2)) * 511 / 10000)
    row = round(4000 * 511 / 10000)
    heights[row - 2:row + 3, column - 2:column + 3] = 100
    np.save(story["terrain"].path, heights[::-1])
    story.update(
        fps=1, hold_seconds=1,
        orbit=lambda target: TerrainOrbitRig(
            target, 1, 2000, 0, 0, clearance=TerrainClearance(minimum_height=25),
        ),
    )
    assert f3d.compile_subject_shot(**story).frame_checks[0]["clearance_m"] > 25
    terrain = replace(story["terrain"], style=RasterHeightSurfaceStyle(100, "m"))
    assert _load_native_heightmap(terrain).max() == 10000
    with pytest.raises(f3d.SubjectShotInfeasible, match="frame 0: terrain occludes summit-a"):
        f3d.compile_subject_shot(**{**story, "terrain": terrain})


def test_styled_shot_matches_the_processed_metre_mesh(story, tmp_path):
    heights = np.arange(512, dtype=np.float32)[None, :].repeat(512, axis=0)
    np.save(story["terrain"].path, heights)
    terrain = replace(story["terrain"], style=RasterHeightSurfaceStyle(2, "m"))
    processed = tmp_path / "processed.npy"
    np.save(processed, _load_native_heightmap(terrain))
    story["features"][0] = replace(story["features"][0], line_xy=((2643990, 1164000), (2644010, 1164000)))
    story["obstacles"] = (f3d.StoryObstacle("shed", "building", (2641000, 1161000, 100, 2641010, 1161010, 150)),)
    expected = f3d.compile_subject_shot(**{**story, "terrain": replace(story["terrain"], path=processed)})
    styled = f3d.compile_subject_shot(**{**story, "terrain": terrain})
    assert styled.frame_checks == expected.frame_checks
    assert styled.frame_scene(0).recipe.terrain.style == terrain.style


@pytest.mark.parametrize(
    ("style", "message"),
    [(RasterHeightSurfaceStyle(1, "m", height_units="ft"), "metre heights"),
     (RasterHeightSurfaceStyle(1, "m", nodata=1), "complete finite")],
)
def test_unsupported_styled_heights_are_rejected_before_the_orbit(story, style, message):
    heights = np.zeros((512, 512), dtype=np.float32)
    heights[0, 0] = 1
    np.save(story["terrain"].path, heights)
    terrain = replace(story["terrain"], style=style)
    with pytest.raises(ValueError, match=message):
        f3d.compile_subject_shot(**{**story, "terrain": terrain, "orbit": None})


def test_render_forwards_certificate_and_cache(story, tmp_path, monkeypatch):
    shot = f3d.compile_subject_shot(**story)
    expected = object()
    out = tmp_path / "frames"
    cache = tmp_path / "cache"

    def render_flythrough(scenes, **kwargs):
        assert len(scenes) == shot.frame_count
        assert kwargs == dict(base_seed=7, samples=1, out_dir=out, certificate=True, cache=cache)
        return expected

    monkeypatch.setattr(subject_story, "render_flythrough", render_flythrough)
    assert shot.render(out, base_seed=7, certificate=True, cache=cache) is expected


def test_ridge_occludes_required_anchor(story):
    heights = np.zeros((512, 512), dtype=np.float32)
    heights[:, 173:177] = 700
    np.save(story["terrain"].path, heights)
    story["reserved_text"] = (0.03, 0.04, 0.75, 0.19)
    story["features"].append(f3d.SubjectFeature("behind-ridge", "Saddle", 2642800, 1164000))
    with pytest.raises(f3d.SubjectShotInfeasible, match="frame 0: terrain occludes behind-ridge"):
        f3d.compile_subject_shot(**story, establish_must_see=("behind-ridge",))


def test_consumer_triangle_clearance_can_fail_after_rig_bake(story):
    heights = np.zeros((512, 512), dtype=np.float32)
    heights[250, 251] = heights[251, 250] = 100
    np.save(story["terrain"].path, heights[::-1])
    bounds = (2640000, 1160000, 2645110, 1165110)
    story.update(
        bounds_wsen=bounds,
        terrain=replace(story["terrain"], metadata=_north_up_grid(bounds)),
        features=[f3d.SubjectFeature("summit-a", "Saddle", 2641505, 1162505)],
        orbit=lambda target: TerrainOrbitRig(
            target, 1, 1000, 0, 0, theta_start_deg=90,
            clearance=TerrainClearance(minimum_height=25),
        ),
    )
    with pytest.raises(f3d.SubjectShotInfeasible, match="frame 0: consumer terrain clearance"):
        f3d.compile_subject_shot(**story)


def test_linear_consumer_path_detects_a_ridge_between_safe_endpoints():
    heights = np.zeros((512, 512), dtype=np.float32)
    heights[:, 250] = 100
    start = np.array([249.0, 250.5, 70.0])
    end = np.array([251.0, 250.5, 70.0])
    endpoints = subject_story._mesh_height(heights, np.array([start[:2], end[:2]]))
    assert (endpoints == 0).all()
    margins = subject_story._terrain_segment_margins(heights, start, end)
    assert margins.min() == pytest.approx(-30)


@pytest.mark.parametrize("shape", ["line", "area"])
@pytest.mark.parametrize("kind", ["building", "vegetation"])
def test_whole_geometry_checks_interior_occlusion(story, shape, kind):
    # East-facing camera looking down toward the origin of these offsets.
    # The small solid blocks the interior ray. No vertex ray hits it.
    center = (2644000, 1164000)
    coords = ((2644000, 1163900), (2644000, 1164100))
    vertices = coords if shape == "line" else (
        (2643900, 1163900), (2644100, 1163900), (2644000, 1164100),
    )
    extent = {"line_xy": vertices} if shape == "line" else {"area_triangles_xy": (vertices,)}
    feature = f3d.SubjectFeature("extent", "Ridge", *center, **extent)
    point_features = [f3d.SubjectFeature(f"end-{i}", "End", *point)
                      for i, point in enumerate(vertices)]
    options = {
        **story, "orbit": lambda target: TerrainOrbitRig(target, 1, 2000, 0, 0, theta_start_deg=30),
        "features": [feature, *point_features], "subject_id": feature.feature_id,
    }
    visible = f3d.compile_subject_shot(**options)
    assert visible.frame_checks[0]["features"]["extent"]["visible_pieces"] > 1

    blocker = f3d.StoryObstacle("middle", kind, (2644495, 1163995, 800, 2644505, 1164005, 950))
    from forge3d._subject_geometry import point_blocker

    eye = np.asarray(visible.frame_checks[0]["eye_xyz_m"])
    lo = np.array([-505, -1005, 799.5])
    hi = np.array([-495, -995, 949.5])
    for easting, northing in vertices:
        target = np.array([easting - 2645000, northing - 1165000, -0.5])
        assert point_blocker(eye, target, [(kind, lo, hi)], 0) is None
    with pytest.raises(f3d.SubjectShotInfeasible, match=f"frame 0: {kind} middle occludes part of extent"):
        f3d.compile_subject_shot(**options, obstacles=(blocker,))


def test_nonlinear_orbit_crosses_obstacle_between_clear_output_poses(story):
    orbit = lambda target: TerrainOrbitRig(target, 1, 1000, 0, 90, theta_start_deg=60)
    options = {**story, "orbit": orbit, "fps": 1, "hold_seconds": 2}
    baseline = f3d.compile_subject_shot(**options)
    assert baseline.frame_checks[1]["nonlinear_terrain_clearance_lower_bound_from_previous_m"] > 0
    assert baseline.frame_checks[2]["nonlinear_terrain_clearance_lower_bound_from_previous_m"] == baseline.frame_checks[2]["clearance_m"]
    # The straight chord is x+y=866, while the arc crosses x=y≈612.
    obstacle = f3d.StoryObstacle("stand", "vegetation", (2644590, 1164590, 450, 2644630, 1164630, 550))
    with pytest.raises(f3d.SubjectShotInfeasible, match=r"frames 0–1: nonlinear path .*vegetation stand"):
        f3d.compile_subject_shot(**options, obstacles=(obstacle,))


def test_between_frame_hermite_dip_violates_terrain_clearance(story, monkeypatch):
    animation = f3d.CameraAnimation()
    for t, radius in [(0, 120), (0.25, 20), (0.75, 20), (1, 120)]:
        animation.add_keyframe(t, 0, 0, radius, 55, target=(4000, 0, 4000))
    assert animation.evaluate(0.5).radius == pytest.approx(7.5)
    monkeypatch.setattr(TerrainOrbitRig, "bake", lambda *args, **kwargs: animation)
    with pytest.raises(f3d.SubjectShotInfeasible, match=r"frames 0–1: nonlinear path consumer terrain clearance"):
        f3d.compile_subject_shot(
            **{**story, "fps": 1, "hold_seconds": 1,
               "orbit": lambda target: TerrainOrbitRig(target, 1, 120, 0, 0,
                                                        clearance=TerrainClearance(minimum_height=10))},
        )


def test_building_clearance_at_output_frame(story):
    # The default orbit's frame-zero eye is directly above the registered anchor.
    obstacle = f3d.StoryObstacle("roof", "building", (2643990, 1163990, 1950, 2644010, 1164010, 2010))
    with pytest.raises(f3d.SubjectShotInfeasible, match="frame 0: consumer camera violates clearance of building roof"):
        f3d.compile_subject_shot(
            **{**story, "orbit": lambda target: TerrainOrbitRig(target, 1, 2000, 0, 0, theta_start_deg=0)},
            obstacles=(obstacle,),
        )


def test_extent_cannot_cross_reserved_text_with_clear_vertices():
    from forge3d._subject_geometry import intersects_rectangle

    reserve = (0.4, 0.4, 0.6, 0.6)
    assert intersects_rectangle(np.array([(0.2, 0.5), (0.8, 0.5)]), reserve)
    assert intersects_rectangle(np.array([(0.1, 0.8), (0.9, 0.8), (0.5, 0.1)]), reserve)
    assert not intersects_rectangle(np.array([(0.2, 0.2), (0.8, 0.2)]), reserve)


@pytest.mark.parametrize("vertices", [
    ((-80., -80., 0.), (-80., 80., 0.)),
    ((-90., -80., 0.), (-70., -80., 0.), (-80., 80., 0.)),
])
def test_native_mesh_ridge_hides_interior_of_draped_extent(vertices):
    from forge3d._subject_geometry import StoryMesh, whole_blocker

    heights = np.zeros((512, 512), dtype=np.float32)
    heights[255:257, 275:277] = 90.
    mesh = StoryMesh(heights, 511.)
    eye = np.array([80., 0., 140.])
    forward = np.array([-160., 0., -140.])
    forward /= np.linalg.norm(forward)
    target = np.array(vertices)
    eye_grid = np.array([*mesh.grid(eye[:2]), eye[2]])
    for vertex in target:
        end = np.array([*mesh.grid(vertex[:2]), vertex[2]])
        assert np.all(subject_story._terrain_segment_margins(heights, eye_grid, end) > 0)
    assert whole_blocker(mesh, eye, target, forward, 0.1, [], []) == "terrain"


def test_invalid_extent_does_not_silently_become_a_point(story):
    story["features"][0] = replace(story["features"][0], line_xy=((),))
    with pytest.raises(ValueError, match="requires a line"):
        f3d.compile_subject_shot(**story)


def test_solid_spanning_near_plane_still_occludes_extent():
    from forge3d._subject_geometry import StoryMesh, box_triangles, whole_blocker

    mesh = StoryMesh(np.full((512, 512), -10.), 511.)
    eye = np.zeros(3)
    forward = np.array([1., 0., 0.])
    lo = np.array([0.05, -10., -10.])
    hi = np.array([10., 10., 10.])
    faces = box_triangles(lo, hi, eye, forward, 0.1)
    target = np.array([[1., -0.2, 0.], [1., 0.2, 0.], [1., 0., 0.2]])
    assert whole_blocker(mesh, eye, target, forward, 0.1, [("solid", faces)], []) == "solid"


def test_safe_curved_orbit_is_not_rejected_by_its_colliding_chord(story):
    heights = np.zeros((512, 512), dtype=np.float32)
    # A hill on the diagonal chord, well inside the circle the consumer follows.
    vertex = round(4433 * 511 / 10000)
    heights[511 - vertex - 1:511 - vertex + 2, vertex - 1:vertex + 2] = 800
    np.save(story["terrain"].path, heights)
    shot = f3d.compile_subject_shot(
        **{**story, "fps": 1, "hold_seconds": 1,
           "orbit": lambda target: TerrainOrbitRig(target, 1, 1000, 0, 90, theta_start_deg=60)},
    )
    assert shot.frame_checks[1]["linear_eye_clearance_from_previous_m"] < 0
    assert shot.frame_checks[1]["nonlinear_terrain_clearance_lower_bound_from_previous_m"] > 0
