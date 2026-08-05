import numpy as np
import pytest

from tools.europe_smoke import basemap, config, hillshade

# Every test that calls basemap.load_engine() reads config.ENGINE_PYC, which is
# the git-ignored [engine] artifact. The `engine_artifact` fixture (conftest)
# skips with the manifest's own Finding when it is not on this machine, so a
# clean checkout reports un-provisioned rather than broken. A .pyc that is
# PRESENT but wrong still runs the test and fails it.


def test_engine_loads_from_the_pinned_bytecode(engine_artifact):
    m = basemap.load_engine()
    assert hasattr(m, "build_base_map")
    assert isinstance(m.SMOKE_TRANSFER, dict)
    assert m.SMOKE_TRANSFER["od_coeff"] == 0.46


def test_configure_installs_the_sentinel_and_stubs_the_downloader(engine_artifact):
    m = basemap.load_engine()
    basemap.configure(m, 4000, 4241)
    assert m.OSM_RELATION_ID == config.SENTINEL_RELATION_ID
    # the stub must return our path without touching the network
    p = m._download_osm_boundary(config.CACHE_DIR)
    assert str(config.SENTINEL_RELATION_ID) in str(p)


def test_configure_uses_the_widened_basemap_window(engine_artifact):
    m = basemap.load_engine()
    basemap.configure(m, *config.BASEMAP_SIZE)
    assert (m.LON_MIN, m.LAT_MIN, m.LON_MAX, m.LAT_MAX) == config.BASEMAP_WINDOW


def test_configure_defeats_the_portrait_branch_defaults(engine_artifact):
    m = basemap.load_engine()
    basemap.configure(m, 4000, 4241)
    # height > width takes the PORTRAIT branch, where these are live
    assert m.TERRAIN_PORTRAIT_WIDTH_FRAC == 1.0
    assert m.TERRAIN_PORTRAIT_TOP_PX == 0
    assert m.TERRAIN_CENTER_X_FRAC == 0.5


def test_assert_frame_rect_rejects_an_overflowing_rect():
    class P:
        frame_rect = (0.0, 0.0, 1.04, 1.0)
    with pytest.raises(AssertionError, match="frame_rect"):
        basemap.assert_frame_rect(P())


def test_assert_frame_rect_accepts_a_contained_rect():
    class P:
        frame_rect = (0.0, 0.0, 1.0, 1.0)
    basemap.assert_frame_rect(P())


# ------------------------------------------------------- gate 9, production
def test_configure_installs_the_hillshade_patch_on_the_shading_pass(engine_artifact):
    """The patch must be wired into the ONLY function that calls
    _dem_hillshade, not left as a helper nobody invokes."""
    m = basemap.load_engine()
    basemap.configure(m, 400, 500)
    state = basemap.patch_state(m)
    assert state["shading_calls"] == 0
    # the seam is the shading pass, and it must be ours
    assert m._render_nadir_hillshade_terrain.__wrapped__ is not None
    # ... while _dem_hillshade is left ALONE outside a shading pass: there is no
    # DEM geometry then, so a bound replacement would be undefined
    assert m._dem_hillshade is state["engine_hillshade"]


def test_configure_owns_the_z_factor_because_the_patch_changes_its_units(engine_artifact):
    m = basemap.load_engine()
    basemap.configure(m, 400, 500)
    assert m.HILLSHADE_EXAGGERATION == config.HILLSHADE_Z_FACTOR


def test_install_hillshade_patch_is_idempotent(engine_artifact):
    m = basemap.load_engine()
    basemap.configure(m, 400, 500)
    wrapper, state = m._render_nadir_hillshade_terrain, basemap.patch_state(m)
    basemap.configure(m, 400, 500)
    basemap.install_hillshade_patch(m)
    assert m._render_nadir_hillshade_terrain is wrapper
    assert basemap.patch_state(m) is state


def test_the_wrapper_installs_a_replacement_built_from_the_ENGINES_geometry():
    """Drive the seam directly, on a stand-in module: the wrapper must hand
    _dem_hillshade a closure made from the bounds and shape it was CALLED with,
    and restore the original afterwards."""
    import types

    seen = {}

    def fake_render(dem, valid, bounds, width, height):
        seen["hillshade"] = fake._dem_hillshade
        # exactly what the engine's shading pass does, both call sites
        fake._dem_hillshade(dem, 5.5, 315.0, 34.0)
        fake._dem_hillshade(dem, 5.5 * 0.7, 315.0, 42.0)
        return ("image", "projector")

    def engine_hillshade(h, exaggeration, azimuth_deg, altitude_deg):
        raise AssertionError("the engine's index-unit hillshade must not run")

    fake = types.SimpleNamespace(_render_nadir_hillshade_terrain=fake_render,
                                 _dem_hillshade=engine_hillshade)
    state = basemap.install_hillshade_patch(fake)
    before = state["shading_calls"]

    dem = np.zeros((64, 96), np.float32)
    bounds = (-1.0e6, 3.5e6, 1.0e6, 1.2e7)
    fake._render_nadir_hillshade_terrain(dem, dem > -1, bounds, 400, 500)

    shade = seen["hillshade"]
    assert shade.is_coslat
    assert shade.shape == (64, 96)
    assert shade.bounds == bounds
    assert shade.calls == 2
    assert state["shape"] == (64, 96) and state["bounds"] == bounds
    basemap.assert_hillshade_patch_ran(state, before)
    # scoped: outside the shading pass the engine's own function is back
    assert fake._dem_hillshade is engine_hillshade


def test_the_wrapper_restores_the_engine_hillshade_even_on_failure():
    import types

    def boom(dem, valid, bounds, width, height):
        raise RuntimeError("shading blew up")

    def engine_hillshade(h, e, a, b):
        return None

    fake = types.SimpleNamespace(_render_nadir_hillshade_terrain=boom,
                                 _dem_hillshade=engine_hillshade)
    basemap.install_hillshade_patch(fake)
    with pytest.raises(RuntimeError, match="blew up"):
        fake._render_nadir_hillshade_terrain(np.zeros((8, 8), np.float32), None,
                                             (-1e6, 3.5e6, 1e6, 1.2e7), 4, 5)
    assert fake._dem_hillshade is engine_hillshade


def test_the_engine_still_routes_through_the_seam_this_patch_wraps(engine_artifact):
    """The whole fix rests on two facts about the compiled engine. Assert them
    against the bytecode so a new .pyc cannot silently orphan the patch."""
    import inspect
    import types as t

    m = basemap.load_engine()

    def codes(fn):
        # unwrap first: our own wrapper stores m._dem_hillshade, so it carries
        # that name in co_names and would satisfy the assertion spuriously
        fn = inspect.unwrap(fn)
        stack, out = [fn.__code__], []
        while stack:
            c = stack.pop()
            out.append(c)
            stack += [k for k in c.co_consts if hasattr(k, "co_code")]
        return out

    def namers(target):
        return {n for n, o in vars(m).items()
                if isinstance(o, t.FunctionType)
                and inspect.unwrap(o).__module__ == m.__name__
                and any(target in c.co_names for c in codes(o))}

    # 1. _dem_hillshade is called from exactly one place
    assert namers("_dem_hillshade") == {"_render_nadir_hillshade_terrain"}
    # 2. and that place is reached by a rebindable global lookup
    assert namers("_render_nadir_hillshade_terrain") == {"render_cpu_terrain_base"}
    import dis
    ops = {i.opname for i in dis.get_instructions(m.render_cpu_terrain_base)
           if i.argval == "_render_nadir_hillshade_terrain"}
    assert ops == {"LOAD_GLOBAL"}
    inner = inspect.unwrap(m._render_nadir_hillshade_terrain)
    ops = {i.opname for i in dis.get_instructions(inner) if i.argval == "_dem_hillshade"}
    assert ops == {"LOAD_GLOBAL"}
    # ... and it looks it up TWICE: hs_fine and hs_broad
    assert sum(1 for i in dis.get_instructions(inner)
               if i.argval == "_dem_hillshade") == 2
    # 3. the shading pass takes the DEM and its bounds, which is why the patch
    #    can be built from the engine's own geometry
    assert list(inspect.signature(inner).parameters) == [
        "dem", "valid", "bounds", "width", "height"]


def test_assert_hillshade_patch_ran_rejects_a_patch_that_never_executed():
    state = {"shading_calls": 3, "bounds": None, "shape": None, "hillshade": None,
             "engine_hillshade": None}
    with pytest.raises(AssertionError, match="wrapper did not run"):
        basemap.assert_hillshade_patch_ran(state, 3)


def test_assert_hillshade_patch_ran_rejects_an_uninvoked_replacement():
    shade = hillshade.make_coslat_hillshade((-1e6, 3.5e6, 1e6, 1.2e7), (64, 64))
    state = {"shading_calls": 1, "bounds": None, "shape": (64, 64),
             "hillshade": shade, "engine_hillshade": None}
    with pytest.raises(AssertionError, match="installed but only invoked"):
        basemap.assert_hillshade_patch_ran(state, 0)


def test_the_production_gate_survives_python_dash_O():
    """ADDED. `python -O` strips the `assert` STATEMENT outright. A production
    gate written with one exists in the source and not in the run -- the same
    class of defect as a patch nobody installs, which is what this task fixes.

    [measured: the previous revision of assert_hillshade_patch_ran printed
    SILENT here -- it did not raise on a patch that never executed.]
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    src = textwrap.dedent("""
        import sys
        sys.path.insert(0, sys.argv[1])
        from tools.europe_smoke import basemap
        assert not __debug__, "the subprocess must be running under -O"
        state = {"shading_calls": 3, "bounds": None, "shape": None,
                 "hillshade": None, "engine_hillshade": None}
        try:
            basemap.assert_hillshade_patch_ran(state, 3)
        except AssertionError:
            print("RAISED")
        else:
            print("SILENT")
    """)
    out = subprocess.run([sys.executable, "-O", "-c", src, str(root)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "RAISED", (
        "assert_hillshade_patch_ran is a no-op under -O; it must raise "
        f"AssertionError explicitly. stdout={out.stdout!r} stderr={out.stderr!r}")


# ------------------------------------------------------ gate 10, the frame fit
def _placement(m):
    return dict(portrait_width_frac=float(m.TERRAIN_PORTRAIT_WIDTH_FRAC),
                portrait_top_px=float(m.TERRAIN_PORTRAIT_TOP_PX),
                fill_frac=float(m.TERRAIN_FILL_FRAC),
                center_x=float(m.TERRAIN_CENTER_X_FRAC),
                center_y=float(m.TERRAIN_CENTER_Y_FRAC))


def test_configure_centres_the_landscape_branch_too(engine_artifact):
    """configure() pinned TERRAIN_CENTER_X_FRAC and left Y at the engine's
    California 0.505, which is the live centre whenever width >= height."""
    m = basemap.load_engine()
    basemap.configure(m, 400, 500)
    assert m.TERRAIN_CENTER_Y_FRAC == 0.5


@pytest.mark.parametrize("width,height", [(400, 500), (500, 400)])
def test_the_frame_geometry_reconstruction_matches_the_engine_bytecode(
        engine_artifact, width, height):
    """The fit is solved from a RECONSTRUCTION of the engine's crop and
    placement maths, so the reconstruction has to be pinned to the bytecode.
    Drives the compiled shading pass on a small synthetic DEM (~20 ms, no
    network, no download) and compares both outputs, in BOTH branches."""
    import inspect

    m = basemap.load_engine()
    basemap.configure(m, width, height)
    inner = inspect.unwrap(m._render_nadir_hillshade_terrain)

    dem = np.random.default_rng(0).normal(300.0, 120.0, (48, 61)).astype(np.float32)
    dem[:5], dem[-9:] = np.nan, np.nan          # an off-centre data extent, so
    dem[:, :3], dem[:, -14:] = np.nan, np.nan   # the crop is not the whole DEM
    valid = np.isfinite(dem)
    bounds = (-5009377.085697312, 2504688.542848652,
              7514065.628545968, 15028131.257091932)

    _, projector = inner(dem, valid, bounds, width, height)
    geom = basemap.engine_frame_geometry(valid, bounds)
    assert geom["crop"] != (0, dem.shape[0] - 1, 0, dem.shape[1] - 1)
    # to the bit, both of them: a last-ULP drift here is what a wrong operator
    # association looks like, and it is the only thing standing between the fit
    # and a silently mis-placed map
    assert geom["map_bounds"] == tuple(projector.map_bounds_mercator)
    assert basemap.engine_frame_rect(geom["merc_w"], geom["merc_h"], width, height,
                                     **_placement(m)) == tuple(projector.frame_rect)


def test_the_contain_fit_shrinks_a_portrait_domain_taller_than_the_canvas():
    """The measured Europe case: a 0.9210992908 domain in a 0.9433962264
    canvas. [merc_w/merc_h read off the delivered 1000x1060 / z5 /
    max_dem_size 1200 render; the whole rect is reproduced to the bit]"""
    merc_w, merc_h = 10843214.150082307, 11772036.151388684
    placement = dict(portrait_width_frac=1.0, portrait_top_px=0, fill_frac=1.0,
                     center_x=0.5, center_y=0.5)
    # unfitted, this is the exact overflow being fixed
    assert basemap.engine_frame_rect(merc_w, merc_h, 1000, 1060,
                                     **placement)[3] == 1.0242068752610456

    fit = basemap.contain_fit(merc_w, merc_h, 1000, 1060, **placement)
    assert fit["global"] == "TERRAIN_PORTRAIT_WIDTH_FRAC" and fit["clamped"]
    assert fit["value"] == pytest.approx(0.9763652482269503, rel=1e-7)
    placement["portrait_width_frac"] = fit["value"]
    rect = basemap.engine_frame_rect(merc_w, merc_h, 1000, 1060, **placement)
    assert basemap.frame_rect_contained(rect)
    assert rect == fit["frame_rect"]
    assert rect[3] == 1.0                # fitted exactly, not letterboxed
    assert fit["backoff_steps"] == 0     # the closed form landed it, no nudging


def test_the_contain_fit_leaves_a_domain_that_already_fits_alone():
    """Shrink-only: a wide domain keeps configure()'s full-width request."""
    fit = basemap.contain_fit(2.0e7, 1.0e7, 1000, 1060, portrait_width_frac=1.0,
                              portrait_top_px=0, fill_frac=1.0, center_x=0.5,
                              center_y=0.5)
    assert fit["value"] == 1.0 and fit["clamped"] is False


def test_the_contain_fit_handles_the_landscape_branch_and_its_offcentre_default():
    merc_w, merc_h = 10843214.150082307, 11772036.151388684
    placement = dict(portrait_width_frac=1.0, portrait_top_px=0, fill_frac=1.0,
                     center_x=0.585, center_y=0.505)     # the engine's own values
    assert basemap.engine_frame_rect(merc_w, merc_h, 1920, 1080, **placement)[3] > 1.0

    fit = basemap.contain_fit(merc_w, merc_h, 1920, 1080, **placement)
    assert fit["global"] == "TERRAIN_FILL_FRAC" and fit["branch"] == "landscape"
    placement["fill_frac"] = fit["value"]
    x0, y0, x1, y1 = basemap.engine_frame_rect(merc_w, merc_h, 1920, 1080, **placement)
    assert basemap.frame_rect_contained((x0, y0, x1, y1))
    # x has slack on this canvas, so 0.505 off-centre is what binds y: 2*0.495
    assert y1 - y0 == pytest.approx(0.99, rel=1e-7)


def test_the_contain_fit_refuses_a_canvas_no_scale_can_rescue():
    """A top inset past the canvas, or a centre off it, is not a scale problem;
    it must raise rather than return a fraction that cannot work."""
    with pytest.raises(ValueError, match="TERRAIN_PORTRAIT_TOP_PX"):
        basemap.contain_fit(1.0e7, 1.1e7, 1000, 1060, portrait_width_frac=1.0,
                            portrait_top_px=1080 * 2, fill_frac=1.0,
                            center_x=0.5, center_y=0.5)
    with pytest.raises(ValueError, match="no positive TERRAIN_FILL_FRAC"):
        basemap.contain_fit(1.0e7, 1.1e7, 1920, 1080, portrait_width_frac=1.0,
                            portrait_top_px=0, fill_frac=1.0,
                            center_x=1.0, center_y=0.5)


def test_the_frame_fit_is_scoped_to_the_shading_call_and_idempotent():
    """configure()'s postcondition must survive a render: the fitted value is
    live only inside the shading pass."""
    import types

    seen = {}

    def fake_render(dem, valid, bounds, width, height):
        seen["width_frac"] = fake.TERRAIN_PORTRAIT_WIDTH_FRAC
        return ("image", "projector")

    fake = types.SimpleNamespace(
        _render_nadir_hillshade_terrain=fake_render,
        TERRAIN_PORTRAIT_WIDTH_FRAC=1.0, TERRAIN_PORTRAIT_TOP_PX=0,
        TERRAIN_FILL_FRAC=1.0, TERRAIN_CENTER_X_FRAC=0.5, TERRAIN_CENTER_Y_FRAC=0.5)
    state = basemap.install_frame_fit_patch(fake)
    wrapper = fake._render_nadir_hillshade_terrain
    assert basemap.install_frame_fit_patch(fake) is state
    assert fake._render_nadir_hillshade_terrain is wrapper
    assert basemap.frame_fit_state(fake) is state

    dem = np.zeros((64, 96), np.float32)
    before = state["fits"]
    fake._render_nadir_hillshade_terrain(
        dem, dem > -1, (-1.0e6, 3.5e6, 1.0e6, 1.2e7), 400, 500)

    assert seen["width_frac"] < 1.0                      # live during the pass
    assert fake.TERRAIN_PORTRAIT_WIDTH_FRAC == 1.0       # restored after it
    assert seen["width_frac"] == state["fit"]["value"]
    assert state["geometry"]["shape"] == (64, 96)
    basemap.assert_frame_fit_ran(state, before)


def test_the_frame_fit_restores_the_configured_fraction_even_on_failure():
    import types

    def boom(dem, valid, bounds, width, height):
        raise RuntimeError("shading blew up")

    fake = types.SimpleNamespace(
        _render_nadir_hillshade_terrain=boom,
        TERRAIN_PORTRAIT_WIDTH_FRAC=1.0, TERRAIN_PORTRAIT_TOP_PX=0,
        TERRAIN_FILL_FRAC=1.0, TERRAIN_CENTER_X_FRAC=0.5, TERRAIN_CENTER_Y_FRAC=0.5)
    basemap.install_frame_fit_patch(fake)
    with pytest.raises(RuntimeError, match="blew up"):
        fake._render_nadir_hillshade_terrain(
            np.zeros((8, 8), np.float32), np.ones((8, 8), bool),
            (-1e6, 3.5e6, 1e6, 1.2e7), 4, 5)
    assert fake.TERRAIN_PORTRAIT_WIDTH_FRAC == 1.0


def test_engine_frame_geometry_rejects_an_all_nodata_dem():
    with pytest.raises(ValueError, match="no valid pixels"):
        basemap.engine_frame_geometry(np.zeros((8, 8), bool),
                                      (-1e6, 3.5e6, 1e6, 1.2e7))


def test_assert_frame_fit_ran_rejects_a_fit_that_never_executed():
    with pytest.raises(AssertionError, match="did not run"):
        basemap.assert_frame_fit_ran({"fits": 3, "fit": None}, 3)


def test_both_frame_gates_survive_python_dash_O():
    """Same defect class as test_the_production_gate_survives_python_dash_O:
    assert_frame_rect and assert_frame_fit_ran are PRODUCTION gates, and the
    `assert` STATEMENT evaporates under -O."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    src = textwrap.dedent("""
        import sys
        sys.path.insert(0, sys.argv[1])
        from tools.europe_smoke import basemap
        assert not __debug__, "the subprocess must be running under -O"

        class P:
            frame_rect = (0.0, 0.0, 1.0, 1.0242068752610456)

        for label, call in (("frame_rect", lambda: basemap.assert_frame_rect(P())),
                            ("frame_fit", lambda: basemap.assert_frame_fit_ran(
                                {"fits": 3, "fit": None}, 3))):
            try:
                call()
            except AssertionError:
                print(label + "=RAISED")
            else:
                print(label + "=SILENT")
    """)
    out = subprocess.run([sys.executable, "-O", "-c", src, str(root)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["frame_rect=RAISED", "frame_fit=RAISED"], (
        f"stdout={out.stdout!r} stderr={out.stderr!r}")


@pytest.fixture(scope="module")
def rendered():
    """One low-zoom render shared by every slow test: it costs minutes.

    The artifact gate is called rather than requested: ``engine_artifact`` is
    function-scoped, and a module-scoped fixture asking for it is a
    ScopeMismatch that ERRORS all three tests before the render is attempted.
    :func:`require_artifacts` is the same gate the fixture wraps -- same live
    manifest check, same skip reason -- so nothing is loosened by calling it
    here, and it also covers the OSM land polygons ``boundary.build_land_union``
    reads inside ``basemap.render``.
    """
    from tools.europe_smoke.tests.conftest import require_artifacts

    require_artifacts("engine", "osm_land")
    img, projector = basemap.render(1000, 1060, config.CACHE_DIR, dem_zoom=5,
                                    max_dem_size=1200)
    m = basemap.load_engine()
    return img, projector, m, basemap.patch_state(m)


@pytest.mark.slow
def test_render_produces_a_contained_frame_rect_and_covers_the_display_window(rendered):
    _, projector, _, _ = rendered
    basemap.assert_frame_rect(projector)
    x0, y0, x1, y1 = projector.map_bounds_mercator
    import math

    def merc_x(lon):
        return math.radians(lon) * 6378137.0

    def merc_y(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6378137.0

    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    assert x0 <= merc_x(lon_min) and x1 >= merc_x(lon_max)
    assert y0 <= merc_y(lat_min) and y1 >= merc_y(lat_max)


@pytest.mark.slow
def test_the_cos_latitude_hillshade_actually_ran_in_the_render(rendered):
    """The finding this test exists for: the patch must have EXECUTED, on the
    real DEM, not merely have been defined.

    The lat 30..72 assertion is now derivable rather than hoped for: the z7
    tile snap of the AREA bbox gives tiles x 50..84 / y 18..55, a 9728x8960
    mosaic rescaled to (4000, 3684), spanning 78.059N..21.955N [measured; the
    same derivation reproduces the cached Greece z9 DEM's rasterio bounds to
    1e-6]. Both inequalities hold with 6.1 / 8.0 degrees to spare.
    """
    _, _, m, state = rendered
    shade = state["hillshade"]
    assert state["shading_calls"] == 1
    assert shade.is_coslat and shade.calls >= 2       # hs_fine + hs_broad
    # it corrected the DEM the engine actually shaded, at that DEM's geometry
    assert shade.shape == state["shape"] and shade.bounds == state["bounds"]
    assert shade.row_metres.size == state["shape"][0]
    # ... and that geometry is the display window's, not some default box
    lat_n, lat_s = shade.latitudes[0], shade.latitudes[-1]
    _, want_s, _, want_n = config.DISPLAY_WINDOW
    assert lat_n >= want_n and lat_s <= want_s, (lat_n, lat_s)
    assert m._dem_hillshade is state["engine_hillshade"]   # rebind was scoped


@pytest.mark.slow
def test_gate_9_on_the_real_rendered_dem(rendered):
    """Gate 9 against the geometry PRODUCTION captured, not a synthetic box."""
    _, _, m, state = rendered
    report = basemap.hillshade_report(state, m)
    assert report["hillshade_invocations"] >= 2
    assert report["uncorrected_ruggedness_bias"] > 1.0
    assert report["gate9_rel_diff"] < config.HILLSHADE_UNIFORMITY_MAX, report
