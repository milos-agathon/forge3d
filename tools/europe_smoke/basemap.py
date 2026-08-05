"""Engine configuration and the Europe hillshade render (S2).

build_base_map() takes NO bbox. The sentinel boundary written by boundary.py
is the only thing defining the map extent, the DEM crop and the nodata mask --
and the engine's default is California relation 165475, so the stub below is
not optional.

THE COS-LATITUDE PATCH (S2, gate 9) IS INSTALLED HERE.
=====================================================
Measured call chain (CPython 3.13 bytecode of examples/wildfire_smoke_engine
.cpython-313.pyc; every arrow is a LOAD_GLOBAL, so every one is rebindable):

  build_base_map(w, h, *, cache_dir, dem_zoom, max_dem_size, force)
    +- prepare_terrain_assets(...)                            LOAD_GLOBAL @2
    |    +- _download_osm_boundary(cache_dir, force=)
    |    +- _build_terrarium_dem(...) -> (path, bounds, span)  # NO shape
    |    +- _build_terrain_overlay(...)
    |    -> TerrainAssets(dem_path, overlay_path, boundary_path,
    |                     bounds_mercator, terrain_span_m)
    +- render_cpu_terrain_base(assets, w, h)                  LOAD_GLOBAL @62
         +- dem, overlay, bounds = _read_dem_and_overlay(assets)
         +- valid = np.isfinite(dem)
         +- if TERRAIN_HILLSHADE:
                return _render_nadir_hillshade_terrain(
                           dem, valid, bounds, w, h)           LOAD_GLOBAL @198
                  +- h = np.where(valid, dem, fill).astype(float32)
                  +- hs_fine  = _dem_hillshade(h, EXAG, AZ, ALT)      @184
                  +- hs_broad = _dem_hillshade(_pil_blur_float(h,3.2),
                                               EXAG*0.7, AZ, ALT+8)   @236

Why the patch is installed by wrapping the SHADING pass:

  * ``_dem_hillshade`` is named in exactly TWO code objects in the whole module
    -- the module body (STORE_NAME) and ``_render_nadir_hillshade_terrain``
    (LOAD_GLOBAL x2) -- verified by walking every nested code object of the
    module's own code object, so no class body, method, lambda or comprehension
    can be hiding a second caller.  A module-level rebind is therefore
    honoured.  But the replacement needs the DEM's mercator bounds and shape,
    which do not exist until the DEM is built, and ``build_base_map`` builds AND
    shades in one call.  There is no seam for a caller.  (This is the whole
    defect: an ``install_hillshade_patch(m, bounds, shape)`` can only be called
    at a moment when its own arguments are still unknown.)
  * ``_render_nadir_hillshade_terrain(dem, valid, bounds, width, height)``
    receives the DEM's exact EPSG:3857 bounds AND the exact array that is about
    to be shaded.  ``render_cpu_terrain_base`` passes the array straight through
    from ``_read_dem_and_overlay`` with no crop or resample, so ``dem.shape`` is
    the raster's shape and ``bounds`` is ``rasterio.open(dem_path).bounds``
    [verified against the cached z9 DEM: 3584x3584, bounds equal to rasterio's
    to the bit].
  * Wrapping ``_build_terrarium_dem`` instead would not do: its return is
    ``(Path, bounds, span)`` -- the shape is never returned, on either the cache
    -hit or the rebuild path -- so it would still need a separate rasterio open.
  * Deriving the geometry independently from bbox + zoom + max_dem_size is NOT
    reproducible.  ``_build_terrarium_dem`` takes the mercator bbox of the
    *boundary polygon*, snaps it out to whole tiles at the requested zoom,
    mosaics, and then rescales with ``scale = max(dem.shape)/max_size`` and two
    INDEPENDENT ``max(1, round(dem.shape[i]/scale))`` calls -- and only when the
    mosaic exceeds ``max_size``, which an outside reconstruction gets wrong on
    its first try (it did here: an unconditional rescale predicted 4000x4000 for
    a raster that is 3584x3584).  Any drift silently produces a wrong correction
    rather than an error.

Both ``_dem_hillshade`` call sites receive full-shape arrays (``h`` and a PIL
blur of ``h``, which preserves shape [verified at (7,11), (301,197) and
(3584,3584)]), so the closure's per-row spacing lines up on both.  The rebind is
scoped to the shading call and undone in a ``finally``, so no stale geometry can
survive into a later render -- verified across two consecutive renders in one
process (wrapper identity, state identity and engine restoration all stable,
shading_calls 1 -> 2, outputs identical).

THE FRAME FIT (S2, gate 10) IS INSTALLED HERE TOO.
====================================================
``_render_nadir_hillshade_terrain`` also owns the placement, and it is the ONLY
code object in the engine that reads any of TERRAIN_PORTRAIT_WIDTH_FRAC,
TERRAIN_PORTRAIT_TOP_PX, TERRAIN_FILL_FRAC, TERRAIN_CENTER_X_FRAC or
TERRAIN_CENTER_Y_FRAC [verified by walking every nested code object of every
module-level function].  Recovered from the bytecode, opcode for opcode:

    dem_h, dem_w = h.shape                       # h, so the DEM's own shape
    r0/r1/c0/c1  = bounding box of `valid`, grown by
                   pad = max(4, round(0.006 * max(dem_h, dem_w)))
    map_west  = west  + (east - west) * c0 / dem_w
    map_east  = west  + (east - west) * (c1 + 1) / dem_w
    map_north = north - (north - south) * r0 / dem_h
    map_south = north - (north - south) * (r1 + 1) / dem_h
    merc_w, merc_h = max(map_east - map_west, 1e-9), max(map_north - map_south, 1e-9)

    if height > width:                                       # PORTRAIT
        img_px_w = TERRAIN_PORTRAIT_WIDTH_FRAC * width
        img_px_h = img_px_w * merc_h / merc_w                # <- derived
        top_px   = TERRAIN_PORTRAIT_TOP_PX * width / 1080.0
        center_x, center_y = 0.5, (top_px + img_px_h / 2.0) / height
    else:                                                    # LANDSCAPE
        img_px_h = TERRAIN_FILL_FRAC * height
        img_px_w = img_px_h * merc_w / merc_h                # <- derived
        center_x, center_y = TERRAIN_CENTER_X_FRAC, TERRAIN_CENTER_Y_FRAC
    frame_rect = (center_x - img_px_w / width / 2, center_y - img_px_h / height / 2,
                  center_x + img_px_w / width / 2, center_y + img_px_h / height / 2)

Note what is NOT there: neither branch fits the map to BOTH canvas axes.  The
portrait branch fits the WIDTH and lets the height fall out of the domain
aspect; the landscape branch does the reverse.  So the scale fraction is the
only lever on the free axis, and its correct value depends on the domain aspect
-- which, exactly as with the hillshade geometry, does not exist until the DEM
has been built and masked.  ``configure()``'s fixed
``TERRAIN_PORTRAIT_WIDTH_FRAC = 1.0`` is therefore a bet that the delivered
domain is at least as wide-for-its-height as the canvas, and for this domain
that bet loses: measured at 1000x1060 / z5 / max_dem_size 1200 the crop spans
merc_w 10843214.150082307 by merc_h 11772036.151388682 (aspect 0.9210992908)
against a canvas aspect of 0.9433962264, and a full-width fit overflows to
``frame_rect (0.0, 0.0, 1.0, 1.0242068752610456)`` -- 1000/1060 / 0.9210992908
to the bit.  ``install_frame_fit_patch`` scopes a CONTAIN fit over the shading
call and restores the configured value in a ``finally``, and
``assert_frame_rect`` then checks the engine's own output rather than trusting
the reconstruction.
"""
from __future__ import annotations

import functools
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np

from . import boundary, config, hillshade

_ENGINE = None
_PATCH_ATTR = "_europe_smoke_hillshade_patch"
_FIT_ATTR = "_europe_smoke_frame_fit"


def load_engine():
    """Load the sourceless engine module. Requires CPython 3.13 exactly."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(
            f"engine bytecode requires CPython 3.13, running {sys.version_info[:2]}")
    loader = importlib.machinery.SourcelessFileLoader(
        "wildfire_engine", str(config.ENGINE_PYC))
    spec = importlib.util.spec_from_loader("wildfire_engine", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wildfire_engine"] = module
    loader.exec_module(module)
    _ENGINE = module
    return module


# --------------------------------------------------------------- gate 9 patch
def install_hillshade_patch(m) -> dict:
    """Make the cos-latitude hillshade actually run inside ``build_base_map``.

    Wraps ``_render_nadir_hillshade_terrain`` so that on every shading pass the
    module global ``_dem_hillshade`` is rebound to a replacement closed over the
    DEM bounds and shape the engine itself resolved, then restored afterwards.

    Idempotent. Returns the shared state dict, also reachable via
    :func:`patch_state`:

        shading_calls   -- times the wrapped shading pass ran
        bounds          -- mercator bounds of the last DEM shaded
        shape           -- (nrows, ncols) of the last DEM shaded
        hillshade       -- the last installed replacement; its ``.calls``
                           counter proves the replacement was INVOKED, not
                           merely installed
        engine_hillshade -- the original, restored after every shading pass
    """
    state = getattr(m, _PATCH_ATTR, None)
    if state is not None:
        return state

    original_render = m._render_nadir_hillshade_terrain
    original_hillshade = m._dem_hillshade
    state = {
        "shading_calls": 0,
        "bounds": None,
        "shape": None,
        "hillshade": None,
        "engine_hillshade": original_hillshade,
    }

    @functools.wraps(original_render)
    def _patched(dem, valid, bounds, width, height):
        # built BEFORE anything is mutated: a degenerate-geometry ValueError
        # here leaves the engine untouched rather than half-patched
        replacement = hillshade.make_coslat_hillshade(
            bounds, dem.shape, dem=dem, valid=valid,
            display_window=config.DISPLAY_WINDOW,
        )
        state["shading_calls"] += 1
        state["bounds"] = tuple(float(v) for v in bounds)
        state["shape"] = (int(dem.shape[0]), int(dem.shape[1]))
        state["hillshade"] = replacement
        m._dem_hillshade = replacement
        try:
            return original_render(dem, valid, bounds, width, height)
        finally:
            m._dem_hillshade = original_hillshade

    setattr(m, _PATCH_ATTR, state)
    m._render_nadir_hillshade_terrain = _patched
    return state


def patch_state(m) -> dict:
    """The hillshade patch's state dict; raises if it was never installed."""
    state = getattr(m, _PATCH_ATTR, None)
    if state is None:
        raise RuntimeError("cos-latitude hillshade patch is not installed on the engine")
    return state


def assert_hillshade_patch_ran(state, shading_calls_before: int) -> None:
    """Fail the render if the cos-latitude replacement did not actually execute.

    Raises ``AssertionError`` explicitly rather than using the ``assert``
    statement: this is a PRODUCTION gate, and ``python -O`` strips ``assert``
    outright.  A gate that evaporates under an interpreter flag is the same
    defect this whole patch exists to close -- a correction that is present in
    the source and absent from the run.  [measured: the statement form does not
    raise under ``-O`` on a patch that never executed.]

    NOTE: ``assert_frame_rect`` below has the identical weakness and is left as
    Task 12 wrote it; it belongs to the frame-rect finding, not this one.
    """
    if state["shading_calls"] != shading_calls_before + 1:
        raise AssertionError(
            "the cos-latitude hillshade wrapper did not run: shading_calls "
            f"{shading_calls_before} -> {state['shading_calls']}; either "
            "TERRAIN_HILLSHADE is false or the engine no longer routes through "
            "_render_nadir_hillshade_terrain"
        )
    shade = state["hillshade"]
    if shade is None or not getattr(shade, "is_coslat", False):
        raise AssertionError(
            "no cos-latitude hillshade was installed for the shading pass")
    if shade.calls < 2:
        raise AssertionError(
            f"the cos-latitude _dem_hillshade was installed but only invoked "
            f"{shade.calls} times (expected >= 2, for hs_fine and hs_broad); the "
            "engine no longer resolves _dem_hillshade from module globals"
        )
    if shade.shape != state["shape"]:
        raise AssertionError(
            f"the installed replacement was built for {shade.shape} but the "
            f"shading pass carried a {state['shape']} DEM")
    if shade.h1_m is None or shade.highland_share is None:
        raise AssertionError("the production hillshade ran without measured ground tiers")
    if not 0.18 <= shade.highland_share <= 0.24:
        raise AssertionError(f"highland share {shade.highland_share:.4f} is outside 0.18..0.24")


def hillshade_report(state, m) -> dict:
    """Gate-9 evidence for the build report (S10.2), measured on the real DEM.

    Task 15 Step 7 must record this; see the amended Step 6 snippet at the foot
    of this file -- the plan's original snippet prints only frame_rect and
    map_bounds_mercator, and fetch.py does not call basemap.render at all, so
    without that amendment this function has no production caller.

    Reflects the LAST shading pass only: ``state['hillshade']`` is overwritten
    per pass, so a process that renders two basemaps gets the more recent one.
    ``assert_hillshade_patch_ran`` uses a before/after delta and stays correct
    per render regardless.
    """
    shade = state["hillshade"]
    if shade is None:
        raise RuntimeError("no shading pass has run yet")
    metric = hillshade.worst_ruggedness_uniformity(
        shade.bounds, shade.shape, float(m.HILLSHADE_EXAGGERATION),
        float(m.HILLSHADE_AZIMUTH_DEG), float(m.HILLSHADE_ALTITUDE_DEG))
    return {
        "dem_shape": list(shade.shape),
        "dem_bounds_mercator": list(shade.bounds),
        "latitude_deg": [float(shade.latitudes[0]), float(shade.latitudes[-1])],
        "ground_metres_per_px": [float(shade.row_metres[0]), float(shade.row_metres[-1])],
        "uncorrected_ruggedness_bias": shade.latitude_span_ratio,
        "z_factor": float(m.HILLSHADE_EXAGGERATION),
        "h1_m": float(shade.h1_m),
        "highland_share": float(shade.highland_share),
        "lowland_relief": hillshade.LOWLAND_RELIEF,
        "shading_calls": state["shading_calls"],
        "hillshade_invocations": shade.calls,
        "gate9_rel_diff": metric["rel_diff"],
        "gate9": metric,
    }


# --------------------------------------------------------------- gate 10 fit
def engine_frame_geometry(valid, bounds) -> dict:
    """Reproduce the engine's data-extent crop and the map bounds it implies.

    Line-for-line from the ``_render_nadir_hillshade_terrain`` bytecode (see the
    module docstring).  Reproduces the engine's own ``map_bounds_mercator`` and
    ``frame_rect`` to the bit on a real shading pass -- pinned by
    ``test_the_frame_geometry_reconstruction_matches_the_engine_bytecode``, which
    drives the compiled shading pass on a synthetic DEM and compares.

    Reconstruction is unavoidable: the fit has to be applied BEFORE the engine
    reads the placement globals, and the engine computes these numbers inside
    the very call that reads them.  It is not trusted, though --
    :func:`assert_frame_rect` scores the engine's real output afterwards, so
    drift fails the render instead of silently mis-placing the map.
    """
    valid = np.asarray(valid)
    if valid.ndim != 2:
        raise ValueError(f"valid mask must be 2-D, got shape {valid.shape}")
    dem_h, dem_w = int(valid.shape[0]), int(valid.shape[1])
    rows_any = np.any(valid, axis=1)
    cols_any = np.any(valid, axis=0)
    if not bool(rows_any.any()):
        raise ValueError(
            "the DEM handed to the shading pass has no valid pixels, so it has "
            "no data extent to place; the boundary polygon does not intersect it")
    r0 = int(np.argmax(rows_any))
    r1 = int(len(rows_any) - np.argmax(rows_any[::-1]) - 1)
    c0 = int(np.argmax(cols_any))
    c1 = int(len(cols_any) - np.argmax(cols_any[::-1]) - 1)
    pad = max(4, int(round(0.006 * max(dem_h, dem_w))))
    r0, r1 = max(0, r0 - pad), min(dem_h - 1, r1 + pad)
    c0, c1 = max(0, c0 - pad), min(dem_w - 1, c1 + pad)

    # The parentheses are load-bearing. The engine divides FIRST and multiplies
    # second (BINARY_OP / then BINARY_OP * at 2728-2731); spelling it
    # ``(east - west) * c0 / dem_w`` associates the other way and lands one ULP
    # off, which is enough to make the reconstruction disagree with the engine.
    west, south, east, north = (float(v) for v in bounds)
    map_west = west + (east - west) * (c0 / dem_w)
    map_east = west + (east - west) * ((c1 + 1) / dem_w)
    map_north = north - (north - south) * (r0 / dem_h)
    map_south = north - (north - south) * ((r1 + 1) / dem_h)
    return {
        "shape": (dem_h, dem_w),
        "pad": pad,
        "crop": (r0, r1, c0, c1),
        "map_bounds": (map_west, map_south, map_east, map_north),
        "merc_w": max(map_east - map_west, 1e-9),
        "merc_h": max(map_north - map_south, 1e-9),
    }


def engine_frame_rect(merc_w: float, merc_h: float, width: int, height: int, *,
                      portrait_width_frac: float, portrait_top_px: float,
                      fill_frac: float, center_x: float, center_y: float
                      ) -> tuple[float, float, float, float]:
    """The engine's placement arithmetic as a pure function.

    ``center_x`` is ignored in the portrait branch: the engine spells it as the
    literal ``0.5`` there, so ``TERRAIN_CENTER_X_FRAC`` is a landscape-only
    lever [bytecode: ``LOAD_CONST 0.5`` at 2743 against ``LOAD_GLOBAL
    TERRAIN_CENTER_X_FRAC`` at 2747].

    As in :func:`engine_frame_geometry`, the associations are the engine's, not
    Python's default left-to-right: the aspect quotient and the ``width/1080``
    ratio are each formed BEFORE the multiply (BINARY_OP / then BINARY_OP * at
    2741, 2742 and 2746).
    """
    if height > width:
        img_px_w = portrait_width_frac * width
        img_px_h = img_px_w * (merc_h / merc_w)
        top_px = portrait_top_px * (width / 1080.0)
        cx, cy = 0.5, (top_px + img_px_h / 2.0) / height
    else:
        img_px_h = fill_frac * height
        img_px_w = img_px_h * (merc_w / merc_h)
        cx, cy = center_x, center_y
    fx_range, fy_range = img_px_w / width, img_px_h / height
    return (cx - fx_range / 2.0, cy - fy_range / 2.0,
            cx + fx_range / 2.0, cy + fy_range / 2.0)


def frame_rect_contained(frame_rect) -> bool:
    """Is the engine's frame rect inside the unit square? Gate 10's predicate."""
    x0, y0, x1, y1 = (float(v) for v in frame_rect)
    return 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


#: engine global -> the :func:`engine_frame_rect` keyword that carries it.
_FIT_KWARG = {"TERRAIN_PORTRAIT_WIDTH_FRAC": "portrait_width_frac",
              "TERRAIN_FILL_FRAC": "fill_frac"}
#: Backoff is for float rounding, so one step should always be enough. More
#: than a handful means the solve is wrong, not that the float is unlucky.
FIT_BACKOFF_STEPS = 8


def contain_fit(merc_w: float, merc_h: float, width: int, height: int, *,
                portrait_width_frac: float, portrait_top_px: float,
                fill_frac: float, center_x: float, center_y: float) -> dict:
    """The largest scale fraction that keeps the engine's frame_rect in [0,1].

    Shrink-only: a domain that already fits keeps the configured fraction, so
    this never silently enlarges the map beyond what ``configure()`` asked for.

    The closed-form limit is solved first and then CHECKED by re-running
    :func:`engine_frame_rect`, backing off by ``config.FRAME_FIT_MARGIN`` until
    it holds. The check is what makes the result trustworthy: the solve puts the
    binding edge exactly ON the unit square, and recomputing that edge through
    the engine's sequence of operations can land it a couple of ULPs outside.
    Backing off unconditionally instead would be worse -- it would shave the
    margin off the common case where the domain already fits and the limit is
    the exactly representable 1.0.

    Returns ``{"global", "configured", "value", "clamped", "limits", "branch",
    "frame_rect", "backoff_steps"}``; ``global`` names the engine attribute to
    rebind for the shading call.

    Raises ``ValueError`` when NO positive fraction can fit -- the placement
    offsets the engine applies before the scale (the portrait top inset, the
    landscape centre fractions) have then already left the canvas, and no
    rescaling of the map can bring them back. That is a genuine dead end for the
    requested canvas, and it is reported as one rather than papered over.
    """
    merc_w, merc_h = float(merc_w), float(merc_h)
    if not (merc_w > 0.0 and merc_h > 0.0):
        raise ValueError(f"degenerate domain extent merc_w={merc_w!r} merc_h={merc_h!r}")

    if height > width:
        branch = "portrait"
        top_px = float(portrait_top_px) * width / 1080.0
        if not 0.0 <= top_px < height:
            raise ValueError(
                f"TERRAIN_PORTRAIT_TOP_PX={portrait_top_px!r} puts the top of the "
                f"map at {top_px:.3f} px on a {width}x{height} canvas, which is "
                "outside it; the portrait branch pins frame_rect.y0 there before "
                "any scaling, so no width fraction can contain the frame")
        limits = {
            # center_x is the literal 0.5 in this branch, so the width fraction
            # IS the x extent of the frame
            "canvas_width": 1.0,
            "canvas_height": (height - top_px) * merc_w / (width * merc_h),
        }
        key, configured = "TERRAIN_PORTRAIT_WIDTH_FRAC", float(portrait_width_frac)
    else:
        branch = "landscape"
        limits = {
            "canvas_height": 2.0 * min(float(center_y), 1.0 - float(center_y)),
            "canvas_width": (2.0 * min(float(center_x), 1.0 - float(center_x))
                             * width * merc_h / (height * merc_w)),
        }
        key, configured = "TERRAIN_FILL_FRAC", float(fill_frac)

    allowed = min(limits.values())
    if allowed <= 0.0:
        raise ValueError(
            f"no positive {key} contains the frame on a {width}x{height} canvas "
            f"for a domain of aspect {merc_w / merc_h:.6f}: the binding limit is "
            f"{allowed!r} from {limits!r}. TERRAIN_CENTER_X_FRAC/"
            "TERRAIN_CENTER_Y_FRAC place the map centre outside the canvas.")

    placement = dict(portrait_width_frac=float(portrait_width_frac),
                     portrait_top_px=float(portrait_top_px),
                     fill_frac=float(fill_frac),
                     center_x=float(center_x), center_y=float(center_y))
    kwarg = _FIT_KWARG[key]
    value, steps = min(configured, allowed), 0
    while steps <= FIT_BACKOFF_STEPS:
        placement[kwarg] = value
        rect = engine_frame_rect(merc_w, merc_h, width, height, **placement)
        if frame_rect_contained(rect):
            break
        value *= 1.0 - config.FRAME_FIT_MARGIN
        steps += 1
    else:
        raise ValueError(
            f"{key} did not converge to a contained frame_rect on a "
            f"{width}x{height} canvas for a domain of aspect "
            f"{merc_w / merc_h:.6f}: the closed-form limit {allowed!r} still "
            f"produces {rect!r} after {FIT_BACKOFF_STEPS} backoff steps, so the "
            "solve and engine_frame_rect disagree by more than rounding")
    return {
        "global": key,
        "branch": branch,
        "configured": configured,
        "value": value,
        "clamped": value < configured,
        "limits": limits,
        "frame_rect": rect,
        "backoff_steps": steps,
        "domain_aspect": merc_w / merc_h,
        "canvas_aspect": width / float(height),
    }


def install_frame_fit_patch(m) -> dict:
    """Contain the engine's frame_rect in the canvas, on every shading pass.

    Same seam and same shape as :func:`install_hillshade_patch`, and for the
    same reason: the value depends on the DEM's masked extent, which exists only
    once the engine is inside ``_render_nadir_hillshade_terrain``.  The fitted
    scale fraction is rebound on the module for the duration of that call and
    restored in a ``finally``, so ``configure()``'s postcondition -- the values a
    reader sees on the module -- is exactly what ``configure()`` set.

    Idempotent. Returns the shared state dict, also reachable via
    :func:`frame_fit_state`:

        fits     -- times the wrapped shading pass ran through this fit
        fit      -- the last :func:`contain_fit` result
        geometry -- the last :func:`engine_frame_geometry` result
    """
    state = getattr(m, _FIT_ATTR, None)
    if state is not None:
        return state

    original_render = m._render_nadir_hillshade_terrain
    state = {"fits": 0, "fit": None, "geometry": None}

    @functools.wraps(original_render)
    def _fitted(dem, valid, bounds, width, height):
        # solved BEFORE anything is mutated: a ValueError here leaves the engine
        # exactly as configure() left it
        geometry = engine_frame_geometry(valid, bounds)
        fit = contain_fit(
            geometry["merc_w"], geometry["merc_h"], width, height,
            portrait_width_frac=float(m.TERRAIN_PORTRAIT_WIDTH_FRAC),
            portrait_top_px=float(m.TERRAIN_PORTRAIT_TOP_PX),
            fill_frac=float(m.TERRAIN_FILL_FRAC),
            center_x=float(m.TERRAIN_CENTER_X_FRAC),
            center_y=float(m.TERRAIN_CENTER_Y_FRAC))
        state["fits"] += 1
        state["geometry"] = geometry
        state["fit"] = fit
        previous = getattr(m, fit["global"])
        setattr(m, fit["global"], fit["value"])
        try:
            return original_render(dem, valid, bounds, width, height)
        finally:
            setattr(m, fit["global"], previous)

    setattr(m, _FIT_ATTR, state)
    m._render_nadir_hillshade_terrain = _fitted
    return state


def frame_fit_state(m) -> dict:
    """The frame-fit patch's state dict; raises if it was never installed."""
    state = getattr(m, _FIT_ATTR, None)
    if state is None:
        raise RuntimeError("frame-fit patch is not installed on the engine")
    return state


def assert_frame_fit_ran(state, fits_before: int) -> None:
    """Fail the render if the frame fit did not run for this shading pass.

    ``assert_frame_rect`` scores the OUTCOME, which is the stronger check -- but
    only for a domain that actually overflows.  A domain that happens to fit
    would pass containment with the fit silently orphaned, and the next domain
    would not.  This asserts the mechanism ran; both gates are cheap.

    Explicit ``raise``, not the ``assert`` statement, which ``python -O``
    strips: see :func:`assert_hillshade_patch_ran`.
    """
    if state["fits"] != fits_before + 1:
        raise AssertionError(
            "the frame-fit wrapper did not run: fits "
            f"{fits_before} -> {state['fits']}; either TERRAIN_HILLSHADE is "
            "false or the engine no longer routes through "
            "_render_nadir_hillshade_terrain")
    fit = state["fit"]
    if not fit or fit.get("value", 0.0) <= 0.0:
        raise AssertionError(f"the frame fit produced no usable scale: {fit!r}")


# --------------------------------------------------------------- engine config
def configure(m, width: int, height: int) -> None:
    """Point the engine at our domain and defeat its country-scale defaults."""
    lon_min, lat_min, lon_max, lat_max = config.BASEMAP_WINDOW
    m.LON_MIN, m.LAT_MIN, m.LON_MAX, m.LAT_MAX = lon_min, lat_min, lon_max, lat_max
    m.OSM_RELATION_ID = config.SENTINEL_RELATION_ID
    m.USER_AGENT = config.USER_AGENT
    m.WIDTH, m.HEIGHT = width, height

    # the fetcher owns the boundary; never let the engine re-download it
    m._download_osm_boundary = (
        lambda cache_dir, force=False: m._osm_boundary_path(cache_dir))

    # height > width takes the PORTRAIT branch, where the first two are live and
    # the engine's country-scale defaults (0.96 of the width, inset 56 px at
    # 1080) both shrink and offset the map for no reason we share.
    #
    # These are the FULL-CANVAS request, not the final placement.  The engine
    # scales one canvas axis by these and derives the other from the domain
    # aspect, so a fixed 1.0 contains the frame only when the domain is at least
    # as wide-for-its-height as the canvas.  This domain is not: measured at
    # 1000x1060 / z5 the delivered crop has aspect 0.9210992908 against a canvas
    # 0.9433962264, and 1.0 overflows to frame_rect.y1 = 1.0242068752610456.
    # install_frame_fit_patch shrinks whichever fraction is live to the contain
    # fit for the shading call it is in, and restores these values after.
    m.TERRAIN_PORTRAIT_WIDTH_FRAC = 1.0
    m.TERRAIN_PORTRAIT_TOP_PX = 0
    m.TERRAIN_CENTER_X_FRAC = 0.5
    # ... and the LANDSCAPE branch centre, which the engine leaves at 0.505.
    # Off-centre by half a percent costs 1% of the map to the contain fit, and
    # nothing here wants the engine's California composition.
    m.TERRAIN_CENTER_Y_FRAC = 0.5
    m.TERRAIN_FILL_FRAC = 1.0

    m.TERRAIN_HILLSHADE = True
    # water-flat detection dies at continental coarseness (0.43% -> 0.00%);
    # sea comes from the OSM land mask instead
    m.WATER_FLAT_RANGE_M = -1.0
    # pixel kernels grow with m/cell at this scale
    m.TERRAIN_AO_STRENGTH = 0.30
    m.TERRAIN_CLARITY = 0.45

    # S2 gate 9. install_hillshade_patch changes what HILLSHADE_EXAGGERATION
    # MEANS -- index-unit slope per pixel becomes a dimensionless z-factor -- so
    # the value has to be ours and named, never the engine's coincidental 5.5.
    install_hillshade_patch(m)
    m.HILLSHADE_EXAGGERATION = config.HILLSHADE_Z_FACTOR

    # S2 gate 10. Installed AFTER the hillshade patch, so the fit is the outer
    # wrapper and the placement globals are already rebound when the shading
    # pass reads them. Both are idempotent, so re-configuring is a no-op.
    install_frame_fit_patch(m)


def assert_frame_rect(projector) -> None:
    """Fail the render if the engine's frame rect leaves the unit square.

    MERGE NOTE (systemic issue (a)): raises ``AssertionError`` explicitly
    rather than using the ``assert`` statement, which ``python -O`` strips.
    The hillshade fix deliberately left this as Task 12 wrote it, on the
    grounds that it belongs to the frame-rect owner; nobody owns it, and an
    optimised build would then accept the exact portrait overflow
    ``configure()`` exists to defeat. Same exception type and same message, so
    ``pytest.raises(AssertionError, match="frame_rect")`` is unchanged.

    This is also the check on ``install_frame_fit_patch``: the fit is solved
    from a RECONSTRUCTION of the engine's crop arithmetic, and this scores the
    engine's own ``frame_rect``. Drift between the two fails the render.
    """
    if not frame_rect_contained(projector.frame_rect):
        raise AssertionError(
            f"frame_rect {projector.frame_rect} is not contained in [0,1]"
        )


def render(width: int, height: int, cache_dir: Path, dem_zoom: int = 7,
           max_dem_size: int = 4000, force: bool = False):
    """Render the Europe nadir hillshade. Returns (PIL image, projector)."""
    cache_dir = Path(cache_dir)
    m = load_engine()
    configure(m, width, height)
    boundary.build_land_union(cache_dir, force=force)

    state = patch_state(m)
    before = state["shading_calls"]
    fit_state = frame_fit_state(m)
    fits_before = fit_state["fits"]

    base, projector = m.build_base_map(
        width, height, cache_dir=cache_dir, dem_zoom=dem_zoom,
        max_dem_size=max_dem_size, force=force)

    assert_frame_rect(projector)
    assert_frame_fit_ran(fit_state, fits_before)
    assert_hillshade_patch_ran(state, before)
    return base, projector
