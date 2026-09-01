#!/usr/bin/env python3
"""Copenhagen city map lit by the PROMETHEUS terrain path tracer (nadir plate).

PT variant of examples/osm_city_demo.py. forge3d's path tracer only traces
heightfields, not triangle meshes, so the buildings are rasterized into an
extruded footprint heightfield (max height per cell) and that surface is lit by
forge3d.path_tracing.hybrid_render_terrain_reference — a converged, tiled GPU
path trace under a NW sun + neutral sky. The result is real soft global-
illumination shadows in the streets and cool sky-lit shadow tones that no
shadow map or baked AO can produce.

Pipeline (mirrors examples/swiss_landcover_pt_3d.py):
1. city.build_city_scene -> surfaces (light-free polygons) + roof_outlines
   (footprint polygon + building height)
2. build_building_heightfield -> extruded footprint grid, ground = 0
3. PT light field: tiled quasi-ortho nadir trace, gray albedo, neutral sky
4. build_lightfree_overlay -> top-down Copenhagen palette, NO baked light
5. modulate the overlay by the PT light field (floor+gain darken + cool tint)
6. city.compose_poster -> circular disc + title + attribution

This is a NADIR top-down plate: depth comes from the physically shaded relief,
not from an oblique camera. Buildings read as lit blocks separated by
path-traced street shadows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import osm_city_demo as city  # noqa: E402

from forge3d.path_tracing import hybrid_render_terrain_reference  # noqa: E402

OUT_DIR = city.PROJECT_ROOT / "examples" / "out" / "osm_city_pt"

# --- PT scene parameters -----------------------------------------------------
SPAN_X = 100.0            # world span of the grid's long axis (matches swiss)
# Vertical exaggeration of the [0,1]-normalized building heightfield. Horizontal
# extent is 2*radius meters over SPAN_X world units (~20 m/unit at r=1000), so
# RELIEF_WORLD ~6 makes a normalized-1.0 (tallest, ~80 m) building read tall
# enough to throw legible street shadows without cliff-like caricature. TUNE.
RELIEF_WORLD = 6.0
CAMERA_FOV_Y = 4.0       # near-orthographic nadir (parallax < ~0.25%, flatter disc)
CAMERA_MARGIN = 1.06
# Rotate the whole scene (geometry + heightfield) to match the CPU vnext
# orientation: Lakes/harbor band across the top, Kongens Have park upper-right.
# Verified by overlay preview against copenhagen_cpu_8k_vnext.png (2026-07-12).
MAP_ROTATION_DEG = 135.0
# PT sun convention (map-maker, measured 2026-07-12): light compass =
# sun_azimuth_deg + 90. With MAP_ROTATION_DEG applied the compass frame turns
# with the geometry, so re-derive via --probe. Verified: azimuth 180 -> SE
# screen shadows with MAP_ROTATION_DEG=135 (90 gave NE).
# ALWAYS verify with --probe before trusting; do not eyeball.
SUN_AZIMUTH = 180.0
# Lighting dimension: lower sun -> longer, more sculptural street shadows that
# reveal massing; a real HDRI dome replaces the analytic sky for richer,
# directional fill and subtle chromatic variation in the shadows.
SUN_ELEVATION = 26.0
SUN_INTENSITY = 2.6
ENV_INTENSITY = 0.92     # HDRI dome fill; lifts the shadowed street floors
ENV_HDR_PATH = city.PROJECT_ROOT / "assets" / "hdri" / "snow_field_1k.hdr"
PT_ALBEDO = (0.62, 0.62, 0.62)
# Finer heightfield raster (technical dimension): rasterize footprints at
# RASTER_SUPERSAMPLE x the trace grid, then area-downsample (BOX) to the trace
# grid. Sub-cell soft building edges -> the ~1 m stepped shadow edges largely
# vanish. Replaces the roof-relief experiment, which read as noise at 1 m/cell.
RASTER_SUPERSAMPLE = 2
# Footprint erosion (m) before extrusion: shrinks each building so touching
# same-height neighbours get a thin valley between them -> the GI casts a
# separating shadow, breaking the heightfield "plateau merge".
FOOTPRINT_EROSION_M = 0.7

# --- light-field -> overlay modulation ---------------------------------------
# Optional soft highlight shoulder, as a knee position in 0..1. None keeps the
# historical hard clip. It matters only when TERRAIN_FLOOR + TERRAIN_GAIN > 1.0,
# i.e. when the sun is allowed to ADD light rather than merely fail to remove
# it: a hard clip then flattens the brightest roofs to solid white and destroys
# exactly the relief the extra range was bought for.
HIGHLIGHT_KNEE = None

SHADE_LOW_PCT = 1.0
SHADE_HIGH_PCT = 99.5
SHADE_GAMMA = 0.90       # <1 lifts midtones
TERRAIN_FLOOR = 0.58     # airier plate; floor + gain <= 1.0
TERRAIN_GAIN = 0.42
# vnext-match grade: the PT's physical cool sky-lit shadows diverge from the CPU
# vnext's warm look, so most of the cool chromatic tint is dropped and a warm
# white balance is applied so shadows read as warm taupe, not blue.
LIGHT_TINT_STRENGTH = 0.25
LIGHT_TINT_CLAMP = (0.82, 1.18)
WARM_BALANCE = (1.03, 1.00, 0.94)  # per-channel final multiplier (warm)
FILMIC = True            # ACES-ish rolloff on the light field for tonal richness
COOL_SHADOW = 0.06       # whisper of cool reintroduced in the DEEPEST shadows only
# Shade-weighted chromatic split: sunlight warmer, sky-lit shadow cooler.
#
# WARM_BALANCE is GLOBAL - it multiplies every pixel by the same per-channel
# constant - so it cannot produce a lit/shadow HUE difference at all, only an
# overall cast. COOL_SHADOW is one-sided (it can cool the dark end but never
# warm the light end) and is weighted by 1-shade, so it is spent almost entirely
# on the deepest 10% of the plate.
#
# Measured on the delivered LoD2 plates the cue was not merely absent, it was
# INVERTED: Prague read B/R 0.826 in the sun against 0.555 in shadow.
#
# The cause is CONTENT, not the grade, and that was settled by measuring the
# RAW light-free overlay - which has no light in it at all - under the same two
# masks: lit (189.8, 198.1, 185.8) B/R 0.979, shadow (209.8, 197.6, 160.7) B/R
# 0.766, i.e. the palette ALONE already carries -0.214 of the -0.270 the
# finished plate shows. With the sun at 26 deg and 2.2x vertical exaggeration a
# shadow runs about 4.5 building-heights, so "shadow" lands overwhelmingly on
# warm ochre ROOF classes while "lit" is open ground - parks, wide roads, water.
# No per-pixel chromatic term can undo a correlation between what is in shadow
# and what colour that thing happens to be; only shorter shadows (a higher sun,
# less exaggeration) or a narrower roof ramp can.
#
# So this lever is a partial, honest correction, not a cure: it buys back a
# measured +0.09 of the -0.21 the palette starts with.
#
# This term is differential: blue is scaled DOWN above SHADE_SPLIT_PIVOT and UP
# below it (with a quarter-weight opposite move on red), so it costs no
# luminance separation and no overall warmth. BLUE carries it because blue is
# the channel with headroom at both ends - red is already deep in the highlight
# shoulder wherever the sun is, so pushing red there buys nothing.
WARM_SHADE_SPLIT = 0.0   # 0 = off (Copenhagen); the LoD2 plates set 0.28
SHADE_SPLIT_PIVOT = 0.55
# Water pixels of the last overlay built by paint_ground_surfaces, so
# _modulate_overlay can hold them out of the split (see there for why).
# A module side-channel rather than an argument because BOTH overlay builders -
# Copenhagen's tier ramp and the LoD2 height ramp - go through
# paint_ground_surfaces, while main() has no reason to know about water at all.
_WATER_MASK: np.ndarray | None = None
# Water painted lighter/desaturated to approach vnext's near-white sky-lit water
# (the flat-ground modulation still darkens it, so paint it bright).
WATER_OVERLAY_RGB = (0xC2, 0xD4, 0xDC)

# --- palette dimension (3) ---------------------------------------------------
# Muted natural green for parks/greenery (the shared 0x74B474 reads cartoonish).
PARK_OVERLAY_RGB = (0x8E, 0xA6, 0x7A)
# Sky-tint water treatment: a gentle lightness ramp toward the light (screen NW)
# plus a whisper of blue, so water reads as a reflective plane, not flat fill.
WATER_GRADIENT = 0.10
WATER_BLUE_LIFT = 7
# Unified grade on the composed plate: a small pull toward luma tames any
# over-saturation into one harmony, with a gentle contrast for cohesion.
SAT_PULL = 0.07
UNIFY_CONTRAST = 1.045

# --- readability pass (composition dimension) --------------------------------
# Road hierarchy + casing: minor roads mid-gray with a darker casing, major
# roads (the CPU "road_hi" crown) near-white. The casing (an outward buffer
# painted first) gives every road a crisp dark edge so the network traces.
# Softened: lighter casing, less-stark major roads, thinner casing so the
# network still traces but no longer dominates (closer to the reference).
ROAD_CASING_RGB = (0xAC, 0xB2, 0xB8)
ROAD_MINOR_RGB = (0xC7, 0xCC, 0xD0)
ROAD_MAJOR_RGB = (0xDE, 0xE2, 0xE5)
ROAD_CASING_M = 1.2          # casing width (m) beyond the road on each side
ROAD_MAJOR_CASING_M = 1.6
# Building tier ramp: wider value separation than the shared palette so building
# HEIGHT reads through color (light cream -> warm sand -> deep ochre); landmark
# deepened from salmon to a true brick/terracotta accent.
TIER_RGB = {
    "low": (0xDE, 0xD4, 0xBF),
    "mid": (0xC6, 0xB1, 0x89),
    "high": (0xA4, 0x8B, 0x63),
    "landmark": (0xBE, 0x60, 0x42),
}
# Per-building edge stroke: paint the full footprint in a darkened tier color,
# then the eroded footprint in the tier color -> a thin dark ring that separates
# adjacent blocks (also helps the heightfield "plateau merge" read as distinct).
BUILDING_EDGE_DARKEN = 0.74
BUILDING_EDGE_M = 0.9        # erosion (m) => stroke width
# Final crispness pass on the composited plate.
UNSHARP = (1.6, 55, 2)       # (radius, percent, threshold)
# Extra keyword arguments for the final Image.save. Empty here (Copenhagen ships
# a screen-resolution PNG); the LoD2 print plates set dpi + an sRGB ICC profile
# through it so the saved file carries its own physical size and colour space.
SAVE_KWARGS: dict = {}
LIGHT_CACHE_KEY = ""


def light_cache_signature(args: argparse.Namespace) -> tuple[str, dict]:
    """Identify every input that changes the traced light field.

    The old cache stored pixels only, so it could not safely be reused after a
    sun or sampler change.  Keep the signature small and JSON-stable; geometry
    and vertical-scale changes are rejected by the LoD2 driver before a
    canonical output is written.
    """
    payload = {
        "schema": 1,
        "sun_azimuth_deg": float(SUN_AZIMUTH),
        "sun_elevation_deg": float(SUN_ELEVATION),
        "sun_intensity": float(SUN_INTENSITY),
        "env_intensity": float(ENV_INTENSITY),
        "env_hdr_path": str(ENV_HDR_PATH),
        "span_x": float(SPAN_X),
        "camera_fov_y": float(CAMERA_FOV_Y),
        "camera_margin": float(CAMERA_MARGIN),
        "map_rotation_deg": float(MAP_ROTATION_DEG),
        "pt_albedo": tuple(float(v) for v in PT_ALBEDO),
        "raster_supersample": int(RASTER_SUPERSAMPLE),
        "footprint_erosion_m": float(FOOTPRINT_EROSION_M),
        "lon": float(args.lon),
        "lat": float(args.lat),
        "radius_m": float(args.radius),
        "grid_max": int(args.grid_max),
        "frame": int(args.frame),
        "tiles": int(args.tiles),
        "spp": int(args.spp),
        "max_frames": int(args.max_frames),
        "min_frames": int(args.min_frames),
        "variance_threshold": float(args.variance_threshold),
        "seed": int(args.seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, default=12.56553)
    parser.add_argument("--lat", type=float, default=55.67594)
    parser.add_argument("--radius", type=float, default=1000.0)
    parser.add_argument("--size", type=int, nargs=2, default=(3840, 2160), metavar=("W", "H"))
    parser.add_argument("--overlay-size", type=int, default=4096)
    parser.add_argument("--grid-max", type=int, default=2048)
    parser.add_argument("--frame", type=int, default=1120)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--min-frames", type=int, default=32)
    parser.add_argument("--variance-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "copenhagen_pt.png")
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--reuse-shade", action="store_true")
    parser.add_argument(
        "--reuse-legacy-shade", action="store_true",
        help="explicitly reuse an unfingerprinted field; output is marked retrace-required",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Fast low-res trace + shadow-direction quadrant probe, then exit.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# geometry -> pixel / grid helpers
# ---------------------------------------------------------------------------


def _rot(geom):
    """Rotate a geometry into the poster's map orientation (about the AOI center)."""
    if geom is None or abs(MAP_ROTATION_DEG) < 1e-6:
        return geom
    return affinity.rotate(geom, MAP_ROTATION_DEG, origin=(0.0, 0.0))


def _polys(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    if isinstance(geom, Polygon):
        return [geom]
    return []


def _to_px(xy: np.ndarray, half_extent_m: float, size: int) -> list[tuple[float, float]]:
    """Meters (x=east, y=north, origin=center) -> image pixels (north up)."""
    t = size / (2.0 * half_extent_m)
    px = (xy[:, 0] + half_extent_m) * t
    py = (half_extent_m - xy[:, 1]) * t
    return list(zip(px.tolist(), py.tolist()))


def _paint(image: Image.Image, geom, rgba, half_extent_m: float) -> None:
    """Paint a (multi)polygon (exterior minus holes) onto an RGBA image."""
    size = image.size[0]
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    painted = False
    for poly in _polys(geom):
        draw.polygon(_to_px(np.asarray(poly.exterior.coords, dtype=np.float64), half_extent_m, size), fill=255)
        painted = True
        for interior in poly.interiors:
            draw.polygon(_to_px(np.asarray(interior.coords, dtype=np.float64), half_extent_m, size), fill=0)
    if painted:
        image.paste(Image.new("RGBA", image.size, tuple(rgba)), mask=mask)


def _apply_water_sky(image: Image.Image, water_mask: Image.Image) -> None:
    """Sky-tint water: a gentle lightness ramp toward the light (screen NW) plus a
    whisper of blue, so water reads as a reflective plane rather than flat fill."""
    ys, xs = np.nonzero(np.asarray(water_mask, dtype=bool))
    if ys.size == 0:
        return
    size = image.size[0]
    ramp = ((size - 1 - xs) + (size - 1 - ys)) / (2.0 * (size - 1))  # 1 at NW, 0 at SE
    factor = (1.0 - WATER_GRADIENT * 0.5) + WATER_GRADIENT * ramp.astype(np.float32)
    arr = np.asarray(image, dtype=np.uint8).copy()
    px = arr[ys, xs, :3].astype(np.float32) * factor[:, None]
    px[:, 2] += WATER_BLUE_LIFT
    arr[ys, xs, :3] = np.clip(px, 0.0, 255.0).astype(np.uint8)
    image.paste(Image.fromarray(arr, "RGBA"))


def build_building_heightfield(scene: "city.SceneLayers", half_extent_m: float, grid: int) -> np.ndarray:
    """Rasterize building footprints into an extruded height grid (meters).

    Draw-tallest-last approximates a per-cell max: sorting by height ascending
    means a taller neighbour overwrites a shorter one where they overlap.
    Ground (streets, water, parks) stays 0.
    """
    hf = Image.new("F", (grid, grid), 0.0)
    draw = ImageDraw.Draw(hf)
    outlines = sorted(scene.roof_outlines, key=lambda ro: float(ro.elevation))
    for ro in outlines:
        h = float(ro.elevation)
        geom = _rot(ro.geometry)
        if FOOTPRINT_EROSION_M > 0.0 and geom is not None and not geom.is_empty:
            eroded = geom.buffer(-FOOTPRINT_EROSION_M)
            if not eroded.is_empty:  # keep tiny buildings that would erode away
                geom = eroded
        for poly in _polys(geom):
            draw.polygon(
                _to_px(np.asarray(poly.exterior.coords, dtype=np.float64), half_extent_m, grid),
                fill=h,
            )
            for interior in poly.interiors:
                draw.polygon(
                    _to_px(np.asarray(interior.coords, dtype=np.float64), half_extent_m, grid),
                    fill=0.0,
                )
    return np.asarray(hf, dtype=np.float32)


def paint_ground_surfaces(scene: "city.SceneLayers", half_extent_m: float, size: int) -> Image.Image:
    """Everything under the buildings: base, landuse, parks, water, roads, rails.

    Split out of build_lightfree_overlay so an alternative building pass (e.g.
    the LoD2 height-ramp plate in city_lod2_plate) can reuse the identical,
    already-tuned ground treatment instead of duplicating it and drifting.
    """
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    road_rgb = tuple(city.COLORS["road"][:3])
    roadhi_rgb = tuple(city.COLORS["road_hi"][:3])
    water_rgb = tuple(city.COLORS["water"][:3])
    park_rgb = tuple(city.COLORS["park"][:3])
    water_mask = Image.new("L", (size, size), 0)
    wm_draw = ImageDraw.Draw(water_mask)
    # Ground surfaces (insertion order), with road hierarchy + casing, light water
    # (sky-tinted below), and muted greens. Casing = an outward buffer under fill.
    for surface in scene.surfaces:
        g = _rot(surface.geometry)
        if g is None or g.is_empty:
            continue
        srgb = tuple(surface.rgba[:3])
        if srgb == road_rgb:
            # Casing width 0 skips the casing entirely rather than painting a
            # zero-width buffer under the fill: buffer(0) returns the same
            # geometry, so it would cost a full-size paste to draw nothing.
            if ROAD_CASING_M > 0.0:
                _paint(image, g.buffer(ROAD_CASING_M), ROAD_CASING_RGB + (255,), half_extent_m)
            _paint(image, g, ROAD_MINOR_RGB + (255,), half_extent_m)
        elif srgb == roadhi_rgb:
            if ROAD_MAJOR_CASING_M > 0.0:
                _paint(image, g.buffer(ROAD_MAJOR_CASING_M), ROAD_CASING_RGB + (255,), half_extent_m)
            _paint(image, g, ROAD_MAJOR_RGB + (255,), half_extent_m)
        elif srgb == water_rgb:
            _paint(image, g, WATER_OVERLAY_RGB + (255,), half_extent_m)
            for poly in _polys(g):
                wm_draw.polygon(_to_px(np.asarray(poly.exterior.coords, dtype=np.float64), half_extent_m, size), fill=255)
                for interior in poly.interiors:
                    wm_draw.polygon(_to_px(np.asarray(interior.coords, dtype=np.float64), half_extent_m, size), fill=0)
        elif srgb == park_rgb:
            _paint(image, g, PARK_OVERLAY_RGB + (255,), half_extent_m)
        else:
            _paint(image, g, srgb + (255,), half_extent_m)
    _apply_water_sky(image, water_mask)
    global _WATER_MASK
    _WATER_MASK = np.asarray(water_mask, dtype=bool)
    return image


def build_lightfree_overlay(scene: "city.SceneLayers", half_extent_m: float, size: int) -> Image.Image:
    """Top-down Copenhagen palette as an RGBA disc, with NO baked lighting.

    Paint in scene insertion order (elevation is a 3D offset in the CPU
    renderer, not a draw order), then building footprints colored by tier.
    """
    image = paint_ground_surfaces(scene, half_extent_m, size)
    # Buildings: tier-ramped fill with a per-footprint dark edge stroke. Direct
    # ImageDraw (no per-building full-size mask) keeps the 8192 overlay fast.
    draw = ImageDraw.Draw(image)
    for ro in scene.roof_outlines:
        tcol = TIER_RGB[city.building_bin(float(ro.elevation))]
        ecol = tuple(int(round(c * BUILDING_EDGE_DARKEN)) for c in tcol)
        for poly in _polys(_rot(ro.geometry)):
            draw.polygon(
                _to_px(np.asarray(poly.exterior.coords, dtype=np.float64), half_extent_m, size),
                fill=ecol + (255,),
            )
            inner = poly.buffer(-BUILDING_EDGE_M)
            for ip in _polys(inner):
                draw.polygon(
                    _to_px(np.asarray(ip.exterior.coords, dtype=np.float64), half_extent_m, size),
                    fill=tcol + (255,),
                )
    return image


# ---------------------------------------------------------------------------
# PT light field (verbatim PROMETHEUS block from swiss_landcover_pt_3d.py)
# ---------------------------------------------------------------------------


def _neutral_sky_env(height: int = 32, width: int = 64) -> np.ndarray:
    zenith = np.array([0.62, 0.72, 0.92], dtype=np.float32) * 1.1
    horizon = np.array([1.00, 0.99, 0.96], dtype=np.float32) * 0.85
    ground = np.array([0.28, 0.28, 0.29], dtype=np.float32)
    rows = np.linspace(1.0, -1.0, height, dtype=np.float32)
    env_rows = np.empty((height, 3), dtype=np.float32)
    up = np.clip(rows, 0.0, 1.0)[:, None] ** 0.65
    env_rows[:] = horizon[None, :] * (1.0 - up) + zenith[None, :] * up
    below = rows < 0.0
    down = np.clip(-rows[below], 0.0, 1.0)[:, None] ** 0.5
    env_rows[below] = horizon[None, :] * (1.0 - down) + ground[None, :] * down
    return np.repeat(env_rows[:, None, :], width, axis=1)


def _read_radiance_hdr(path) -> np.ndarray:
    """Minimal Radiance .hdr (RGBE) reader: flat and new-style RLE scanlines.
    Ported from examples/forest_cover_copernicus/italy_forest_pt_3d.py."""
    data = Path(path).read_bytes()
    if not data.startswith(b"#?"):
        raise ValueError(f"Not a Radiance HDR file: {path}")
    pos = data.index(b"\n\n") + 2
    dim_end = data.index(b"\n", pos)
    dims = data[pos:dim_end].split()
    if dims[0] != b"-Y" or dims[2] != b"+X":
        raise ValueError(f"Unsupported HDR orientation: {data[pos:dim_end]!r}")
    height, width = int(dims[1]), int(dims[3])
    pos = dim_end + 1
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        if (width >= 8 and width < 32768 and data[pos] == 2 and data[pos + 1] == 2
                and (data[pos + 2] << 8 | data[pos + 3]) == width):
            pos += 4
            for c in range(4):
                x = 0
                while x < width:
                    count = data[pos]
                    pos += 1
                    if count > 128:
                        rgbe[y, x: x + count - 128, c] = data[pos]
                        pos += 1
                        x += count - 128
                    else:
                        rgbe[y, x: x + count, c] = np.frombuffer(data, dtype=np.uint8, count=count, offset=pos)
                        pos += count
                        x += count
        else:
            row = np.frombuffer(data, dtype=np.uint8, count=width * 4, offset=pos)
            rgbe[y] = row.reshape(width, 4)
            pos += width * 4
    exp = rgbe[:, :, 3].astype(np.int32)
    scale = np.where(exp == 0, 0.0, np.ldexp(1.0, exp - 136)).astype(np.float32)
    return rgbe[:, :, :3].astype(np.float32) * scale[:, :, None]


def _load_hdr_env(path, target_h: int = 64) -> np.ndarray:
    env = _read_radiance_hdr(path)
    h, w = env.shape[:2]
    step = max(1, h // target_h)
    env = env[: (h // step) * step, : (w // step) * step]
    env = env.reshape(h // step, step, w // step, step, 3).mean(axis=(1, 3))
    lum = env @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    env = env / max(float(lum.mean()), 1e-6)  # mean-luminance -> 1 so ENV_INTENSITY keeps meaning
    return np.ascontiguousarray(env, dtype=np.float32)


_ENV_CACHE = None


def _env_map() -> np.ndarray:
    """Real HDRI dome (cached), falling back to the analytic sky if unreadable."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        try:
            _ENV_CACHE = _load_hdr_env(ENV_HDR_PATH)
            print(f"[PT] env: HDRI {ENV_HDR_PATH.name} {_ENV_CACHE.shape[1]}x{_ENV_CACHE.shape[0]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[PT] env: HDRI load failed ({exc}); analytic sky")
            _ENV_CACHE = _neutral_sky_env()
    return _ENV_CACHE


def _camera_for_grid(rows: int, cols: int, *, tiles: int = 1, tx: int = 0, ty: int = 0) -> dict:
    span_z = SPAN_X * rows / cols
    half_extent = 0.5 * max(SPAN_X, span_z) * CAMERA_MARGIN
    distance = half_extent / math.tan(math.radians(CAMERA_FOV_Y / 2.0))
    tile_half = half_extent / tiles
    center_x = -half_extent + (tx + 0.5) * 2.0 * tile_half
    center_z = -half_extent + (ty + 0.5) * 2.0 * tile_half
    fov_y = math.degrees(2.0 * math.atan(math.tan(math.radians(CAMERA_FOV_Y / 2.0)) / tiles))
    return {
        "origin": (center_x, distance, center_z),
        "look_at": (center_x, 0.0, center_z),
        "up": (0.0, 0.0, -1.0),
        "fov_y": fov_y,
        "exposure": 1.0,
    }


def _normalize_heightfield(hf: np.ndarray, grid_max: int) -> np.ndarray:
    data = hf.astype(np.float32)
    scale = min(1.0, grid_max / float(max(data.shape)))
    if scale < 1.0:
        new_size = (max(2, round(data.shape[1] * scale)), max(2, round(data.shape[0] * scale)))
        # BOX = area averaging: anti-aliases the hard building-edge cliffs into
        # sub-cell ramps so shadow edges stop stair-stepping at the trace grid.
        data = np.asarray(
            Image.fromarray(data, mode="F").resize(new_size, Image.Resampling.BOX),
            dtype=np.float32,
        )
    data = data - data.min()
    data = data / max(float(data.max()), 1e-6)
    return np.ascontiguousarray(data, dtype=np.float32)


def _pt_pass(grid_hf, frame, camera, args, *, label="full"):
    rows, cols = grid_hf.shape
    spacing = SPAN_X / (cols - 1)
    out = hybrid_render_terrain_reference(
        grid_hf,
        frame,
        frame,
        camera,
        spacing=(spacing, spacing),
        exaggeration=RELIEF_WORLD,
        albedo=PT_ALBEDO,
        sun_azimuth_deg=SUN_AZIMUTH,
        sun_elevation_deg=SUN_ELEVATION,
        sun_intensity=SUN_INTENSITY,
        env_map=_env_map(),
        env_intensity=ENV_INTENSITY,
        spp=int(args.spp),
        max_frames=int(args.max_frames),
        min_frames=int(args.min_frames),
        variance_threshold=float(args.variance_threshold),
        seed=int(args.seed),
    )
    print(
        f"[PT:{label}] converged={out['converged']} frames={out['frames']} "
        f"variance={out['variance']:.3e} "
        f"peak_host_visible={out.get('peak_host_visible_bytes', 0) / 2**20:.1f} MiB"
    )
    rgba = out["rgba"].astype(np.float32) / 255.0
    hit = np.isfinite(out["depth"])
    rgb = np.where(hit[:, :, None], rgba[:, :, :3], np.nan).astype(np.float32)
    return rgb, hit


def _pt_light_field(grid_hf, frame, args):
    rows, cols = grid_hf.shape
    tiles = max(1, int(args.tiles))
    if tiles == 1:
        return _pt_pass(grid_hf, frame, _camera_for_grid(rows, cols), args)
    rgb = np.full((tiles * frame, tiles * frame, 3), np.nan, dtype=np.float32)
    hit = np.zeros((tiles * frame, tiles * frame), dtype=bool)
    for ty in range(tiles):
        for tx in range(tiles):
            camera = _camera_for_grid(rows, cols, tiles=tiles, tx=tx, ty=ty)
            tile_rgb, tile_hit = _pt_pass(grid_hf, frame, camera, args, label=f"tile {tx},{ty}")
            rgb[ty * frame : (ty + 1) * frame, tx * frame : (tx + 1) * frame] = tile_rgb
            hit[ty * frame : (ty + 1) * frame, tx * frame : (tx + 1) * frame] = tile_hit
    if not hit.any():
        raise RuntimeError("Path tracer produced no terrain hits — check camera framing")
    return rgb, hit


def _shade_on_overlay_grid(rgb_field, hit, overlay_size):
    ys, xs = np.nonzero(hit)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    crop = rgb_field[top:bottom, left:right, :]
    lum = crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    finite = np.isfinite(lum)
    lum = np.where(finite, lum, float(np.nanmedian(lum)))
    low = float(np.percentile(lum[finite], SHADE_LOW_PCT))
    high = float(np.percentile(lum[finite], SHADE_HIGH_PCT))
    shade = np.clip((lum - low) / max(high - low, 1e-6), 0.0, 1.0)
    if FILMIC:
        # ACES-ish filmic rolloff for richer mids + soft highlight shoulder,
        # renormalized so full light still maps to 1.
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        shade = (shade * (a * shade + b)) / (shade * (c * shade + d) + e)
        shade = np.clip(shade / ((a + b) / (c + d + e)), 0.0, 1.0)
    shade = np.power(shade, SHADE_GAMMA, dtype=np.float32)

    tint = crop / np.maximum(lum, 1e-4)[:, :, None]
    tint = np.where(finite[:, :, None], tint, 1.0)
    lit = shade > 0.75
    if lit.any():
        anchor = np.median(tint[lit], axis=0)
        tint = tint / np.maximum(anchor[None, None, :], 1e-4)
    tint = np.clip(tint, LIGHT_TINT_CLAMP[0], LIGHT_TINT_CLAMP[1]).astype(np.float32)

    def _resize(field: np.ndarray) -> np.ndarray:
        img = Image.fromarray(field, mode="F").resize(overlay_size, Image.Resampling.BICUBIC)
        return np.asarray(img, dtype=np.float32)

    shade_hi = np.clip(_resize(shade), 0.0, 1.0)
    tint_hi = np.stack([_resize(np.ascontiguousarray(tint[:, :, c])) for c in range(3)], axis=-1)
    tint_hi = np.clip(tint_hi, LIGHT_TINT_CLAMP[0], LIGHT_TINT_CLAMP[1])
    return shade_hi, tint_hi


def _modulate_overlay(overlay: Image.Image, shade: np.ndarray, tint: np.ndarray) -> Image.Image:
    arr = np.asarray(overlay.convert("RGBA"), dtype=np.uint8).copy()
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    scale = TERRAIN_FLOOR + TERRAIN_GAIN * shade
    tint_mix = 1.0 + LIGHT_TINT_STRENGTH * (tint - 1.0)
    warm = np.asarray(WARM_BALANCE, dtype=np.float32)
    rgb = rgb * scale[:, :, None] * tint_mix * warm[None, None, :]
    if COOL_SHADOW > 0.0:
        # whisper of cool ONLY in the deepest shadows (chromatic depth) while the
        # plate stays warm overall.
        depth = np.clip(1.0 - shade, 0.0, 1.0)[:, :, None]
        cool = np.array([1.0 - COOL_SHADOW, 1.0, 1.0 + 0.6 * COOL_SHADOW], dtype=np.float32)
        rgb = rgb * (1.0 + depth * (cool[None, None, :] - 1.0))
    if WARM_SHADE_SPLIT != 0.0:
        # Applied BEFORE the highlight shoulder so the shoulder still guarantees
        # nothing clips: measured, 0.28 lands at 0.00% clipped, and the same
        # split applied after the knee would push the sunlit reds back over 1.0.
        d = (shade - SHADE_SPLIT_PIVOT)[:, :, None]
        axis = np.array([0.25 * WARM_SHADE_SPLIT, 0.0, -WARM_SHADE_SPLIT], dtype=np.float32)
        split = 1.0 + d * axis[None, None, :]
        if _WATER_MASK is not None and _WATER_MASK.shape == shade.shape:
            # WATER IS EXEMPT, for two independent reasons.
            #
            # Physical: water is a sky MIRROR, not a diffuse surface. Sunlit
            # water reflects the sky, not the sun, so warming it in proportion
            # to direct light is simply wrong.
            #
            # Practical: water is the one colour on the plate that was solved
            # BACKWARDS through this exact chain, and the split is a new term
            # that solution never saw. Water sits at median shade 0.92, i.e.
            # the end where the split takes blue away. Measured on Prague,
            # unprotected: B/R 1.147 -> 1.053 at split 0.28, with GREEN
            # becoming the top channel - precisely the "water reads green"
            # failure the house style was built to avoid. It cannot be
            # re-solved either: the source blue is already 0xF4 with no
            # headroom left for WATER_BLUE_LIFT.
            split = np.where(_WATER_MASK[:, :, None], 1.0, split)
        rgb = rgb * split
    if HIGHLIGHT_KNEE is not None:
        # Smooth rolloff above the knee: continuous in value AND slope there
        # (the derivative is exactly 1 at x == knee), asymptotic to 1, so no
        # value ever clips and the ordering of the roof classes survives.
        k = float(HIGHLIGHT_KNEE)
        span = max(1.0 - k, 1e-6)
        rgb = np.where(rgb > k,
                       k + span * (1.0 - np.exp(-(rgb - k) / span)),
                       rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    # Unified grade (palette dim): pull toward luma to tame over-saturation into
    # one harmony, then a gentle contrast for cohesion.
    luma = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[:, :, None]
    rgb = luma + (rgb - luma) * (1.0 - SAT_PULL)
    rgb = np.clip(0.5 + (rgb - 0.5) * UNIFY_CONTRAST, 0.0, 1.0)
    arr[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# shadow-direction quadrant probe (map-maker: verify, never eyeball)
# ---------------------------------------------------------------------------


def _quadrant_probe(scene, half_extent_m, args) -> None:
    """Fast low-res trace; report which diagonal quadrant around the tallest
    buildings is darkest (= shadow direction). Want SE for classic NW light."""
    grid = _normalize_heightfield(
        build_building_heightfield(scene, half_extent_m, 768), 768
    )
    probe_args = argparse.Namespace(**{**vars(args), "tiles": 1, "max_frames": 1024,
                                       "min_frames": 16, "variance_threshold": 4e-3})
    rgb, hit = _pt_pass(grid, 640, _camera_for_grid(*grid.shape), probe_args, label="probe")
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # tallest cells in the (downsampled) heightfield
    flat = grid.ravel()
    idx = np.argsort(flat)[-40:]
    rr, cc = np.unravel_index(idx, grid.shape)
    sy, sx = lum.shape[0] / grid.shape[0], lum.shape[1] / grid.shape[1]
    d = 10
    sums = {"SE": [], "SW": [], "NE": [], "NW": []}
    for r, c in zip(rr, cc):
        pr, pc = int(r * sy), int(c * sx)
        for name, (dr, dc) in {"SE": (d, d), "SW": (d, -d), "NE": (-d, d), "NW": (-d, -d)}.items():
            y0, x0 = pr + dr, pc + dc
            if 3 <= y0 < lum.shape[0] - 3 and 3 <= x0 < lum.shape[1] - 3:
                sums[name].append(float(np.nanmean(lum[y0 - 3:y0 + 4, x0 - 3:x0 + 4])))
    means = {k: (np.mean(v) if v else float("nan")) for k, v in sums.items()}
    darkest = min(means, key=lambda k: means[k])
    print("== Quadrant probe (mean luminance around tallest buildings) ==")
    for k in ("NW", "NE", "SW", "SE"):
        print(f"   {k}: {means[k]:.4f}")
    print(f"   -> darkest quadrant = shadows fall {darkest} "
          f"(want SE for classic NW light; SUN_AZIMUTH={SUN_AZIMUTH})")


def main() -> int:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    half_extent_m = float(args.radius)

    print(f"[PT-City] center=({args.lon:.5f}, {args.lat:.5f}) radius={args.radius:.0f}m")
    print("== Building city scene (shared with CPU pipeline) ==")
    scene = city.build_city_scene(args.lon, args.lat, args.radius, refresh_osm=bool(args.refresh_osm))
    if not scene.roof_outlines:
        raise SystemExit("No buildings produced from the requested AOI.")

    if args.probe:
        _quadrant_probe(scene, half_extent_m, args)
        return 0

    shade_cache = OUT_DIR / "pt_light_field.npz"
    cache_key, cache_payload = light_cache_signature(args)
    global LIGHT_CACHE_KEY, SUN_AZIMUTH, SUN_ELEVATION
    LIGHT_CACHE_KEY = cache_key
    pnginfo = SAVE_KWARGS.get("pnginfo")
    cached = None
    legacy_cache = False
    if (args.reuse_shade or args.reuse_legacy_shade) and shade_cache.is_file():
        with np.load(shade_cache, allow_pickle=False) as candidate:
            stored_key = str(candidate["cache_key"].item()) if "cache_key" in candidate.files else ""
            if stored_key == cache_key and "rgb" in candidate.files and "hit" in candidate.files:
                cached = (candidate["rgb"].copy(), candidate["hit"].copy())
            elif args.reuse_legacy_shade and "rgb" in candidate.files and "hit" in candidate.files:
                cached = (candidate["rgb"].copy(), candidate["hit"].copy())
                legacy_cache = True
                cache_key = "legacy-unfingerprinted"
                # The old LoD2 caches were made by the shared north-up default;
                # keep the perspective warp consistent and label it unverified.
                SUN_AZIMUTH = 0.0
                SUN_ELEVATION = 60.0
            else:
                print(f"== Ignoring stale/unfingerprinted PT light field: {shade_cache} ==")
    if pnginfo is not None:
        pnginfo.add_text("light_cache_fingerprint", cache_key)
        if legacy_cache:
            pnginfo.add_text("lighting_status", "legacy cache; retrace required before launch")
        pnginfo.add_text("sun_azimuth_deg", f"{SUN_AZIMUTH:g}")
        pnginfo.add_text("sun_elevation_deg", f"{SUN_ELEVATION:g}")
    LIGHT_CACHE_KEY = cache_key
    if cached is not None:
        suffix = "; RETRACE REQUIRED" if legacy_cache else ""
        print(f"== Reusing cached PT light field: {shade_cache} ({cache_key}{suffix}) ==")
        rgb_field, hit = cached
    else:
        print("== Rasterizing building heightfield ==")
        raster_grid = int(args.grid_max) * RASTER_SUPERSAMPLE
        hf = build_building_heightfield(scene, half_extent_m, raster_grid)
        grid = _normalize_heightfield(hf, int(args.grid_max))
        print(f"[PT] raster={raster_grid} -> grid={grid.shape[1]}x{grid.shape[0]} frame={args.frame}px "
              f"spp={args.spp} tiles={args.tiles}x{args.tiles} "
              f"tallest={float(hf.max()):.0f}m relief={RELIEF_WORLD}")
        print("== Path tracing the light field (PROMETHEUS) ==")
        rgb_field, hit = _pt_light_field(grid, int(args.frame), args)
        np.savez_compressed(shade_cache, rgb=rgb_field, hit=hit,
                            cache_key=np.asarray(cache_key),
                            cache_payload=np.asarray(json.dumps(cache_payload, sort_keys=True)))
        print(f"[PT] light field cached: {shade_cache} ({cache_key})")

    print("== Building light-free overlay ==")
    overlay = build_lightfree_overlay(scene, half_extent_m, int(args.overlay_size))

    print("== Modulating overlay by the PT light field ==")
    shade, tint = _shade_on_overlay_grid(rgb_field, hit, overlay.size)
    subject = _modulate_overlay(overlay, shade, tint)

    print("== Composing poster ==")
    width, height = int(args.size[0]), int(args.size[1])
    poster = city.compose_poster(subject, width=width, height=height, radius_m=args.radius)
    # Final crispness pass (map-maker recipe): light unsharp on the composed plate.
    from PIL import ImageFilter
    poster = poster.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=UNSHARP[0], percent=UNSHARP[1], threshold=UNSHARP[2])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster.save(args.output, **SAVE_KWARGS)
    print(f"Success! Poster saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
