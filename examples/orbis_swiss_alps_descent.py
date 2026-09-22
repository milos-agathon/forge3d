#!/usr/bin/env python3
"""ORBIS globe descent: from a whole-Earth view down onto the Jungfrau.

The flight opens 7,500 km straight above Switzerland with north up, the way a
seat-back flight map shows the Earth: Europe, North Africa and the Middle
East, the Atlantic, Greenland and the edge of North America. It falls towards
the Alps and, once the Earth no longer fills the frame, swings around to the
north side, so the final approach crosses the Swiss plateau onto the north
faces of the Eiger, Moench and Jungfrau with Swiss terrain in the foreground.

Two data sources, one palette (Crameri's *fes*, split at sea level):

* ``assets/tif/switzerland_dem.tif`` streams into :class:`forge3d.GlobeScene`
  as real 3D terrain on the curved WGS84 Earth, with fine COG pages arriving
  at source resolution as the camera descends.
* NOAA ETOPO 2022 (60 arc-second surface elevation, land + bathymetry) is baked
  once into an equirectangular *fes* + hillshade image and passed as the
  scene's ``earth_texture``: the Earth everywhere outside the Swiss DEM.
  Download (478 MB, public domain) from
  https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc
  and pass it with ``--etopo`` (or set ``FORGE3D_ETOPO``). Without it the
  Earth is drawn as a plain lit sphere.

Outputs (under ``--output-dir``):

* ``frames/frame_####.png`` - the descent, with an altitude HUD
* ``orbis_swiss_alps_descent.mp4`` - when ``ffmpeg`` is on PATH
* ``contact_sheet.png`` - a 4x3 overview of the descent
* ``orbis_metrics.json`` - with ``--metrics``: the physical metrics from
  ``GlobeScene.scripted_descent()`` for the same oriented path

Usage::

    python examples/orbis_swiss_alps_descent.py --etopo path/to/ETOPO_2022_v1_60s_N90W180_surface.nc
    python examples/orbis_swiss_alps_descent.py --frames 180 --size 1280 720
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    # Load the real package before ensure_repo_import() puts the repo's
    # python/ first on sys.path, where a lightweight rasterio test stub lives.
    import rasterio
    from rasterio.enums import Resampling
except ImportError:  # only needed to bake the ETOPO Earth texture
    rasterio = None

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d

ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "assets" / "tif" / "switzerland_dem.tif"
FONT_PATH = ROOT / "assets" / "fonts" / "NotoSansLatin-subset.ttf"
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "out" / "orbis_swiss_alps_descent"

# Scene target: centre of the LOD-10 tile holding the Jungfrau-Aletsch massif.
JUNGFRAU_LON = 7.910
JUNGFRAU_LAT = 46.494
TARGET_NAME = "Jungfrau"

# Every waypoint targets a point inside the Swiss DEM (GlobeScene places the
# camera altitude * tan(pitch) back along the heading, so the camera itself
# may be far outside it). The target glides from the middle of Switzerland,
# which centres the whole-Earth opening, onto the Eiger / Moench / Jungfrau
# wall. The heading swings from north (map orientation, camera south of the
# target) through east to south (camera north of the target): outside the DEM
# the Earth is a textured sphere at sea level, so the low approach keeps Swiss
# terrain, not a sunken neighbouring country, in the foreground.
START_TARGET = (8.20, 46.80)
FINAL_TARGET = (7.985, 46.560)
START_ALTITUDE_M = 7_500_000.0
START_HEADING_DEG = 0.0
FINAL_HEADING_DEG = 180.0
START_PITCH_DEG = 0.0
FINAL_PITCH_DEG = 77.0
# Fractions of the descent over which the camera swings to the north side.
SWING_START = 0.40
SWING_END = 0.78
# GlobeScene keeps the camera-to-target distance (altitude / cos(pitch))
# within 9,800 km, the terrain runtime contract's render-space bound.
MAX_CAMERA_TARGET_DISTANCE_M = 9_800_000.0

# Crameri "fes" (Scientific colour maps, https://www.fabiocrameri.ch/colourmaps/)
# is split at its midpoint: greys for the sea floor, then dark green -> olive ->
# tan -> white on land. Each half is sampled at 17 positions of the 256-entry
# map, and the domain is symmetric around 0 m, so the jump sits exactly at sea
# level (interpolating across it would paint continental shelves grey-green).
FES_SEA_HEX = (
    "#0d0d0d", "#1d1d1d", "#2b2b2b", "#393939", "#464646", "#535353", "#606060",
    "#6b6b6b", "#777777", "#818181", "#8d8d8d", "#9a9a9a", "#a9a9a9", "#b9b9b9",
    "#cacaca", "#dddddd", "#f1f1f1",
)
FES_LAND_HEX = (
    "#024026", "#1b4921", "#345220", "#4a5922", "#5e5e26", "#716229", "#83672d",
    "#976d32", "#ab773e", "#b88550", "#c19769", "#c8a882", "#d0ba9c", "#d7cab7",
    "#dfd7cf", "#e6e2e5", "#ededfc",
)
# The last sea stop sits this far below 0 m so the jump is effectively sharp.
SEA_LEVEL_EPSILON_M = 0.5
ELEVATION_LIMIT_M = 5_000.0
ELEVATION_DOMAIN = (-ELEVATION_LIMIT_M, ELEVATION_LIMIT_M)

SUN_AZIMUTH_DEG = 150.0
SUN_ELEVATION_DEG = 40.0
# With the colormap pre-linearised (see terrain_colormap_stops), this sun
# intensity makes lit terrain match the textured Earth around it.
SUN_INTENSITY = 3.5

EARTH_TEXTURE_WIDTH = 8192


def _hex_rgb(value: str) -> np.ndarray:
    return np.array([int(value[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def fes_stops() -> list[tuple[float, str]]:
    """(elevation m, hex) stops over ELEVATION_DOMAIN with the fes split at 0 m."""
    low, high = ELEVATION_DOMAIN
    n = len(FES_SEA_HEX) - 1
    sea = [(low + (-SEA_LEVEL_EPSILON_M - low) * i / n, c) for i, c in enumerate(FES_SEA_HEX)]
    n = len(FES_LAND_HEX) - 1
    land = [(high * i / n, c) for i, c in enumerate(FES_LAND_HEX)]
    return sea + land


def fes_rgb(elevation_m: np.ndarray) -> np.ndarray:
    """Interpolate fes colours (0-255 floats) for elevations in metres."""
    stops = fes_stops()
    xs = np.array([s[0] for s in stops])
    rgb = np.stack([_hex_rgb(s[1]) for s in stops])
    z = np.clip(np.asarray(elevation_m, dtype=np.float64), xs[0], xs[-1])
    return np.stack([np.interp(z, xs, rgb[:, c]) for c in range(3)], axis=-1)


def srgb_to_linear_255(value: np.ndarray) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4) * 255.0


def terrain_colormap_stops() -> list[tuple[float, str]]:
    """fes stops for the terrain colormap, pre-linearised.

    The terrain shader treats colormap values as linear albedo and encodes its
    lit result to sRGB, so display colours are decoded here first; otherwise
    the DEM renders visibly paler than the same palette on the Earth texture.
    """
    out = []
    for elevation, color in fes_stops():
        r, g, b = np.round(srgb_to_linear_255(_hex_rgb(color))).astype(int)
        out.append((elevation, f"#{r:02x}{g:02x}{b:02x}"))
    return out


def bake_earth_texture(elevation_m: np.ndarray, *, exaggeration: float = 8.0) -> np.ndarray:
    """fes + hillshade RGB (uint8) for an equirectangular elevation grid.

    ``elevation_m`` is (height, width) with row 0 at 90 N and column 0 at
    180 W. Hillshade uses a north-west light; east spacing shrinks with
    cos(latitude). Flat ground keeps its palette colour.
    """
    z = np.asarray(elevation_m, dtype=np.float64)
    height, width = z.shape
    lat = np.deg2rad(90.0 - (np.arange(height) + 0.5) * 180.0 / height)
    cell = 2.0 * math.pi * 6_371_000.0 / width
    dzdx = np.gradient(z, axis=1) / (cell * np.maximum(np.cos(lat), 0.02))[:, None]
    dzdy = -np.gradient(z, axis=0) / cell
    nx, ny = -dzdx * exaggeration, -dzdy * exaggeration
    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    azimuth, altitude = np.deg2rad(315.0), np.deg2rad(45.0)
    light = (np.sin(azimuth) * np.cos(altitude), np.cos(azimuth) * np.cos(altitude), np.sin(altitude))
    shade = np.clip((nx * light[0] + ny * light[1] + light[2]) / norm, 0.0, 1.0)
    factor = np.clip(0.45 + 0.55 * shade / light[2], 0.0, 1.25)
    return np.clip(fes_rgb(z) * factor[..., None], 0, 255).astype(np.uint8)


def load_earth_texture(etopo: Path | None, cache_dir: Path) -> np.ndarray | None:
    """Bake (or reuse) the ETOPO Earth texture; None without a source."""
    if etopo is None or not Path(etopo).is_file():
        print(
            "[ORBIS] no ETOPO source (--etopo / FORGE3D_ETOPO): the Earth outside "
            "Switzerland stays a plain lit sphere. See the module docstring for the download."
        )
        return None
    from PIL import Image

    import hashlib

    # The cache key covers everything that shapes the bake.
    key = hashlib.sha256(repr((FES_SEA_HEX, FES_LAND_HEX, ELEVATION_DOMAIN, EARTH_TEXTURE_WIDTH)).encode()).hexdigest()[:10]
    cache = cache_dir / f"earth_fes_{EARTH_TEXTURE_WIDTH}_{key}.png"
    if cache.is_file():
        return np.asarray(Image.open(cache).convert("RGB"))
    if rasterio is None:
        raise SystemExit("baking the ETOPO Earth texture needs rasterio (pip install rasterio)")

    started = time.perf_counter()
    height = EARTH_TEXTURE_WIDTH // 2
    with rasterio.open(f'NETCDF:"{Path(etopo).as_posix()}":z') as src:
        z = src.read(1, out_shape=(height, EARTH_TEXTURE_WIDTH), resampling=Resampling.average)
    texture = bake_earth_texture(z)
    cache_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture).save(cache)
    print(f"[ORBIS] baked {EARTH_TEXTURE_WIDTH}x{height} Earth texture in {time.perf_counter() - started:.1f} s")
    return texture


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def descent_waypoints(
    frame_count: int,
    *,
    end_altitude_m: float = 2_500.0,
    knee_m: float = 1_500.0,
) -> list[tuple[float, float, float, float, float]]:
    """Oriented ``(lon, lat, altitude, heading, pitch)`` waypoints.

    Altitude is log-spaced in ``altitude + knee`` from the whole-Earth view to
    ``end_altitude_m`` (each decade gets similar screen time). Pitch eases from
    straight down to oblique; the heading swings through east once the camera
    is tilted, so the turn reads as an orbit around the Alps, not a spin.
    """
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not 0.0 <= end_altitude_m < START_ALTITUDE_M:
        raise ValueError(f"end_altitude_m must be in [0, {START_ALTITUDE_M:.0f})")
    if knee_m <= 0.0:
        raise ValueError("knee_m must be positive")

    log_start = math.log(START_ALTITUDE_M + knee_m)
    log_end = math.log(end_altitude_m + knee_m)
    heading_delta = FINAL_HEADING_DEG - START_HEADING_DEG
    waypoints = []
    for index in range(frame_count):
        t = index / (frame_count - 1)
        altitude = math.exp(log_start + (log_end - log_start) * t) - knee_m
        if index == 0:
            altitude = START_ALTITUDE_M
        elif index == frame_count - 1:
            altitude = end_altitude_m
        glide = _smoothstep((t - 0.25) / 0.75)
        lon = START_TARGET[0] + (FINAL_TARGET[0] - START_TARGET[0]) * glide
        lat = START_TARGET[1] + (FINAL_TARGET[1] - START_TARGET[1]) * glide
        swing = _smoothstep((t - SWING_START) / (SWING_END - SWING_START))
        heading = (START_HEADING_DEG + heading_delta * swing) % 360.0
        pitch = START_PITCH_DEG + (FINAL_PITCH_DEG - START_PITCH_DEG) * _smoothstep(t * 1.1)
        waypoints.append((lon, lat, max(altitude, 0.0), heading, pitch))
    return waypoints


def format_altitude(altitude_m: float) -> str:
    if altitude_m >= 10_000.0:
        return f"{altitude_m / 1000.0:,.0f} km"
    if altitude_m >= 1_000.0:
        return f"{altitude_m / 1000.0:.1f} km"
    if altitude_m >= 10.0:
        return f"{altitude_m:,.0f} m"
    return f"{altitude_m:.1f} m"


def build_render_params(size: tuple[int, int], *, msaa: int = 4, render_scale: float = 1.0):
    """Look for the globe path: size, lighting and the fes elevation palette.

    GlobeScene owns the camera (pose, clip planes, clipmap mode and span).
    ``render_scale`` supersamples the internal target and blit-resolves down
    to ``size`` for crisper edges.
    """
    from forge3d.terrain_params import make_terrain_params_config

    colormap = f3d.Colormap1D.from_stops(stops=terrain_colormap_stops(), domain=ELEVATION_DOMAIN)
    overlays = [
        f3d.OverlayLayer.from_colormap1d(
            colormap, strength=1.0, offset=0.0, blend_mode="Alpha", domain=ELEVATION_DOMAIN
        )
    ]
    config = make_terrain_params_config(
        size_px=size,
        render_scale=render_scale,
        terrain_span=1000.0,  # replaced by GlobeScene with the tile extent
        msaa_samples=msaa,
        z_scale=1.0,
        exposure=1.0,
        domain=ELEVATION_DOMAIN,
        albedo_mode="colormap",
        colormap_strength=1.0,
        light_azimuth_deg=SUN_AZIMUTH_DEG,
        light_elevation_deg=SUN_ELEVATION_DEG,
        sun_intensity=SUN_INTENSITY,
        # The palette already encodes elevation; slope hue shifts would tint
        # every steep face red.
        hue_variation_strength=0.0,
        overlays=overlays,
    )
    return f3d.TerrainRenderParams(config)


def _load_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default(size=size)


def annotate_frame(rgba: np.ndarray, waypoint: Sequence[float], progress: float):
    """Draw the altitude HUD onto a rendered RGBA frame; returns a PIL image."""
    from PIL import Image, ImageDraw

    lon, lat, altitude, heading, pitch = waypoint
    image = Image.fromarray(np.ascontiguousarray(rgba[..., :3]), mode="RGB")
    width, height = image.size
    scale = height / 720.0
    pad = int(28 * scale)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(int(20 * scale))
    value_font = _load_font(int(44 * scale))
    small_font = _load_font(int(16 * scale))

    panel_w, panel_h = int(340 * scale), int(150 * scale)
    draw.rounded_rectangle(
        (pad, pad, pad + panel_w, pad + panel_h), radius=int(10 * scale), fill=(10, 14, 22, 150)
    )
    x = pad + int(16 * scale)
    y = pad + int(12 * scale)
    draw.text((x, y), "ORBIS  |  Earth to Jungfrau", font=title_font, fill=(225, 232, 240))
    y += int(28 * scale)
    draw.text((x, y), format_altitude(altitude), font=value_font, fill=(255, 255, 255))
    y += int(56 * scale)
    draw.text((x, y), f"target {lat:.4f} N   {lon:.4f} E", font=small_font, fill=(190, 200, 212))
    y += int(22 * scale)
    draw.text((x, y), f"heading {heading:.0f}   pitch {pitch:.0f} from nadir", font=small_font, fill=(190, 200, 212))

    bar_y = height - pad
    bar_h = max(int(4 * scale), 2)
    draw.rectangle((pad, bar_y, width - pad, bar_y + bar_h), fill=(255, 255, 255, 60))
    draw.rectangle(
        (pad, bar_y, pad + int((width - 2 * pad) * progress), bar_y + bar_h), fill=(255, 255, 255, 210)
    )
    return image


def contact_sheet(frame_paths: Sequence[Path], output: Path, columns: int = 4, rows: int = 3) -> None:
    from PIL import Image

    count = columns * rows
    picks = [frame_paths[round(i * (len(frame_paths) - 1) / (count - 1))] for i in range(count)]
    first = Image.open(picks[0])
    cell_w, cell_h = first.width // 2, first.height // 2
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (0, 0, 0))
    for i, path in enumerate(picks):
        cell = Image.open(path).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(cell, ((i % columns) * cell_w, (i // columns) * cell_h))
    sheet.save(output)


def encode_video(frames_dir: Path, output: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        print("[ORBIS] ffmpeg not found; leaving the PNG frame sequence on disk.")
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "14",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr[-1200:]}")
    return True


def make_scene(params, earth_texture: np.ndarray | None):
    return f3d.GlobeScene(
        DEM_PATH, JUNGFRAU_LON, JUNGFRAU_LAT, TARGET_NAME, params=params, earth_texture=earth_texture
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--etopo",
        type=Path,
        default=Path(os.environ["FORGE3D_ETOPO"]) if os.environ.get("FORGE3D_ETOPO") else None,
        help="ETOPO 2022 60s surface netCDF for the Earth outside Switzerland (default: $FORGE3D_ETOPO).",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    parser.add_argument("--frames", type=int, default=360, help="Descent frames (default 360 = 12 s at 30 fps).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-start", type=int, default=45, help="Copies of the opening Earth view.")
    parser.add_argument("--hold", type=int, default=45, help="Extra copies of the final frame.")
    parser.add_argument("--end-altitude", type=float, default=2_500.0, help="Final altitude above ground (m).")
    parser.add_argument("--msaa", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="Internal supersample factor; output is blit-resolved back to --size.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=240,
        help="Max unrecorded stream steps at the first waypoint before recording.",
    )
    parser.add_argument(
        "--settle",
        type=int,
        default=24,
        help="Max extra stream steps per waypoint until streaming converges (0 disables).",
    )
    parser.add_argument("--metrics", action="store_true", help="Also run scripted_descent() and write orbis_metrics.json.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 encoding.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not DEM_PATH.is_file() or DEM_PATH.stat().st_size < 1024:
        raise SystemExit(f"missing DEM (unrestored LFS pointer?): {DEM_PATH}")

    size = (int(args.size[0]), int(args.size[1]))
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    earth_texture = load_earth_texture(args.etopo, args.output_dir / ".cache")
    scene = make_scene(
        build_render_params(size, msaa=args.msaa, render_scale=args.render_scale),
        earth_texture,
    )
    path = descent_waypoints(args.frames, end_altitude_m=args.end_altitude)
    print(f"[ORBIS] {scene!r}")
    print(f"[ORBIS] {len(path)} waypoints, {format_altitude(path[0][2])} -> {format_altitude(path[-1][2])}")

    # Streaming is bounded to one step per rendered frame. Hold the opening
    # waypoint (unrecorded) until the visible ring has streamed in, so the
    # flight opens on the whole DEM instead of watching it load.
    for step in range(args.warmup):
        scene.fly_to(*path[0])
        if scene.streaming_stats().get("converged"):
            break
    print(f"[ORBIS] warm-up: {step + 1} stream steps, converged={scene.streaming_stats().get('converged')}")

    frame_paths: list[Path] = []
    started = time.perf_counter()
    settled = 0
    for index, waypoint in enumerate(path):
        frame = scene.fly_to(*waypoint)
        # One bounded stream step runs inside each fly_to. Keep polling the
        # same waypoint until the loader is drained and the clipmap's demanded
        # ring tiles are resident, so the recorded frame is not shaded from
        # coarse fallback tiles.
        settle_steps = 0
        while (
            settle_steps < args.settle
            and not scene.streaming_stats().get("converged", False)
        ):
            frame = scene.fly_to(*waypoint)
            settle_steps += 1
        settled += settle_steps
        image = annotate_frame(frame.to_numpy(), waypoint, index / (len(path) - 1))
        out = frames_dir / f"frame_{len(frame_paths) + (args.hold_start if index else 0):04d}.png"
        image.save(out)
        if index == 0:
            for extra in range(1, args.hold_start + 1):
                shutil.copyfile(out, frames_dir / f"frame_{extra:04d}.png")
        frame_paths.append(out)
        if index % 30 == 0 or index == len(path) - 1:
            print(f"[ORBIS] frame {index:4d}  {format_altitude(waypoint[2]):>9}  pitch {waypoint[4]:4.1f}")
    print(f"[ORBIS] settle: {settled} extra stream steps across {len(path)} waypoints")
    last = len(path) + args.hold_start
    for extra in range(args.hold):
        shutil.copyfile(frame_paths[-1], frames_dir / f"frame_{last + extra:04d}.png")
    print(f"[ORBIS] rendered {len(path)} frames in {time.perf_counter() - started:.1f} s")
    stats = scene.streaming_stats()
    print(
        f"[ORBIS] streaming: {stats.get('resident_fine_tiles')} fine tiles resident, "
        f"source spacing {stats.get('source_sample_spacing_m')} m"
    )

    sheet = args.output_dir / "contact_sheet.png"
    contact_sheet(frame_paths, sheet)
    print(f"[ORBIS] contact sheet: {sheet}")

    if not args.no_video:
        video = args.output_dir / "orbis_swiss_alps_descent.mp4"
        if encode_video(frames_dir, video, args.fps):
            print(f"[ORBIS] video: {video}")

    if args.metrics:
        # The physical coverage probe needs single-sample rendering.
        metrics = make_scene(build_render_params(size, msaa=1), earth_texture).scripted_descent(path)
        payload = {
            "source": DEM_PATH.relative_to(ROOT).as_posix(),
            "target": {"name": TARGET_NAME, "lon": JUNGFRAU_LON, "lat": JUNGFRAU_LAT},
            "metrics": metrics.as_dict(),
        }
        metrics_path = args.output_dir / "orbis_metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[ORBIS] max jitter {metrics.max_vertex_jitter_px:.2e} px "
            f"(naive {metrics.naive_max_vertex_jitter_px:.2e} px), "
            f"LOD crack px {metrics.lod_crack_pixels}, "
            f"peak GPU {metrics.peak_gpu_visible_bytes / 2**20:.0f} MiB -> {metrics_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
