#!/usr/bin/env python3
"""ORBIS globe descent: from ISS altitude to the north faces of the Eiger, Moench and Jungfrau.

The Swiss terrain examples (``swiss_terrain_landcover_viewer.py``,
``swiss_landcover_pt_oblique.py``) render ``assets/tif/switzerland_dem.tif`` as
a flat heightmap patch. This example streams the same GeoTIFF into the ORBIS
:class:`forge3d.GlobeScene`: the DEM sits on the curved WGS84 Earth in f64
ECEF, its real outline rests on the lit Earth, and fine COG pages stream in at
source resolution as the camera descends.

Every waypoint targets the Eiger / Moench / Jungfrau wall; ``GlobeScene``
places the camera ``altitude * tan(pitch)`` back along the heading, so the
camera itself may sit far outside the DEM. The flight starts 408 km up (ISS
orbit) looking south across Switzerland to the curved Earth limb, then falls
while pitching towards the horizon and ends on an oblique view of the north
faces above Grindelwald.

Outputs (under ``--output-dir``):

* ``frames/frame_####.png`` - the descent, with an altitude HUD
* ``orbis_swiss_alps_descent.mp4`` - when ``ffmpeg`` is on PATH
* ``contact_sheet.png`` - a 4x3 overview of the descent
* ``orbis_metrics.json`` - with ``--metrics``: the physical metrics from
  ``GlobeScene.scripted_descent()`` for the same oriented path

Usage::

    python examples/orbis_swiss_alps_descent.py
    python examples/orbis_swiss_alps_descent.py --frames 120 --size 960 540
    python examples/orbis_swiss_alps_descent.py --metrics
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d

ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "assets" / "tif" / "switzerland_dem.tif"
FONT_PATH = ROOT / "assets" / "fonts" / "NotoSansLatin-subset.ttf"
DEFAULT_OUTPUT_DIR = ROOT / "examples" / "out" / "orbis_swiss_alps_descent"

START_ALTITUDE_M = 408_000.0

# Scene target: centre of the LOD-10 tile holding the Jungfrau-Aletsch massif.
JUNGFRAU_LON = 7.910
JUNGFRAU_LAT = 46.494
TARGET_NAME = "Jungfrau"

# Every waypoint targets the Eiger / Moench / Jungfrau wall, looking south.
AIM_LONLAT = (7.985, 46.560)
HEADING_DEG = 180.0

# Alpine hypsometric palette (metres): lowland -> pasture -> forest -> scree
# -> rock -> firn -> glacier ice. Switzerland's lowest point is 193 m.
ELEVATION_DOMAIN = (190.0, 4200.0)
ALPINE_STOPS: Sequence[tuple[float, str]] = (
    (190.0, "#2f5d33"),
    (700.0, "#4f7f3c"),
    (1300.0, "#7c9a4e"),
    (1800.0, "#a2a466"),
    (2300.0, "#96916f"),
    (2800.0, "#8a8580"),
    (3200.0, "#c9c6c2"),
    (3600.0, "#eef2f6"),
    (4200.0, "#ffffff"),
)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def descent_waypoints(
    frame_count: int,
    *,
    end_altitude_m: float = 2_500.0,
    start_pitch_deg: float = 55.0,
    end_pitch_deg: float = 77.0,
    knee_m: float = 1_500.0,
) -> list[tuple[float, float, float, float, float]]:
    """Oriented ``(lon, lat, altitude, heading, pitch)`` waypoints.

    Every waypoint targets the Jungfrau wall. Altitude is log-spaced in
    ``altitude + knee`` (each decade gets similar screen time, the final
    approach is short) and pitch eases from the orbital limb view towards the
    horizon. The first and last altitudes are exact.
    """
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not 0.0 <= end_altitude_m < START_ALTITUDE_M:
        raise ValueError("end_altitude_m must be in [0, 408000)")
    if not 0.0 <= start_pitch_deg <= end_pitch_deg < 90.0:
        raise ValueError("pitch must satisfy 0 <= start <= end < 90 degrees")
    if knee_m <= 0.0:
        raise ValueError("knee_m must be positive")

    log_start = math.log(START_ALTITUDE_M + knee_m)
    log_end = math.log(end_altitude_m + knee_m)
    waypoints = []
    for index in range(frame_count):
        t = index / (frame_count - 1)
        altitude = math.exp(log_start + (log_end - log_start) * t) - knee_m
        if index == 0:
            altitude = START_ALTITUDE_M
        elif index == frame_count - 1:
            altitude = end_altitude_m
        pitch = start_pitch_deg + (end_pitch_deg - start_pitch_deg) * _smoothstep(t)
        waypoints.append((AIM_LONLAT[0], AIM_LONLAT[1], max(altitude, 0.0), HEADING_DEG, pitch))
    return waypoints


def format_altitude(altitude_m: float) -> str:
    if altitude_m >= 10_000.0:
        return f"{altitude_m / 1000.0:,.0f} km"
    if altitude_m >= 1_000.0:
        return f"{altitude_m / 1000.0:.1f} km"
    if altitude_m >= 10.0:
        return f"{altitude_m:,.0f} m"
    return f"{altitude_m:.1f} m"


def build_render_params(
    size: tuple[int, int],
    *,
    msaa: int = 4,
    sun_azimuth_deg: float = 150.0,
    sun_elevation_deg: float = 32.0,
):
    """Look for the globe path: size, lighting and the elevation palette.

    GlobeScene owns the camera (pose, clip planes, clipmap mode and span).
    """
    from forge3d.terrain_params import make_terrain_params_config

    colormap = f3d.Colormap1D.from_stops(stops=list(ALPINE_STOPS), domain=ELEVATION_DOMAIN)
    overlays = [
        f3d.OverlayLayer.from_colormap1d(
            colormap, strength=1.0, offset=0.0, blend_mode="Alpha", domain=ELEVATION_DOMAIN
        )
    ]
    config = make_terrain_params_config(
        size_px=size,
        render_scale=1.0,
        terrain_span=1000.0,  # replaced by GlobeScene with the tile extent
        msaa_samples=msaa,
        z_scale=1.0,
        exposure=1.0,
        domain=ELEVATION_DOMAIN,
        albedo_mode="colormap",
        colormap_strength=1.0,
        light_azimuth_deg=sun_azimuth_deg,
        light_elevation_deg=sun_elevation_deg,
        sun_intensity=3.0,
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
    draw.text((x, y), "ORBIS  |  Bernese Oberland", font=title_font, fill=(225, 232, 240))
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
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr[-1200:]}")
    return True


def make_scene(params):
    return f3d.GlobeScene(DEM_PATH, JUNGFRAU_LON, JUNGFRAU_LAT, TARGET_NAME, params=params)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--frames", type=int, default=300, help="Descent frames (default 300 = 10 s at 30 fps).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold", type=int, default=45, help="Extra copies of the final frame.")
    parser.add_argument("--end-altitude", type=float, default=2_500.0, help="Final altitude above ground (m).")
    parser.add_argument("--msaa", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--warmup",
        type=int,
        default=240,
        help="Max unrecorded stream steps at the first waypoint before recording.",
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

    scene = make_scene(build_render_params(size, msaa=args.msaa))
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
    for index, waypoint in enumerate(path):
        frame = scene.fly_to(*waypoint)
        image = annotate_frame(frame.to_numpy(), waypoint, index / (len(path) - 1))
        out = frames_dir / f"frame_{index:04d}.png"
        image.save(out)
        frame_paths.append(out)
        if index % 30 == 0 or index == len(path) - 1:
            print(f"[ORBIS] frame {index:4d}  {format_altitude(waypoint[2]):>9}  pitch {waypoint[4]:4.1f}")
    for extra in range(args.hold):
        shutil.copyfile(frame_paths[-1], frames_dir / f"frame_{len(path) + extra:04d}.png")
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
        metrics = make_scene(build_render_params(size, msaa=1)).scripted_descent(path)
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
