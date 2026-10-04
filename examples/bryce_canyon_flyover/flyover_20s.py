"""Step 3 of 3: render the 20 s Bryce Canyon flyover (1920x1080, 30 fps) to an MP4.

The camera flies 0.6-3.0 km of a ground track up the valley, over the amphitheatre and
across the rim, at full speed from the first frame and easing out over the last 1.5 s.
Flying height and aim follow low-passed terrain profiles, so single hoodoos cannot jolt
the camera. Frames go straight from the forge3d viewer into ffmpeg.

Run fetch_data.py and prep_look.py first. Needs forge3d (with its interactive viewer;
set FORGE3D_VIEWER_BINARY to pick a specific build), numpy, rasterio, Pillow and ffmpeg.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path

import forge3d as f3d
import numpy as np
import rasterio
from forge3d.viewer import open_viewer_async
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "bryce_canyon_flyover_20s_1080p.mp4"

TILE_LEFT, TILE_TOP, TILE, CTX = 395998.5, 4165574.5, 3800, 12288.0  # must match fetch_data.py
LAT, LON, SUN_UTC = 37.6283, -112.1677, (2026, 6, 21, 13, 30, 0)     # 07:30 local, low side light
# Ground track in flight-tile pixels (x = column east, y = row south).
TRACK = [(3650, 1450), (3000, 1560), (2300, 1650), (1750, 1720), (1450, 1950), (1350, 2350), (1500, 2750),
         (1850, 3050), (1650, 3330), (1250, 3330), (850, 3000), (720, 2500), (850, 2050), (1050, 1700)]
CLEARANCE, LOOKAHEAD = 180.0, 700.0  # metres above the terrain envelope; aim distance along the track
S0, S1 = 600.0, 3000.0               # flown section of the track, metres
FPS, SECONDS, EASE_OUT_S = 30, 20, 1.5
N = FPS * SECONDS
SIZE, FOV = (1920, 1080), 52.0
ZENITH, HORIZON = np.array([0.24, 0.42, 0.70]), np.array([0.70, 0.79, 0.88])


def gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(x, r, mode="edge"), k / k.sum(), mode="valid")


def max_filter(x: np.ndarray, half: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(np.pad(x, half, mode="edge"), 2 * half + 1).max(axis=1)


def catmull_rom(points: np.ndarray, samples: int = 200) -> np.ndarray:
    p = np.vstack([points[0], points, points[-1]])
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    segments = [0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
                for p0, p1, p2, p3 in zip(p[:-3], p[1:-2], p[2:-1], p[3:])]
    return np.vstack(segments + [points[-1:]])


def flight_profiles(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track position, eye altitude and smoothed ground height at every metre of the track."""
    dense = catmull_rom(np.asarray(TRACK, dtype=float))
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])
    stations = np.arange(0.0, arc[-1], 1.0)
    xy = np.column_stack([np.interp(stations, arc, dense[:, k]) for k in (0, 1)])
    col, row = np.clip(np.rint(xy), 0, TILE - 1).astype(int).T
    coarse = dem[::10, ::10]  # highest point within ~60 m of each station
    local_max = np.array([coarse[max(0, int((y - 60) // 10)):int((y + 60) // 10 + 1),
                                 max(0, int((x - 60) // 10)):int((x + 60) // 10 + 1)].max() for x, y in xy])
    eye_alt = gauss(max_filter(local_max, 250), 120.0) + CLEARANCE
    return xy, eye_alt, gauss(dem[row, col], 150.0)


def distance(f: int) -> float:
    """Track distance at frame f: constant speed from frame 0, C1 ease-out ending exactly at S1."""
    t, T, te = f / FPS, (N - 1) / FPS, EASE_OUT_S
    v = (S1 - S0) / (T - te / 2)
    a = max(0.0, t - (T - te))
    return S0 + v * (t - a * a / (2 * te))


def at(values: np.ndarray, s: float):
    s = float(np.clip(s, 0, len(values) - 1))
    i = int(s)
    w = s - i
    return values[i] * (1 - w) + values[min(i + 1, len(values) - 1)] * w


def world(col: float, row: float, h: float, min_h: float) -> np.ndarray:
    """Flight-tile pixel + elevation -> viewer world (x east, y up from the DEM minimum, z south)."""
    return np.array([TILE_LEFT + 0.5 + col, h - min_h, -(TILE_TOP - 0.5 - row)])


def camera_commands(dem: np.ndarray, min_h: float) -> list[dict]:
    xy, eye_alt, ground = flight_profiles(dem)
    eyes, aims = [], []
    for f in range(N):
        s, j = distance(f), distance(f) + LOOKAHEAD
        eyes.append(world(*at(xy, s), at(eye_alt, s), min_h))
        aims.append(world(*at(xy, j), 0.6 * at(ground, j) + 0.4 * at(eye_alt, j), min_h))
    aims = np.column_stack([gauss(c, 8.0) for c in np.array(aims).T])  # ~0.25 s low-pass on the aim
    commands = []
    for eye, aim in zip(eyes, aims):
        off = eye - aim
        r = float(np.linalg.norm(off))
        commands.append({"cmd": "set_terrain_camera", "phi_deg": math.degrees(math.atan2(off[2], off[0])),
                         "theta_deg": math.degrees(math.acos(off[1] / r)), "radius": r, "fov_deg": FOV,
                         "target": [float(v) for v in aim]})
    return commands


def with_sky(png: Path) -> np.ndarray:
    """Replace the viewer's flat background with a zenith-to-horizon gradient."""
    img = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    mask = np.abs(img - img[0, img.shape[1] // 2]).max(axis=2) <= 2
    terrain_rows = np.flatnonzero(~mask.all(axis=1))
    horizon = int(terrain_rows[0]) if terrain_rows.size else img.shape[0] // 2
    t = np.clip(np.arange(img.shape[0]) / max(horizon, 1), 0.0, 1.0)[:, None] ** 1.6
    sky = (ZENITH * (1 - t) + HORIZON * t) * 255.0
    out = img.astype(np.float64)
    out[mask] = np.broadcast_to(sky[:, None, :], out.shape)[mask]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg must be on PATH")
    context_path = DATA / "context_dem_2m_smooth.tif"
    with rasterio.open(DATA / "Bryce_Canyon.tif") as src:
        dem = src.read(1)
    with rasterio.open(context_path) as src:
        ctx_left, ctx_top = src.transform.c, src.transform.f
        min_h = float(src.read(1).min())
    cams = camera_commands(dem, min_h)
    u0, v0 = (TILE_LEFT - ctx_left) / CTX, (ctx_top - TILE_TOP) / CTX
    sun = f3d.sun_position_utc(LAT, LON, *SUN_UTC)

    partial, snapshot = OUTPUT.with_suffix(".part.mp4"), HERE / "_frame.png"
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}",
         "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-vf", "scale=out_color_matrix=bt709", "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-movflags", "+faststart", "-f", "mp4", str(partial)],
        stdin=subprocess.PIPE)
    start = time.perf_counter()
    viewer = open_viewer_async(width=SIZE[0], height=SIZE[1], terrain_path=str(context_path), fov_deg=FOV, timeout=120.0)
    try:
        viewer.load_overlay("naip_context", DATA / "naip_context_4m_graded.png", extent=(0, 0, 1, 1), z_order=0)
        viewer.load_overlay("naip_tile_1m", DATA / "naip_tile_1m_graded.png",
                            extent=(u0, v0, u0 + TILE / CTX, v0 + TILE / CTX), z_order=1)
        viewer.send_ipc({"cmd": "set_terrain_sun", "azimuth_deg": float(sun.azimuth),
                         "elevation_deg": float(sun.elevation), "intensity": 1.0})
        viewer.send_ipc({"cmd": "set_terrain_pbr", "enabled": True, "exposure": 0.6,
                         "height_ao": {"enabled": True, "strength": 0.8, "max_distance": 150.0},
                         "sun_visibility": {"enabled": True, "mode": "soft", "max_distance": 2000.0}})
        viewer.send_ipc({"cmd": "set_terrain", "ambient": 0.1, "zscale": 1.0})
        for f, cam in enumerate(cams):
            viewer.send_ipc(cam)
            viewer.snapshot(snapshot, *SIZE)
            encoder.stdin.write(with_sky(snapshot).tobytes())
            if f % 100 == 0:
                print(f"frame {f}/{N}  {time.perf_counter() - start:.0f}s", flush=True)
    finally:
        viewer.close()
        encoder.stdin.close()
        snapshot.unlink(missing_ok=True)
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    partial.replace(OUTPUT)
    print(f"wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
