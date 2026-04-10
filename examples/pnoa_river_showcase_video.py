#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter
from rasterio.transform import from_origin

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.terrain_scatter import viewer_orbit_radius
from pnoa_river_showcase import IMHOF_PALETTE, PROJECT_ROOT, display_path, read_dem, smooth_heightmap

DEFAULT_DEM = PROJECT_ROOT / "assets" / "tif" / "dem_rainier.tif"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "out" / "pnoa_river_showcase" / "pnoa_river_reference.mp4"
FRAME_SIZE = (1920, 1920)
DURATION = 12.0
FPS = 30
CROP_SIZE = 1800
MAX_DEM_SIZE = 2600
WATER_PATH = [
    (0.15, 0.58),
    (0.26, 0.54),
    (0.40, 0.55),
    (0.54, 0.57),
    (0.66, 0.61),
    (0.69, 0.74),
    (0.60, 0.82),
]


@dataclass(frozen=True)
class SceneConfig:
    phi_deg: float
    theta_deg: float
    radius: float
    fov_deg: float
    zscale: float
    sun_intensity: float
    ambient: float
    shadow: float
    target: tuple[float, float, float]
    display_orbit_radius: float


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


IMHOF_RGB = np.asarray([_hex_to_rgb01(color) for color in IMHOF_PALETTE], dtype=np.float32)
IMHOF_STOPS = np.linspace(0.0, 1.0, len(IMHOF_RGB), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the reference PNOA river showcase MP4.")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def crop_hero_heightmap(heightmap: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    peak_row, peak_col = np.unravel_index(int(np.argmax(heightmap)), heightmap.shape)
    row0 = int(np.clip(peak_row - 520, 0, max(heightmap.shape[0] - crop_size, 0)))
    col0 = int(np.clip(peak_col - 820, 0, max(heightmap.shape[1] - crop_size, 0)))
    cropped = np.asarray(heightmap[row0 : row0 + crop_size, col0 : col0 + crop_size], dtype=np.float32)
    if cropped.shape != (crop_size, crop_size):
        row_pad = crop_size - cropped.shape[0]
        col_pad = crop_size - cropped.shape[1]
        cropped = np.pad(
            cropped,
            ((row_pad // 2, row_pad - row_pad // 2), (col_pad // 2, col_pad - col_pad // 2)),
            mode="edge",
        )
    cropped -= float(cropped.min())
    return smooth_heightmap(cropped, passes=0)


def build_overlay_rgba(heightmap: np.ndarray) -> np.ndarray:
    floor = float(np.percentile(heightmap, 1.0))
    ceiling = float(np.percentile(heightmap, 99.6))
    normalized = np.clip((heightmap - floor) / max(ceiling - floor, 1e-6), 0.0, 1.0)
    clipped = np.power(normalized.astype(np.float32), 1.03).reshape(-1)
    terrain = np.stack([np.interp(clipped, IMHOF_STOPS, IMHOF_RGB[:, i]) for i in range(3)], axis=1)
    terrain = terrain.reshape(heightmap.shape + (3,)).astype(np.float32)
    terrain = np.clip(terrain * np.array([0.98, 0.95, 0.90], dtype=np.float32), 0.0, 1.0)

    size = heightmap.shape[0]
    water = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(water)
    draw.line([(x * size, y * size) for x, y in WATER_PATH], fill=255, width=max(8, int(size * 0.045)), joint="curve")
    draw.ellipse((0.45 * size, 0.50 * size, 0.68 * size, 0.66 * size), fill=230)
    draw.ellipse((0.20 * size, 0.51 * size, 0.31 * size, 0.58 * size), fill=220)
    water = water.filter(ImageFilter.GaussianBlur(radius=max(4.0, size * 0.008)))
    water_mix = (np.asarray(water, dtype=np.float32) / 255.0)[..., None] * 0.34
    terrain = terrain * (1.0 - water_mix) + np.array([0.58, 0.60, 0.57], dtype=np.float32) * water_mix
    return np.dstack(
        [np.round(np.clip(terrain, 0.0, 1.0) * 255.0).astype(np.uint8), np.full((size, size), 255, dtype=np.uint8)]
    )


def write_dem(path: Path, heightmap: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=heightmap.shape[1],
        height=heightmap.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0.0, float(heightmap.shape[0]), 1.0, 1.0),
        compress="lzw",
    ) as dst:
        dst.write(np.asarray(heightmap, dtype=np.float32), 1)


def build_scene_config(heightmap: np.ndarray) -> SceneConfig:
    cutoff = float(np.percentile(heightmap, 72.0))
    weights = np.clip(heightmap - cutoff, 0.0, None)
    row_grid, col_grid = np.indices(heightmap.shape, dtype=np.float32)
    if float(weights.sum()) > 1e-6:
        focus_row = float((weights * row_grid).sum() / weights.sum())
        focus_col = float((weights * col_grid).sum() / weights.sum())
    else:
        focus_row = float(heightmap.shape[0] * 0.5)
        focus_col = float(heightmap.shape[1] * 0.5)
    zscale = 0.17
    return SceneConfig(
        phi_deg=42.0,
        theta_deg=57.0,
        radius=viewer_orbit_radius(float(heightmap.shape[0]), scale=3.8),
        fov_deg=25.0,
        zscale=zscale,
        sun_intensity=2.35,
        ambient=0.03,
        shadow=1.0,
        target=(focus_col, float(np.percentile(heightmap, 56.0) * zscale), focus_row),
        display_orbit_radius=float(heightmap.shape[0]) * 0.50,
    )


def project_world_to_screen(world_point: np.ndarray, *, scene: SceneConfig, width: int, height: int) -> tuple[float, float]:
    def unit(vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        return np.zeros_like(vector, dtype=np.float32) if length <= 1e-6 or not np.isfinite(length) else vector / length

    target = np.asarray(scene.target, dtype=np.float32)
    phi, theta = np.deg2rad([scene.phi_deg, scene.theta_deg])
    eye = target + np.array(
        [scene.radius * np.sin(theta) * np.cos(phi), scene.radius * np.cos(theta), scene.radius * np.sin(theta) * np.sin(phi)],
        dtype=np.float32,
    )
    forward = unit(target - eye)
    side = unit(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
    up = unit(np.cross(side, forward))
    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = side, up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    focal = 1.0 / np.tan(np.deg2rad(scene.fov_deg) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0], proj[1, 1] = focal / max(float(width) / max(float(height), 1.0), 1e-6), focal
    proj[2, 2], proj[2, 3], proj[3, 2] = scene.radius * 10.0 / (1.0 - scene.radius * 10.0), scene.radius * 10.0 / (1.0 - scene.radius * 10.0), -1.0
    clip = proj @ view @ np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float32)
    if abs(float(clip[3])) <= 1e-6:
        return float(width) * 0.5, float(height) * 0.28
    ndc = clip[:3] / clip[3]
    return (float(ndc[0]) * 0.5 + 0.5) * float(width), (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(height)


def sun_state(t: float, duration: float = DURATION) -> tuple[float, float]:
    phase = (t / max(duration, 1e-6)) * (2.0 * np.pi)
    return float((42.0 + np.degrees(phase)) % 360.0), float(24.0 + 8.0 * np.sin(phase - 0.55))


def orb_position(t: float, *, scene: SceneConfig, width: int, height: int) -> tuple[float, float, tuple[float, float]]:
    azimuth_deg, elevation_deg = sun_state(t)
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    display_point = np.asarray(scene.target, dtype=np.float32) + np.array(
        [np.cos(elevation) * np.sin(azimuth), np.sin(elevation), np.cos(elevation) * np.cos(azimuth)],
        dtype=np.float32,
    ) * scene.display_orbit_radius
    return azimuth_deg, elevation_deg, project_world_to_screen(display_point, scene=scene, width=width, height=height)


def build_vignette(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx / max(width - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(height - 1, 1)) * 2.0 - 1.0
    edge = np.clip((np.sqrt((nx * 0.95) ** 2 + (ny * 1.05) ** 2) - 0.44) / 0.54, 0.0, 1.0)
    coarse = (np.random.default_rng(7).random((36, 36)) * 255.0).astype(np.uint8)
    noise = Image.fromarray(coarse, mode="L").resize((width, height), resample=Image.Resampling.BICUBIC)
    noise = np.asarray(noise.filter(ImageFilter.GaussianBlur(radius=max(18, width // 24))), dtype=np.float32) / 255.0
    return np.clip(1.0 - edge * (0.18 + 0.10 * noise), 0.74, 1.0)


def light_maps(width: int, height: int, orb_xy: tuple[float, float], target_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    def unit(vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        return np.zeros_like(vector, dtype=np.float32) if length <= 1e-6 or not np.isfinite(length) else vector / length

    def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
        scaled = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
        return scaled * scaled * (3.0 - 2.0 * scaled)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    orb = np.asarray(orb_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    min_dim = float(min(width, height))
    to_light = unit(orb - target)
    if not np.any(to_light):
        to_light = np.array([0.0, -1.0], dtype=np.float32)
    away = -to_light
    perp = np.array([-away[1], away[0]], dtype=np.float32)

    base = np.full((height, width, 3), 0.036, dtype=np.float32)
    base += (0.018 + 0.030 * (1.0 - yy / max(float(height - 1), 1.0)) ** 1.35)[:, :, None]
    wall = np.exp(-(((xx - width * 0.56) / max(width * 0.72, 1.0)) ** 2 + ((yy - height * 0.42) / max(height * 0.58, 1.0)) ** 2)) * 0.10
    base += wall[:, :, None]

    def radial(center: np.ndarray, radius: float, strength: float) -> np.ndarray:
        dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        return strength * np.clip(1.0 - dist / max(radius, 1.0), 0.0, 1.0) ** 2

    dx = xx - target[0]
    dy = yy - target[1]
    cone_along = np.clip(dx * to_light[0] + dy * to_light[1], 0.0, None)
    cone_across = np.abs(dx * perp[0] + dy * perp[1])
    cone_width = min_dim * 0.14 + cone_along * 0.36
    cone = np.exp(-((cone_along / max(min_dim * 0.95, 1.0)) ** 2 + (cone_across / np.maximum(cone_width, 1.0)) ** 2)) * 0.18
    base += np.clip(radial(orb, min_dim * 0.92, 0.22) + radial(orb, min_dim * 0.48, 0.18) + cone, 0.0, 0.58)[:, :, None]

    along = np.clip(dx * away[0] + dy * away[1], 0.0, None)
    across = np.abs(dx * perp[0] + dy * perp[1])
    spread = min_dim * 0.06 + along * 0.40
    wedge = np.clip(
        np.exp(-((across / np.maximum(spread, 1.0)) ** 2)) * 0.72
        + np.exp(-((across / np.maximum(spread * 0.52, 1.0)) ** 2)) * 0.48,
        0.0,
        1.0,
    )
    shadow = wedge * smoothstep(min_dim * 0.03, min_dim * 0.14, along) * (1.0 - smoothstep(min_dim * 0.92, min_dim * 1.30, along))
    background = np.clip(base - shadow[:, :, None] * 0.52, 0.0, 1.0)

    footprint = target + to_light * (min_dim * 0.10)
    dx = xx - footprint[0]
    dy = yy - footprint[1]
    toward = dx * to_light[0] + dy * to_light[1]
    across = dx * perp[0] + dy * perp[1]
    pool = np.clip(
        np.exp(-((toward / max(min_dim * 0.15, 1.0)) ** 2 + (across / max(min_dim * 0.095, 1.0)) ** 2)) * 0.80
        + np.exp(-((toward / max(min_dim * 0.28, 1.0)) ** 2 + (across / max(min_dim * 0.18, 1.0)) ** 2)) * 0.35,
        0.0,
        1.0,
    )
    terrain_light = np.clip(0.06 + pool * 0.86, 0.0, 1.0) * (1.0 - smoothstep(min_dim * 0.02, min_dim * 0.48, along) * 0.92)
    return background, terrain_light


def postprocess_frame(raw_path: Path, out_path: Path, orb_xy: tuple[float, float], target_xy: tuple[float, float], vignette: np.ndarray) -> None:
    def compress(rgb: np.ndarray, threshold: float, rolloff: float) -> np.ndarray:
        clipped = np.clip(rgb, 0.0, 1.0)
        excess = np.clip(clipped - threshold, 0.0, None)
        return np.where(clipped > threshold, threshold + excess / (1.0 + excess * rolloff), clipped)

    image = Image.open(raw_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    bg, terrain_light = light_maps(image.width, image.height, orb_xy, target_xy)
    terrain_seed = (rgb.max(axis=2) > 0.040).astype(np.uint8) * 255
    terrain_mask = np.asarray(Image.fromarray(terrain_seed, mode="L").filter(ImageFilter.GaussianBlur(radius=5.0)), dtype=np.float32) / 255.0
    water_seed = (
        (rgb[:, :, 2] > rgb[:, :, 1] + 0.03) & (rgb[:, :, 1] > rgb[:, :, 0] + 0.02) & (rgb[:, :, 2] > 0.28)
    )
    water_glow = np.zeros_like(rgb)
    if water_seed.any():
        glow = Image.fromarray(np.where(water_seed, 255, 0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=12))
        water_glow = (np.asarray(glow, dtype=np.float32) / 255.0)[:, :, None] * np.array([0.010, 0.011, 0.012], dtype=np.float32)

    light_gain = 0.24 + terrain_light[:, :, None] * 0.76
    warm_mix = 0.16 + terrain_light[:, :, None] * 0.64
    relit = rgb * light_gain * ((1.0 - warm_mix) + np.array([0.98, 0.95, 0.90], dtype=np.float32) * warm_mix)
    blur = Image.fromarray(np.round(np.clip(relit, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    blur = np.asarray(blur.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32) / 255.0
    relit = compress(np.clip(relit + (relit - blur) * 0.36, 0.0, 1.0), 0.70, 5.5)

    mask = terrain_mask[:, :, None]
    staged = bg * (1.0 - mask) + relit * 0.96 * mask
    staged += water_glow * mask
    staged = np.clip(np.power(compress(staged * vignette[:, :, None], 0.62, 7.0), 1.02), 0.0, 1.0)

    layer = Image.new("RGBA", (image.width, image.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = orb_xy
    glow_r = max(8.0, image.width * 0.018)
    haze_r = glow_r * 1.7
    core_r = max(2.0, image.width * 0.0038)
    draw.ellipse((x - haze_r, y - haze_r, x + haze_r, y + haze_r), fill=(255, 255, 255, 12))
    draw.ellipse((x - glow_r, y - glow_r, x + glow_r, y + glow_r), fill=(255, 255, 255, 22))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(4.0, image.width * 0.0055)))
    draw = ImageDraw.Draw(layer)
    draw.ellipse((x - core_r, y - core_r, x + core_r, y + core_r), fill=(255, 255, 255, 255))
    orb_rgba = np.asarray(layer, dtype=np.float32) / 255.0
    staged = np.clip(staged * (1.0 - orb_rgba[:, :, 3:4]) + orb_rgba[:, :, :3] * orb_rgba[:, :, 3:4], 0.0, 1.0)
    Image.fromarray(np.round(staged * 255.0).astype(np.uint8), mode="RGB").save(out_path)


def encode_mp4(frames_dir: Path, output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the MP4 output.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {result.stderr[-800:]}")


def configure_viewer(viewer: f3d.ViewerHandle, overlay_path: Path, scene: SceneConfig) -> None:
    azimuth, elevation = sun_state(0.0, 1.0)
    viewer.send_ipc(
        {
            "cmd": "set_terrain",
            "phi": scene.phi_deg,
            "theta": scene.theta_deg,
            "radius": scene.radius,
            "fov": scene.fov_deg,
            "sun_azimuth": azimuth,
            "sun_elevation": elevation,
            "sun_intensity": scene.sun_intensity,
            "ambient": scene.ambient,
            "zscale": scene.zscale,
            "shadow": scene.shadow,
            "background": [0.01, 0.01, 0.012],
            "water_level": -999999.0,
            "target": list(scene.target),
        }
    )
    viewer.send_ipc(
        {
            "cmd": "set_terrain_pbr",
            "enabled": True,
            "shadow_technique": "pcss",
            "shadow_map_res": 4096,
            "exposure": 0.44,
            "msaa": 8,
            "ibl_intensity": 0.0,
            "normal_strength": 1.38,
            "height_ao": {"enabled": True, "directions": 12, "steps": 24, "max_distance": 150.0, "strength": 0.16, "resolution_scale": 1.0},
            "sun_visibility": {
                "enabled": True,
                "mode": "soft",
                "samples": 2,
                "steps": 72,
                "max_distance": 500.0,
                "softness": 0.28,
                "bias": 0.0025,
                "resolution_scale": 1.0,
            },
            "tonemap": {"operator": "aces", "white_point": 1.85, "white_balance_enabled": True, "temperature": 5050.0, "tint": 0.0},
            "sky": {"enabled": False},
            "materials": {"snow": 0.0, "rock": 1.0, "grass": 0.0, "wetness": 0.14},
        }
    )
    viewer.load_overlay("hero_albedo", overlay_path, extent=(0.0, 0.0, 1.0, 1.0), opacity=1.0, preserve_colors=False)
    viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
    viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
    viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": False})
    time.sleep(2.0)


def render_video(args: argparse.Namespace) -> tuple[Path | None, Path]:
    width, height = FRAME_SIZE
    dem_path = args.dem.resolve()
    output_path = args.output.resolve()
    preview_path = args.preview.resolve() if args.preview is not None else output_path.with_name(f"{output_path.stem}_preview.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    heightmap, _, _, _, _ = read_dem(dem_path, MAX_DEM_SIZE)
    cropped = crop_hero_heightmap(heightmap)
    scene = build_scene_config(cropped)

    with tempfile.TemporaryDirectory(prefix="forge3d_pnoa_video_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        dem_crop_path = temp_dir / "hero_crop.tif"
        overlay_path = temp_dir / "hero_overlay.png"
        raw_dir = temp_dir / "raw"
        frames_dir = temp_dir / "frames"
        raw_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        write_dem(dem_crop_path, cropped)
        Image.fromarray(build_overlay_rgba(cropped), mode="RGBA").save(overlay_path)
        vignette = build_vignette(width, height)
        target_xy = project_world_to_screen(np.asarray(scene.target, dtype=np.float32), scene=scene, width=width, height=height)

        with f3d.open_viewer_async(terrain_path=dem_crop_path, width=width, height=height, timeout=45.0) as viewer:
            configure_viewer(viewer, overlay_path, scene)
            total_frames = 1 if args.preview_only else max(1, int(round(DURATION * FPS)))
            for frame in range(total_frames):
                t = 0.0 if total_frames == 1 else frame / float(FPS)
                azimuth, elevation, orb_xy = orb_position(t, scene=scene, width=width, height=height)
                viewer.send_ipc({"cmd": "set_terrain_sun", "azimuth_deg": azimuth, "elevation_deg": elevation, "intensity": scene.sun_intensity})
                raw_path = raw_dir / f"raw_{frame:04d}.png"
                final_path = frames_dir / f"frame_{frame:04d}.png"
                viewer.snapshot(raw_path, width=width, height=height)
                postprocess_frame(raw_path, final_path, orb_xy, target_xy, vignette)
                if frame == 0:
                    shutil.copyfile(final_path, preview_path)
                print(f"\r[Video] frame {frame + 1}/{total_frames}", end="", flush=True)
            print()

        if args.preview_only:
            return None, preview_path
        encode_mp4(frames_dir, output_path)
        return output_path, preview_path


def main() -> int:
    args = parse_args()
    output_path, preview_path = render_video(args)
    print(f"[Video] DEM: {display_path(args.dem)}")
    print(f"[Video] Preview: {display_path(preview_path)}")
    if output_path is not None:
        print(f"[Video] MP4: {display_path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
