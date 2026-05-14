"""Render a Sentinel-2 timelapse over the Khumbu Icefall.

The module keeps deterministic helper functions importable for tests while the
network, raster, viewer, and ffmpeg work stays behind the CLI path.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import forge3d as f3d


KHUMBU_BBOX = (86.74, 27.86, 86.99, 28.06)
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PLANETARY_COMPUTER_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COPERNICUS_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-02-28"
DEFAULT_MAX_SCENES = 24
DEFAULT_CLOUD_COVER = 35.0
DEFAULT_FPS = 10
DEFAULT_DURATION = 10.0
DEFAULT_DEM_RESOLUTION = 5.0
DEFAULT_CROSSFADE_FRAMES = 10

TIMELAPSE_ORBIT_START_PHI = -18.0
TIMELAPSE_ORBIT_DEGREES = 104.0
FRAME_BACKGROUND_RGB = (252, 253, 255)
BASE_SLAB_RGB = (24, 27, 30)
COOL_ALPINE_HDRI_NAME = "cool_alpine_128x64.hdr"


@dataclass(frozen=True)
class SceneItem:
    item_id: str
    date: str
    datetime: str
    cloud_cover: float
    mgrs_tile: str
    assets: dict[str, str]


@dataclass(frozen=True)
class FramePlanItem:
    index: int
    scene: SceneItem
    label_scene: SceneItem
    blend_scene: SceneItem | None = None
    blend_alpha: float = 0.0
    label_opacity: float = 1.0
    blend_peer_index: int | None = None


@dataclass(frozen=True)
class RgbStretch:
    low: tuple[float, float, float]
    high: tuple[float, float, float]


@dataclass(frozen=True)
class RenderScene:
    terrain_width: float
    terrain_height: float
    zscale: float
    target: tuple[float, float, float]
    radius: float
    fov_deg: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--max-scenes", type=int, default=DEFAULT_MAX_SCENES)
    parser.add_argument("--cloud-cover", type=float, default=DEFAULT_CLOUD_COVER)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--dem-resolution", type=float, default=DEFAULT_DEM_RESOLUTION)
    parser.add_argument("--size", type=int, nargs=2, default=(1600, 1000), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "out" / "khumbu_icefall_sentinel_timelapse")
    return parser.parse_args(argv)


def build_stac_payload(
    *,
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    cloud_cover: float,
    limit: int,
) -> dict[str, object]:
    return {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": float(cloud_cover)}},
        "limit": int(limit),
    }


def _scene_from_feature(item: dict[str, object]) -> SceneItem:
    props = item.get("properties", {})
    assets = item.get("assets", {})
    if not isinstance(props, dict) or not isinstance(assets, dict):
        raise ValueError("STAC feature is missing properties or assets")
    timestamp = str(props["datetime"])
    asset_hrefs = {
        str(key): str(value["href"])
        for key, value in assets.items()
        if isinstance(value, dict) and "href" in value
    }
    return SceneItem(
        item_id=str(item["id"]),
        date=timestamp[:10],
        datetime=timestamp,
        cloud_cover=float(props.get("eo:cloud_cover", 100.0)),
        mgrs_tile=str(props.get("s2:mgrs_tile", "")),
        assets=asset_hrefs,
    )


def _snow_season_score(scene: SceneItem) -> int:
    month = int(scene.date[5:7])
    if month in (1, 2, 3, 10, 11, 12):
        return 0
    if month in (4, 5, 9):
        return 1
    return 2


def select_scenes(features: Iterable[dict[str, object]], max_scenes: int | None) -> list[SceneItem]:
    by_date: dict[str, SceneItem] = {}
    for feature in features:
        scene = _scene_from_feature(feature)
        current = by_date.get(scene.date)
        if current is None or (scene.cloud_cover, scene.item_id) < (current.cloud_cover, current.item_id):
            by_date[scene.date] = scene

    unique = sorted(by_date.values(), key=lambda scene: scene.date)
    if max_scenes is None or len(unique) <= max_scenes:
        return unique
    if max_scenes <= 0:
        return []

    if max_scenes <= 2:
        selected = sorted(unique, key=lambda scene: (_snow_season_score(scene), scene.cloud_cover, scene.date))[:max_scenes]
        return sorted(selected, key=lambda scene: scene.date)

    span = len(unique) - 1
    indexes: list[int] = []
    for index in range(max_scenes):
        candidate = math.floor(index * span / max(1, max_scenes - 1))
        if candidate not in indexes:
            indexes.append(candidate)
    probe = 0
    while len(indexes) < max_scenes and probe < len(unique):
        if probe not in indexes:
            indexes.append(probe)
        probe += 1
    return [unique[index] for index in sorted(indexes[:max_scenes])]


def unique_scenes_by_id(scenes: Iterable[SceneItem]) -> list[SceneItem]:
    seen: set[str] = set()
    unique: list[SceneItem] = []
    for scene in scenes:
        if scene.item_id not in seen:
            seen.add(scene.item_id)
            unique.append(scene)
    return unique


def frame_path(frames_dir: Path, index: int) -> Path:
    return frames_dir / f"frame_{index:04d}.png"


def _format_date_label(date_text: str) -> str:
    import datetime as _dt

    return _dt.date.fromisoformat(date_text).strftime("%Y %B %d")


def build_frame_plan(
    scenes: Sequence[SceneItem],
    *,
    fps: int,
    duration: float,
    preview_only: bool,
    crossfade_frames: int = DEFAULT_CROSSFADE_FRAMES,
) -> list[FramePlanItem]:
    if not scenes:
        return []
    total_frames = 1 if preview_only else max(1, int(round(float(fps) * float(duration))))
    if len(scenes) == 1:
        return [FramePlanItem(index=i, scene=scenes[0], label_scene=scenes[0]) for i in range(total_frames)]

    segments = len(scenes) - 1
    base = total_frames // segments
    remainder = total_frames % segments
    plan: list[FramePlanItem] = []
    for segment_index in range(segments):
        length = base + (1 if segment_index < remainder else 0)
        current = scenes[segment_index]
        nxt = scenes[segment_index + 1]
        fade = min(max(0, crossfade_frames), max(0, length - 1))
        for offset in range(length):
            global_index = len(plan)
            in_fade = fade > 0 and offset >= length - fade - 1 and offset < length - 1
            if in_fade:
                fade_offset = offset - (length - fade - 1)
                alpha = (fade_offset + 1) / (fade + 1)
                label_scene = nxt if alpha >= 0.5 else current
                opacity = 1.0 - 0.7 * math.sin(math.pi * alpha)
                plan.append(FramePlanItem(global_index, current, label_scene, nxt, alpha, opacity, global_index))
            else:
                scene = nxt if offset == length - 1 else current
                plan.append(FramePlanItem(global_index, scene, scene))
    return plan[:total_frames]


def _blend_transition_images(current, peer, alpha: float):
    from PIL import Image

    alpha = min(1.0, max(0.0, float(alpha)))
    return Image.blend(current.convert("RGB"), peer.convert("RGB"), alpha)


def _label_image(image, date_text: str, *, opacity: float = 1.0):
    from PIL import Image, ImageDraw, ImageFont

    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    text = _format_date_label(date_text)
    x, y = 34, 30
    badge = (x - 12, y - 10, x + 190, y + 46)
    alpha = int(230 * min(1.0, max(0.0, float(opacity))))
    draw.rectangle(badge, fill=(18, 22, 26, alpha))
    draw.text((x, y), text, fill=(255, 255, 255, alpha), font=font)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _label_frame(path: Path, date_text: str, *, opacity: float = 1.0) -> None:
    from PIL import Image

    image = Image.open(path)
    _label_image(image, date_text, opacity=opacity).save(path)


def _lighten_map_image(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    bg = np.array(FRAME_BACKGROUND_RGB, dtype=np.float32)
    mask_bg = np.all(np.abs(arr - bg) <= 1.0, axis=2)
    luma = arr.mean(axis=2)
    lift = np.where(luma < 80.0, 28.0, np.where(luma < 180.0, 18.0, 2.0))
    arr = np.clip(arr + lift[..., None], 0, 255)
    arr[mask_bg] = bg
    from PIL import Image

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def copernicus_dem_urls(bbox: Sequence[float]) -> list[str]:
    _west, south, _east, north = bbox
    urls: list[str] = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        lat_prefix = "N" if lat >= 0 else "S"
        lat_token = f"{lat_prefix}{abs(lat):02d}_00"
        lon_token = "E086_00"
        name = f"Copernicus_DSM_COG_10_{lat_token}_{lon_token}_DEM"
        urls.append(f"{COPERNICUS_BASE_URL}/{name}/{name}.tif")
    return urls


def _dem_grid_cache_token(profile: dict[str, object]) -> tuple[object, ...]:
    return (profile.get("crs"), tuple(profile.get("transform", ())), profile.get("width"), profile.get("height"))


def normalize_rgb_to_rgba(rgb: np.ndarray, stretch: RgbStretch | None = None) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if stretch is None:
        low = np.percentile(arr.reshape(-1, 3), 2, axis=0)
        high = np.percentile(arr.reshape(-1, 3), 98, axis=0)
    else:
        low = np.array(stretch.low, dtype=np.float32)
        high = np.array(stretch.high, dtype=np.float32)
    scaled = (arr - low) / np.maximum(high - low, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = _apply_khumbu_color_grade(scaled)
    rgba = np.empty(arr.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = np.clip(scaled * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def _apply_khumbu_color_grade(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    blue_lift = np.array([0.84, 0.96, 1.24], dtype=np.float32)
    graded = np.clip(arr * blue_lift + np.array([0.040, 0.090, 0.125], dtype=np.float32), 0.0, 1.0)
    return graded


def ensure_cool_alpine_hdri(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / COOL_ALPINE_HDRI_NAME
    if path.exists():
        return path
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 128\n"
    pixels = bytearray()
    for y in range(64):
        t = y / 63.0
        for _x in range(128):
            r = int(115 + 40 * (1.0 - t))
            g = int(135 + 35 * (1.0 - t))
            b = int(185 + 45 * (1.0 - t))
            pixels.extend((r, g, b, 129))
    path.write_bytes(header + bytes(pixels))
    return path


def _camera_progress_for_frame(index: int, total_frames: int) -> float:
    if total_frames <= 1:
        return 0.5
    t = min(1.0, max(0.0, index / (total_frames - 1)))
    return round(t * t * (3.0 - 2.0 * t), 3)


def _terrain_state(scene: RenderScene, *, progress: float) -> dict[str, object]:
    phi = TIMELAPSE_ORBIT_START_PHI + TIMELAPSE_ORBIT_DEGREES * float(progress)
    return {
        "cmd": "set_terrain",
        "target": list(scene.target),
        "radius": scene.radius,
        "fov": scene.fov_deg,
        "theta": 34.0,
        "phi": phi,
        "sun_elevation": 38.0,
        "sun_intensity": 0.82,
        "ambient": 0.48,
        "shadow": 0.55,
        "background": [value / 255.0 for value in FRAME_BACKGROUND_RGB],
        "progress": float(progress),
    }


def _terrain_pbr_state(*, hdri_path: Path) -> dict[str, object]:
    return {
        "cmd": "set_pbr",
        "hdr_path": str(hdri_path),
        "ibl_intensity": 0.55,
        "exposure": 1.35,
        "height_ao": {"strength": 0.18},
        "tonemap": {"temperature": 7200.0},
    }


def build_render_scene(dem_path: Path) -> RenderScene:
    import rasterio

    with rasterio.open(dem_path) as dataset:
        dem = dataset.read(1).astype(np.float32)
        width = float(dataset.width)
        height = float(dataset.height)
    low = float(np.percentile(dem, 2.0))
    target_level = float(np.percentile(dem, 58.0))
    zscale = 0.34
    return RenderScene(
        terrain_width=width,
        terrain_height=height,
        zscale=zscale,
        target=(width * 0.52, (target_level - low) * zscale, height * 0.46),
        radius=max(width, height) * 2.48,
        fov_deg=29.0,
    )


def _ground_frame(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    image = Image.open(path).convert("RGB")
    bg = Image.new("RGB", image.size, FRAME_BACKGROUND_RGB)
    ground = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ground)
    w, h = image.size
    draw.ellipse((w * 0.18, h * 0.66, w * 0.82, h * 0.88), fill=(0, 0, 0, 76))
    ground = ground.filter(ImageFilter.GaussianBlur(max(6, w // 45)))
    base = Image.new("RGBA", image.size, (0, 0, 0, 0))
    base_draw = ImageDraw.Draw(base)
    base_draw.rectangle((w * 0.18, h * 0.78, w * 0.82, h * 0.84), fill=(*BASE_SLAB_RGB, 255))
    bg = Image.alpha_composite(bg.convert("RGBA"), ground)
    arr = np.asarray(image)
    mask = np.any(np.abs(arr.astype(np.int16) - np.array(FRAME_BACKGROUND_RGB, dtype=np.int16)) > 4, axis=2)
    bg.paste(image, mask=Image.fromarray(mask.astype(np.uint8) * 255))
    bg = Image.alpha_composite(bg, base)
    bg.convert("RGB").save(path)


def _read_reprojected_asset(href: str, dem_profile: dict[str, object], *, indexes: int | list[int]) -> np.ndarray:
    import rasterio

    with rasterio.open(href) as src:
        return src.read(indexes)


def prepare_overlay(scene: SceneItem, *, dem_path: Path, cache_dir: Path, force: bool = False) -> Path:
    from PIL import Image
    import rasterio

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{scene.item_id}.png"
    if path.exists() and not force:
        return path
    with rasterio.open(dem_path) as dem:
        profile = dict(dem.profile)
    if {"B04", "B03", "B02"} <= set(scene.assets):
        bands = [
            _read_reprojected_asset(scene.assets[name], profile, indexes=1)
            for name in ("B04", "B03", "B02")
        ]
        rgb = np.dstack(bands)
    else:
        visual = _read_reprojected_asset(scene.assets["visual"], profile, indexes=[1, 2, 3])
        rgb = np.moveaxis(visual, 0, -1)
    Image.fromarray(normalize_rgb_to_rgba(rgb), mode="RGBA").save(path)
    return path


def sign_planetary_computer_url(href: str) -> str:
    import requests

    for _attempt in range(3):
        response = requests.get(PLANETARY_COMPUTER_SIGN_URL, params={"href": href}, timeout=30)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", "1")))
            continue
        response.raise_for_status()
        return str(response.json().get("href", href))
    response.raise_for_status()
    return href


def search_sentinel_scenes(
    *,
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    cloud_cover: float,
    max_scenes: int | None,
) -> list[SceneItem]:
    import requests

    payload: dict[str, object] = build_stac_payload(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        cloud_cover=cloud_cover,
        limit=100,
    )
    features: list[dict[str, object]] = []
    while True:
        response = requests.post(STAC_SEARCH_URL, json=payload, timeout=60)
        response.raise_for_status()
        page = response.json()
        features.extend(page.get("features", []))
        next_payload = None
        for link in page.get("links", []):
            if link.get("rel") == "next":
                next_payload = dict(link.get("body", {}))
                break
        if next_payload is None:
            break
        payload = next_payload
    return select_scenes(features, max_scenes=max_scenes)


def render_frames(
    *,
    dem_path: Path,
    overlays: dict[str, Path],
    frame_plan: Sequence[FramePlanItem],
    frames_dir: Path,
    size: tuple[int, int],
) -> None:
    if not frame_plan:
        raise ValueError("frame_plan must contain at least one frame")
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    scene = build_render_scene(dem_path)
    total = len(frame_plan)
    with f3d.open_viewer_async(timeout=180.0) as viewer:
        for item in frame_plan:
            progress = _camera_progress_for_frame(item.index, total)
            output = frame_path(frames_dir, item.index)
            viewer.send_ipc(_terrain_state(scene, progress=progress))
            viewer.load_overlay("sentinel", overlays[item.scene.item_id])
            viewer.snapshot(output, width=size[0], height=size[1])
            if item.blend_scene is not None:
                peer_path = frames_dir / f"frame_{item.index:04d}_peer.png"
                viewer.send_ipc(_terrain_state(scene, progress=progress))
                viewer.load_overlay("sentinel", overlays[item.blend_scene.item_id])
                viewer.snapshot(peer_path, width=size[0], height=size[1])
                blended = _blend_transition_images(Image.open(output), Image.open(peer_path), item.blend_alpha)
                blended.save(output)
                peer_path.unlink(missing_ok=True)
            _ground_frame(output)
            lifted = _lighten_map_image(Image.open(output))
            _label_image(lifted, item.label_scene.date, opacity=item.label_opacity).save(output)


def encode_mp4(frames_dir: Path, output_path: Path, *, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    raise SystemExit(
        "Full Khumbu rendering requires live data preparation; use the tested helpers or extend render_timelapse()."
    )


if __name__ == "__main__":
    raise SystemExit(main())
