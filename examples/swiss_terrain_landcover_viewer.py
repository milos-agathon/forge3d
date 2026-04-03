#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from forge3d.viewer import open_viewer_async


TARGET_CRS = "EPSG:2056"
CACHE_KEY = "epsg_2056"
TITLE = "Land cover in 2024 SWITZERLAND"
CAPTION_LINES = [
    "©2026 Milos Popovic (https://milospopovic.net)",
    "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft",
]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


LANDCOVER_CLASSES = [
    ("#419bdf", "#2a86d8", "Water"),
    ("#397d49", "#2b6a3d", "Trees"),
    ("#7a87c6", "#6677c8", "Flooded vegetation"),
    ("#e49635", "#d97f22", "Crops"),
    ("#c4281b", "#b72218", "Built area"),
    ("#a59b8f", "#a59b8f", "Bare ground"),
    ("#a8ebff", "#8cddff", "Snow"),
    ("#e3e2c3", "#e3e2c3", "Rangeland"),
]
LANDCOVER_LABELS = [label for _, _, label in LANDCOVER_CLASSES]
LANDCOVER_SOURCE_PALETTE_RGB = np.array([_hex_to_rgb(source) for source, _, _ in LANDCOVER_CLASSES], dtype=np.uint8)
LANDCOVER_BASE_PALETTE_RGB = np.array([_hex_to_rgb(display) for _, display, _ in LANDCOVER_CLASSES], dtype=np.uint8)
LANDCOVER_INVALID_CHANNEL_MAX = 15
LANDCOVER_DESPECKLE_PASSES = 2
LANDCOVER_DESPECKLE_MAX_SUPPORT = 2
LANDCOVER_DESPECKLE_MIN_MAJORITY = 4
LANDCOVER_CLASS_CACHE_KEY = "src-v1"
LANDCOVER_OPACITY = 0.70
LANDCOVER_OVERLAY_CACHE_KEY = f"display-v4-op{int(round(LANDCOVER_OPACITY * 1000.0)):04d}"

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets" / "tif"
CACHE_DIR = ROOT / ".tmp" / "swiss_terrain_landcover_viewer" / CACHE_KEY

TERRAIN_CONFIG = {
    "phi": 90.0,
    "theta": 31.0,
    "radius": 17600.0,
    "fov": 17.0,
    "zscale": 0.072,
    "sun_azimuth": 315.0,
    "sun_elevation": 24.0,
    "sun_intensity": 1.0,
    "ambient": 0.29,
    "shadow": 0.72,
    "background": [1.0, 1.0, 1.0],
}
PBR_CONFIG = {
    "enabled": True,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.0,
    "msaa": 8,
    "ibl_intensity": 0.32,
    "normal_strength": 3.4,
    "height_ao": {
        "enabled": True,
        "directions": 12,
        "steps": 32,
        "max_distance": 200.0,
        "strength": 0.75,
        "resolution_scale": 0.75,
    },
    "sun_visibility": {
        "enabled": True,
        "mode": "hard",
        "samples": 1,
        "steps": 48,
        "max_distance": 2000.0,
        "softness": 0.0,
        "bias": 0.005,
        "resolution_scale": 1.0,
    },
    "tonemap": {
        "operator": "aces",
        "white_point": 5.0,
        "white_balance_enabled": True,
        "temperature": 6500.0,
        "tint": 0.0,
    },
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def landcover_palette_rgb(opacity: float = LANDCOVER_OPACITY) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity >= 0.999:
        return LANDCOVER_BASE_PALETTE_RGB.copy()
    base = LANDCOVER_BASE_PALETTE_RGB.astype(np.float32)
    mixed = np.round(255.0 - (255.0 - base) * opacity)
    return mixed.astype(np.uint8)


def landcover_legend(opacity: float = LANDCOVER_OPACITY) -> list[tuple[str, str]]:
    palette = landcover_palette_rgb(opacity)
    return [(_rgb_to_hex(tuple(int(channel) for channel in palette[index])), label) for index, label in enumerate(LANDCOVER_LABELS)]


def crop_subject(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    corners = np.asarray([arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]], dtype=np.int16)
    background = np.median(corners, axis=0)
    mask = (arr[:, :, 3] > 0) & (np.abs(arr[:, :, :3].astype(np.int16) - background).max(axis=2) > 8)
    if not np.any(mask):
        return image
    ys, xs = np.nonzero(mask)
    pad = max(12, round(max(image.size) * 0.02))
    return image.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
    )


def compose_snapshot(raw_path: Path, output_path: Path) -> None:
    raw_path = raw_path.resolve()
    output_path = output_path.resolve()
    if raw_path == output_path:
        raise ValueError("compose_snapshot requires separate raw and output paths")

    raw = Image.open(raw_path).convert("RGBA")
    subject = crop_subject(raw)
    width, height = raw.size
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    margin = max(24, width // 32)

    title_font = load_font(max(28, width // 28), bold=True)
    legend_title_font = load_font(max(18, width // 48), bold=True)
    legend_font = load_font(max(14, width // 70))
    caption_font = load_font(max(12, width // 90))

    title_box = draw.textbbox((0, 0), TITLE, font=title_font)
    title_y = max(18, height // 36)
    draw.text((width // 2, title_y), TITLE, fill=(38, 38, 42), font=title_font, anchor="ma")

    legend_x = margin
    legend_y = title_y + (title_box[3] - title_box[1]) + 24
    draw.text((legend_x, legend_y), "Land Cover", fill=(54, 54, 60), font=legend_title_font)
    legend_cursor = legend_y + legend_title_font.size + 10
    dot = max(10, legend_font.size - 2)
    row_gap = max(6, legend_font.size // 3)
    for color, label in landcover_legend():
        draw.ellipse((legend_x, legend_cursor, legend_x + dot, legend_cursor + dot), fill=color, outline=(160, 160, 160))
        draw.text((legend_x + dot + 10, legend_cursor - 2), label, fill=(72, 72, 78), font=legend_font)
        legend_cursor += dot + row_gap

    caption_gap = max(4, caption_font.size // 3)
    caption_top = height - margin - len(CAPTION_LINES) * caption_font.size - (len(CAPTION_LINES) - 1) * caption_gap
    for index, line in enumerate(CAPTION_LINES):
        y = caption_top + index * (caption_font.size + caption_gap)
        draw.text((width // 2, y), line, fill=(82, 82, 88), font=caption_font, anchor="ma")

    available_top = legend_cursor + max(16, height // 80)
    available_height = max(1, caption_top - available_top - max(12, height // 60))
    subject.thumbnail((max(1, width - margin * 2), available_height), Image.Resampling.LANCZOS)
    map_x = (width - subject.width) // 2
    map_y = available_top + max(0, (available_height - subject.height) // 8)
    canvas.alpha_composite(subject, dest=(map_x, map_y))
    canvas.save(output_path)


def ensure_dem_in_target_crs(dem_path: Path) -> Path:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CACHE_DIR / f"{dem_path.stem}_{CACHE_KEY}.tif"
    with rasterio.open(dem_path) as src:
        if src.crs == CRS.from_user_input(TARGET_CRS):
            return dem_path
        if output_path.exists() and output_path.stat().st_mtime >= dem_path.stat().st_mtime:
            return output_path
        transform, width, height = calculate_default_transform(src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=src.nodata if src.nodata is not None else -9999.0,
            compress="lzw",
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=profile["nodata"],
                init_dest_nodata=True,
            )
    return output_path


def snap_overlay_to_legend(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    valid = rgb8.max(axis=2) > LANDCOVER_INVALID_CHANNEL_MAX
    if not np.any(valid):
        return rgb8, valid
    flat_rgb = rgb8.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    flat_codes = (
        (flat_rgb[:, 0].astype(np.uint32) << 16)
        | (flat_rgb[:, 1].astype(np.uint32) << 8)
        | flat_rgb[:, 2].astype(np.uint32)
    )
    palette = LANDCOVER_SOURCE_PALETTE_RGB.astype(np.int32)
    for code in np.unique(flat_codes[flat_valid]):
        color = np.array([(int(code) >> 16) & 0xFF, (int(code) >> 8) & 0xFF, int(code) & 0xFF], dtype=np.int32)
        index = int(np.argmin(np.sum((palette - color) ** 2, axis=1)))
        flat_rgb[flat_codes == code] = LANDCOVER_SOURCE_PALETTE_RGB[index]
    return rgb8, valid


def rgb_to_classes(rgb: np.ndarray) -> np.ndarray:
    snapped, valid = snap_overlay_to_legend(rgb)
    classes = np.full(valid.shape, -1, dtype=np.int16)
    for index, color in enumerate(LANDCOVER_SOURCE_PALETTE_RGB):
        classes[valid & np.all(snapped == color, axis=2)] = index
    return classes


def despeckle_landcover_classes(classes: np.ndarray, passes: int = LANDCOVER_DESPECKLE_PASSES) -> np.ndarray:
    cleaned = classes.copy()
    height, width = cleaned.shape
    for _ in range(max(0, passes)):
        same_count = np.zeros((height, width), dtype=np.uint8)
        best_count = np.zeros((height, width), dtype=np.uint8)
        best_class = np.full((height, width), -1, dtype=np.int16)
        for index in range(len(LANDCOVER_BASE_PALETTE_RGB)):
            mask = cleaned == index
            if not np.any(mask):
                continue
            padded = np.pad(mask, 1, mode="constant", constant_values=False).astype(np.uint8)
            count = (
                padded[0:height, 0:width]
                + padded[0:height, 1 : width + 1]
                + padded[0:height, 2 : width + 2]
                + padded[1 : height + 1, 0:width]
                + padded[1 : height + 1, 1 : width + 1]
                + padded[1 : height + 1, 2 : width + 2]
                + padded[2 : height + 2, 0:width]
                + padded[2 : height + 2, 1 : width + 1]
                + padded[2 : height + 2, 2 : width + 2]
            )
            same_count[mask] = count[mask]
            replace = count > best_count
            best_count[replace] = count[replace]
            best_class[replace] = index
        replace = (
            (cleaned >= 0)
            & (same_count <= LANDCOVER_DESPECKLE_MAX_SUPPORT)
            & (best_count >= LANDCOVER_DESPECKLE_MIN_MAJORITY)
            & (best_class != cleaned)
        )
        if not np.any(replace):
            break
        cleaned[replace] = best_class[replace]
    return cleaned


def classes_to_rgba(classes: np.ndarray) -> np.ndarray:
    rgba = np.zeros(classes.shape + (4,), dtype=np.uint8)
    valid = classes >= 0
    palette = landcover_palette_rgb()
    rgba[valid, :3] = palette[classes[valid]]
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def build_landcover_classes(landcover_path: Path) -> Path:
    import rasterio

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    classes_path = CACHE_DIR / f"{landcover_path.stem}_{LANDCOVER_CLASS_CACHE_KEY}_classes.tif"
    if classes_path.exists() and classes_path.stat().st_mtime >= landcover_path.stat().st_mtime:
        return classes_path
    with rasterio.open(landcover_path) as src:
        if src.count < 3:
            raise ValueError(f"Expected at least 3 land-cover bands, got {src.count}")
        profile = src.profile.copy()
        profile.update(count=1, dtype="int16", nodata=-1, compress="lzw")
        with rasterio.open(classes_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                rgb = np.moveaxis(
                    src.read([1, 2, 3], window=window, masked=True, out_dtype="uint8").filled(0),
                    0,
                    -1,
                )
                dst.write(rgb_to_classes(rgb), 1, window=window)
    return classes_path


def build_overlay(landcover_path: Path, terrain_path: Path) -> Path:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    classes_path = build_landcover_classes(landcover_path)
    overlay_path = CACHE_DIR / f"{landcover_path.stem}_{CACHE_KEY}_{LANDCOVER_OVERLAY_CACHE_KEY}_overlay.png"
    newest_input = max(terrain_path.stat().st_mtime, classes_path.stat().st_mtime)
    if overlay_path.exists() and overlay_path.stat().st_mtime >= newest_input:
        return overlay_path
    with rasterio.open(terrain_path) as terrain, rasterio.open(classes_path) as landcover_classes:
        classes = np.full((terrain.height, terrain.width), -1, dtype=np.int16)
        reproject(
            source=rasterio.band(landcover_classes, 1),
            destination=classes,
            src_transform=landcover_classes.transform,
            src_crs=landcover_classes.crs,
            dst_transform=terrain.transform,
            dst_crs=terrain.crs,
            resampling=Resampling.mode,
            src_nodata=-1,
            dst_nodata=-1,
            init_dest_nodata=True,
        )
    Image.fromarray(classes_to_rgba(despeckle_landcover_classes(classes)), "RGBA").save(overlay_path)
    return overlay_path


def render(snapshot_path: Path, dem_path: Path, landcover_path: Path, width: int, height: int) -> None:
    terrain_path = ensure_dem_in_target_crs(dem_path)
    overlay_path = build_overlay(landcover_path, terrain_path)
    snapshot_path = snapshot_path.resolve()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.unlink(missing_ok=True)
    raw_snapshot_path = CACHE_DIR / f"{snapshot_path.stem}_{width}x{height}_raw.png"
    raw_snapshot_path.unlink(missing_ok=True)

    with open_viewer_async(width=width, height=height, timeout=30.0) as viewer:
        viewer.load_terrain(terrain_path)
        viewer.send_ipc({"cmd": "set_terrain", **TERRAIN_CONFIG})
        viewer.send_ipc({"cmd": "set_terrain_pbr", **PBR_CONFIG})
        viewer.load_overlay(
            "landcover",
            overlay_path,
            extent=(0.0, 0.0, 1.0, 1.0),
            opacity=1.0,
            preserve_colors=True,
        )
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": True})
        time.sleep(2.0)
        viewer.snapshot(raw_snapshot_path, width=width, height=height)

    compose_snapshot(raw_snapshot_path, snapshot_path)
    print(f"Saved snapshot: {snapshot_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("swiss_render.png"))
    parser.add_argument("--width", type=int, default=1920*3)
    parser.add_argument("--height", type=int, default=1920*3)
    parser.add_argument("--dem", type=Path, default=ASSETS_DIR / "switzerland_dem.tif")
    parser.add_argument("--landcover", type=Path, default=ASSETS_DIR / "switzerland_land_cover.tif")
    args = parser.parse_args()

    if not args.dem.exists():
        print(f"Error: DEM file not found: {args.dem}")
        return 1
    if not args.landcover.exists():
        print(f"Error: land-cover file not found: {args.landcover}")
        return 1

    try:
        render(args.snapshot, args.dem, args.landcover, args.width, args.height)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
