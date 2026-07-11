#!/usr/bin/env python3
"""
Poland Population Density -- Vintage Contour Terrain

Python/forge3d version of ``contour-population-map.r`` for a local Poland
example. It keeps the R script's density bands and vintage colors, then
renders the bands as a stepped 3D population surface with forge3d.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poland_population_spikes as base
import poland_population_spikes_height_shade as height_shade


REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "pol_pd_2020_1km_UNadj.tif"
OUTPUT_DIR = REPO / "examples" / "out"
CLEAN_DEM = OUTPUT_DIR / "poland_population_contours_height.tif"
OVERLAY_PATH = OUTPUT_DIR / "poland_population_contours_overlay.png"
SNAPSHOT_PATH = OUTPUT_DIR / "poland_population_contours_raw.png"
FINAL_PATH = OUTPUT_DIR / "poland_population_contours_3d.png"
TARGET_CRS = "EPSG:3035"

CONTOUR_BREAKS = (0.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2500.0, 5000.0, float("inf"))
CONTOUR_LABELS = ("<25", "25-50", "50-100", "100-200", "200-500", "500-1k", "1k-2.5k", "2.5k-5k", ">5k")
VINTAGE_COLOURS = ("#b4c79d", "#c8cf8f", "#e3d685", "#e8c374", "#dfb39b", "#d88b83", "#b85c62", "#954957", "#733745")
CONTOUR_LINE_DARKEN = 0.92
CONTOUR_LINE_WIDTH = 0

WINDOW_SIZE = 2200
SNAP_SIZE = 8192
FINAL_WIDTH = 8192
FINAL_HEIGHT = 6000
BG_COLOR = [0.020, 0.018, 0.015]
PLATE_BG_RGB = (245, 240, 229)
TEXT_RGB = (31, 37, 48)
MUTED_RGB = (95, 103, 119)
RENDER_BG_RGB = (3, 3, 5)
MIN_DETAIL_LUMA = 76.0

CAMERA_PHI = 78.0
CAMERA_THETA = 47.0
CAMERA_RADIUS = 25000.0
CAMERA_FOV = 29.0
CAMERA_ZSCALE = 0.088
HEIGHT_STEP_SHARE = 0.82
SMOOTH_RADIUS = 18.0
GENERALIZE_KERNEL = 13
# 9x+ exceeds the current viewer TIFF decoder limit for this Poland raster.
RENDER_SUPERSAMPLE = 8
CONTOUR_RENDER_AMBIENT = 1.0
CONTOUR_RENDER_SHADOW = 0.0
CONTOUR_RENDER_NORMAL_STRENGTH = 0.0
CONTOUR_RENDER_HEIGHT_AO_STRENGTH = 0.0
RELIEF_BLUR_RADIUS = 5
RELIEF_STRENGTH = 0.13

MAP_MARGIN = 60
MAP_TOP = 650
MAP_BOTTOM = 110
TITLE_FONT_SIZE = 190
SUBTITLE_FONT_SIZE = 100
CAPTION_FONT_SIZE = 68
TITLE_Y = 72
SUBTITLE_Y = 310
HEADER_RULE_Y = 420
CAPTION_Y = FINAL_HEIGHT - 180
LEGEND_Y = 72
LEGEND_SWATCH_W = 300
LEGEND_SWATCH_H = 110
LEGEND_GAP = 32
LEGEND_LABEL_FONT_SIZE = 80
LEGEND_TITLE_FONT_SIZE = 92


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"Expected #rrggbb color, got {value!r}")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def darken_rgb(rgb: tuple[int, int, int], factor: float = CONTOUR_LINE_DARKEN) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in rgb)


def apply_soft_contour_relief(rgb: np.ndarray, bg_mask: np.ndarray | None = None) -> np.ndarray:
    """Add a bounded cartographic bevel from palette classes, without dark wall strokes."""
    source = np.asarray(rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Expected an RGB image array")

    if min(source.shape[:2]) < 3:
        return source.copy()

    palette = np.array([hex_to_rgb(color) for color in VINTAGE_COLOURS], dtype=np.int16)
    src_i16 = source.astype(np.int16)
    best_dist = np.full(source.shape[:2], np.iinfo(np.int32).max, dtype=np.int32)
    classes = np.zeros(source.shape[:2], dtype=np.uint8)
    for idx, color in enumerate(palette):
        diff = src_i16 - color.reshape(1, 1, 3)
        dist = np.sum(diff * diff, axis=2, dtype=np.int32)
        take = dist < best_dist
        classes[take] = idx
        best_dist[take] = dist[take]

    if bg_mask is None:
        bg_mask = np.zeros(source.shape[:2], dtype=bool)
    else:
        bg_mask = np.asarray(bg_mask, dtype=bool)
    classes[bg_mask] = 0

    # Replace only near-certain one-pixel class speckles. Seven agreeing
    # neighbours is conservative enough to preserve legitimate narrow bands.
    padded = np.pad(classes, 1, mode="edge")
    neighbour_counts = np.zeros((len(VINTAGE_COLOURS), *classes.shape), dtype=np.uint8)
    for y_offset, x_offset in (
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)
    ):
        neighbour = padded[
            y_offset:y_offset + classes.shape[0],
            x_offset:x_offset + classes.shape[1],
        ]
        for class_index in range(len(VINTAGE_COLOURS)):
            neighbour_counts[class_index] += neighbour == class_index
    majority_class = np.argmax(neighbour_counts, axis=0).astype(np.uint8)
    majority_count = np.max(neighbour_counts, axis=0)
    speckle = (~bg_mask) & (majority_count >= 7) & (classes != majority_class)
    classes[speckle] = majority_class[speckle]
    cleaned_source = source.copy()
    cleaned_source[speckle] = palette[majority_class[speckle]].astype(np.uint8)

    height_img = Image.fromarray((classes.astype(np.float32) * (255.0 / (len(VINTAGE_COLOURS) - 1))).astype(np.uint8))
    height = np.array(height_img.filter(ImageFilter.GaussianBlur(RELIEF_BLUR_RADIUS)), dtype=np.float32) / 255.0
    # ponytail: screen-space bevel; real vector terraces if printed-map accuracy matters.
    dx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    dy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    shade = 1.0 + RELIEF_STRENGTH * np.clip((-dx - dy) * 7.0, -1.0, 1.0)
    shade[bg_mask] = 1.0

    relieved = np.clip(cleaned_source.astype(np.float32) * shade[:, :, np.newaxis], MIN_DETAIL_LUMA, 245.0)
    relieved[bg_mask] = source[bg_mask]
    return relieved.astype(np.uint8)


def filled_valid_mask(valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    try:
        from scipy.ndimage import binary_fill_holes
    except ImportError:
        inv = ~valid
        outside = np.zeros_like(inv)
        stack = [(y, x) for y in range(inv.shape[0]) for x in (0, inv.shape[1] - 1) if inv[y, x]]
        stack += [(y, x) for y in (0, inv.shape[0] - 1) for x in range(inv.shape[1]) if inv[y, x]]
        while stack:
            y, x = stack.pop()
            if outside[y, x]:
                continue
            outside[y, x] = True
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < inv.shape[0] and 0 <= xx < inv.shape[1] and inv[yy, xx] and not outside[yy, xx]:
                    stack.append((yy, xx))
        return valid | (inv & ~outside)
    return binary_fill_holes(valid)


def contour_indices(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Map density values to the same five bands used by the R script."""
    values = np.asarray(data, dtype=np.float32)
    valid = filled_valid_mask(valid_mask)
    indices = np.digitize(values, CONTOUR_BREAKS[1:-1], right=False).astype(np.uint8)
    indices = np.clip(indices, 0, len(CONTOUR_LABELS) - 1)
    indices[~valid] = 0
    return indices


def generalized_contour_indices(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    indices = contour_indices(data, valid_mask)
    if min(indices.shape) < 5:
        return indices
    # ponytail: median generalization removes raster-cell islands; use vector polygons if this needs cartographic topology.
    indices = np.array(Image.fromarray(indices, mode="L").filter(ImageFilter.MedianFilter(GENERALIZE_KERNEL)), dtype=np.uint8)
    indices[~filled_valid_mask(valid_mask)] = 0
    return indices


def contour_overlay_rgba(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    palette = np.array([hex_to_rgb(color) for color in VINTAGE_COLOURS], dtype=np.uint8)
    indices = generalized_contour_indices(data, valid_mask)
    valid = filled_valid_mask(valid_mask)
    overlay = np.zeros((*indices.shape, 4), dtype=np.uint8)
    overlay[:, :, :3] = palette[indices]
    overlay[:, :, 3] = valid.astype(np.uint8) * 255
    return overlay


def smooth_density(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = np.where(valid_mask, np.asarray(data, dtype=np.float32), 0.0)
    radius = int(round(SMOOTH_RADIUS))
    if radius <= 0 or min(values.shape) < radius * 2 + 1:
        smoothed = values.copy()
    else:
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
            summed = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
            size = radius * 2 + 1
            smoothed = (
                summed[size:, size:]
                - summed[:-size, size:]
                - summed[size:, :-size]
                + summed[:-size, :-size]
            ) / float(size * size)
        else:
            smoothed = gaussian_filter(values, sigma=SMOOTH_RADIUS, mode="nearest")
    smoothed[~np.asarray(valid_mask, dtype=bool)] = 0.0
    return smoothed.astype(np.float32)


def prepare_render_density(data: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smoothed = smooth_density(data, valid_mask)
    if RENDER_SUPERSAMPLE <= 1:
        return smoothed, np.asarray(valid_mask, dtype=bool)

    height, width = smoothed.shape
    size = (width * RENDER_SUPERSAMPLE, height * RENDER_SUPERSAMPLE)
    up_data = np.array(Image.fromarray(smoothed, mode="F").resize(size, Image.BICUBIC), dtype=np.float32)
    up_valid = np.array(
        Image.fromarray(filled_valid_mask(valid_mask).astype(np.uint8) * 255, mode="L").resize(size, Image.NEAREST),
        dtype=np.uint8,
    ) > 0
    up_data[~up_valid] = 0.0
    return up_data, up_valid


def generate_contour_overlay(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> Path:
    del clip_max
    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    render_data, render_valid = prepare_render_density(data, valid_mask)
    Image.fromarray(contour_overlay_rgba(render_data, render_valid)).save(str(OVERLAY_PATH))
    print(f"  Overlay saved  : {OVERLAY_PATH}")
    return OVERLAY_PATH


def build_contour_height_dem(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> np.ndarray:
    """Create stepped relief: contour class gives the shelf, density shapes peaks."""
    smoothed, valid = prepare_render_density(data, valid_mask)
    bands = generalized_contour_indices(smoothed, valid).astype(np.float32) / (len(CONTOUR_LABELS) - 1)
    density = np.log1p(np.clip(smoothed, 0.0, clip_max)) / max(np.log1p(clip_max), 1e-6)
    height = (HEIGHT_STEP_SHARE * bands + (1.0 - HEIGHT_STEP_SHARE) * density) * clip_max
    height[~valid] = 0.0
    return height.astype(np.float32)


def cleanup_snapshot(raw: np.ndarray) -> np.ndarray:
    """Light cleanup only; the contour colors are already encoded in the overlay."""
    rgb = raw[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    bg_ref = rgb[0, 0]
    bg_ref_lum = 0.2126 * bg_ref[0] + 0.7152 * bg_ref[1] + 0.0722 * bg_ref[2]
    if bg_ref_lum < 40.0:
        bg_mask = np.max(np.abs(rgb - bg_ref.reshape(1, 1, 3)), axis=2) <= 3.0
    else:
        bg_mask = np.zeros(lum.shape, dtype=bool)

    lum_after = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    dark_detail = (~bg_mask) & (lum_after < MIN_DETAIL_LUMA)
    lift = MIN_DETAIL_LUMA / np.maximum(lum_after, 1.0)
    rgb[dark_detail] = np.clip(rgb[dark_detail] * lift[dark_detail, np.newaxis], 0.0, 255.0)
    result = apply_soft_contour_relief(rgb.astype(np.uint8), bg_mask)
    result[bg_mask] = RENDER_BG_RGB
    return result


def make_contour_legend() -> Image.Image:
    swatch_w = LEGEND_SWATCH_W
    swatch_h = LEGEND_SWATCH_H
    gap = LEGEND_GAP
    label_h = LEGEND_LABEL_FONT_SIZE + 18
    title_h = LEGEND_TITLE_FONT_SIZE + 22
    width = len(CONTOUR_LABELS) * swatch_w + (len(CONTOUR_LABELS) - 1) * gap
    height = title_h + swatch_h + label_h
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(base.FONT_REGULAR, LEGEND_TITLE_FONT_SIZE)
    label_font = ImageFont.truetype(base.FONT_LIGHT, LEGEND_LABEL_FONT_SIZE)

    draw.text((0, 0), "people / km2", font=title_font, fill=TEXT_RGB + (255,))
    for i, (label, color) in enumerate(zip(CONTOUR_LABELS, VINTAGE_COLOURS)):
        x = i * (swatch_w + gap)
        draw.rectangle((x, title_h, x + swatch_w, title_h + swatch_h), fill=hex_to_rgb(color) + (255,))
        draw.text((x, title_h + swatch_h + 8), label, font=label_font, fill=TEXT_RGB + (255,))
    return canvas


def replace_render_background(render: Image.Image) -> Image.Image:
    arr = np.array(render.convert("RGB"))
    bg = np.array(RENDER_BG_RGB, dtype=np.int16)
    diff = np.abs(arr.astype(np.int16) - bg.reshape(1, 1, 3))
    arr[np.all(diff <= 4, axis=2)] = PLATE_BG_RGB
    return Image.fromarray(arr)


def compose_final_plate(contour_img: np.ndarray, clip_max: float) -> None:
    del clip_max
    canvas = Image.new("RGB", (FINAL_WIDTH, FINAL_HEIGHT), PLATE_BG_RGB)
    draw = ImageDraw.Draw(canvas)

    margin = MAP_MARGIN
    top = MAP_TOP
    bottom = MAP_BOTTOM
    map_box = (FINAL_WIDTH - margin * 2, FINAL_HEIGHT - top - bottom)

    render = replace_render_background(base.crop_render_to_content(contour_img, pad=130))
    render = ImageOps.contain(render, map_box, Image.LANCZOS)
    canvas.paste(render, (margin + (map_box[0] - render.width) // 2, top))

    title_font = ImageFont.truetype(base.FONT_BOLD, TITLE_FONT_SIZE)
    subtitle_font = ImageFont.truetype(base.FONT_LIGHT, SUBTITLE_FONT_SIZE)
    caption_font = ImageFont.truetype(base.FONT_LIGHT, CAPTION_FONT_SIZE)

    draw.text((margin, TITLE_Y), "POLAND POPULATION CONTOURS", font=title_font, fill=TEXT_RGB)
    draw.text((margin, SUBTITLE_Y), "WorldPop 2020 - 1 km grid - color extrusion with forge3d", font=subtitle_font, fill=MUTED_RGB)
    draw.line([(margin, HEADER_RULE_Y), (margin + 1120, HEADER_RULE_Y)], fill=(92, 82, 64), width=5)

    legend = make_contour_legend()
    canvas.paste(legend, (FINAL_WIDTH - margin - legend.width, LEGEND_Y), legend)

    caption = "Data: WorldPop 2020 UN-adjusted - contour breaks adapted from contour-population-map.r"
    draw.text((margin, CAPTION_Y), caption, font=caption_font, fill=MUTED_RGB)

    canvas.save(str(FINAL_PATH), "PNG")
    print(f"  Final plate    : {FINAL_PATH}")


def configure_base_module() -> None:
    """Point the shared Poland forge3d helpers at this contour-band variant."""
    base.DATA_PATH = DATA_PATH
    base.TARGET_CRS = TARGET_CRS
    base.OUTPUT_DIR = OUTPUT_DIR
    base.CLEAN_DEM = CLEAN_DEM
    base.OVERLAY_PATH = OVERLAY_PATH
    base.SNAPSHOT_PATH = SNAPSHOT_PATH
    base.FINAL_PATH = FINAL_PATH
    base.generate_magma_overlay = generate_contour_overlay
    base.compose_final_plate = compose_final_plate

    height_shade.OUTPUT_DIR = OUTPUT_DIR
    height_shade.CLEAN_DEM = CLEAN_DEM
    height_shade.OVERLAY_PATH = OVERLAY_PATH
    height_shade.SNAPSHOT_PATH = SNAPSHOT_PATH
    height_shade.FINAL_PATH = FINAL_PATH
    height_shade.WINDOW_SIZE = WINDOW_SIZE
    height_shade.SNAP_SIZE = SNAP_SIZE
    height_shade.BG_COLOR = BG_COLOR
    height_shade.CAMERA_PHI = CAMERA_PHI
    height_shade.CAMERA_THETA = CAMERA_THETA
    height_shade.CAMERA_RADIUS = CAMERA_RADIUS
    height_shade.CAMERA_FOV = CAMERA_FOV
    height_shade.CAMERA_ZSCALE = CAMERA_ZSCALE
    height_shade.AMBIENT = CONTOUR_RENDER_AMBIENT
    height_shade.SHADOW = CONTOUR_RENDER_SHADOW
    height_shade.NORMAL_STRENGTH = CONTOUR_RENDER_NORMAL_STRENGTH
    height_shade.HEIGHT_AO_STRENGTH = CONTOUR_RENDER_HEIGHT_AO_STRENGTH
    height_shade.build_height_dem = build_contour_height_dem
    height_shade.cleanup_snapshot = cleanup_snapshot
    height_shade.configure_base_module()


def main() -> int:
    configure_base_module()
    return height_shade.main()


if __name__ == "__main__":
    raise SystemExit(main())
