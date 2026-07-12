"""
Germany Population Density -- Lajolla Height-Shade Style

Based on ``poland_population_spikes_height_shade.py``.  The rendering and
cleanup pipeline is shared with the Poland example, while this script swaps in
Germany's WorldPop density raster, Lajolla coloring, and Germany-specific
output labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poland_population_spikes as base
import poland_population_spikes_height_shade as height_shade


_POLAND_HEIGHT_SHADE_CLEANUP = height_shade.cleanup_snapshot

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "deu_pd_2020_1km_UNadj.tif"
OUTPUT_DIR = REPO / "examples" / "out"
CLEAN_DEM = OUTPUT_DIR / "deu_pd_clean_height_shade.tif"
OVERLAY_PATH = OUTPUT_DIR / "germany_lajolla_height_shade_overlay.png"
SNAPSHOT_PATH = OUTPUT_DIR / "germany_spikes_lajolla_height_shade_raw.png"
FINAL_PATH = OUTPUT_DIR / "germany_population_spikes_lajolla_4k.png"

WINDOW_SIZE = base.WINDOW_SIZE
SNAP_SIZE = base.SNAP_SIZE
BG_COLOR = base.BG_COLOR
HEIGHT_SHADE_PALETTE_GAMMA = 1.7
LEGEND_SCALE = 1.5
GERMANY_CAMERA_PHI = 90.0
GERMANY_CAMERA_THETA = 55.0
PLATE_BG_RGB = (255, 255, 255)
TEXT_RGB = (0, 0, 0)
RENDER_BG_RGB = (3, 3, 5)
TITLE_FONT_SIZE = 128
SUBTITLE_FONT_SIZE = 120
CAPTION_FONT_SIZE = 68
CITY_LABEL_FONT_SIZE = 50
GERMANY_LONLAT_BOUNDS = (5.70, 47.05, 15.35, 55.25)
GERMANY_CITY_LABELS = (
    ("Berlin", 13.4050, 52.5200, 42, -74),
    ("Hamburg", 9.9937, 53.5511, -244, -76),
    ("Munich", 11.5820, 48.1351, 42, 38),
    ("Cologne", 6.9603, 50.9375, -224, -70),
    ("Frankfurt", 8.6821, 50.1109, 42, 34),
)


def get_lajolla_lut() -> np.ndarray:
    """Return cmcrameri Lajolla in its standard display RGB order."""
    import cmcrameri.cm as cm

    return np.array(cm.lajolla.colors, dtype=np.float32)


def build_lajolla_lut_srgb() -> np.ndarray:
    """Build a 256-entry sRGB uint8 Lajolla LUT."""
    lajolla = get_lajolla_lut()
    return (np.clip(lajolla, 0.0, 1.0) * 255).astype(np.uint8)


def apply_palette_value_gamma(lut: np.ndarray, gamma: float) -> np.ndarray:
    """Return a LUT with low values compressed toward the palette floor."""
    values = np.linspace(0.0, 1.0, lut.shape[0], dtype=np.float32)
    indices = np.clip(
        (np.power(values, gamma) * (lut.shape[0] - 1)).astype(np.int32),
        0,
        lut.shape[0] - 1,
    )
    return lut[indices]


def build_lajolla_height_shade_lut_srgb() -> np.ndarray:
    """Build the display LUT used by the height-shade render and legend."""
    return apply_palette_value_gamma(build_lajolla_lut_srgb(), HEIGHT_SHADE_PALETTE_GAMMA)


def generate_lajolla_overlay(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> Path:
    """Render Lajolla-colored RGBA overlay from population density data."""
    lajolla = build_lajolla_lut_srgb()

    norm = np.clip(data / max(clip_max, 1e-6), 0.0, 1.0)
    indices = np.clip((norm * 255).astype(np.int32), 0, 255)

    h, w = data.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(3):
        overlay[:, :, c] = lajolla[indices, c]
    overlay[:, :, 3] = 255 * valid_mask.astype(np.uint8)

    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(str(OVERLAY_PATH))
    print(f"  Overlay saved  : {OVERLAY_PATH}")
    return OVERLAY_PATH


def make_lajolla_legend(bar_w: int = 36, bar_h: int = 700, clip_max: float = 2072.0) -> Image.Image:
    """Vertical Lajolla gradient bar with tick labels."""
    lajolla_lut = build_lajolla_height_shade_lut_srgb()
    scale = LEGEND_SCALE
    scaled = lambda value: int(round(value * scale))
    bar_w = scaled(bar_w)
    bar_h = scaled(bar_h)

    bar = np.zeros((bar_h, bar_w, 4), dtype=np.uint8)
    for y in range(bar_h):
        t = 1.0 - (y / (bar_h - 1))
        idx = int(np.clip(t * 255, 0, 255))
        bar[y, :, :3] = lajolla_lut[idx]
        bar[y, :, 3] = 255
    bar_img = Image.fromarray(bar)

    font = ImageFont.truetype(base.FONT_LIGHT, scaled(40))
    title_font = ImageFont.truetype(base.FONT_REGULAR, scaled(36))

    label_gap = scaled(16)
    label_w = scaled(240)
    title_h = scaled(72)
    total_w = bar_w + label_gap + label_w
    total_h = title_h + bar_h + scaled(28)

    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), "people / km\u00b2", font=title_font, fill=TEXT_RGB + (255,))
    canvas.paste(bar_img, (0, title_h))

    n_ticks = 6
    for i in range(n_ticks):
        t = i / (n_ticks - 1)
        value = t * clip_max
        y_pos = title_h + int((1.0 - t) * (bar_h - 1))
        label = f"{value:,.0f}"
        draw.text(
            (bar_w + scaled(18), y_pos - scaled(18)),
            label,
            font=font,
            fill=TEXT_RGB + (255,),
        )
        draw.line(
            [(bar_w, y_pos), (bar_w + scaled(10), y_pos)],
            fill=TEXT_RGB + (255,),
            width=max(1, scaled(2)),
        )

    return canvas


def replace_render_background(render: Image.Image) -> Image.Image:
    """Replace cropped viewer background pixels with the plate background."""
    arr = np.array(render.convert("RGB"))
    bg = np.array(RENDER_BG_RGB, dtype=np.uint8)
    arr[np.all(arr == bg.reshape(1, 1, 3), axis=2)] = PLATE_BG_RGB
    return Image.fromarray(arr)


def content_bounds(render: np.ndarray, pad: int = 140) -> tuple[int, int, int, int, int, int, int, int]:
    """Return content and padded crop bounds for the rendered map."""
    bg = np.array(RENDER_BG_RGB, dtype=np.int16)
    diff = np.abs(render.astype(np.int16) - bg)
    content_mask = np.any(diff > 2, axis=2)
    if not content_mask.any():
        height, width = render.shape[:2]
        return 0, 0, width, height, 0, 0, width, height

    ys, xs = np.where(content_mask)
    content_x0 = int(xs.min())
    content_x1 = int(xs.max()) + 1
    content_y0 = int(ys.min())
    content_y1 = int(ys.max()) + 1
    crop_x0 = max(0, content_x0 - pad)
    crop_x1 = min(render.shape[1], content_x1 + pad)
    crop_y0 = max(0, content_y0 - pad)
    crop_y1 = min(render.shape[0], content_y1 + pad)
    return content_x0, content_y0, content_x1, content_y1, crop_x0, crop_y0, crop_x1, crop_y1


def content_rect_on_plate(
    render: np.ndarray,
    map_rect: tuple[int, int, int, int],
    *,
    pad: int = 140,
) -> tuple[float, float, float, float]:
    """Map the render content bounds into final-plate coordinates."""
    content_x0, content_y0, content_x1, content_y1, crop_x0, crop_y0, crop_x1, crop_y1 = (
        content_bounds(render, pad=pad)
    )
    map_x, map_y, map_w, map_h = map_rect
    crop_w = max(1, crop_x1 - crop_x0)
    crop_h = max(1, crop_y1 - crop_y0)
    scale_x = map_w / crop_w
    scale_y = map_h / crop_h
    return (
        map_x + (content_x0 - crop_x0) * scale_x,
        map_y + (content_y0 - crop_y0) * scale_y,
        map_x + (content_x1 - crop_x0) * scale_x,
        map_y + (content_y1 - crop_y0) * scale_y,
    )


def project_city_to_plate(lon: float, lat: float, content_rect: tuple[float, float, float, float]) -> tuple[int, int]:
    """Project lon/lat into the upright Germany map content rectangle."""
    min_lon, min_lat, max_lon, max_lat = GERMANY_LONLAT_BOUNDS
    x0, y0, x1, y1 = content_rect
    x = x0 + ((lon - min_lon) / (max_lon - min_lon)) * (x1 - x0)
    y = y0 + ((max_lat - lat) / (max_lat - min_lat)) * (y1 - y0)
    return int(round(x)), int(round(y))


def draw_city_labels(canvas: Image.Image, content_rect: tuple[float, float, float, float]) -> None:
    """Draw the top five German city labels on the final plate."""
    label_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(label_layer)
    font = ImageFont.truetype(base.FONT_REGULAR, CITY_LABEL_FONT_SIZE)
    label_fill = (255, 255, 255, 238)
    label_outline = (0, 0, 0, 210)
    leader_fill = (0, 0, 0, 235)
    text_fill = (0, 0, 0, 255)
    pad_x = 18
    pad_y = 10
    radius = 8

    for name, lon, lat, offset_x, offset_y in GERMANY_CITY_LABELS:
        point_x, point_y = project_city_to_plate(lon, lat, content_rect)
        label_x = point_x + offset_x
        label_y = point_y + offset_y
        text_box = draw.textbbox((0, 0), name, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        box = (
            label_x,
            label_y,
            label_x + text_w + pad_x * 2,
            label_y + text_h + pad_y * 2,
        )
        draw.line(
            [(point_x, point_y), (label_x + pad_x, label_y + pad_y + text_h // 2)],
            fill=leader_fill,
            width=4,
        )
        draw.ellipse(
            (point_x - 12, point_y - 12, point_x + 12, point_y + 12),
            fill=(255, 255, 255, 245),
            outline=leader_fill,
            width=4,
        )
        draw.rounded_rectangle(box, radius=radius, fill=label_fill, outline=label_outline, width=3)
        draw.text((label_x + pad_x, label_y + pad_y - text_box[1]), name, font=font, fill=text_fill)

    canvas.paste(label_layer, (0, 0), label_layer)


def zero_floor_mask(raw: np.ndarray) -> np.ndarray:
    """Find flat, low-saturation floor pixels that represent zero density."""
    rgb = raw[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    warmth = rgb[:, :, 0] - rgb[:, :, 2]
    bg_mask = (lum < 25.0) & (warmth < 8.0)

    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)

    return (~bg_mask) & (lum >= 85.0) & (lum < 165.0) & (sat < 0.18) & (warmth < 36.0)


def cleanup_snapshot(raw: np.ndarray) -> np.ndarray:
    """Clean up Germany render while preserving shared height-shade variation."""
    result = _POLAND_HEIGHT_SHADE_CLEANUP(raw)
    result[zero_floor_mask(raw)] = build_lajolla_height_shade_lut_srgb()[0]
    return result



def configure_base_module() -> None:
    """Point the shared Poland helpers at Germany-specific data and outputs."""
    base.DATA_PATH = DATA_PATH
    base.OUTPUT_DIR = OUTPUT_DIR
    base.CLEAN_DEM = CLEAN_DEM
    base.OVERLAY_PATH = OVERLAY_PATH
    base.SNAPSHOT_PATH = SNAPSHOT_PATH
    base.FINAL_PATH = FINAL_PATH
    base.compose_final_plate = compose_final_plate
    base.generate_magma_overlay = generate_lajolla_overlay
    base.build_magma_lut_srgb = build_lajolla_height_shade_lut_srgb
    base.make_magma_legend = make_lajolla_legend
    height_shade.cleanup_snapshot = cleanup_snapshot

    height_shade.OUTPUT_DIR = OUTPUT_DIR
    height_shade.CLEAN_DEM = CLEAN_DEM
    height_shade.OVERLAY_PATH = OVERLAY_PATH
    height_shade.SNAPSHOT_PATH = SNAPSHOT_PATH
    height_shade.FINAL_PATH = FINAL_PATH
    height_shade.WINDOW_SIZE = WINDOW_SIZE
    height_shade.SNAP_SIZE = SNAP_SIZE
    height_shade.BG_COLOR = BG_COLOR
    height_shade.CAMERA_PHI = GERMANY_CAMERA_PHI
    height_shade.CAMERA_THETA = GERMANY_CAMERA_THETA
    height_shade.configure_base_module()


def compose_final_plate(magma_img: np.ndarray, clip_max: float) -> None:
    """Compose post-processed render + title + legend into final 4K plate."""
    plate_size = SNAP_SIZE
    canvas = Image.new("RGB", (plate_size, plate_size), PLATE_BG_RGB)
    draw = ImageDraw.Draw(canvas)

    margin_top = 220
    margin_bottom = 120
    margin_left = 90
    margin_right = 90

    map_w = plate_size - margin_left - margin_right
    map_h = plate_size - margin_top - margin_bottom

    map_rect = (margin_left, margin_top, map_w, map_h)
    label_content_rect = content_rect_on_plate(magma_img, map_rect, pad=140)
    render = replace_render_background(base.crop_render_to_content(magma_img, pad=140))
    render = render.resize((map_w, map_h), Image.LANCZOS)
    canvas.paste(render, (margin_left, margin_top))

    draw_city_labels(canvas, label_content_rect)

    title_font = ImageFont.truetype(base.FONT_BOLD, TITLE_FONT_SIZE)
    subtitle_font = ImageFont.truetype(base.FONT_LIGHT, SUBTITLE_FONT_SIZE)
    title_text = "POPULATION DENSITY"
    subtitle_text = "GERMANY  \u00b7  2020  \u00b7  1 km resolution"
    title_pos = (margin_left, 42)
    draw.text(title_pos, title_text, font=title_font, fill=TEXT_RGB)
    title_bbox = draw.textbbox(title_pos, title_text, font=title_font)
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
    subtitle_x = title_bbox[2] + 56
    title_center_y = (title_bbox[1] + title_bbox[3]) / 2
    subtitle_center_offset = (subtitle_bbox[1] + subtitle_bbox[3]) / 2
    subtitle_y = int(round(title_center_y - subtitle_center_offset))
    draw.text(
        (subtitle_x, subtitle_y),
        subtitle_text,
        font=subtitle_font,
        fill=TEXT_RGB,
    )

    draw.line([(margin_left, 258), (margin_left + 760, 258)], fill=TEXT_RGB, width=3)

    legend = make_lajolla_legend(bar_w=46, bar_h=400, clip_max=clip_max)
    legend_x = margin_left
    legend_y = 316
    canvas.paste(legend, (legend_x, legend_y), legend)

    attr_font = ImageFont.truetype(base.FONT_LIGHT, CAPTION_FONT_SIZE)
    attr_text = "Data: WorldPop 2020 UN-adjusted  \u00b7  Visualization: forge3d  \u00b7  milos makes maps"
    attr_bbox = draw.textbbox((0, 0), attr_text, font=attr_font)
    attr_h = attr_bbox[3] - attr_bbox[1]
    draw.text(
        (legend_x + legend.width + 70, plate_size - 74 - attr_h - attr_bbox[1]),
        attr_text,
        font=attr_font,
        fill=TEXT_RGB,
    )

    canvas.save(str(FINAL_PATH), "PNG")
    print(f"  Final plate    : {FINAL_PATH}")


def main() -> int:
    configure_base_module()
    return height_shade.main()


if __name__ == "__main__":
    raise SystemExit(main())
