"""
France Population Density -- Roma Height-Shade Style

Based on ``germany_population_spikes_height_shade.py``. The rendering and
cleanup pipeline is shared with the Poland/Germany examples, while this script
uses France's WorldPop density raster and a reversed cmcrameri Roma palette so
low values are cold and high values are warm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import germany_population_spikes_height_shade as country_style
import poland_population_spikes as base
import poland_population_spikes_height_shade as height_shade

_BASE_LOAD_POPULATION_DATA = base.load_population_data
_HEIGHT_SHADE_BUILD_HEIGHT_DEM = height_shade.build_height_dem

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data" / "fra_pd_2020_1km_UNadj.tif"
DATA_DOWNLOAD_URL = (
    "https://data.worldpop.org/GIS/Population_Density/"
    "Global_2000_2020_1km_UNadj/2020/FRA/fra_pd_2020_1km_UNadj.tif"
)
OUTPUT_DIR = REPO / "examples" / "out"
CLEAN_DEM = OUTPUT_DIR / "fra_pd_clean_height_shade.tif"
OVERLAY_PATH = OUTPUT_DIR / "france_roma_height_shade_overlay.png"
SNAPSHOT_PATH = OUTPUT_DIR / "france_spikes_roma_height_shade_raw.png"
FINAL_PATH = OUTPUT_DIR / "france_population_spikes_roma_4k.png"

WINDOW_SIZE = base.WINDOW_SIZE
SNAP_SIZE = 6144
BG_COLOR = base.BG_COLOR
HEIGHT_SHADE_PALETTE_GAMMA = 1.62
RASTER_SUPERSAMPLE = 5
SPIKE_HEIGHT_MULTIPLIER = 5.75
LEGEND_SCALE = 1.75
LEGEND_Y = 135
FRANCE_CAMERA_PHI = 88.0
FRANCE_CAMERA_THETA = 55.0
FRANCE_CAMERA_RADIUS = 9300.0
FRANCE_CAMERA_FOV = 42.0
FINAL_CROP_PAD = 144
FINAL_PLATE_W = 4800
FINAL_PLATE_H = 3200
MAP_AREA_X = 63
MAP_AREA_Y = 155
MAP_AREA_W = 4680
MAP_AREA_H = 2985
INSET_AREA_W = 960
INSET_AREA_H = 750
INSET_AREA_X = 3765
INSET_AREA_Y = 2135
PLATE_BG_RGB = (250, 249, 246)
TEXT_RGB = (20, 24, 28)
MUTED_TEXT_RGB = (82, 86, 91)
RULE_RGB = (168, 164, 156)
RENDER_BG_RGB = (3, 3, 5)
TITLE_FONT_SIZE = 156
SUBTITLE_FONT_SIZE = 81
LEGEND_NOTE_FONT_SIZE = 51
CAPTION_FONT_SIZE = 57
INSET_LABEL_FONT_SIZE = 51

# Exact natural-order samples from R scico::scico(33, palette = "roma").
# Keeping the reference samples local makes this example deterministic without
# turning cmcrameri into a mandatory runtime dependency.
SCICO_ROMA_33 = np.array(
    [
        [0.494118, 0.090196, 0.000000],
        [0.521569, 0.168627, 0.019608],
        [0.556863, 0.231373, 0.043137],
        [0.584314, 0.290196, 0.066667],
        [0.611765, 0.341176, 0.090196],
        [0.635294, 0.392157, 0.113725],
        [0.658824, 0.443137, 0.133333],
        [0.686275, 0.494118, 0.160784],
        [0.709804, 0.545098, 0.192157],
        [0.737255, 0.603922, 0.231373],
        [0.764706, 0.662745, 0.286275],
        [0.792157, 0.725490, 0.360784],
        [0.811765, 0.788235, 0.439216],
        [0.819608, 0.839216, 0.533333],
        [0.819608, 0.882353, 0.623529],
        [0.796078, 0.905882, 0.701961],
        [0.752941, 0.913725, 0.760784],
        [0.698039, 0.913725, 0.803922],
        [0.627451, 0.894118, 0.827451],
        [0.549020, 0.862745, 0.839216],
        [0.466667, 0.819608, 0.843137],
        [0.388235, 0.772549, 0.831373],
        [0.321569, 0.717647, 0.815686],
        [0.262745, 0.666667, 0.796078],
        [0.219608, 0.611765, 0.776471],
        [0.188235, 0.560784, 0.752941],
        [0.164706, 0.513725, 0.733333],
        [0.145098, 0.462745, 0.713725],
        [0.129412, 0.411765, 0.690196],
        [0.113725, 0.360784, 0.666667],
        [0.094118, 0.305882, 0.643137],
        [0.062745, 0.247059, 0.619608],
        [0.007843, 0.192157, 0.596078],
    ],
    dtype=np.float32,
)


def get_roma_lut() -> np.ndarray:
    """Return a 256-sample Roma LUT with low cold and high warm."""
    sample_x = np.linspace(0.0, 1.0, SCICO_ROMA_33.shape[0], dtype=np.float32)
    lut_x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    reversed_samples = SCICO_ROMA_33[::-1]
    return np.column_stack(
        [np.interp(lut_x, sample_x, reversed_samples[:, channel]) for channel in range(3)]
    ).astype(np.float32)


def build_roma_lut_srgb() -> np.ndarray:
    """Build a 256-entry sRGB uint8 Roma LUT."""
    roma = get_roma_lut()
    return (np.clip(roma, 0.0, 1.0) * 255).astype(np.uint8)


def apply_palette_value_gamma(lut: np.ndarray, gamma: float) -> np.ndarray:
    """Return a LUT with low values compressed toward the palette floor."""
    values = np.linspace(0.0, 1.0, lut.shape[0], dtype=np.float32)
    indices = np.clip(
        (np.power(values, gamma) * (lut.shape[0] - 1)).astype(np.int32),
        0,
        lut.shape[0] - 1,
    )
    return lut[indices]


def build_roma_height_shade_lut_srgb() -> np.ndarray:
    """Build the display LUT used by the height-shade render and legend."""
    return apply_palette_value_gamma(build_roma_lut_srgb(), HEIGHT_SHADE_PALETTE_GAMMA)


def generate_roma_overlay(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> Path:
    """Render Roma-colored RGBA overlay from population density data."""
    roma = build_roma_lut_srgb()

    norm = np.clip(data / max(clip_max, 1e-6), 0.0, 1.0)
    indices = np.clip((norm * 255).astype(np.int32), 0, 255)

    h, w = data.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(3):
        overlay[:, :, c] = roma[indices, c]
    overlay[:, :, 3] = 255 * valid_mask.astype(np.uint8)

    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(str(OVERLAY_PATH))
    print(f"  Overlay saved  : {OVERLAY_PATH}")
    return OVERLAY_PATH


def supersample_population_grid(data: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upsample population data and mask for a denser rendered spike raster."""
    if RASTER_SUPERSAMPLE <= 1:
        return data.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)

    height, width = data.shape
    size = (width * RASTER_SUPERSAMPLE, height * RASTER_SUPERSAMPLE)
    data_img = Image.fromarray(np.ascontiguousarray(data, dtype=np.float32), mode="F")
    valid_img = Image.fromarray(valid_mask.astype(np.uint8) * 255, mode="L")
    up_data = np.array(data_img.resize(size, Image.BICUBIC), dtype=np.float32)
    up_valid = np.array(valid_img.resize(size, Image.NEAREST), dtype=np.uint8) > 0
    up_data[~up_valid] = 0.0
    return up_data, up_valid


def load_population_data_supersampled() -> tuple[np.ndarray, float, np.ndarray]:
    """Load France raster through the shared path, then supersample it."""
    data, clip_max, valid_mask = _BASE_LOAD_POPULATION_DATA()
    if RASTER_SUPERSAMPLE <= 1:
        return data, clip_max, valid_mask

    up_data, up_valid = supersample_population_grid(data, valid_mask)
    print(f"  Supersample    : {RASTER_SUPERSAMPLE}x -> {up_data.shape[1]}x{up_data.shape[0]}")
    return up_data, clip_max, up_valid


def build_france_height_dem(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> np.ndarray:
    """Build France height data with taller population spikes."""
    height = _HEIGHT_SHADE_BUILD_HEIGHT_DEM(data, clip_max, valid_mask)
    return (height * SPIKE_HEIGHT_MULTIPLIER).astype(np.float32)


def make_roma_legend(bar_w: int = 36, bar_h: int = 700, clip_max: float = 2072.0) -> Image.Image:
    """Vertical Roma gradient bar with tick labels."""
    roma_lut = build_roma_height_shade_lut_srgb()
    scale = LEGEND_SCALE
    scaled = lambda value: int(round(value * scale))
    bar_w = scaled(bar_w)
    bar_h = scaled(bar_h)

    bar = np.zeros((bar_h, bar_w, 4), dtype=np.uint8)
    for y in range(bar_h):
        t = 1.0 - (y / (bar_h - 1))
        idx = int(np.clip(t * 255, 0, 255))
        bar[y, :, :3] = roma_lut[idx]
        bar[y, :, 3] = 255
    bar_img = Image.fromarray(bar)

    font = ImageFont.truetype(base.FONT_LIGHT, scaled(36))
    title_font = ImageFont.truetype(base.FONT_REGULAR, scaled(34))
    note_font = ImageFont.truetype(base.FONT_LIGHT, scaled(22))

    label_gap = scaled(16)
    label_w = scaled(240)
    title_h = scaled(72)
    note_h = scaled(64)
    total_w = bar_w + label_gap + label_w
    total_h = title_h + bar_h + note_h

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
        if i == n_ticks - 1:
            label = f"\u2265 {label}"
        draw.text(
            (bar_w + scaled(18), y_pos - scaled(18)),
            label,
            font=font,
            fill=MUTED_TEXT_RGB + (255,),
        )
        draw.line(
            [(bar_w, y_pos), (bar_w + scaled(10), y_pos)],
            fill=RULE_RGB + (255,),
            width=max(1, scaled(2)),
        )

    note_y = title_h + bar_h + scaled(18)
    draw.text((0, note_y), "color and height capped at p99", font=note_font, fill=MUTED_TEXT_RGB + (255,))

    return canvas


def normalize_plate_background(render: Image.Image) -> Image.Image:
    """Convert inherited pure-white crop background to the France plate tone."""
    arr = np.array(render.convert("RGB"))
    white_mask = np.all(arr >= 248, axis=2)
    arr[white_mask] = PLATE_BG_RGB
    return Image.fromarray(arr)


def render_content_mask(render: Image.Image) -> np.ndarray:
    """Return the non-background mask for a cropped render on the plate color."""
    arr = np.array(render.convert("RGB"), dtype=np.int16)
    bg = np.array(PLATE_BG_RGB, dtype=np.int16)
    diff = np.abs(arr - bg.reshape(1, 1, 3))
    return np.any(diff > 10, axis=2)


def mask_bounds(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    """Return padded bounds for a boolean mask."""
    if not mask.any():
        height, width = mask.shape[:2]
        return 0, 0, width, height
    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def largest_southeast_component_mask(mask: np.ndarray) -> np.ndarray:
    """Find Corsica as the largest detached southeast component."""
    try:
        from scipy.ndimage import label
    except ImportError:
        height, width = mask.shape
        yy, xx = np.mgrid[0:height, 0:width]
        return mask & (xx > width * 0.58) & (yy > height * 0.54)

    labels, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if count < 2:
        return np.zeros_like(mask, dtype=bool)

    areas = np.bincount(labels.ravel())
    areas[0] = 0
    main_label = int(np.argmax(areas))

    height, width = mask.shape
    best_label = 0
    best_score = 0.0
    for component_label in range(1, count + 1):
        if component_label == main_label or areas[component_label] < 800:
            continue
        ys, xs = np.where(labels == component_label)
        centroid_x = float(xs.mean()) / max(1, width)
        centroid_y = float(ys.mean()) / max(1, height)
        southeast = max(0.0, centroid_x - 0.48) + max(0.0, centroid_y - 0.48)
        score = float(areas[component_label]) * (1.0 + southeast * 3.0)
        if score > best_score:
            best_score = score
            best_label = component_label

    if best_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == best_label


def split_mainland_and_corsica(render: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    """Crop mainland France tightly and return Corsica as a separate inset image."""
    arr = np.array(render.convert("RGB"))
    content = render_content_mask(render)
    corsica_mask = largest_southeast_component_mask(content)
    if not corsica_mask.any():
        x0, y0, x1, y1 = mask_bounds(content, pad=64)
        return Image.fromarray(arr[y0:y1, x0:x1]), None

    mainland_arr = arr.copy()
    mainland_arr[corsica_mask] = PLATE_BG_RGB
    mainland_mask = content & ~corsica_mask
    mx0, my0, mx1, my1 = mask_bounds(mainland_mask, pad=72)
    cx0, cy0, cx1, cy1 = mask_bounds(corsica_mask, pad=140)
    return Image.fromarray(mainland_arr[my0:my1, mx0:mx1]), Image.fromarray(arr[cy0:cy1, cx0:cx1])


def fit_rect(render_size: tuple[int, int], area: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Fit an image size inside an x/y/w/h area while preserving aspect."""
    render_w, render_h = render_size
    area_x, area_y, area_w, area_h = area
    scale = min(area_w / max(1, render_w), area_h / max(1, render_h))
    out_w = int(round(render_w * scale))
    out_h = int(round(render_h * scale))
    out_x = area_x + (area_w - out_w) // 2
    out_y = area_y + (area_h - out_h) // 2
    return out_x, out_y, out_w, out_h


def draw_corsica_inset(canvas: Image.Image, corsica: Image.Image | None) -> None:
    """Place Corsica as a labeled inset instead of letting it drive the main crop."""
    if corsica is None:
        return
    draw = ImageDraw.Draw(canvas)
    rect = fit_rect(corsica.size, (INSET_AREA_X, INSET_AREA_Y, INSET_AREA_W, INSET_AREA_H))
    inset_x, inset_y, inset_w, inset_h = rect
    inset = corsica.resize((inset_w, inset_h), Image.LANCZOS)
    inset_mask = Image.fromarray((render_content_mask(inset) * 255).astype(np.uint8), mode="L")
    canvas.paste(inset, (inset_x, inset_y), inset_mask)

    label_font = ImageFont.truetype(base.FONT_REGULAR, INSET_LABEL_FONT_SIZE)
    label = "Corsica"
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_x = inset_x + inset_w - (label_bbox[2] - label_bbox[0])
    label_y = inset_y + inset_h + 12
    draw.line([(inset_x, inset_y + inset_h + 2), (inset_x + inset_w, inset_y + inset_h + 2)], fill=RULE_RGB, width=2)
    draw.text((label_x, label_y), label, font=label_font, fill=MUTED_TEXT_RGB)


def zero_floor_mask(raw: np.ndarray) -> np.ndarray:
    """Find flat, low-saturation floor pixels that represent zero density."""
    rgb = raw[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    coldness = rgb[:, :, 2] - rgb[:, :, 0]
    bg_mask = (lum < 25.0) & (coldness > -8.0)

    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)

    return (~bg_mask) & (lum >= 55.0) & (lum < 150.0) & (sat < 0.24) & (coldness > 8.0)


def cleanup_snapshot(raw: np.ndarray) -> np.ndarray:
    """Clean up France render by remapping lit pixels back onto Roma."""
    rgb = raw[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    mean_c = np.mean(rgb, axis=2, keepdims=True)
    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)
    bg_mask = (lum < 35.0) & (sat < 0.28)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    blue_dominance = np.clip((blue - red - 18.0) / 85.0, 0.0, 1.0)
    warm_dominance = np.clip((red - blue + 18.0) / 105.0, 0.0, 1.0)
    green_mid = np.clip((green - red + 18.0) / 95.0, 0.0, 1.0) * np.clip(
        (green - blue + 42.0) / 95.0, 0.0, 1.0
    )
    brightness = np.clip((lum - 55.0) / 175.0, 0.0, 1.0)

    value = brightness * (1.0 - 0.92 * blue_dominance)
    value = np.maximum(value, 1.35 * green_mid)
    value = np.maximum(value, warm_dominance)
    value = np.where(
        (green_mid > 0.30) & (warm_dominance < 0.35),
        np.minimum(value, 0.58),
        value,
    )
    value = np.clip(value, 0.0, 1.0)

    floor_mask = ((blue_dominance > 0.42) & (red < 170.0) & (lum < 185.0)) | zero_floor_mask(raw)
    value[floor_mask] = 0.0

    lut = build_roma_height_shade_lut_srgb()
    indices = np.clip((value * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
    result = lut[indices].copy()
    result[bg_mask] = RENDER_BG_RGB
    return result


def configure_base_module() -> None:
    """Point the shared Poland helpers at France-specific data and outputs."""
    base.DATA_PATH = DATA_PATH
    base.load_population_data = load_population_data_supersampled
    base.OUTPUT_DIR = OUTPUT_DIR
    base.CLEAN_DEM = CLEAN_DEM
    base.OVERLAY_PATH = OVERLAY_PATH
    base.SNAPSHOT_PATH = SNAPSHOT_PATH
    base.FINAL_PATH = FINAL_PATH
    base.compose_final_plate = compose_final_plate
    base.generate_magma_overlay = generate_roma_overlay
    base.build_magma_lut_srgb = build_roma_height_shade_lut_srgb
    base.make_magma_legend = make_roma_legend
    height_shade.cleanup_snapshot = cleanup_snapshot

    height_shade.OUTPUT_DIR = OUTPUT_DIR
    height_shade.CLEAN_DEM = CLEAN_DEM
    height_shade.OVERLAY_PATH = OVERLAY_PATH
    height_shade.SNAPSHOT_PATH = SNAPSHOT_PATH
    height_shade.FINAL_PATH = FINAL_PATH
    height_shade.WINDOW_SIZE = WINDOW_SIZE
    height_shade.SNAP_SIZE = SNAP_SIZE
    height_shade.BG_COLOR = BG_COLOR
    height_shade.CAMERA_PHI = FRANCE_CAMERA_PHI
    height_shade.CAMERA_THETA = FRANCE_CAMERA_THETA
    height_shade.CAMERA_RADIUS = FRANCE_CAMERA_RADIUS
    height_shade.CAMERA_FOV = FRANCE_CAMERA_FOV
    height_shade.build_height_dem = build_france_height_dem
    height_shade.configure_base_module()


def ensure_france_raster_available() -> None:
    """Fail early with the exact WorldPop raster location if data is missing."""
    if DATA_PATH.exists():
        return

    print("ERROR: Missing France population raster")
    print(f"  Expected path : {DATA_PATH}")
    print(f"  Download URL  : {DATA_DOWNLOAD_URL}")
    print(
        "  PowerShell    : "
        f"Invoke-WebRequest -Uri \"{DATA_DOWNLOAD_URL}\" -OutFile \"{DATA_PATH}\""
    )
    raise SystemExit(1)


def fit_map_rect(render_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return an aspect-preserving final-plate rect for the rendered map."""
    return fit_rect(render_size, (MAP_AREA_X, MAP_AREA_Y, MAP_AREA_W, MAP_AREA_H))


def content_rect_for_crop(
    bounds: tuple[int, int, int, int, int, int, int, int],
    map_rect: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map raw-render content bounds into the fitted plate rectangle."""
    content_x0, content_y0, content_x1, content_y1, crop_x0, crop_y0, crop_x1, crop_y1 = bounds
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


def compose_final_plate(roma_img: np.ndarray, clip_max: float) -> None:
    """Compose post-processed render + title + legend into final 4K plate."""
    plate_w = FINAL_PLATE_W
    plate_h = FINAL_PLATE_H
    canvas = Image.new("RGB", (plate_w, plate_h), PLATE_BG_RGB)
    draw = ImageDraw.Draw(canvas)

    bounds = country_style.content_bounds(roma_img, pad=FINAL_CROP_PAD)
    _, _, _, _, crop_x0, crop_y0, crop_x1, crop_y1 = bounds
    render = country_style.replace_render_background(
        Image.fromarray(roma_img).crop((crop_x0, crop_y0, crop_x1, crop_y1))
    )
    render = normalize_plate_background(render)
    mainland, corsica = split_mainland_and_corsica(render)
    map_rect = fit_map_rect(mainland.size)
    map_x, map_y, map_w, map_h = map_rect
    map_render = mainland.resize((map_w, map_h), Image.LANCZOS)
    canvas.paste(map_render, (map_x, map_y))

    draw_corsica_inset(canvas, corsica)

    title_font = ImageFont.truetype(base.FONT_BOLD, TITLE_FONT_SIZE)
    subtitle_font = ImageFont.truetype(base.FONT_LIGHT, SUBTITLE_FONT_SIZE)
    title_text = "Population density"
    subtitle_text = "France  \u00b7  2020  \u00b7  1 km grid"
    title_pos = (135, 69)
    draw.text(title_pos, title_text, font=title_font, fill=TEXT_RGB)
    title_bbox = draw.textbbox(title_pos, title_text, font=title_font)
    draw.text(
        (title_pos[0], title_bbox[3] + 14),
        subtitle_text,
        font=subtitle_font,
        fill=MUTED_TEXT_RGB,
    )

    legend = make_roma_legend(bar_w=46, bar_h=400, clip_max=clip_max)
    legend_x = plate_w - legend.width - 135
    legend_y = LEGEND_Y
    canvas.paste(legend, (legend_x, legend_y), legend)

    attr_font = ImageFont.truetype(base.FONT_LIGHT, CAPTION_FONT_SIZE)
    attr_text = "Data: WorldPop 2020 UN-adjusted  \u00b7  Rendered with forge3d  \u00b7  milos makes maps"
    attr_bbox = draw.textbbox((0, 0), attr_text, font=attr_font)
    attr_w = attr_bbox[2] - attr_bbox[0]
    attr_h = attr_bbox[3] - attr_bbox[1]
    draw.text(
        ((plate_w - attr_w) // 2, plate_h - 87 - attr_h - attr_bbox[1]),
        attr_text,
        font=attr_font,
        fill=MUTED_TEXT_RGB,
    )

    canvas.save(str(FINAL_PATH), "PNG")
    print(f"  Final plate    : {FINAL_PATH}")


def main() -> int:
    ensure_france_raster_available()
    configure_base_module()
    return height_shade.main()


if __name__ == "__main__":
    raise SystemExit(main())
