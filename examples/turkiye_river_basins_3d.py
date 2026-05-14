"""3D-style Turkiye river basins poster helpers.

The rendering pipeline can be extended from these deterministic helpers; tests
cover the poster aspect, camera defaults, river styling, and cache invalidation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


REFERENCE_ASPECT = 7706 / 5274
POSTER_CREDIT = "River basins of Turkiye. Reference cartography ©2023 Milos Popovic."
OVERLAY_STYLE_VERSION = "turkiye-river-basins-overlay-v2"

BASIN_PALETTE = [
    "#33a4db",
    "#69c17d",
    "#f0c75e",
    "#e58b63",
    "#9f7fd1",
]

CAMERA = {
    "exaggeration": 0.82,
    "radius": 3.15,
    "target": [0.0, 0.0, 0.0],
}
TERRAIN = {
    "theta": 26.0,
    "phi": 38.0,
    "ambient": 0.56,
    "shadow": 0.42,
}
RELIEF_TERRAIN = {
    **TERRAIN,
    "ambient": 0.34,
    "shadow": 0.72,
}
PBR = {
    "normal_strength": 0.85,
    "sun_visibility": {"mode": "soft"},
}
RELIEF_PBR = {
    "normal_strength": 1.45,
    "sun_visibility": {"mode": "hard"},
}

RIVER_ALPHA_MAP = {
    1: 64,
    2: 78,
    3: 92,
    4: 112,
    5: 136,
    6: 160,
    7: 188,
    8: 216,
    9: 238,
}


def _snapshot_dimensions(width: int) -> tuple[int, int]:
    return int(width), int(round(int(width) / REFERENCE_ASPECT))


def _subject_mask_from_white(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"), dtype=np.int16)
    rgb = arr[..., :3]
    alpha = arr[..., 3] > 0
    return alpha & np.any(rgb < 248, axis=2)


def _combine_render_passes(color_path: Path, relief_path: Path) -> Image.Image:
    color = np.asarray(Image.open(color_path).convert("RGBA"), dtype=np.float32)
    relief = np.asarray(Image.open(relief_path).convert("RGBA"), dtype=np.float32)
    relief_luma = np.mean(relief[..., :3], axis=2, keepdims=True) / 255.0
    shade = 0.58 + relief_luma * 0.62
    combined = color.copy()
    combined[..., :3] = np.clip(color[..., :3] * shade, 0, 255)
    combined[..., 2] = np.maximum(combined[..., 2], combined[..., 0] + 4)
    combined[..., 3] = color[..., 3]
    return Image.fromarray(combined.astype(np.uint8), mode="RGBA")


def _reframe_snapshot(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    src = source.convert("RGBA")
    mask = _subject_mask_from_white(src)
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    if not mask.any():
        return canvas
    ys, xs = np.nonzero(mask)
    crop = src.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    target_w = int(size[0] * 0.98)
    target_h = int(size[1] * 0.96)
    crop = ImageOps.contain(crop, (target_w, target_h))
    x = (size[0] - crop.width) // 2
    y = (size[1] - crop.height) // 2
    canvas.alpha_composite(crop, (x, y))
    return canvas


def _river_width_px(order: int, canvas_width: int) -> float:
    order = max(1, min(9, int(order)))
    return (0.00022 * canvas_width) * (order**1.8)


def _brighten_subject(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32)
    mask = _subject_mask_from_white(Image.fromarray(arr.astype(np.uint8), mode="RGBA"))
    rgb = arr[..., :3]
    luma = np.mean(rgb, axis=2)
    lift = np.where(luma < 80.0, 8.0, 22.0)
    rgb[mask] = np.clip(rgb[mask] + lift[mask, None], 0, 255)
    cool_mask = mask & (luma >= 80.0)
    rgb[cool_mask, 2] = np.maximum(rgb[cool_mask, 2], rgb[cool_mask, 0] + 86.0)
    arr[..., :3] = rgb
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _overlay_style_path(overlay_path: Path) -> Path:
    return overlay_path.with_suffix(overlay_path.suffix + ".style")


def _overlay_is_current(overlay_path: Path) -> bool:
    if not overlay_path.exists():
        return False
    sidecar = _overlay_style_path(overlay_path)
    if not sidecar.exists():
        return False
    return sidecar.read_text(encoding="utf-8").strip() == OVERLAY_STYLE_VERSION


def main() -> int:
    width, height = _snapshot_dimensions(4200)
    print(f"Turkiye river basins poster helpers ready for {width}x{height}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
