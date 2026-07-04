"""SDF font atlas baking and validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_LATIN_CHARSET = "".join(chr(codepoint) for codepoint in range(32, 127))


@dataclass(frozen=True)
class BakedAtlas:
    image: np.ndarray
    metrics: dict[str, Any]


def _font(font_path: str | Path | None, font_size: int) -> Any:
    from PIL import ImageFont

    if font_path is not None:
        return ImageFont.truetype(str(font_path), int(font_size))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", int(font_size))
    except Exception:
        return ImageFont.load_default()


def _glyph_bbox(font: Any, char: str) -> tuple[int, int, int, int]:
    try:
        bbox = font.getbbox(char)
    except Exception:
        bbox = None
    if bbox is None:
        width, height = font.getsize(char)
        return (0, 0, int(width), int(height))
    return tuple(int(value) for value in bbox)


def _glyph_advance(font: Any, char: str, fallback: int) -> float:
    try:
        return float(font.getlength(char))
    except Exception:
        return float(fallback)


def _sdf(mask: np.ndarray, px_range: int) -> np.ndarray:
    inside = mask > 0
    if not inside.any():
        return np.full(mask.shape, 128, dtype=np.uint8)
    try:
        from scipy.ndimage import distance_transform_edt

        dist_in = distance_transform_edt(inside)
        dist_out = distance_transform_edt(~inside)
        signed = dist_in - dist_out
        sdf = 0.5 + signed / max(1.0, float(px_range) * 2.0)
        return np.clip(sdf * 255.0, 0.0, 255.0).astype(np.uint8)
    except Exception:
        return np.where(inside, 255, 0).astype(np.uint8)


def bake_atlas(
    font_path: str | Path | None = None,
    charset: Sequence[str] | str = DEFAULT_LATIN_CHARSET,
    *,
    font_size: int = 32,
    px_range: int = 8,
    padding: int = 4,
) -> BakedAtlas:
    """Bake a single-channel SDF atlas image and metrics JSON payload."""

    from PIL import Image, ImageDraw

    glyphs = sorted({str(char)[0] for char in charset})
    font = _font(font_path, int(font_size))
    boxes = {char: _glyph_bbox(font, char) for char in glyphs}
    max_w = max((box[2] - box[0] for box in boxes.values()), default=font_size)
    max_h = max((box[3] - box[1] for box in boxes.values()), default=font_size)
    cell_w = int(max_w + padding * 2 + px_range * 2)
    cell_h = int(max_h + padding * 2 + px_range * 2)
    columns = max(1, int(np.ceil(np.sqrt(max(1, len(glyphs))))))
    rows = int(np.ceil(len(glyphs) / columns))
    width = columns * cell_w
    height = rows * cell_h
    atlas = np.zeros((height, width, 4), dtype=np.uint8)
    metrics: dict[str, Any] = {
        "kind": "sdf_font_atlas",
        "font_size": int(font_size),
        "line_height": int(round(font_size * 4 / 3)),
        "baseline": int(font_size),
        "px_range": int(px_range),
        "channels": 1,
        "width": int(width),
        "height": int(height),
        "glyphs": {},
    }

    for index, char in enumerate(glyphs):
        col = index % columns
        row = index // columns
        x = col * cell_w
        y = row * cell_h
        bbox = boxes[char]
        glyph_w = max(1, bbox[2] - bbox[0])
        glyph_h = max(1, bbox[3] - bbox[1])
        mask_image = Image.new("L", (cell_w, cell_h), 0)
        draw = ImageDraw.Draw(mask_image)
        draw_x = padding + px_range - bbox[0]
        draw_y = padding + px_range - bbox[1]
        draw.text((draw_x, draw_y), char, font=font, fill=255)
        sdf = _sdf(np.asarray(mask_image, dtype=np.uint8), int(px_range))
        atlas[y : y + cell_h, x : x + cell_w, 0] = sdf
        atlas[y : y + cell_h, x : x + cell_w, 1] = sdf
        atlas[y : y + cell_h, x : x + cell_w, 2] = sdf
        atlas[y : y + cell_h, x : x + cell_w, 3] = 255
        metrics["glyphs"][str(ord(char))] = {
            "x": int(x),
            "y": int(y),
            "w": int(cell_w),
            "h": int(cell_h),
            "ox": int(-padding - px_range),
            "oy": int(-padding - px_range),
            "adv": float(_glyph_advance(font, char, glyph_w)),
        }

    return BakedAtlas(image=atlas, metrics=metrics)


def validate_atlas_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate atlas metrics and return a normalized dict."""

    required_top = ("font_size", "line_height", "baseline", "glyphs")
    missing = [key for key in required_top if key not in metrics]
    if missing:
        raise ValueError(f"Atlas metrics missing field(s): {', '.join(missing)}")
    glyphs = metrics.get("glyphs")
    if not isinstance(glyphs, Mapping) or not glyphs:
        raise ValueError("Atlas metrics require a non-empty glyphs mapping")
    normalized = dict(metrics)
    normalized["glyphs"] = {}
    for key, value in glyphs.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Glyph {key!r} metrics must be a mapping")
        glyph = {}
        for field in ("x", "y", "w", "h", "ox", "oy", "adv"):
            if field not in value:
                raise ValueError(f"Glyph {key!r} missing metric {field!r}")
            glyph[field] = float(value[field])
        normalized["glyphs"][str(int(key))] = glyph
    return normalized


def save_atlas(atlas: BakedAtlas, png_path: str | Path, json_path: str | Path) -> tuple[Path, Path]:
    """Save atlas image and metrics JSON."""

    from PIL import Image

    png = Path(png_path)
    metrics = Path(json_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas.image, mode="RGBA").save(png)
    metrics.write_text(json.dumps(validate_atlas_metrics(atlas.metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return png, metrics


def load_atlas_metrics(path: str | Path) -> dict[str, Any]:
    """Load and validate an atlas metrics JSON file."""

    return validate_atlas_metrics(json.loads(Path(path).read_text(encoding="utf-8")))


def default_latin_atlas_paths() -> tuple[Path, Path]:
    base = Path(__file__).resolve().parent / "data" / "fonts"
    return base / "atlas_latin_default.png", base / "atlas_latin_default.json"


__all__ = [
    "BakedAtlas",
    "DEFAULT_LATIN_CHARSET",
    "bake_atlas",
    "default_latin_atlas_paths",
    "load_atlas_metrics",
    "save_atlas",
    "validate_atlas_metrics",
]
