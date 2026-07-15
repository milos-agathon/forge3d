"""Compatibility adapters for the native RGB MSDF atlas baker."""

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


def _default_font_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "fonts" / "NotoSansLatin-subset.ttf"


def bake_atlas(
    font_path: str | Path | None = None,
    charset: Sequence[str] | str = DEFAULT_LATIN_CHARSET,
    *,
    font_size: int = 32,
    px_range: int = 8,
    padding: int = 4,
) -> BakedAtlas:
    """Bake deterministic RGB MSDF glyphs through the native outline pipeline."""
    from .text import bake_msdf_atlas

    font = Path(font_path) if font_path is not None else _default_font_path()
    characters = "".join(sorted({str(value)[0] for value in charset if str(value)}))
    baked = bake_msdf_atlas(
        [font], characters, float(font_size), float(px_range), int(padding)
    )
    metrics = dict(baked["metrics"])
    metrics["font_source"] = (
        str(font) if font_path is not None else "forge3d/data/fonts/NotoSansLatin-subset.ttf"
    )
    return BakedAtlas(
        image=np.asarray(baked["image"], dtype=np.uint8),
        metrics=validate_atlas_metrics(metrics),
    )


def validate_atlas_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native MSDF metadata and normalize glyph number fields."""
    required_top = (
        "kind",
        "font_size",
        "line_height",
        "baseline",
        "px_range",
        "padding",
        "channels",
        "width",
        "height",
        "bake_ms",
        "byte_count",
        "font_source",
        "glyphs",
    )
    missing = [key for key in required_top if key not in metrics]
    if missing:
        raise ValueError(f"Atlas metrics missing field(s): {', '.join(missing)}")
    if metrics["kind"] != "msdf_font_atlas" or int(metrics["channels"]) != 3:
        raise ValueError("Atlas metrics require kind='msdf_font_atlas' and channels=3")
    glyphs = metrics["glyphs"]
    if not isinstance(glyphs, Mapping) or not glyphs:
        raise ValueError("Atlas metrics require a non-empty glyphs mapping")
    normalized = dict(metrics)
    normalized["channels"] = 3
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


def save_atlas(
    atlas: BakedAtlas, png_path: str | Path, json_path: str | Path
) -> tuple[Path, Path]:
    """Save the RGB image with the in-tree encoder and canonical metadata JSON."""
    from ._png import save_png

    png = Path(png_path)
    metrics = Path(json_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    save_png(png, atlas.image)
    metrics.write_text(
        json.dumps(validate_atlas_metrics(atlas.metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return png, metrics


def load_atlas_metrics(path: str | Path) -> dict[str, Any]:
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
