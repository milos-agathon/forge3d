"""Deterministic native text shaping."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Mapping, Sequence

from ._native import get_native_module, get_native_symbol


class TextShapingError(ValueError):
    def __init__(self, message: str, diagnostics: list[dict]):
        super().__init__(message)
        self.diagnostics = diagnostics


# Exported types retain their import-time identities.
ShapedText = get_native_symbol("ShapedText", object)


def shape(
    text: str,
    font_chain: Sequence[str | Path],
    size: float,
    script: str | None = None,
    language: str | None = None,
    features: Mapping[str, bool] | None = None,
) -> ShapedText:
    _native = get_native_module()
    fonts = [str(Path(font)) for font in font_chain]
    for font in fonts:
        if not Path(font).is_file():
            raise TextShapingError(
                f"font not found: {font}",
                [{"status": "diagnostic_block", "reason": "font_not_found", "font": font}],
            )
    if _native is None or not hasattr(_native, "text_shape"):
        raise TextShapingError(
            "native text shaper is unavailable",
            [{"status": "diagnostic_block", "reason": "native_text_unavailable"}],
        )
    try:
        return _native.text_shape(text, fonts, size, script, language, dict(features or {}))
    except ValueError as error:
        diagnostics = list(getattr(error, "diagnostics", ()))
        if not diagnostics:
            diagnostics = [{
                "status": "diagnostic_block",
                "reason": "shaping_failed",
                "message": str(error),
            }]
        raise TextShapingError(
            str(error),
            diagnostics,
        ) from error


def rasterize_shaped_run(
    shaped: ShapedText,
    width: int,
    height: int,
    origin: tuple[float, float] = (0.0, 0.0),
    line_ranges: Sequence[tuple[int, int]] | None = None,
    positioned_glyphs: Sequence[Mapping] | None = None,
    halo_width: float = 0.0,
):
    _native = get_native_module()
    return _native.rasterize_shaped_run(
        shaped, width, height, origin, line_ranges,
        None if positioned_glyphs is None else [dict(glyph) for glyph in positioned_glyphs],
        halo_width,
    )


def rasterize_svg_path(
    path_data: str, width: int, height: int,
    origin: tuple[float, float] = (0.0, 0.0), halo_width: float = 0.0,
):
    """Rasterize absolute M/L/Q/C/Z outline paths, including a round halo."""
    _native = get_native_module()
    return _native.rasterize_svg_path(path_data, width, height, origin, halo_width)


def bake_msdf_atlas(
    font_chain: Sequence[str | Path],
    charset: str | ShapedText,
    font_size: float,
    px_range: float = 8.0,
    padding: int = 4,
    *,
    channels: int = 3,
):
    if channels not in (3, 4):
        raise ValueError("MSDF atlas channels must be 3 or 4")
    _native = get_native_module()
    fonts = [str(Path(font)) for font in font_chain]
    native_shaped_type = get_native_symbol("ShapedText")
    if (
        (native_shaped_type is not None and isinstance(charset, native_shaped_type))
        or (ShapedText is not object and isinstance(charset, ShapedText))
    ):
        shaped_payload = charset.to_dict()
        actual_sources = [str(source) for source in shaped_payload.get("font_sources", ())]
        actual_hashes = [str(value) for value in shaped_payload.get("font_sha256", ())]
        supplied_hashes = [
            hashlib.sha256(Path(font).read_bytes()).hexdigest() for font in fonts
        ]
        if supplied_hashes != actual_hashes:
            error = TextShapingError(
                "font_chain does not match the immutable fonts held by ShapedText",
                [{
                    "status": "diagnostic_block",
                    "reason": "atlas_font_mismatch",
                    "expected_sha256": actual_hashes,
                    "supplied_sha256": supplied_hashes,
                }],
            )
            raise error
        baked = _native.bake_msdf_atlas_shaped(
            charset,
            font_size,
            px_range,
            padding,
        )
        baked["metrics"]["font_sources"] = actual_sources
        baked["metrics"]["font_sha256"] = actual_hashes
    else:
        baked = _native.bake_msdf_atlas(
            fonts,
            charset,
            font_size,
            px_range,
            padding,
        )
        baked["metrics"]["font_sources"] = fonts
        baked["metrics"]["font_sha256"] = [
            hashlib.sha256(Path(font).read_bytes()).hexdigest() for font in fonts
        ]
    if channels == 4:
        import numpy as np
        rgb = baked["image"]
        image = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
        image[..., :3] = rgb
        image[..., 3] = baked["sdf_image"]
        baked["image"] = image
        baked["metrics"]["channels"] = 4
        baked["metrics"]["byte_count"] = image.nbytes
    return baked


__all__ = [
    "ShapedText",
    "TextShapingError",
    "bake_msdf_atlas",
    "rasterize_shaped_run",
    "rasterize_svg_path",
    "shape",
]
