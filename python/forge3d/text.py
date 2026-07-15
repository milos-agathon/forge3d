"""Deterministic native text shaping."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ._native import get_native_module


class TextShapingError(ValueError):
    def __init__(self, message: str, diagnostics: list[dict]):
        super().__init__(message)
        self.diagnostics = diagnostics


_native = get_native_module()
ShapedText = getattr(_native, "ShapedText", object)


def shape(
    text: str,
    font_chain: Sequence[str | Path],
    size: float,
    script: str | None = None,
    language: str | None = None,
    features: Mapping[str, bool] | None = None,
) -> ShapedText:
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
):
    return _native.rasterize_shaped_run(
        shaped, width, height, origin, line_ranges
    )


def bake_msdf_atlas(
    font_chain: Sequence[str | Path],
    charset: str | ShapedText,
    font_size: float,
    px_range: float = 8.0,
    padding: int = 4,
):
    fonts = [str(Path(font)) for font in font_chain]
    if hasattr(_native, "ShapedText") and isinstance(charset, ShapedText):
        baked = _native.bake_msdf_atlas_shaped(
            charset,
            font_size,
            px_range,
            padding,
        )
    else:
        baked = _native.bake_msdf_atlas(
            fonts,
            charset,
            font_size,
            px_range,
            padding,
        )
    baked["metrics"]["font_sources"] = fonts
    return baked


__all__ = [
    "ShapedText",
    "TextShapingError",
    "bake_msdf_atlas",
    "rasterize_shaped_run",
    "shape",
]
