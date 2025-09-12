# python/forge3d/tiles/overlay.py
# Attribution overlay rendering for text and optional logo on RGBA images.
# Offers position presets and simple DPI/extent-aware sizing with contrast outline.
# RELEVANT FILES:python/forge3d/tiles/client.py,examples/xyz_tile_compose_demo.py,tests/test_tiles_overlay.py

from __future__ import annotations

from typing import Literal, Optional, Tuple


def draw_attribution(
    image: "Image.Image",
    text: str,
    logo: Optional["Image.Image"] = None,
    position: Literal["tl", "tr", "bl", "br", "outside"] = "br",
    dpi: int = 96,
    margin: int = 8,
) -> "Image.Image":
    """Draw attribution text and optional logo onto an image.

    Returns the input image for chaining.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError("Pillow is required for attribution overlays") from e

    scale = max(1.0, dpi / 96.0)
    pad = int(round(margin * scale))
    font_size = max(10, int(round(11 * scale)))
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None  # type: ignore

    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]

    logo_img = None
    lw = lh = 0
    if logo is not None:
        # Fit logo to approx text height
        target_h = int(round(th * 1.2)) if th > 0 else int(round(16 * scale))
        ratio = target_h / max(1, logo.height)
        logo_img = logo.resize((int(logo.width * ratio), target_h), Image.Resampling.LANCZOS)
        lw, lh = logo_img.size

    total_w = lw + (pad if logo_img else 0) + tw
    total_h = max(lh, th)

    W, H = image.size
    x: int
    y: int
    if position == "tl":
        x, y = pad, pad
    elif position == "tr":
        x, y = W - total_w - pad, pad
    elif position == "bl":
        x, y = pad, H - total_h - pad
    elif position == "br":
        x, y = W - total_w - pad, H - total_h - pad
    else:  # "outside"
        x, y = pad, H + pad

    # Draw text with stroke for contrast
    tx = x + (lw + (pad if logo_img else 0))
    ty = y + (total_h - th) // 2
    draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font, stroke_width=max(1, int(scale)), stroke_fill=(0, 0, 0, 255))

    if logo_img is not None:
        ly = y + (total_h - lh) // 2
        image.paste(logo_img, (x, ly), mask=logo_img if logo_img.mode == "RGBA" else None)

    return image

