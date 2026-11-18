#!/usr/bin/env python3
"""Generate the Studio app icons from code, so they are reproducible.

Run from `studio/`:  python3 scripts/make_icons.py

The mark is an isometric cube on the Forge3D amber gradient: the product is a
3D editor, and the cube is the Phase 1 starter object.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SS = 4  # supersampling factor
AMBER_TOP = (245, 181, 95)
AMBER_BOTTOM = (217, 124, 43)
INK = (26, 18, 6)


def rounded_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def gradient(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        t = y / max(1, size - 1)
        colour = tuple(
            round(AMBER_TOP[i] + (AMBER_BOTTOM[i] - AMBER_TOP[i]) * t) for i in range(3)
        )
        draw.line(((0, y), (size, y)), fill=colour)
    return image


def draw_cube(draw: ImageDraw.ImageDraw, size: int) -> None:
    cx, cy = size / 2, size / 2
    u = size * 0.105
    w = size * 0.205
    stroke = max(2, round(size * 0.032))
    top = (cx, cy - 2 * u)
    right_top = (cx + w, cy - u)
    centre = (cx, cy)
    left_top = (cx - w, cy - u)
    right_bottom = (cx + w, cy + u)
    bottom = (cx, cy + 2 * u)
    left_bottom = (cx - w, cy + u)

    draw.polygon([top, right_top, centre, left_top], fill=(255, 255, 255, 56))
    draw.polygon([right_top, right_bottom, bottom, centre], fill=(26, 18, 6, 40))
    draw.polygon([left_top, centre, bottom, left_bottom], fill=(26, 18, 6, 20))

    for a, b in [
        (top, right_top), (right_top, centre), (centre, left_top), (left_top, top),
        (right_top, right_bottom), (right_bottom, bottom), (bottom, centre),
        (left_top, left_bottom), (left_bottom, bottom),
    ]:
        draw.line([a, b], fill=INK, width=stroke, joint="curve")


def render(size: int) -> Image.Image:
    big = size * SS
    base = gradient(big).convert("RGBA")
    overlay = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw_cube(ImageDraw.Draw(overlay), big)
    base = Image.alpha_composite(base, overlay)
    base.putalpha(rounded_mask(big, big * 0.22))
    return base.resize((size, size), Image.LANCZOS)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "src-tauri" / "icons"
    out.mkdir(parents=True, exist_ok=True)

    for size, name in [
        (32, "32x32.png"),
        (128, "128x128.png"),
        (256, "128x128@2x.png"),
        (512, "icon.png"),
    ]:
        render(size).save(out / name)
        print(f"wrote {name}")

    ico = out / "icon.ico"
    render(256).save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote icon.ico")


if __name__ == "__main__":
    main()
