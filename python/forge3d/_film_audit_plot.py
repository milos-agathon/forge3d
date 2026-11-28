"""Dependency-free, deterministic raster plot for an ARGUS contact sheet."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_NUMERALS = {
    "0": "111101101101111",
    "1": "010110010010111",
    "2": "111001111100111",
    "3": "111001111001111",
    "4": "101101111001001",
    "5": "111100111001111",
    "6": "111100111101111",
    "7": "111001010010010",
    "8": "111101111101111",
    "9": "111101111001111",
    ".": "000000000000010",
}


def _label(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    pos = min(max(0, x), image.shape[1] - 4 * len(text))
    for digit in text:
        mask = np.array(list(_NUMERALS[digit])) == "1"
        image[y : y + 5, pos : pos + 3][mask.reshape(5, 3)] = color
        pos += 4


def _stroke(image: np.ndarray, p: tuple[int, int], q: tuple[int, int], color) -> None:
    count = 1 + max(abs(p[0] - q[0]), abs(p[1] - q[1]))
    x = np.rint(np.linspace(p[0], q[0], count)).astype(np.intp)
    y = np.rint(np.linspace(p[1], q[1], count)).astype(np.intp)
    image[y, x] = color


def time_plot(
    points: Sequence[tuple[int, float | None]],
    event: int,
    threshold: float,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Render values, cutoff and selected frame against source frame numbers."""
    background, ink = (24, 30, 38), (160, 174, 190)
    curve, cutoff, cursor = (118, 230, 191), (248, 186, 96), (106, 145, 223)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background
    first, last = points[0][0], points[-1][0]
    maximum = max(threshold, *(value or 0 for _, value in points))
    label = f"{maximum:.3f}".rstrip("0").rstrip(".") or "0"
    left, top, right, bottom = 4 * len(label) + 4, 6, width - 7, height - 12

    def x_at(index: int) -> int:
        return left + round((right - left) * (index - first) / max(1, last - first))

    def y_at(value: float) -> int:
        return bottom - round((bottom - top) * value / maximum) if maximum else bottom

    _stroke(image, (left, top), (left, bottom), ink)
    _stroke(image, (left, bottom), (right, bottom), ink)
    event_x, limit_y = x_at(event), y_at(threshold)
    image[limit_y, left : right + 1 : 3] = cutoff
    image[top : bottom + 1 : 2, event_x] = cursor
    previous = None
    for index, value in points:
        point = (x_at(index), y_at(value)) if value is not None else None
        if point is not None:
            if previous is not None:
                _stroke(image, previous, point, curve)
            image[point[1] - 1 : point[1] + 2, point[0] - 1 : point[0] + 2] = curve
        previous = point
    _label(image, label, 2, top, ink)
    for index, color in ((first, ink), (last, ink), (event, cursor)):
        _label(image, str(index), x_at(index) - 2 * len(str(index)), bottom + 3, color)
    return image
