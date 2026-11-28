"""Temporal spectra and persistent screen-space structure for ARGUS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._deltae import relative_luminance, srgb_to_linear

_CODE = np.arange(256, dtype=np.float64) / 255
_WEIGHTS = relative_luminance(np.eye(3, dtype=np.uint8) * 255)
_OPAQUE = _WEIGHTS[:, None] * srgb_to_linear(_CODE)
_COMPOSITED = _WEIGHTS[:, None, None] * srgb_to_linear(_CODE[:, None] * _CODE)


def source_luminance(rgba: np.ndarray) -> np.ndarray:
    """Rec.709 linear luminance for every decoded source pixel over black."""
    alpha = rgba[..., 3]
    if np.all(alpha == 255):
        luminance = _OPAQUE[0, rgba[..., 0]]
        luminance += _OPAQUE[1, rgba[..., 1]]
        luminance += _OPAQUE[2, rgba[..., 2]]
    else:
        luminance = _COMPOSITED[0, alpha, rgba[..., 0]]
        luminance += _COMPOSITED[1, alpha, rgba[..., 1]]
        luminance += _COMPOSITED[2, alpha, rgba[..., 2]]
    return luminance


def temporal_amplitudes(tile_luminance: np.ndarray) -> np.ndarray:
    """Single-sided amplitudes in cycles/frame after removing a linear trend."""
    count = len(tile_luminance)
    if count < 2:
        return np.zeros_like(tile_luminance)
    time = np.arange(count, dtype=np.float64) - (count - 1) / 2
    centered = tile_luminance - tile_luminance.mean(axis=0)
    time = time[:, None, None]
    slope = (centered * time).sum(axis=0) / np.square(time).sum()
    amplitudes = np.abs(np.fft.rfft(centered - time * slope, axis=0)) * (2 / count)
    amplitudes[0] = 0
    if count % 2 == 0:
        amplitudes[-1] *= (
            0.5  # The Nyquist bin has no distinct negative-frequency partner.
        )
    return amplitudes


def line_ridges(luminance: np.ndarray) -> np.ndarray:
    """Signed contrast where a pixel is an extremum on opposite sides of a ridge."""
    ridge = np.zeros_like(luminance)
    # A two-sample normal spans the width of a thin two-pixel ridge.
    for values, across in ((luminance, ridge), (luminance.T, ridge.T)):
        center = values[:, 2:-2]
        left = center - values[:, :-4]
        right = center - values[:, 4:]
        response = np.maximum(0, np.minimum(left, right)) + np.minimum(
            0, np.maximum(left, right)
        )
        target = across[:, 2:-2]
        np.copyto(target, response, where=np.abs(response) > np.abs(target))
    return ridge


def structural_change(before: np.ndarray, after: np.ndarray) -> float:
    """Mean sRGB-code difference after allowing a gain and offset per channel."""
    a, b = (image - image.mean(axis=(0, 1)) for image in (before, after))
    energy = np.sum(a * a, axis=(0, 1))
    gain = np.divide(
        np.sum(a * b, axis=(0, 1)), energy, out=np.zeros_like(energy), where=energy > 0
    )
    return float(np.mean(np.abs(b - a * gain)))


@dataclass(frozen=True)
class ScreenLine:
    # Coordinates are half-open analysis pixels. The caller maps them to source pixels.
    region: tuple[int, int, int, int]
    contrast: float
    polarity: int


def connected_components(mask: np.ndarray, *, diagonal: bool):
    remaining = mask.copy()
    height, width = mask.shape
    offsets = (
        tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)
        if diagonal
        else ((0, -1), (0, 1), (-1, 0), (1, 0))
    )
    for row, col in zip(*np.nonzero(mask), strict=True):
        if not remaining[row, col]:
            continue
        remaining[row, col] = False
        component = [(row, col)]
        for y, x in component:
            for dy, dx in offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and remaining[yy, xx]:
                    remaining[yy, xx] = False
                    component.append((yy, xx))
        yield np.asarray(component)


def fixed_lines(
    luminance: tuple[np.ndarray, np.ndarray, np.ndarray],
    ridge: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    contrast_floor: float,
    variation_limit: float,
    tile_edge: int,
) -> list[ScreenLine]:
    """Find straight ridges common to three unregistered images of a moving shot."""
    stable = (
        np.maximum.reduce(luminance) - np.minimum.reduce(luminance) <= variation_limit
    )
    findings = []
    for polarity in (1, -1):
        response = np.minimum.reduce([polarity * r for r in ridge])
        for pixels in connected_components(
            stable & (response > contrast_floor), diagonal=True
        ):
            span = np.ptp(pixels, axis=0) + 1
            if np.linalg.norm(span) < 2 * tile_edge:
                continue
            # Fit the principal axis so diagonals and axis-aligned seams use the
            # same test. The transverse support fits inside the ridge stencil.
            centered = pixels - pixels.mean(axis=0)
            _, axes = np.linalg.eigh(centered.T @ centered)
            transverse, longitudinal = np.ptp(centered @ axes, axis=0) + 1
            if transverse > 5 or longitudinal < 2 * tile_edge:
                continue
            y, x = pixels.T
            findings.append(
                ScreenLine(
                    (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)),
                    float(response[y, x].mean()),
                    polarity,
                )
            )
    return findings
