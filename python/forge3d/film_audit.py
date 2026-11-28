"""ARGUS temporal audit of source PNG frames."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any
import zlib

import numpy as np

from ._deltae import delta_e_2000, relative_luminance, srgb_to_lab
from ._film_audit_plot import time_plot
from ._film_audit_signals import (
    ScreenLine,
    connected_components,
    fixed_lines,
    line_ridges,
    source_luminance,
    structural_change,
    temporal_amplitudes,
)
from ._png import decode_png, load_png_rgba, save_png
from .chronos import FlythroughManifest, _CHRONOS_MANIFEST_SCHEMA

__all__ = ["AuditSettings", "audit"]

_FRAME_NUMBER = re.compile(r"([0-9]+)$")
_PREVIEW_EDGE = 192
_TILE_EDGE = 12
_PULSE_PERIOD = 3
_PULSE_CYCLES = 3
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class AuditSettings:
    """Thresholds for the source-frame audit. See the film audit guide for units."""

    flash_floor: float = 0.01
    pop_floor: float = 3.0
    duplicate_ratio: float = 0.12
    motion_floor: float = 1.0
    window_radius: int = 3
    mad_multiplier: float = 1.0
    changed_luminance: float = 0.01
    lighting_step_floor: float = 0.02
    pulse_floor: float = 0.01
    drop_motion_ratio: float = 1.5
    line_contrast_floor: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "flash_floor",
            "pop_floor",
            "motion_floor",
            "mad_multiplier",
            "changed_luminance",
            "lighting_step_floor",
            "pulse_floor",
            "line_contrast_floor",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.duplicate_ratio < 1:
            raise ValueError("duplicate_ratio must be between 0 and 1")
        if not math.isfinite(self.drop_motion_ratio) or self.drop_motion_ratio <= 1:
            raise ValueError("drop_motion_ratio must be finite and greater than 1")
        if type(self.window_radius) is not int or self.window_radius < 1:
            raise ValueError("window_radius must be a positive integer")


@dataclass(frozen=True)
class _Source:
    index: int
    path: Path
    size: tuple[int, int] | None = None


def _integer(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{what} must be a nonnegative integer")
    return value


def _sources(
    frames: str | Path | FlythroughManifest, frames_dir: str | Path | None
) -> tuple[list[_Source], dict[str, Any], Path | None]:
    manifest_path = None
    if isinstance(frames, FlythroughManifest):
        if frames_dir is None:
            raise ValueError("frames_dir is required with a FlythroughManifest object")
        manifest, root = frames.to_dict(), Path(frames_dir)
    else:
        path = Path(frames)
        if path.is_dir():
            if frames_dir is not None:
                raise ValueError("frames_dir is only used with a CHRONOS manifest")
            sources = []
            for image in path.iterdir():
                number = _FRAME_NUMBER.search(image.stem)
                if number and image.suffix.lower() == ".png" and image.is_file():
                    sources.append(_Source(int(number[1]), image))
            return sorted(sources, key=lambda source: source.index), {}, None
        manifest_path = path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = path.parent if frames_dir is None else Path(frames_dir)

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _CHRONOS_MANIFEST_SCHEMA
    ):
        raise ValueError("expected a CHRONOS flythrough/1 manifest")
    records = manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("CHRONOS manifest frames must be a list")
    root = root.resolve()
    sources = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            raise ValueError("CHRONOS frame record must include a PNG image path")
        index = _integer(record.get("frame_index"), "frame_index")
        image = Path(record["image"])
        path = (root / image).resolve()
        if (
            image.anchor
            or path.suffix.lower() != ".png"
            or not path.is_relative_to(root)
        ):
            raise ValueError(
                f"frame {index}: image must be a relative PNG path inside frames_dir"
            )
        size = (
            _integer(record.get("width"), "width"),
            _integer(record.get("height"), "height"),
        )
        sources.append(_Source(index, path, size))
    return sources, manifest, manifest_path


def _timeline(
    sources: list[_Source],
    cuts: Sequence[int],
    holds: Sequence[tuple[int, int]],
) -> tuple[list[int], list[tuple[int, int]], set[int], list[int]]:
    positions = {source.index: i for i, source in enumerate(sources)}
    if not positions:
        raise ValueError("no numbered source PNGs found")
    if len(positions) != len(sources) or list(positions) != sorted(positions):
        raise ValueError("source frame numbers must be sorted and unique")

    cuts = sorted({_integer(index, "cut frame") for index in cuts})
    if any(index not in positions for index in cuts):
        raise ValueError("cut frame must exist in the source sequence")
    boundaries = sorted({0, len(sources), *(positions[index] for index in cuts)})

    hold_ranges = set()
    for hold in holds:
        if not isinstance(hold, (tuple, list)) or len(hold) != 2:
            raise ValueError("each hold must have a first and last frame")
        first, last = (_integer(index, "hold frame") for index in hold)
        if first not in positions or last not in positions or first > last:
            raise ValueError("hold endpoints must exist in source order")
        hold_ranges.add((first, last))
    held_transitions = {
        i
        for first, last in hold_ranges
        for i in range(positions[first] + 1, positions[last] + 1)
    }
    return cuts, sorted(hold_ranges), held_transitions, boundaries


def _read(source: _Source) -> np.ndarray:
    try:
        with source.path.open("rb") as png:
            header = png.read(29)
            if (
                len(header) != 29
                or header[:8] != _PNG_SIGNATURE
                or header[12:16] != b"IHDR"
            ):
                raise ValueError("expected a PNG with an IHDR chunk")
            # The repo decoder scales 16-bit grayscale correctly to its 8-bit RGBA output.
            rgba = decode_png(header + png.read()) if header[24] == 16 else None
        if rgba is None:
            rgba = load_png_rgba(source.path)
    except (OSError, ValueError, zlib.error) as error:
        raise ValueError(f"frame {source.index} ({source.path}): {error}") from error
    if source.size is not None and source.size != rgba.shape[1::-1]:
        raise ValueError(
            f"frame {source.index}: PNG size disagrees with CHRONOS manifest"
        )
    return rgba


def _bins(length: int, count: int) -> np.ndarray:
    return np.arange(count + 1, dtype=np.intp) * length // count


@dataclass(frozen=True)
class _Grid:
    y: np.ndarray
    x: np.ndarray
    tile_y: np.ndarray
    tile_x: np.ndarray
    window: np.ndarray

    @classmethod
    def for_shape(cls, shape: tuple[int, ...]) -> _Grid:
        height, width = shape[:2]
        scale = min(1.0, _PREVIEW_EDGE / max(height, width))
        h, w = max(1, round(height * scale)), max(1, round(width * scale))
        window = np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1])
        return cls(
            _bins(height, h),
            _bins(width, w),
            _bins(h, math.ceil(h / _TILE_EDGE)),
            _bins(w, math.ceil(w / _TILE_EDGE)),
            window,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.y[-1]), int(self.x[-1])


def _sum_bins(
    values: np.ndarray, y: np.ndarray, x: np.ndarray, dtype=None
) -> np.ndarray:
    columns = np.add.reduceat(values, x[:-1], axis=1, dtype=dtype)
    return np.add.reduceat(columns, y[:-1], axis=0, dtype=dtype)


@dataclass
class _Preview:
    digest: bytes
    rgb: np.ndarray
    lab_luma: np.ndarray
    spectrum: np.ndarray
    ridge: np.ndarray


def _preview(rgba: np.ndarray, grid: _Grid) -> _Preview:
    if rgba.shape[:2] != grid.shape:
        raise ValueError(f"all source PNGs must have size {grid.shape[::-1]}")
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if not np.all(alpha == 255):
        rgb = rgb.astype(np.float32) * (alpha[..., None].astype(np.float32) / 255)
    if rgb.shape[:2] != grid.window.shape:
        dtype = np.uint32 if rgb.dtype == np.uint8 else np.float32
        area = np.diff(grid.y)[:, None] * np.diff(grid.x)[None, :]
        rgb = (_sum_bins(rgb, grid.y, grid.x, dtype) / area[..., None]).astype(
            np.float32
        )
    rgb = rgb.astype(np.float32, copy=False)
    srgb = rgb / 255
    luminance = relative_luminance(srgb)
    lab_luma = np.concatenate((srgb_to_lab(srgb), luminance[..., None]), axis=2)
    spectrum = np.fft.rfft2((luminance - luminance.mean()) * grid.window)
    return _Preview(
        hashlib.sha256(rgba).digest(), rgb, lab_luma, spectrum, line_ridges(luminance)
    )


def _shift(
    before: _Preview, after: _Preview, shape: tuple[int, int]
) -> tuple[float, ...]:
    """Estimate where before-frame content translated to in the after frame."""
    if before.digest == after.digest:
        return 0.0, 0.0
    cross = after.spectrum * before.spectrum.conj()
    magnitude = np.abs(cross)
    np.divide(
        cross,
        magnitude,
        out=cross,
        where=magnitude > np.finfo(magnitude.dtype).eps * magnitude.max(),
    )
    correlation = np.fft.irfft2(cross, s=shape, axes=(0, 1))
    peak = np.unravel_index(correlation.argmax(), shape)
    shift = []
    for axis, (i, length) in enumerate(zip(peak, shape)):
        line = correlation[:, peak[1]] if axis == 0 else correlation[peak[0], :]
        left, middle, right = line[(i - 1) % length], line[i], line[(i + 1) % length]
        curvature = left - 2 * middle + right
        fraction = (left - right) / (2 * curvature) if curvature < 0 else 0.0
        offset = (i + fraction + length / 2) % length - length / 2
        shift.append(float(offset))
    return tuple(shift)


def _sample(
    image: np.ndarray, y: np.ndarray, x: np.ndarray, dy: float, dx: float
) -> np.ndarray:
    """Bilinear reprojection on the shared valid crop, as in the flythrough pop test."""
    sy, sx = y + dy, x + dx
    y0, x0 = np.floor(sy).astype(np.intp), np.floor(sx).astype(np.intp)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    fy, fx = (sy - y0)[:, None, None], (sx - x0)[None, :, None]
    a, b = image[y0[:, None], x0], image[y1[:, None], x0]
    top = a + (image[y0[:, None], x1] - a) * fx
    bottom = b + (image[y1[:, None], x1] - b) * fx
    return top + (bottom - top) * fy


def _triplet(
    before: _Preview, at: _Preview, after: _Preview, grid: _Grid, *, held: bool = False
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.window.shape
    dy, dx = (
        (0.0, 0.0) if held else (axis / 2 for axis in _shift(before, after, (h, w)))
    )
    # A centered 2-frame motion estimate stays independent of the suspect frame.
    # Use only overlap, including for the neighbor-to-neighbor control.
    y = np.arange(math.ceil(abs(dy)), h - math.ceil(abs(dy)))
    x = np.arange(math.ceil(abs(dx)), w - math.ceil(abs(dx)))
    heat = np.zeros((h, w))
    warp_improves_agreement = False
    if (dy or dx) and y.size and x.size:
        a = _sample(before.lab_luma, y, x, -dy, -dx)
        c = _sample(after.lab_luma, y, x, dy, dx)
        crop = np.ix_(y, x)
        registered = a[..., 3] - c[..., 3]
        unregistered = before.lab_luma[..., 3][crop] - after.lab_luma[..., 3][crop]
        # Repeated structure can give phase correlation the wrong peak. Admit
        # the warp only when the neighbors agree better on the same pixels.
        warp_improves_agreement = np.mean(
            np.abs(registered - np.median(registered))
        ) < np.mean(np.abs(unregistered - np.median(unregistered)))
    if not warp_improves_agreement:
        y, x = np.arange(h), np.arange(w)
        a, c = before.lab_luma, after.lab_luma
    crop = np.ix_(y, x)
    b = at.lab_luma[crop]
    valid = np.zeros_like(heat)
    valid[crop] = 1
    counts = _sum_bins(valid, grid.tile_y, grid.tile_x)
    supported = counts > 0

    def tile_mean(values: np.ndarray) -> np.ndarray:
        heat[crop] = values
        totals = _sum_bins(heat, grid.tile_y, grid.tile_x)
        return totals / np.maximum(counts, 1)

    from_before = tile_mean(b[..., 3] - a[..., 3])
    from_after = tile_mean(b[..., 3] - c[..., 3])
    disagreement = np.abs(from_before - from_after)
    bright = np.minimum(from_before, from_after) - disagreement
    dark = -np.maximum(from_before, from_after) - disagreement
    flash = max(0.0, np.median(bright[supported]), np.median(dark[supported]))
    impulse = np.maximum(bright, 0) - np.maximum(dark, 0)
    a, b, c = a[..., :3], b[..., :3], c[..., :3]
    # Excess over the neighbors' own disagreement rejects motion, occlusion,
    # and monotone lighting changes that remain after translation compensation.
    excess = np.minimum(delta_e_2000(a, b), delta_e_2000(c, b)) - delta_e_2000(a, c)
    tiles = tile_mean(np.maximum(excess, 0))
    return float(flash), tiles, heat, impulse


@dataclass
class _Measures:
    decoded_sha256: str = ""
    pair_difference: float | None = None
    exact_repeat: bool | None = None
    flash: float | None = None
    pop: np.ndarray | None = None
    tile_luminance: np.ndarray | None = None
    luminance: float = 0.0
    mean_absolute_luminance_change: float | None = None
    tile_absolute_luminance_change: np.ndarray | None = None
    changed_fraction: float | None = None
    impulse: np.ndarray | None = None
    lines: list[ScreenLine] | None = None
    lighting_step: float | None = None
    drop_ratio: float | None = None
    stride_ratio: float | None = None

    @property
    def peak_pop(self) -> float | None:
        return None if self.pop is None else float(self.pop.max())


def _measure(
    sources: list[_Source],
    boundaries: list[int],
    held: set[int],
    settings: AuditSettings,
) -> tuple[_Grid, list[_Measures]]:
    first = _read(sources[0])
    grid = _Grid.for_shape(first.shape)
    measures = [_Measures() for _ in sources]
    tile_y, tile_x = grid.y[grid.tile_y], grid.x[grid.tile_x]
    area = np.diff(tile_y)[:, None] * np.diff(tile_x)[None, :]

    def tiled(luma: np.ndarray) -> np.ndarray:
        return _sum_bins(luma, tile_y, tile_x) / area

    for start, stop in pairwise(boundaries):
        recent: deque[_Preview] = deque(maxlen=3)
        previous_luma: np.ndarray | None = None
        for i in range(start, stop):
            rgba = first if i == 0 else _read(sources[i])
            current = _preview(rgba, grid)
            measure = measures[i]
            luma = (
                current.lab_luma[..., 3]
                if rgba.shape[:2] == grid.window.shape
                else source_luminance(rgba)
            )
            measure.decoded_sha256 = current.digest.hex()
            measure.luminance = float(luma.mean())
            measure.tile_luminance = tiled(luma)
            if recent:
                measure.exact_repeat = current.digest == recent[-1].digest
                measure.pair_difference = (
                    0.0
                    if measure.exact_repeat
                    else float(
                        np.abs(current.rgb - recent[-1].rgb).mean(dtype=np.float64)
                    )
                )
                change = np.abs(luma - previous_luma)
                measure.mean_absolute_luminance_change = float(change.mean())
                measure.tile_absolute_luminance_change = tiled(change)
                measure.changed_fraction = float(
                    np.mean(change > settings.changed_luminance)
                )
            previous_luma = luma
            recent.append(current)
            if len(recent) == 3:
                center = measures[i - 1]
                if recent[0].digest == recent[1].digest == recent[2].digest:
                    pop = impulse = np.zeros_like(area, dtype=np.float64)
                    flash = 0.0
                else:
                    flash, pop, _, impulse = _triplet(
                        *recent, grid, held=_at_hold(i - 1, held)
                    )
                center.flash, center.pop, center.impulse = flash, pop, impulse
                if (
                    i not in held
                    and i - 1 not in held
                    and min(center.pair_difference, measure.pair_difference)
                    > settings.motion_floor
                ):
                    center.lines = fixed_lines(
                        tuple(p.lab_luma[..., 3] for p in recent),
                        tuple(p.ridge for p in recent),
                        contrast_floor=settings.line_contrast_floor,
                        variation_limit=settings.changed_luminance,
                        tile_edge=_TILE_EDGE,
                    )
                    # Exposure changes alone can also move the raw code
                    # difference. A fixed line requires moving surroundings.
                    if center.lines and any(
                        structural_change(a.rgb, b.rgb) <= settings.motion_floor
                        for a, b in pairwise(recent)
                    ):
                        center.lines = []
    return grid, measures


def _at_hold(center: int, held: set[int]) -> bool:
    # Motion can stop or resume abruptly at a hold. Those stencils cannot
    # assume the center is halfway through an evenly spaced translation.
    return any(i in held for i in (center - 1, center, center + 1))


def _threshold(value: float, neighbors: list[float | None], multiplier: float) -> float:
    sample = np.array(
        [observation for observation in neighbors if observation is not None]
    )
    if sample.size == 0:
        return value
    median = np.median(sample)
    return max(value, float(median + multiplier * np.median(np.abs(sample - median))))


def _regions(scores: np.ndarray, threshold: float, grid: _Grid):
    """Yield the peak tile and source bounds of every edge-connected component."""
    for cells in connected_components(scores > threshold, diagonal=False):
        rows, cols = cells.T
        values = scores[rows, cols]
        y, x = cells[values.argmax()]
        region = [
            int(grid.x[grid.tile_x[cols.min()]]),
            int(grid.y[grid.tile_y[rows.min()]]),
            int(grid.x[grid.tile_x[cols.max() + 1]]),
            int(grid.y[grid.tile_y[rows.max() + 1]]),
        ]
        yield float(scores[y, x]), region, [int(y), int(x)]


def _nearby(i: int, start: int, stop: int, radius: int) -> range:
    return range(max(start, i - radius), min(stop, i + radius + 1))


def _pulse_runs(
    measures: list[_Measures], start: int, stop: int, settings: AuditSettings
):
    # Three cycles establish two repeated gaps. Each cycle contains an event and
    # its before/after references, giving an integral period-3 Fourier bin.
    span = _PULSE_PERIOD * (_PULSE_CYCLES - 1)
    sampled = np.stack([m.tile_luminance for m in measures[start:stop]])
    zero = np.zeros_like(sampled[0])
    impulses = np.stack(
        [m.impulse if m.impulse is not None else zero for m in measures[start:stop]]
    )
    next_start = start + 1
    for i in range(start + 1, stop - span - 1):
        if i < next_start:
            continue
        index = i - start
        phase = impulses[index : index + span + 1 : _PULSE_PERIOD]
        same_sign = np.all(phase > 0, axis=0) | np.all(phase < 0, axis=0)
        strength = np.where(same_sign, np.abs(phase).min(axis=0), 0)
        if float(strength.max()) <= settings.pulse_floor:
            continue
        window = range(
            max(0, index - settings.window_radius),
            min(len(impulses), index + span + settings.window_radius + 1),
        )
        background = np.abs(
            impulses[[j for j in window if (j - index) % _PULSE_PERIOD]]
        )
        median = np.median(background, axis=0)
        limit = np.maximum(
            settings.pulse_floor,
            median
            + settings.mad_multiplier * np.median(np.abs(background - median), axis=0),
        )
        spectrum = temporal_amplitudes(sampled[index - 1 : index + span + 2])
        fundamental = spectrum[_PULSE_CYCLES]
        other_bins = np.delete(spectrum, _PULSE_CYCLES, axis=0)
        supported = (strength > limit) & (fundamental > other_bins.max(axis=0))
        impulse_map = np.where(supported, strength, 0)
        if not supported.any():
            continue
        y, x = np.unravel_index(impulse_map.argmax(), impulse_map.shape)
        frames = list(range(i, i + span + 1, _PULSE_PERIOD))
        sign = np.sign(phase[0, y, x])
        for j in range(frames[-1] + _PULSE_PERIOD, stop - 1, _PULSE_PERIOD):
            if sign * impulses[j - start, y, x] <= limit[y, x]:
                break
            frames.append(j)
        next_start = frames[-1] + 1
        yield (
            frames,
            (int(y), int(x)),
            impulse_map,
            float(limit[y, x]),
            float(fundamental[y, x]),
        )


def _overlaps(a: Sequence[int], b: Sequence[int], margin: int = 0) -> bool:
    return min(a[2], b[2] + margin) > max(a[0], b[0] - margin) and min(
        a[3], b[3] + margin
    ) > max(a[1], b[1] - margin)


@dataclass
class _LineRun:
    start: int
    end: int
    line: ScreenLine


def _line_runs(measures: list[_Measures], start: int, stop: int) -> list[_LineRun]:
    runs: list[_LineRun] = []
    active: list[_LineRun] = []
    for center in range(start + 1, stop - 1):
        active = [run for run in active if run.end >= center - 1]
        for line in measures[center].lines or ():
            match = next(
                (
                    run
                    for run in active
                    if line.polarity == run.line.polarity
                    and _overlaps(line.region, run.line.region)
                ),
                None,
            )
            if match is None:
                match = _LineRun(center - 1, center + 1, line)
                runs.append(match)
                active.append(match)
            else:
                match.end = center + 1
                if line.contrast > match.line.contrast:
                    match.line = line
    return runs


def _transition_signals(
    sources: list[_Source],
    measures: list[_Measures],
    start: int,
    stop: int,
    held: set[int],
    motion_floor: float,
) -> None:
    if stop - start < 2:
        return
    luma = np.stack([m.tile_luminance for m in measures[start:stop]])
    signed = np.median(np.diff(luma, axis=0), axis=(1, 2))
    for i in range(start + 2, stop - 1):
        p = i - start - 1
        center = measures[i]
        center.lighting_step = float(
            max(0, abs(signed[p]) - max(abs(signed[p - 1]), abs(signed[p + 1])))
        )
        before, after = sources[i - 1].index, sources[i].index
        stride = before - sources[i - 2].index
        if stride == sources[i + 1].index - after and (after - before) % stride == 0:
            center.stride_ratio = (after - before) / stride
        if any(j in held for j in (i - 1, i, i + 1)):
            continue
        left, incoming, right = (measures[j].pair_difference for j in (i - 1, i, i + 1))
        assert left is not None and incoming is not None and right is not None
        if min(left, right) > motion_floor:
            center.drop_ratio = incoming / max(left, right)


def _find(
    sources: list[_Source],
    grid: _Grid,
    measures: list[_Measures],
    boundaries: list[int],
    held_transitions: set[int],
    settings: AuditSettings,
) -> list[tuple[int, dict[str, Any]]]:
    found = []
    height, width = grid.shape
    for start, stop in pairwise(boundaries):

        def add(i, kind, score, threshold, *, region=None, frame_index=None, **details):
            roles = {
                role: sources[j].index
                for role, j in (("before", i - 1), ("at", i), ("after", i + 1))
                if start <= j < stop
            }
            found.append(
                (
                    i,
                    {
                        "frame_index": sources[i].index
                        if frame_index is None
                        else frame_index,
                        "class": kind,
                        "region": [0, 0, width, height] if region is None else region,
                        "score": score,
                        "threshold": threshold,
                        "context_frames": list(roles.values()),
                        "context_roles": roles,
                        **details,
                    },
                )
            )

        _transition_signals(
            sources, measures, start, stop, held_transitions, settings.motion_floor
        )

        pulsing = set()
        for peaks, tile, strength, cutoff, amplitude in _pulse_runs(
            measures, start, stop, settings
        ):
            _, region, _ = max(
                _regions(strength, 0, grid), key=lambda component: component[0]
            )
            add(
                peaks[0],
                "periodic_pulse",
                float(strength[tile]),
                cutoff,
                region=region,
                tile=list(tile),
                period_frames=_PULSE_PERIOD,
                pulse_frame_indices=[sources[j].index for j in peaks],
                end_frame_index=sources[peaks[-1]].index,
                spectral_amplitude=amplitude,
            )
            pulsing.update(j for peak in peaks for j in (peak - 1, peak, peak + 1))

        detected_lines = []
        for run in _line_runs(measures, start, stop):
            x0, y0, x1, y1 = run.line.region
            region = [
                int(grid.x[x0]),
                int(grid.y[y0]),
                int(grid.x[x1]),
                int(grid.y[y1]),
            ]
            reference = [
                max(
                    (
                        view.contrast
                        for view in measures[j].lines or ()
                        if _overlaps(view.region, run.line.region)
                    ),
                    default=0.0,
                )
                for j in _nearby(run.start, start, stop, settings.window_radius)
                if j < run.start
            ]
            cutoff = _threshold(
                settings.line_contrast_floor, reference, settings.mad_multiplier
            )
            if run.line.contrast > cutoff:
                detected_lines.append((run, region))
                add(
                    run.start,
                    "fixed_line",
                    run.line.contrast,
                    cutoff,
                    region=region,
                    end_frame_index=sources[run.end].index,
                    polarity=run.line.polarity,
                )

        duplicate_limits = {}
        dropped = set()
        for i in range(start + 2, stop - 1):
            ratio = measures[i].stride_ratio
            if ratio is not None and ratio >= 2:
                expected = sources[i - 1].index + (
                    sources[i - 1].index - sources[i - 2].index
                )
                add(
                    i,
                    "dropped_frame",
                    ratio,
                    2.0,
                    frame_index=expected,
                    missing_index_is_explicit=True,
                    observed_frame_index=sources[i].index,
                    missing_frame_count=int(ratio) - 1,
                )
                dropped.add(i)
                continue
            if any(j in held_transitions for j in (i - 1, i, i + 1)):
                continue
            left, incoming, right = (
                measures[j].pair_difference for j in (i - 1, i, i + 1)
            )
            assert left is not None and incoming is not None and right is not None
            flank_motion = min(left, right)
            pair_reference = [
                measures[j].pair_difference
                for j in _nearby(i, start + 1, stop, settings.window_radius)
                if j != i and j not in held_transitions
            ]
            # An unexpectedly low transition also has to clear the lower
            # median/MAD envelope, so a jittery context cannot invent a repeat.
            median = float(np.median(pair_reference))
            mad = float(np.median(np.abs(np.asarray(pair_reference) - median)))
            limit = settings.duplicate_ratio * min(
                flank_motion, median - settings.mad_multiplier * mad
            )
            if flank_motion > settings.motion_floor and incoming < limit:
                duplicate_limits[i] = limit

        def neighbors(i):
            return [
                measures[j]
                for j in _nearby(i, start, stop, settings.window_radius)
                if j != i
            ]

        flash_limits = {
            i: cutoff
            for i in range(start + 1, stop - 1)
            if i not in pulsing
            and measures[i].flash is not None
            and measures[i].flash
            > (
                cutoff := _threshold(
                    settings.flash_floor,
                    [n.flash for n in neighbors(i)],
                    settings.mad_multiplier,
                )
            )
        }

        # Detect image-only gaps before checking pops. The midpoint is invalid
        # immediately on either side of a cadential break.
        for i in range(start + 2, stop - 1):
            if (
                i in dropped
                or i in pulsing
                or i in flash_limits
                or i - 1 in flash_limits
                or any(j in duplicate_limits for j in (i - 1, i, i + 1))
            ):
                continue
            m = measures[i]
            nearby = neighbors(i)
            step_limit = _threshold(
                settings.lighting_step_floor,
                [n.lighting_step for n in nearby],
                settings.mad_multiplier,
            )
            drop_limit = _threshold(
                settings.drop_motion_ratio,
                [n.drop_ratio for n in nearby],
                settings.mad_multiplier,
            )
            if m.lighting_step is not None and m.lighting_step > step_limit:
                add(i, "lighting_step", m.lighting_step, step_limit)
                dropped.add(i)
            elif m.drop_ratio is not None and m.drop_ratio > drop_limit:
                add(
                    i,
                    "dropped_frame",
                    m.drop_ratio,
                    drop_limit,
                    missing_index_is_explicit=False,
                )
                dropped.add(i)

        for i in range(start + 1, stop - 1):
            if i in pulsing or i in dropped:
                continue
            m = measures[i]
            peak_pop = m.peak_pop
            nearby = neighbors(i)
            pop_limit = _threshold(
                settings.pop_floor,
                [n.peak_pop for n in nearby],
                settings.mad_multiplier,
            )
            if i in duplicate_limits:
                add(i, "duplicate", m.pair_difference, duplicate_limits[i])
            elif i in flash_limits:
                add(i, "flash", m.flash, flash_limits[i])
            # A confirmed repeat changes the motion cadence in both adjacent
            # stencils, making their midpoint comparison unreliable.
            elif (
                i - 1 not in duplicate_limits
                and i + 1 not in duplicate_limits
                and i - 1 not in dropped
                and i + 1 not in dropped
                and m.pop is not None
                and peak_pop is not None
                and peak_pop > pop_limit
            ):
                regions = list(_regions(m.pop, pop_limit, grid))
                for score, region, tile in regions:
                    if not any(
                        run.start - 1 <= i <= run.end + 1
                        and _overlaps(region, line_region)
                        for run, line_region in detected_lines
                    ):
                        add(
                            i,
                            "localized_pop",
                            score,
                            pop_limit,
                            region=region,
                            **({"tile": tile} if len(regions) > 1 else {}),
                        )
    return sorted(found, key=lambda entry: entry[1]["frame_index"])


def _evidence(
    out_dir: Path,
    folder: str,
    sources: dict[str, tuple[_Source, str]],
    grid: _Grid,
    finding: dict[str, Any],
    *,
    held: bool,
) -> None:
    target = out_dir / folder
    target.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = finding["region"]
    previews, images = {}, {}
    for label, (source, expected) in sources.items():
        rgba = _read(source)
        preview = _preview(rgba, grid)
        if preview.digest.hex() != expected:
            raise ValueError(
                f"frame {source.index}: source pixels changed during audit"
            )
        previews[label] = preview
        images[label] = f"{folder}/{label}.png"
        save_png(out_dir / images[label], rgba[y0:y1, x0:x1])
    if finding["class"] == "fixed_line":
        heat = np.maximum(0, finding["polarity"] * previews["at"].ridge)
        measure = "screen_space_ridge_luminance_contrast"
    elif finding["class"] in {"duplicate", "dropped_frame", "lighting_step"}:
        heat = delta_e_2000(
            previews["before"].lab_luma[..., :3], previews["at"].lab_luma[..., :3]
        )
        measure = "incoming_delta_e_2000"
    else:
        _, _, heat, _ = _triplet(*previews.values(), grid, held=held)
        measure = "motion_compensated_temporal_excess_delta_e_2000"

    ys = np.searchsorted(grid.y, np.arange(y0, y1), side="right") - 1
    xs = np.searchsorted(grid.x, np.arange(x0, x1), side="right") - 1
    crop = heat[np.ix_(ys, xs)]
    maximum = float(crop.max())
    intensity = (
        np.rint(crop * (255 / maximum)).astype(np.uint8)
        if maximum
        else crop.astype(np.uint8)
    )
    heatmap = np.stack((intensity, intensity // 2, np.zeros_like(intensity)), axis=-1)
    images["heatmap"] = f"{folder}/heatmap.png"
    save_png(out_dir / images["heatmap"], heatmap)

    yslice, xslice = slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1)
    heat_preview = heatmap[
        np.ix_(np.unique(ys, return_index=True)[1], np.unique(xs, return_index=True)[1])
    ]
    panels = np.concatenate(
        [np.rint(p.rgb[yslice, xslice]).astype(np.uint8) for p in previews.values()]
        + [heat_preview],
        axis=1,
    )
    canvas_width = max(panels.shape[1], _PREVIEW_EDGE)
    plot = time_plot(
        finding["time_series"]["points"],
        finding["frame_index"],
        finding["threshold"],
        width=canvas_width,
        height=_PREVIEW_EDGE // 3,
    )
    contact = np.empty(
        (panels.shape[0] + plot.shape[0], canvas_width, 3), dtype=np.uint8
    )
    contact[:] = plot[-1, 0]
    offset = (canvas_width - panels.shape[1]) // 2
    contact[: panels.shape[0], offset : offset + panels.shape[1]] = panels
    contact[panels.shape[0] :] = plot
    images["contact_sheet"] = f"{folder}/contact.png"
    save_png(out_dir / images["contact_sheet"], contact)
    finding.update(evidence=images, heatmap_measure=measure, heatmap_max=maximum)


def _json_bytes(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_output(
    target: Path,
    sources: list[_Source],
    grid: _Grid,
    found: list[tuple[int, dict[str, Any]]],
    held: set[int],
    report: dict[str, Any],
) -> None:
    # Different input pixels or settings own different evidence paths. Publish
    # the report last so a failed rerun leaves its predecessor's links valid.
    snapshot = hashlib.sha256(_json_bytes(report)).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    input_paths = {source.path.resolve() for source in sources}
    positions = {source.index: j for j, source in enumerate(sources)}
    with TemporaryDirectory(dir=target, prefix=".film-audit-") as temp:
        stage = Path(temp)
        for number, (i, finding) in enumerate(found):
            folder = f"evidence/{snapshot}/{finding['frame_index']}/{number}"
            evidence_sources = {
                role: (
                    sources[positions[index]],
                    report["measures"][positions[index]]["decoded_sha256"],
                )
                for role, index in finding["context_roles"].items()
            }
            _evidence(
                stage,
                folder,
                evidence_sources,
                grid,
                finding,
                held=_at_hold(i, held),
            )
        paths = [
            Path(name) for _, finding in found for name in finding["evidence"].values()
        ]
        for relative in paths:
            output = (target / relative).resolve()
            if output in input_paths or not output.is_relative_to(target):
                raise ValueError(
                    f"evidence destination overlaps the source or escapes out_dir: {output}"
                )
        for relative in paths:
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative).replace(output)
        (stage / "audit.json").write_bytes(_json_bytes(report))
        (stage / "audit.json").replace(target / "audit.json")


def _measure_rows(
    sources: list[_Source], measures: list[_Measures]
) -> list[dict[str, Any]]:
    rows = []
    for i, (source, m) in enumerate(zip(sources, measures, strict=True)):
        previous = measures[i - 1] if m.pair_difference is not None else None
        rows.append(
            {
                "frame_index": source.index,
                "image": source.path.name,
                "decoded_sha256": m.decoded_sha256,
                "exact_repeat": m.exact_repeat,
                "incoming_mean_code_difference": m.pair_difference,
                "mean_luminance": m.luminance,
                "incoming_mean_luminance_difference": m.luminance - previous.luminance
                if previous
                else None,
                "incoming_mean_absolute_luminance_difference": m.mean_absolute_luminance_change,
                "incoming_tile_luminance_difference": (
                    m.tile_luminance - previous.tile_luminance
                ).tolist()
                if previous
                else None,
                "incoming_tile_mean_absolute_luminance_difference": m.tile_absolute_luminance_change.tolist()
                if previous
                else None,
                "incoming_changed_pixel_fraction": m.changed_fraction,
                "temporal_luminance_second_difference": (
                    measures[i + 1].luminance
                    + measures[i - 1].luminance
                    - 2 * m.luminance
                    if m.flash is not None
                    else None
                ),
                "flash_luminance": m.flash,
                "peak_tile_pop_delta_e_2000": m.peak_pop,
                "lighting_step_luminance": m.lighting_step,
                "drop_motion_ratio": m.drop_ratio,
                "source_stride_ratio": m.stride_ratio,
                "fixed_line_contrast": (
                    max((line.contrast for line in m.lines), default=0.0)
                    if m.lines is not None
                    else None
                ),
            }
        )
    return rows


def _attach_series(
    found: list[tuple[int, dict[str, Any]]],
    rows: list[dict[str, Any]],
    measures: list[_Measures],
    boundaries: list[int],
    settings: AuditSettings,
) -> None:
    metrics = {
        "duplicate": ("incoming_mean_code_difference", "sRGB code"),
        "dropped_frame": ("drop_motion_ratio", "ratio"),
        "flash": ("flash_luminance", "relative luminance"),
        "localized_pop": ("peak_tile_pop_delta_e_2000", "CIEDE2000"),
        "lighting_step": ("lighting_step_luminance", "relative luminance"),
        "fixed_line": ("fixed_line_contrast", "relative luminance"),
        "periodic_pulse": ("tile_impulse_luminance", "relative luminance"),
    }
    positions = {row["frame_index"]: j for j, row in enumerate(rows)}
    for i, finding in found:
        boundary = bisect_right(boundaries, i)
        start, stop = boundaries[boundary - 1 : boundary + 1]
        end = (
            positions[finding["end_frame_index"]] if "end_frame_index" in finding else i
        )
        metric, units = metrics[finding["class"]]
        if finding.get("missing_index_is_explicit"):
            metric = "source_stride_ratio"
        elif finding["class"] == "localized_pop" and "tile" in finding:
            metric = "tile_pop_delta_e_2000"
        tile_measure = {
            "tile_impulse_luminance": "impulse",
            "tile_pop_delta_e_2000": "pop",
        }.get(metric)
        points = []
        for j in range(
            max(start, i - settings.window_radius),
            min(stop, end + settings.window_radius + 1),
        ):
            if tile_measure is not None:
                tile = tuple(finding["tile"])
                values = getattr(measures[j], tile_measure)
                value = float(abs(values[tile])) if values is not None else None
            else:
                value = rows[j][metric]
            points.append([rows[j]["frame_index"], value])
        finding["time_series"] = {"metric": metric, "units": units, "points": points}


def audit(
    frames: str | Path | FlythroughManifest,
    *,
    out_dir: str | Path | None = None,
    frames_dir: str | Path | None = None,
    cuts: Sequence[int] = (),
    holds: Sequence[tuple[int, int]] = (),
    settings: AuditSettings = AuditSettings(),
) -> dict[str, Any]:
    """Audit a numbered PNG directory or CHRONOS manifest and return a JSON-ready report.

    A cut names the first frame of a new shot. Holds are inclusive frame-number
    pairs and permit repeats within each pair. Manifest annotations and keyword
    declarations are combined. With out_dir, write audit.json and evidence PNGs.
    """
    sources, manifest, manifest_path = _sources(frames, frames_dir)
    target = Path(out_dir).resolve() if out_dir is not None else None
    if target is not None and manifest_path is not None:
        if (target / "audit.json").resolve() == manifest_path:
            raise ValueError("report destination overlaps the input manifest")
    manifest_cuts, manifest_holds = manifest.get("cuts", []), manifest.get("holds", [])
    if not isinstance(manifest_cuts, list) or not isinstance(manifest_holds, list):
        raise ValueError("manifest cuts and holds must be lists")
    cuts, hold_ranges, held, boundaries = _timeline(
        sources, (*cuts, *manifest_cuts), (*holds, *manifest_holds)
    )
    grid, measures = _measure(sources, boundaries, held, settings)
    found = _find(sources, grid, measures, boundaries, held, settings)
    rows = _measure_rows(sources, measures)
    _attach_series(found, rows, measures, boundaries, settings)
    report = {
        "schema": "forge3d.film_audit/1",
        "frame_count": len(sources),
        "resolution": list(grid.shape[::-1]),
        "analysis_resolution": list(grid.window.shape[::-1]),
        "analysis_tile_bounds": {
            "x": grid.x[grid.tile_x].tolist(),
            "y": grid.y[grid.tile_y].tolist(),
        },
        "cuts": cuts,
        "holds": [list(hold) for hold in hold_ranges],
        "settings": asdict(settings),
        "findings": [finding for _, finding in found],
        "measures": rows,
        "tile_luminance_spectra": [
            {
                "first_frame_index": sources[start].index,
                "last_frame_index": sources[stop - 1].index,
                "frequencies_cycles_per_frame": np.fft.rfftfreq(stop - start).tolist(),
                "amplitudes": temporal_amplitudes(
                    np.stack([m.tile_luminance for m in measures[start:stop]])
                ).tolist(),
            }
            for start, stop in pairwise(boundaries)
        ],
    }
    if target is not None:
        _write_output(target, sources, grid, found, held, report)
    return report


def _hold_argument(value: str) -> tuple[int, int]:
    first, separator, last = value.partition(":")
    if (
        not separator
        or not first.isascii()
        or not first.isdecimal()
        or not last.isascii()
        or not last.isdecimal()
    ):
        raise argparse.ArgumentTypeError("hold must be FIRST:LAST using frame numbers")
    return int(first), int(last)


def _main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frames", type=Path, help="a numbered PNG directory or CHRONOS manifest"
    )
    parser.add_argument(
        "--frames-dir", type=Path, help="override manifest PNG directory"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("film-audit"))
    parser.add_argument("--cut", type=int, action="append", default=[], metavar="FRAME")
    parser.add_argument(
        "--hold", type=_hold_argument, action="append", default=[], metavar="FIRST:LAST"
    )
    parser.add_argument(
        "--mad-multiplier", type=float, default=AuditSettings.mad_multiplier
    )
    options = parser.parse_args(args)
    try:
        report = audit(
            options.frames,
            out_dir=options.out_dir,
            frames_dir=options.frames_dir,
            cuts=options.cut,
            holds=options.hold,
            settings=AuditSettings(mad_multiplier=options.mad_multiplier),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    count = len(report["findings"])
    print(
        f"{report['frame_count']} frames, {count} findings. {options.out_dir / 'audit.json'}"
    )
    return int(count > 0)


if __name__ == "__main__":
    raise SystemExit(_main())
