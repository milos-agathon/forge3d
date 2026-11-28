"""Deterministic source-PNG controls and injections for ARGUS."""

from collections.abc import Iterator
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from forge3d._png import save_png
from forge3d.film_audit import AuditSettings, audit

SIZE = (192, 108)
COUNT = 21
INJECT_AT = COUNT // 2
POP_REGION = (119, 63, 137, 78)
CONTROLS = ("static", "orbit", "fast_orbit", "zoom", "daylight", "hold", "cut")
CASES = (
    ("static", "bright_flash", "flash"),
    ("static", "mild_flash", "flash"),
    ("orbit", "dark_flash", "flash"),
    ("daylight", "bright_flash", "flash"),
    ("orbit", "duplicate", "duplicate"),
    ("zoom", "near_duplicate", "duplicate"),
    ("orbit", "periodic_pulse", "periodic_pulse"),
    ("orbit", "localized_reseed", "periodic_pulse"),
    ("orbit", "drop_numbered", "dropped_frame"),
    ("orbit", "drop_renumbered", "dropped_frame"),
    ("zoom", "drop_renumbered", "dropped_frame"),
    ("orbit", "bright_step", "lighting_step"),
    ("static", "dark_step", "lighting_step"),
    ("orbit", "vertical_line", "fixed_line"),
    ("zoom", "diagonal_line", "fixed_line"),
    ("orbit", "pop", "localized_pop"),
    ("orbit", "small_pop", "localized_pop"),
    ("zoom", "edge_pop", "localized_pop"),
    ("hold", "pop", "localized_pop"),
)


def _scene(story: str, frame: int, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    t = frame - INJECT_AT
    x = (np.arange(width, dtype=np.float32) - width / 2)[None, :] / width
    y = (np.arange(height, dtype=np.float32) - height / 2)[:, None] / width
    if story in ("orbit", "fast_orbit"):
        speed = 1 if story == "orbit" else 2
        angle = speed * (0.025 * t + 0.0003 * t * t)
        ca, sa = np.cos(angle), np.sin(angle)
        x, y = ca * x - sa * y - speed * 0.012 * t, sa * x + ca * y + 0.006 * t
    elif story == "zoom":
        scale = np.exp(-0.032 * t)
        x, y = (x + 0.08) * scale - 0.08, (y - 0.06) * scale + 0.06
    elif story == "cut" and t >= 0:
        x, y = x + 0.37, y - 0.21

    relief = 22 * np.sin(37 * x + 9 * y) + 17 * np.cos(18 * y - 20 * x)
    detail = 9 * np.sin(79 * x + 23 * y) * np.cos(93 * y - 31 * x)
    ridge = (y > 0.05 + 0.06 * np.sin(16 * x)) * 44
    rock = (((x + 0.16) / 0.12) ** 2 + ((y - 0.02) / 0.07) ** 2 < 1) * 60
    rgb = np.stack(
        (
            102 + relief + detail + rock,
            123 + relief + ridge,
            120 + 0.8 * relief - detail - ridge + rock,
        ),
        axis=-1,
    )
    if story == "daylight":
        amount = (1 - np.cos(np.pi * frame / (COUNT - 1))) / 2
        rgb = rgb * (1 - 0.45 * amount) + np.array([12, 3, 10]) * amount
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def declarations(story: str) -> dict:
    if story == "hold":
        return {"holds": [(INJECT_AT - 2, INJECT_AT + 2)]}
    if story == "cut":
        return {"cuts": [INJECT_AT]}
    return {}


def region_for(defect: str, size: tuple[int, int] = SIZE) -> tuple[int, ...]:
    regions = {
        "edge_pop": (3, 81, 17, 97),
        "small_pop": (116, 68, 124, 76),
        "vertical_line": (84, 8, 86, 100),
        "diagonal_line": (45, 13, 113, 80),
        "localized_reseed": (51, 25, 107, 68),
    }
    region = regions.get(defect, POP_REGION)
    return tuple(
        round(i * size[axis % 2] / SIZE[axis % 2]) for axis, i in enumerate(region)
    )


def frames(
    story: str, defect: str | None = None, *, size: tuple[int, int] = SIZE
) -> Iterator[np.ndarray]:
    last = None
    for index in range(COUNT):
        if index == INJECT_AT and defect in {"drop_numbered", "drop_renumbered"}:
            continue
        t = index
        if story == "hold":
            first, end = declarations("hold")["holds"][0]
            t = min(t, first) if t <= end else t - (end - first)
        image = _scene("orbit" if story == "hold" else story, t, size)
        if index >= INJECT_AT and defect in {"bright_step", "dark_step"}:
            shift = 48 if defect == "bright_step" else -37
            image = (image.astype(np.int16) + shift).clip(0, 255).astype(np.uint8)
        if index >= INJECT_AT and (index - INJECT_AT) % 3 == 0:
            if defect == "periodic_pulse":
                image = (image.astype(np.int16) + 48).clip(0, 255).astype(np.uint8)
            elif defect == "localized_reseed":
                x0, y0, x1, y1 = region_for(defect, size)
                rng = np.random.default_rng(index)
                patch = image[y0:y1, x0:x1]
                noise = rng.integers(26, 62, patch.shape, dtype=np.int16)
                patch[:] = (patch.astype(np.int16) + noise).clip(0, 255)
        if INJECT_AT <= index < INJECT_AT + 10:
            if defect == "vertical_line":
                x0, y0, x1, y1 = region_for(defect, size)
                image[y0:y1, x0:x1] = 255
            elif defect == "diagonal_line":
                x0, y0, x1, y1 = region_for(defect, size)
                y = np.arange(y0, y1)
                x = x0 + (x1 - x0 - 2) * (y - y0) // (y1 - y0 - 1)
                image[y, x] = image[y, x + 1] = 0
        if index == INJECT_AT:
            if defect in ("bright_flash", "dark_flash", "mild_flash"):
                delta = {"bright_flash": 75, "dark_flash": -57, "mild_flash": 8}[defect]
                image = (image.astype(np.int16) + delta).clip(0, 255).astype(np.uint8)
            elif defect in ("duplicate", "near_duplicate"):
                image = last.copy()
                if defect == "near_duplicate":
                    # A small decoded difference despite repeating the previous pose.
                    image[::2, ::2, 0] -= 1
            elif defect in ("pop", "edge_pop", "small_pop"):
                x0, y0, x1, y1 = region_for(defect, size)
                image[y0:y1, x0:x1] = [235, 35, 201]
        yield image
        last = image


def write(
    directory: Path,
    story: str,
    defect: str | None = None,
    *,
    size: tuple[int, int] = SIZE,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames(story, defect, size=size)):
        frame_number = i + 1 if defect == "drop_numbered" and i >= INJECT_AT else i
        save_png(directory / f"frame_{frame_number}.png", frame)


def benchmark(root: Path) -> None:
    for width, height in ((1920, 1080), (3840, 2160)):
        source = root / f"{width}x{height}"
        write(source, "orbit", size=(width, height))
        start = perf_counter()
        report = audit(source, out_dir=root / f"{width}x{height}-audit")
        seconds = perf_counter() - start
        assert report["frame_count"] == COUNT and not report["findings"], report[
            "findings"
        ]
        print(
            f"{width}x{height}, {COUNT} frames, {seconds:.3f} s, {COUNT / seconds:.2f} FPS"
        )


def calibrate(root: Path) -> None:
    keys = (
        "flash_luminance",
        "peak_tile_pop_delta_e_2000",
        "lighting_step_luminance",
        "drop_motion_ratio",
        "fixed_line_contrast",
    )
    for story, defect, expected in [(s, None, None) for s in CONTROLS] + list(CASES):
        source = root / f"{story}-{defect}"
        write(source, story, defect)
        timeline = declarations(story)
        report = audit(source, **timeline)
        finding_rows = report["findings"]
        result = {
            "story": story,
            "injection": defect,
            "expected": expected,
            "findings": [
                {
                    key: finding[key]
                    for key in ("frame_index", "class", "score", "threshold", "region")
                }
                for finding in finding_rows
            ],
            "peaks": {
                key: max(
                    (row[key] for row in report["measures"] if row[key] is not None),
                    default=0,
                )
                for key in keys
            },
        }
        if defect is None:
            # Measure which line and pulse scores survive the structural tests
            # with their absolute floors disabled. Keep the other gates active.
            relaxed = audit(
                source,
                settings=AuditSettings(line_contrast_floor=0, pulse_floor=0),
                **timeline,
            )
            result["clean_pulse_peak_without_floor"] = max(
                (
                    finding["score"]
                    for finding in relaxed["findings"]
                    if finding["class"] == "periodic_pulse"
                ),
                default=0,
            )
            result["clean_line_peak_without_floor"] = max(
                (row["fixed_line_contrast"] or 0 for row in relaxed["measures"]),
                default=0,
            )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure ARGUS on generated source PNGs"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--benchmark", type=Path)
    modes.add_argument("--calibrate", type=Path)
    args = parser.parse_args()
    if args.benchmark is not None:
        benchmark(args.benchmark)
    else:
        calibrate(args.calibrate)
