#!/usr/bin/env python3
"""Recreate the Humanity Globe population-density video with forge3d helpers.

Reference concept and original rayshader/rayrender recipe:
https://gist.github.com/tylermorganwall/3ee1c6e2a5dff19aca7836c05cbbf9ac

Data source family: GPW-v4 population density, revision 11, 2020.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "out" / "humanity_globe"
DEFAULT_CACHE_DIR = REPO_ROOT / "examples" / ".cache" / "humanity_globe"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "humanity_globe_forge3d.mp4"
DEFAULT_PREVIEW = DEFAULT_OUTPUT_DIR / "humanity_globe_preview.png"
DEFAULT_GPW_15MIN = REPO_ROOT / "data" / "gpw_v4_population_density_rev11_2020_15_min.tif"
DEFAULT_SIZE = 720
DEFAULT_FPS = 25
DEFAULT_DURATION = 28.8
GPW_30SEC_URL = "https://pacific-data.sprep.org/system/files/Global_2020_PopulationDensity30sec_GPWv4.tiff"
EXPECTED_15MIN_SHAPE = (720, 1440)
EXPECTED_30SEC_SHAPE = (21600, 43200)
AGGREGATION_FACTOR = 30
DENSITY_THRESHOLDS = np.array([1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
LEGEND_TITLE = "People per 30km^2"
LEGEND_LABELS = ("0", "1>", "5>", "10>", "50>", "100>", "500>", "1000>")
TITLE_TEXT = "The Humanity Globe: World Population Density, 30km^2 Grid"
DATA_CREDIT = "Data: 2020 GPW-v4"
CREATED_CREDIT = "Created with forge3d"
REFERENCE_CREDIT = "Reference: @tylermorganwall"
INITIAL_CENTER_LONGITUDE = -100.0
FALLBACK_TURBO7 = np.array(
    [
        [109, 76, 134],
        [116, 173, 239],
        [52, 214, 207],
        [183, 230, 121],
        [247, 222, 91],
        [246, 142, 64],
        [209, 55, 36],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class GpwData:
    path: Path
    derived: bool
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a forge3d recreation of the Humanity Globe population-density MP4."
    )
    parser.add_argument("--gpw-tif", type=Path, default=None, help="Explicit GPW-v4 2020 population-density GeoTIFF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def frame_count(args: argparse.Namespace) -> int:
    if args.frames is not None:
        return max(1, int(args.frames))
    return max(1, int(round(float(args.fps) * float(args.duration))))


def classify_density(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    safe = np.where(np.isfinite(data), data, 0.0)
    classes = np.zeros(safe.shape, dtype=np.uint8)
    for idx, threshold in enumerate(DENSITY_THRESHOLDS, start=1):
        classes[safe > threshold] = idx
    return classes


def validate_15min_grid(data: np.ndarray) -> None:
    if tuple(data.shape) != EXPECTED_15MIN_SHAPE:
        raise ValueError(
            f"Expected 15-minute GPW grid with shape {EXPECTED_15MIN_SHAPE}, got {tuple(data.shape)}"
        )


def turbo_class_palette() -> np.ndarray:
    base = np.array([[175, 175, 175]], dtype=np.uint8)
    try:
        import matplotlib.colormaps as colormaps

        cmap = colormaps["turbo"]
        xs = np.linspace(0.04, 0.92, 7, dtype=np.float32)
        turbo = np.round(np.asarray(cmap(xs), dtype=np.float32)[:, :3] * 255.0).astype(np.uint8)
    except Exception:
        turbo = FALLBACK_TURBO7.copy()
    return np.vstack([base, turbo]).astype(np.uint8)


def resolve_gpw_source(args: argparse.Namespace) -> GpwData:
    candidates = []
    if args.gpw_tif is not None:
        candidates.append(Path(args.gpw_tif))
    candidates.extend([DEFAULT_GPW_15MIN, Path(args.cache_dir) / DEFAULT_GPW_15MIN.name])
    for candidate in candidates:
        if candidate.exists():
            return GpwData(candidate, derived=False, source=str(candidate))
    derived_path = derive_15min_gpw(Path(args.cache_dir), force=bool(args.force))
    return GpwData(derived_path, derived=True, source=GPW_30SEC_URL)


def derive_15min_gpw(cache_dir: Path, *, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / DEFAULT_GPW_15MIN.name
    metadata_path = out_path.with_suffix(out_path.suffix + ".json")
    if out_path.exists() and not force:
        return out_path
    source_path = cache_dir / "Global_2020_PopulationDensity30sec_GPWv4.tiff"
    if not source_path.exists() or force:
        download_path = source_path.with_suffix(source_path.suffix + ".download")
        if download_path.exists():
            download_path.unlink()
        try:
            urllib.request.urlretrieve(GPW_30SEC_URL, download_path)
            download_path.replace(source_path)
        except Exception as exc:
            if download_path.exists():
                download_path.unlink()
            raise SystemExit(
                "Missing GPW-v4 15-minute raster. Provide --gpw-tif pointing to "
                "gpw_v4_population_density_rev11_2020_15_min.tif, or allow download "
                f"from {GPW_30SEC_URL}. Download failed: {exc}"
            ) from exc
    aggregate_30sec_to_15min(source_path, out_path)
    metadata_path.write_text(
        json.dumps(
            {"source": GPW_30SEC_URL, "aggregation": "mean 30x30 cells", "shape": list(EXPECTED_15MIN_SHAPE)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def aggregate_30sec_to_15min(source_path: Path, out_path: Path) -> None:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "rasterio is required to derive the 15-minute GPW raster. Install rasterio or pass --gpw-tif."
        ) from exc

    with rasterio.open(source_path) as src:
        if (src.height, src.width) != EXPECTED_30SEC_SHAPE:
            raise ValueError(
                f"Expected 30-arc-second GPW raster shape {EXPECTED_30SEC_SHAPE}, got {(src.height, src.width)}"
            )
        profile = src.profile.copy()
        out = np.zeros(EXPECTED_15MIN_SHAPE, dtype=np.float32)
        for out_row in range(EXPECTED_15MIN_SHAPE[0]):
            window = rasterio.windows.Window(0, out_row * AGGREGATION_FACTOR, src.width, AGGREGATION_FACTOR)
            block = src.read(1, window=window, masked=True).astype(np.float32)
            row = block.reshape(AGGREGATION_FACTOR, EXPECTED_15MIN_SHAPE[1], AGGREGATION_FACTOR).mean(axis=(0, 2))
            out[out_row, :] = np.asarray(row.filled(0.0) if hasattr(row, "filled") else row, dtype=np.float32)
    profile.update(
        width=EXPECTED_15MIN_SHAPE[1],
        height=EXPECTED_15MIN_SHAPE[0],
        dtype="float32",
        count=1,
        compress="deflate",
        transform=from_origin(-180.0, 90.0, 0.25, 0.25),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def orbit_longitude(frame_index: int, total_frames: int) -> float:
    total = max(1, int(total_frames))
    return INITIAL_CENTER_LONGITUDE + 360.0 * (int(frame_index) / float(total))


def sphere_lat_lon(size: int, center_lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = int(size)
    coords = (np.arange(size, dtype=np.float32) + 0.5) / float(size)
    x = (coords[None, :] - 0.5) * 2.0
    y = (0.5 - coords[:, None]) * 2.0
    radius = 0.72
    sx = x / radius
    sy = y / radius
    rr = sx * sx + sy * sy
    visible = rr <= 1.0
    z = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0)).astype(np.float32)
    lat_column = np.degrees(np.arcsin(np.clip(sy, -1.0, 1.0))).astype(np.float32)
    lon_offset = np.degrees(np.arctan2(sx, np.maximum(z, 1e-6))).astype(np.float32)
    lon = ((float(center_lon) + lon_offset + 180.0) % 360.0 - 180.0).astype(np.float32)
    lat = np.broadcast_to(lat_column, z.shape).astype(np.float32)
    sx_full = np.broadcast_to(sx, z.shape).astype(np.float32)
    sy_full = np.broadcast_to(sy, z.shape).astype(np.float32)
    normals = np.stack([sx_full, sy_full, z], axis=2)
    normals[~visible] = 0.0
    return visible, lat, lon, normals


def sample_classes(classes: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    h, w = classes.shape
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(np.int32), 0, h - 1)
    col = np.clip(((lon + 180.0) / 360.0 * w).astype(np.int32), 0, w - 1)
    return classes[row, col]


def _light(normals: np.ndarray) -> np.ndarray:
    light_dir = np.array([-0.45, 0.35, 0.82], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    lambert = np.clip((normals * light_dir.reshape(1, 1, 3)).sum(axis=2), 0.0, 1.0)
    rim = np.power(np.clip(1.0 - normals[:, :, 2], 0.0, 1.0), 2.0)
    return np.clip(0.38 + 0.52 * lambert + 0.12 * rim, 0.0, 1.0)


def render_frame(
    density: np.ndarray,
    *,
    frame_index: int,
    total_frames: int,
    size: int = DEFAULT_SIZE,
    include_text: bool = True,
) -> np.ndarray:
    size = int(size)
    classes = classify_density(density)
    visible, lat, lon, normals = sphere_lat_lon(size, orbit_longitude(frame_index, total_frames))
    sampled = sample_classes(classes, lat, lon)
    palette = turbo_class_palette()
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    base = palette[0].astype(np.float32)
    light = _light(normals)
    rgb[visible] = base * light[visible, None]
    positive = visible & (sampled > 0)
    if positive.any():
        color = palette[sampled]
        boost = np.clip(0.92 + 0.20 * (1.0 - normals[:, :, 2]), 0.85, 1.18)
        rgb[positive] = np.clip(color[positive].astype(np.float32) * boost[positive, None], 0.0, 255.0)
    alpha = np.full((size, size, 1), 255, dtype=np.uint8)
    frame = np.dstack([np.clip(rgb, 0.0, 255.0).astype(np.uint8), alpha])
    if include_text:
        frame = compose_frame_text(frame)
    return frame


def compose_frame_text(frame: np.ndarray) -> np.ndarray:
    return frame


def main() -> int:
    parse_args()
    raise SystemExit("Rendering is implemented in later tasks.")


if __name__ == "__main__":
    raise SystemExit(main())
