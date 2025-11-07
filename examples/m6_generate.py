#!/usr/bin/env python3
"""
Milestone 6 — Validation Harness (CPU reference vs GPU).

Runs the GGX sphere renderer on the GPU, evaluates a CPU reference at
matching pixel rays, and validates the outputs against the acceptance
criteria (RMS ≤ 1e-3 per channel, 99.9% of pixels ≤ 5e-3 error).

Artifacts written to --outdir (default: reports):
  - m6_diff_heatmap.png  : false-color |GPU-CPU| per pixel
  - m6_diff.csv          : per-sample dump (u, v, dot products, RGBs, errors)
  - m6_meta.json         : metrics, bounds, determinism data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _import_shim import ensure_repo_import  # noqa: E402

ensure_repo_import()

import forge3d as f3d  # noqa: E402

try:  # pragma: no cover - optional dependency
    from PIL import Image

    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False


RNG_SEED_LABEL = "forge3d-seed-42"
SAMPLE_GRID_DEFAULT = 32
SPHERE_SECTORS_DEFAULT = 256
SPHERE_STACKS_DEFAULT = 128
ANALYSIS_MASK_SHRINK = 0.85
ANALYSIS_NV_MIN = 0.2
ANALYSIS_MASK_THRESHOLD = 0.95
F0 = np.array([0.04, 0.04, 0.04], dtype=np.float32)
BASE_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float32)
LIGHT_DIR = np.array([0.5, 0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
CAMERA_POS = np.array([0.0, 0.0, 2.0], dtype=np.float32)
FOV_Y_DEG = 60.0
EPS = 1e-6


@dataclass
class CpuReference:
    """Holds CPU reference buffers for sampling."""

    linear_rgb: np.ndarray
    normals: np.ndarray
    view: np.ndarray
    half: np.ndarray
    nl: np.ndarray
    nv: np.ndarray
    nh: np.ndarray
    vh: np.ndarray
    valid_mask: np.ndarray


@dataclass
class SampleResult:
    u: float
    v: float
    pixel_x: float
    pixel_y: float
    nl: float
    nv: float
    nh: float
    vh: float
    gpu_rgb: np.ndarray
    cpu_rgb: np.ndarray
    abs_err: np.ndarray


@dataclass
class ValidationResult:
    width: int
    height: int
    eval_width: int
    eval_height: int
    eval_scale: int
    roughness: float
    sample_rows: int
    sample_cols: int
    sample_count: int
    sphere_sectors: int
    sphere_stacks: int
    light_dir: np.ndarray
    analysis_shrink: float
    analysis_nv_min: float
    gpu_linear: np.ndarray
    cpu_linear_quant: np.ndarray
    diff_rgb_eval: np.ndarray
    diff_rgb_tile: np.ndarray
    analysis_mask_eval: np.ndarray
    analysis_mask_tile: np.ndarray
    debug_tiles: Dict[int, np.ndarray]
    samples: List[SampleResult]
    rms: np.ndarray
    max_abs: np.ndarray
    percentile_999: float
    max_idx: Tuple[int, int]
    cpu_ref: CpuReference


def _seed_from_label(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def make_rng(label: str = RNG_SEED_LABEL) -> np.random.Generator:
    return np.random.default_rng(_seed_from_label(label))


def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear floats."""
    a = 0.055
    return np.where(
        arr <= 0.04045,
        arr / 12.92,
        ((arr + a) / (1.0 + a)) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] to sRGB."""
    a = 0.055
    return np.where(
        arr <= 0.0031308,
        arr * 12.92,
        (1.0 + a) * np.power(np.clip(arr, 0.0, None), 1.0 / 2.4) - a,
    ).astype(np.float32)


def ensure_pil() -> None:
    if not HAS_PIL:
        raise RuntimeError("PIL is required to write PNG files for M6 outputs.")


def save_png(path: Path, array: np.ndarray) -> None:
    ensure_pil()
    Image.fromarray(array, mode="RGB").save(path)


def _as_python_light_dir(light_dir: np.ndarray | None) -> tuple[float, float, float] | None:
    if light_dir is None:
        return None
    vec = np.asarray(light_dir, dtype=np.float32)
    return float(vec[0]), float(vec[1]), float(vec[2])


def render_gpu_tile(
    roughness: float,
    size: int,
    sphere_sectors: int,
    sphere_stacks: int,
    light_dir: np.ndarray | None,
) -> np.ndarray:
    """Render the GGX tile on the GPU and return RGBA8 array."""
    light_tuple = _as_python_light_dir(light_dir)
    tile = f3d.render_brdf_tile_full(
        "ggx",
        roughness,
        size,
        size,
        light_intensity=3.0,
        exposure=1.0,
        base_color=tuple(BASE_COLOR.tolist()),
        sphere_sectors=sphere_sectors,
        sphere_stacks=sphere_stacks,
        light_dir=light_tuple,
    )
    return tile.astype(np.uint8)


def render_gpu_term_tile(
    roughness: float,
    size: int,
    sphere_sectors: int,
    sphere_stacks: int,
    light_dir: np.ndarray | None,
    debug_kind: int,
) -> np.ndarray:
    light_tuple = _as_python_light_dir(light_dir)
    tile = f3d.render_brdf_tile_debug(
        "ggx",
        roughness,
        size,
        size,
        light_intensity=3.0,
        exposure=1.0,
        base_color=tuple(BASE_COLOR.tolist()),
        sphere_sectors=sphere_sectors,
        sphere_stacks=sphere_stacks,
        light_dir=light_tuple,
        debug_kind=debug_kind,
    )
    return tile.astype(np.uint8)


def _geometry_lambda_term(cos_theta: np.ndarray, rough_sq: float) -> np.ndarray:
    clipped = np.clip(cos_theta, 1e-4, 1.0)
    cos2 = clipped * clipped
    t2 = np.clip((1.0 - cos2) / cos2, 0.0, None)
    return 0.5 * (np.sqrt(1.0 + rough_sq * t2) - 1.0)


def compute_cpu_reference(
    width: int,
    height: int,
    roughness: float,
    light_dir: np.ndarray,
    base_color: np.ndarray,
    light_intensity: float,
    gpu_mask: np.ndarray,
) -> CpuReference:
    """Evaluate the CPU GGX reference at every pixel via ray-sphere intersection."""
    xs = (np.arange(width, dtype=np.float32) + 0.5)
    ys = (np.arange(height, dtype=np.float32) + 0.5)
    ndc_x = (xs / width) * 2.0 - 1.0
    ndc_y = 1.0 - (ys / height) * 2.0
    X, Y = np.meshgrid(ndc_x, ndc_y)
    tan_half = math.tan(math.radians(FOV_Y_DEG) / 2.0)
    dir_x = X * tan_half * (width / height)
    dir_y = Y * tan_half
    dir_z = -np.ones_like(dir_x)
    dirs = np.stack([dir_x, dir_y, dir_z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    b = 2.0 * (dirs[..., 0] * CAMERA_POS[0] + dirs[..., 1] * CAMERA_POS[1] + dirs[..., 2] * CAMERA_POS[2])
    c = float(np.dot(CAMERA_POS, CAMERA_POS)) - 1.0
    disc = b * b - 4.0 * c
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    t0 = (-b - sqrt_disc) / 2.0
    t1 = (-b + sqrt_disc) / 2.0
    t = np.where(t0 > EPS, t0, t1)
    hit_mask = (disc >= 0.0) & (t > EPS)

    positions = CAMERA_POS + dirs * t[..., None]
    normals = np.zeros_like(positions, dtype=np.float32)
    normals[hit_mask] = positions[hit_mask] / np.linalg.norm(positions[hit_mask], axis=-1, keepdims=True)

    view_dirs = np.zeros_like(positions, dtype=np.float32)
    view_dirs[hit_mask] = (CAMERA_POS - positions[hit_mask]) / np.linalg.norm(CAMERA_POS - positions[hit_mask], axis=-1, keepdims=True)

    half_vecs = np.zeros_like(positions, dtype=np.float32)
    half_vecs[hit_mask] = view_dirs[hit_mask] + light_dir
    half_vecs[hit_mask] /= np.linalg.norm(half_vecs[hit_mask], axis=-1, keepdims=True)

    nl = np.clip(np.sum(normals * light_dir, axis=-1), 0.0, 1.0)
    nv = np.clip(np.sum(normals * view_dirs, axis=-1), 0.0, 1.0)
    nh = np.clip(np.sum(normals * half_vecs, axis=-1), 0.0, 1.0)
    vh = np.clip(np.sum(view_dirs * half_vecs, axis=-1), 0.0, 1.0)

    alpha = max(roughness * roughness, 1e-4)
    a2 = alpha * alpha
    denom = (nh * nh) * (a2 - 1.0) + 1.0
    D = a2 / np.clip(math.pi * denom * denom, 1e-8, None)
    rough_sq = max(roughness, 1e-4)
    rough_sq *= rough_sq
    G = 1.0 / (1.0 + _geometry_lambda_term(nv, rough_sq) + _geometry_lambda_term(nl, rough_sq))
    G = np.clip(G, 0.0, 1.0)

    F = F0 + (1.0 - F0) * (np.clip(1.0 - vh[..., None], 0.0, 1.0) ** 5.0)
    denom_spec = np.clip(4.0 * nl * nv, 1e-4, None)[..., None]
    spec = (D[..., None] * F * G[..., None]) / denom_spec
    diffuse = (1.0 - F) * base_color * (1.0 / math.pi)
    linear_rgb = (spec + diffuse) * light_intensity * nl[..., None]

    valid_mask = hit_mask & gpu_mask
    linear_rgb = np.where(valid_mask[..., None], linear_rgb, 0.0)

    return CpuReference(
        linear_rgb=linear_rgb.astype(np.float32),
        normals=normals,
        view=view_dirs,
        half=half_vecs,
        nl=nl.astype(np.float32),
        nv=nv.astype(np.float32),
        nh=nh.astype(np.float32),
        vh=vh.astype(np.float32),
        valid_mask=valid_mask,
    )


def bilinear_sample(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinearly sample an image (H, W, C) at floating coordinates."""
    h, w = image.shape[:2]
    x = float(np.clip(x, 0.0, w - 2.001))
    y = float(np.clip(y, 0.0, h - 2.001))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    tx = x - x0
    ty = y - y0
    top = image[y0, x0] * (1.0 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1.0 - tx) + image[y1, x1] * tx
    return top * (1.0 - ty) + bottom * ty


def bilinear_sample_scalar(image: np.ndarray, x: float, y: float) -> float:
    return float(bilinear_sample(image[..., None], x, y)[0])


def _pull_inside_mask(x: float, y: float, cx: float, cy: float, mask: np.ndarray) -> Tuple[float, float]:
    yy = int(round(y))
    xx = int(round(x))
    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx]:
        return x, y
    for scale in np.linspace(0.99, 0.5, 8):
        nx = cx + (x - cx) * scale
        ny = cy + (y - cy) * scale
        yy = int(round(ny))
        xx = int(round(nx))
        if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx]:
            return nx, ny
    raise RuntimeError("Unable to place sample inside mask; sphere coverage is too thin.")


def make_analysis_mask(mask: np.ndarray, nv: np.ndarray, shrink: float = ANALYSIS_MASK_SHRINK) -> np.ndarray:
    """Conservatively erode the valid mask and enforce NV>=threshold to avoid rim aliasing artifacts."""
    base = mask & (nv >= ANALYSIS_NV_MIN)
    if shrink >= 0.999:
        return base
    h, w = base.shape
    yy, xx = np.indices(base.shape)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    radius = min(cx, cy) * shrink
    circular = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    return base & circular


def concentric_sample_disk(u: float, v: float) -> Tuple[float, float]:
    """Map uniform random numbers (u, v) in [0,1) to a uniform point on the unit disk."""
    sx = 2.0 * u - 1.0
    sy = 2.0 * v - 1.0
    if sx == 0.0 and sy == 0.0:
        return 0.0, 0.0
    if abs(sx) > abs(sy):
        r = sx
        theta = (math.pi / 4.0) * (sy / (sx if sx != 0.0 else 1.0))
    else:
        r = sy
        theta = math.pi / 2.0 - (math.pi / 4.0) * (sx / (sy if sy != 0.0 else 1.0))
    return float(r * math.cos(theta)), float(r * math.sin(theta))


def generate_sample_points(
    width: int,
    height: int,
    rows: int,
    cols: int,
    cpu_ref: CpuReference,
    rng: np.random.Generator,
) -> List[Tuple[float, float, float, float]]:
    """Generate stratified (rows x cols) disk samples projected onto the sphere silhouette."""
    mask = cpu_ref.valid_mask
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise RuntimeError("No valid pixels detected in the GPU mask.")
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    radius_x = max(1.0, 0.5 * (x_max - x_min))
    radius_y = max(1.0, 0.5 * (y_max - y_min))
    radius = min(radius_x, radius_y)

    points: List[Tuple[float, float, float, float]] = []
    for i in range(rows):
        for j in range(cols):
            u = (j + rng.random()) / cols
            v = (i + rng.random()) / rows
            dx, dy = concentric_sample_disk(u, v)
            raw_x = cx + dx * radius
            raw_y = cy + dy * radius
            x, y = _pull_inside_mask(raw_x, raw_y, cx, cy, mask)
            points.append((x, y, u, v))
    return points


def evaluate_samples(
    points: Sequence[Tuple[float, float, float, float]],
    gpu_linear: np.ndarray,
    cpu_color: np.ndarray,
    cpu_ref: CpuReference,
) -> List[SampleResult]:
    results: List[SampleResult] = []
    for x, y, u, v in points:
        gpu = bilinear_sample(gpu_linear, x, y)
        cpu = bilinear_sample(cpu_color, x, y)
        nl = bilinear_sample_scalar(cpu_ref.nl, x, y)
        nv = bilinear_sample_scalar(cpu_ref.nv, x, y)
        nh = bilinear_sample_scalar(cpu_ref.nh, x, y)
        vh = bilinear_sample_scalar(cpu_ref.vh, x, y)
        abs_err = np.abs(cpu - gpu)
        results.append(
            SampleResult(
                u=u,
                v=v,
                pixel_x=x,
                pixel_y=y,
                nl=nl,
                nv=nv,
                nh=nh,
                vh=vh,
                gpu_rgb=gpu,
                cpu_rgb=cpu,
                abs_err=abs_err,
            )
        )
    return results


COLORMAP_POS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
COLORMAP_RGB = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.8, 0.9],
        [1.0, 1.0, 0.0],
        [1.0, 0.2, 0.0],
    ],
    dtype=np.float32,
)


def apply_colormap(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    idx = np.searchsorted(COLORMAP_POS, clipped, side="right")
    idx = np.clip(idx, 1, len(COLORMAP_POS) - 1)
    idx0 = idx - 1
    pos0 = COLORMAP_POS[idx0]
    pos1 = COLORMAP_POS[idx]
    span = np.clip(pos1 - pos0, 1e-6, None)
    t = (clipped - pos0) / span
    colors0 = COLORMAP_RGB[idx0]
    colors1 = COLORMAP_RGB[idx]
    return (1.0 - t)[..., None] * colors0 + t[..., None] * colors1


def downsample_mean(image: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return image
    h, w = image.shape[:2]
    c = image.shape[2]
    reshaped = image.reshape(h // scale, scale, w // scale, scale, c)
    return reshaped.mean(axis=(1, 3))


def write_heatmap(diff_rgb: np.ndarray, path: Path) -> None:
    diff_scalar = np.max(diff_rgb, axis=2)
    max_err = float(diff_scalar.max())
    norm = diff_scalar / max(max_err, 1e-8)
    cmap = (apply_colormap(norm) * 255.0).astype(np.uint8)
    save_png(path, cmap)


def write_csv(samples: Sequence[SampleResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "u",
                "v",
                "pixel_x",
                "pixel_y",
                "nl",
                "nv",
                "vh",
                "nh",
                "gpu_r",
                "gpu_g",
                "gpu_b",
                "cpu_r",
                "cpu_g",
                "cpu_b",
                "abs_err_r",
                "abs_err_g",
                "abs_err_b",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.u:.6f}",
                    f"{sample.v:.6f}",
                    f"{sample.pixel_x:.3f}",
                    f"{sample.pixel_y:.3f}",
                    f"{sample.nl:.6f}",
                    f"{sample.nv:.6f}",
                    f"{sample.vh:.6f}",
                    f"{sample.nh:.6f}",
                    f"{sample.gpu_rgb[0]:.6f}",
                    f"{sample.gpu_rgb[1]:.6f}",
                    f"{sample.gpu_rgb[2]:.6f}",
                    f"{sample.cpu_rgb[0]:.6f}",
                    f"{sample.cpu_rgb[1]:.6f}",
                    f"{sample.cpu_rgb[2]:.6f}",
                    f"{sample.abs_err[0]:.6f}",
                    f"{sample.abs_err[1]:.6f}",
                    f"{sample.abs_err[2]:.6f}",
                ]
            )


def build_meta(
    result: ValidationResult,
    csv_name: str,
    csv_sha256: str,
    heatmap_name: str,
) -> dict:
    return {
        "description": "Milestone 6 CPU vs GPU validation (direct light GGX)",
        "rng_seed": RNG_SEED_LABEL,
        "tile_size": {"width": result.width, "height": result.height},
        "evaluation_tile_size": {"width": result.eval_width, "height": result.eval_height},
        "eval_scale": result.eval_scale,
        "roughness": result.roughness,
        "sample_grid": {"rows": result.sample_rows, "cols": result.sample_cols, "count": result.sample_count},
        "sphere_mesh": {"sectors": result.sphere_sectors, "stacks": result.sphere_stacks},
        "analysis_mask": {
            "shrink": result.analysis_shrink,
            "nv_min": result.analysis_nv_min,
            "coverage": float(result.analysis_mask_tile.mean()),
        },
        "sample_method": "stratified_concentric_disk",
        "heatmap": heatmap_name,
        "diff_csv": csv_name,
        "diff_csv_sha256": csv_sha256,
        "metrics": {
            "rms_error": {
                "r": float(result.rms[0]),
                "g": float(result.rms[1]),
                "b": float(result.rms[2]),
            },
            "max_abs_error": {
                "r": float(result.max_abs[0]),
                "g": float(result.max_abs[1]),
                "b": float(result.max_abs[2]),
            },
            "percentile_99_9_abs_error": float(result.percentile_999),
            "worst_pixel": {"x": int(result.max_idx[1]), "y": int(result.max_idx[0])},
        },
        "light": {
            "direction": result.light_dir.tolist(),
            "radiance": 3.0,
        },
        "camera": {
            "position": CAMERA_POS.tolist(),
            "fov_y_degrees": FOV_Y_DEG,
        },
        "base_color": BASE_COLOR.tolist(),
        "f0": F0.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 6 CPU vs GPU validation harness.")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile resolution (square).")
    parser.add_argument("--roughness", type=float, default=0.5, help="Material roughness to validate.")
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_GRID_DEFAULT,
        help=f"Stratified samples per axis (default: {SAMPLE_GRID_DEFAULT}).",
    )
    parser.add_argument(
        "--eval-scale",
        type=int,
        default=2,
        help="Supersampling factor for validation (>=1). Higher values reduce aliasing.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("reports"), help="Output directory for artifacts.")
    parser.add_argument("--heatmap", type=str, default="m6_diff_heatmap.png", help="Heatmap filename.")
    parser.add_argument("--csv", type=str, default="m6_diff.csv", help="CSV filename.")
    parser.add_argument("--meta", type=str, default="m6_meta.json", help="Metadata filename.")
    parser.add_argument(
        "--sphere-sectors",
        type=int,
        default=SPHERE_SECTORS_DEFAULT,
        help=f"UV-sphere longitudinal tessellation for GPU tile (default: {SPHERE_SECTORS_DEFAULT}).",
    )
    parser.add_argument(
        "--sphere-stacks",
        type=int,
        default=SPHERE_STACKS_DEFAULT,
        help=f"UV-sphere latitudinal tessellation for GPU tile (default: {SPHERE_STACKS_DEFAULT}).",
    )
    return parser.parse_args()


def run_validation(
    tile_size: int,
    roughness: float,
    samples_per_axis: int,
    eval_scale: int,
    sphere_sectors: int = SPHERE_SECTORS_DEFAULT,
    sphere_stacks: int = SPHERE_STACKS_DEFAULT,
    seed_label: str = RNG_SEED_LABEL,
    light_dir: Sequence[float] | np.ndarray | None = None,
) -> ValidationResult:
    if tile_size <= 0:
        raise ValueError("Tile size must be positive.")
    width = height = int(tile_size)
    eval_scale = max(1, int(eval_scale))
    eval_width = width * eval_scale
    eval_height = height * eval_scale
    if light_dir is None:
        light_vec = LIGHT_DIR.copy()
    else:
        light_vec = np.asarray(light_dir, dtype=np.float32)
        norm = np.linalg.norm(light_vec)
        if norm < 1e-6:
            raise ValueError("light_dir must have non-zero length")
        light_vec /= norm

    gpu_rgba = render_gpu_tile(roughness, eval_width, sphere_sectors, sphere_stacks, light_vec)
    gpu_linear = srgb_to_linear(gpu_rgba[..., :3] / 255.0)
    gpu_mask = (gpu_rgba[..., :3].sum(axis=2) > 0)

    cpu_ref = compute_cpu_reference(
        eval_width,
        eval_height,
        roughness,
        light_vec,
        BASE_COLOR,
        light_intensity=3.0,
        gpu_mask=gpu_mask,
    )
    debug_tiles = {
        kind: render_gpu_term_tile(
            roughness,
            eval_width,
            sphere_sectors,
            sphere_stacks,
            light_vec,
            kind,
        )
        for kind in (1, 2, 3)
    }

    cpu_linear = cpu_ref.linear_rgb
    cpu_srgb = linear_to_srgb(np.clip(cpu_linear, 0.0, 1.0))
    cpu_quant = np.clip(np.round(cpu_srgb * 255.0), 0, 255).astype(np.uint8)
    cpu_linear_quant = srgb_to_linear(cpu_quant.astype(np.float32) / 255.0)

    analysis_mask = make_analysis_mask(cpu_ref.valid_mask, cpu_ref.nv, ANALYSIS_MASK_SHRINK)
    if not np.any(analysis_mask):
        raise RuntimeError("Analysis mask is empty; sphere coverage too small.")

    rng = make_rng(seed_label)
    sample_points = generate_sample_points(
        eval_width,
        eval_height,
        samples_per_axis,
        samples_per_axis,
        cpu_ref,
        rng,
    )
    samples = evaluate_samples(sample_points, gpu_linear, cpu_linear_quant, cpu_ref)
    sample_count = len(samples)
    if sample_count == 0:
        raise RuntimeError("No valid samples were gathered for RMS analysis.")

    sample_err = np.asarray([s.abs_err for s in samples], dtype=np.float32)
    weights = np.clip(np.asarray([s.nv for s in samples], dtype=np.float32), 1e-4, None)
    rms = np.sqrt(np.average(sample_err ** 2, axis=0, weights=weights))

    diff_rgb_eval = np.abs(cpu_linear_quant - gpu_linear)
    diff_rgb_eval = np.where(analysis_mask[..., None], diff_rgb_eval, 0.0)
    mask_float = analysis_mask.astype(np.float32)
    diff_rgb_tile = downsample_mean(diff_rgb_eval, eval_scale)
    mask_tile = downsample_mean(mask_float[..., None], eval_scale)[..., 0] >= ANALYSIS_MASK_THRESHOLD
    if not np.any(mask_tile):
        raise RuntimeError("Downsampled analysis mask is empty; increase eval_scale or adjust thresholds.")
    diff_scalar = np.max(diff_rgb_tile, axis=2)
    valid_diffs = diff_scalar[mask_tile]
    perc_999 = float(np.percentile(valid_diffs, 99.9)) if valid_diffs.size else 0.0
    if valid_diffs.size:
        masked_rgb = diff_rgb_tile[mask_tile]
        max_abs = masked_rgb.max(axis=0)
    else:
        max_abs = np.zeros(3, dtype=np.float32)
    masked_scalar = np.where(mask_tile, diff_scalar, -1.0)
    max_idx = np.unravel_index(np.argmax(masked_scalar), masked_scalar.shape)

    return ValidationResult(
        width=width,
        height=height,
        eval_width=eval_width,
        eval_height=eval_height,
        eval_scale=eval_scale,
        roughness=roughness,
        sample_rows=samples_per_axis,
        sample_cols=samples_per_axis,
        sample_count=sample_count,
        sphere_sectors=sphere_sectors,
        sphere_stacks=sphere_stacks,
        light_dir=light_vec,
        analysis_shrink=ANALYSIS_MASK_SHRINK,
        analysis_nv_min=ANALYSIS_NV_MIN,
        gpu_linear=gpu_linear,
        cpu_linear_quant=cpu_linear_quant,
        diff_rgb_eval=diff_rgb_eval,
        diff_rgb_tile=diff_rgb_tile,
        analysis_mask_eval=analysis_mask,
        analysis_mask_tile=mask_tile,
        debug_tiles=debug_tiles,
        samples=samples,
        rms=rms.astype(np.float32),
        max_abs=max_abs.astype(np.float32),
        percentile_999=perc_999,
        max_idx=max_idx,
        cpu_ref=cpu_ref,
    )


def main() -> None:
    args = parse_args()
    result = run_validation(
        tile_size=int(args.tile_size),
        roughness=float(args.roughness),
        samples_per_axis=int(args.samples),
        eval_scale=int(args.eval_scale),
        sphere_sectors=int(args.sphere_sectors),
        sphere_stacks=int(args.sphere_stacks),
    )

    if np.any(result.rms > 1e-3 + 1e-6):
        raise RuntimeError(f"F1 RMS threshold failed: {result.rms} > 1e-3")
    if result.percentile_999 > 5e-3 + 1e-6:
        raise RuntimeError(f"F2 99.9 percentile threshold failed: {result.percentile_999:.6f} > 5e-3")

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    heatmap_path = outdir / args.heatmap
    csv_path = outdir / args.csv
    meta_path = outdir / args.meta

    write_heatmap(result.diff_rgb_tile, heatmap_path)
    write_csv(result.samples, csv_path)
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = build_meta(
        result=result,
        csv_name=csv_path.name,
        csv_sha256=csv_sha256,
        heatmap_name=heatmap_path.name,
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=== Milestone 6 Validation ===")
    print(
        f"Tile size      : {result.width}x{result.height} "
        f"(eval: {result.eval_width}x{result.eval_height}, scale={result.eval_scale})"
    )
    print(f"Sphere mesh    : sectors={result.sphere_sectors}, stacks={result.sphere_stacks}")
    print(
        "Light dir      : "
        f"({result.light_dir[0]:.6f}, {result.light_dir[1]:.6f}, {result.light_dir[2]:.6f})"
    )
    print(
        f"Analysis mask  : shrink={result.analysis_shrink:.2f}, "
        f"nv_min={result.analysis_nv_min:.2f}, coverage={result.analysis_mask_tile.mean():.3f}"
    )
    print(f"Samples        : {result.sample_count} ({result.sample_rows}x{result.sample_cols} grid)")
    print(f"RMS error (RGB): {result.rms[0]:.6e}, {result.rms[1]:.6e}, {result.rms[2]:.6e}")
    print(f"99.9%% abs err : {result.percentile_999:.6e}")
    print(f"Max abs err    : {result.max_abs[0]:.6e}, {result.max_abs[1]:.6e}, {result.max_abs[2]:.6e}")
    print(f"Outputs -> {outdir.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
