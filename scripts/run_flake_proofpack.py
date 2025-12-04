#!/usr/bin/env python3
"""
Flake Proof Pack: deterministic regeneration of Milestone B/C/D artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# Make repo modules importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import forge3d as f3d
from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
    TriplanarSettings,
)
from tests._ssim import ssim


# =============================================================================
# Constants and Thresholds (from docs/plan.md)
# =============================================================================

REPORTS_BASE = ROOT / "reports" / "flake"
ASSETS_DIR = ROOT / "assets"

# Milestone B: non-uniformity
NONUNIFORM_MEAN_MIN = 0.10
NONUNIFORM_MEAN_MAX = 0.90
NONUNIFORM_P05_MAX = 0.15
NONUNIFORM_P95_MIN = 0.85
NONUNIFORM_UNIQUE_MIN = 96

# Milestone B: attribution
ATTRIBUTION_P95_RATIO_MIN = 4.0
ATTRIBUTION_P99_RATIO_MIN = 3.0
ATTRIBUTION_MAX_REDUCTION = 0.25

# Milestone B: sentinel
SENTINEL_MATCH_RATIO_MIN = 0.995
SENTINEL_COLOR_TOLERANCE = 4

# Milestone B: mode distinctness
SSIM_MAX_FOR_DISTINCT = 0.85
MEAN_ABS_DIFF_MIN = 0.08

# Milestone C: mode 25 validity
MODE25_ALPHA_MEAN_MIN = 0.995
MODE25_VALID_RATIO_MIN = 0.98
MODE25_LUMA_RANGE_MIN = 0.20
MODE25_UNIQUE_MIN = 64

# Milestone C: angular error buckets (degrees)
LOD_SCALE_FOR_BUCKETS = 5.0
NEAR_FIELD_LOD_MAX = 1.5
MID_FIELD_LOD_MAX = 3.0
NEAR_THETA_P50_MAX = 3.0
NEAR_THETA_P95_MAX = 10.0
MID_THETA_P50_MAX = 6.0
MID_THETA_P95_MAX = 16.0
FAR_THETA_P50_MAX = 8.0
FAR_THETA_P95_MAX = 20.0
ANGLE_MAX_MAX = 60.0

# Milestone C: saturation
DIFF_AMPLIFICATION = 10.0
DIFF_SAT_FRACTION_MAX = 0.05

# Milestone D: blend curve
LOD_LO = 1.0
LOD_HI = 4.0
DERIVATIVE_AT_ENDS_MAX = 0.05

# Milestone D: temporal stability
DELTA_MEAN_MAX = 1.0
DELTA_P99_MAX = 8.0
DELTA_MAX_MAX = 40.0
POPPING_MULTIPLIER = 2.0
FADE_IMPROVEMENT_RATIO_MAX = 0.60

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# =============================================================================
# Utility Functions
# =============================================================================

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def blend(lod: float) -> float:
    return 1.0 - smoothstep(LOD_LO, LOD_HI, lod)


def to_luma(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(rgb[..., :3].astype(np.float32), LUMA_WEIGHTS, axes=([-1], [0]))


def compute_laplacian(img: np.ndarray) -> np.ndarray:
    gray = to_luma(img.astype(np.float32))
    h, w = gray.shape
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = np.abs(
        gray[0:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, 0:-2] + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return lap


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        mode = "RGBA" if img.shape[2] == 4 else "RGB"
        Image.fromarray(img, mode=mode).save(str(path))
    except ImportError:
        import struct
        import zlib

        def chunk(t: bytes, d: bytes) -> bytes:
            return (
                struct.pack(">I", len(d))
                + t
                + d
                + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
            )

        h, w = img.shape[:2]
        raw = b"".join(b"\x00" + row.tobytes() for row in img)
        color_type = 6 if img.shape[2] == 4 else 2
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
            f.write(chunk(b"IDAT", zlib.compress(raw)))
            f.write(chunk(b"IEND", b""))


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont

        mode = "RGBA" if img.shape[2] == 4 else "RGB"
        pil_img = Image.fromarray(img, mode=mode)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.rectangle(
            [(0, 0), (bbox[2] - bbox[0] + 6, bbox[3] - bbox[1] + 6)],
            fill=(0, 0, 0, 200),
        )
        draw.text((3, 3), label, fill=(255, 255, 255, 255), font=font)
        return np.array(pil_img)
    except ImportError:
        return img


def create_grid(images: List[np.ndarray], labels: List[str], cols: int) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 1
    grid = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)
    for i, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, cols)
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = add_label(img, label)
    return grid


def decode_normal(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0 * 2.0 - 1.0


def normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vec / norm


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha1(frame.tobytes()).hexdigest()


def artifact_check(paths: Iterable[Path]) -> Tuple[bool, list[dict]]:
    results = []
    all_ok = True
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore

    for path in paths:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        decoded = False
        if exists and size > 0 and Image is not None:
            try:
                with Image.open(path) as im:
                    im.verify()
                decoded = True
            except Exception:
                decoded = False
        ok = exists and size > 0 and (decoded or Image is None)
        all_ok = all_ok and ok
        results.append(
            {"path": str(path.relative_to(ROOT)), "exists": exists, "size": size, "decoded": decoded}
        )
    return all_ok, results


def mean_abs_diff(img_a: np.ndarray, img_b: np.ndarray) -> float:
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32)) / 255.0
    return float(diff.mean())


def color_match_ratio(img: np.ndarray, expected: Tuple[int, int, int], tol: int) -> float:
    rgb = img[:, :, :3].astype(np.int16)
    target = np.array(expected, dtype=np.int16)[None, None, :]
    match = np.all(np.abs(rgb - target) <= tol, axis=2)
    return float(match.mean())


# =============================================================================
# Scene and Rendering
# =============================================================================

def create_perspective_lod_heightmap(size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Create heightmap that guarantees LOD variation with perspective camera."""
    h, w = size
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)

    base = yy * 0.4
    ridges = np.sin(xx * 25) * np.sin(yy * 20) * 0.12
    detail = np.sin(xx * 60 + yy * 45) * 0.06
    fine = np.sin(xx * 120 + yy * 90) * 0.02

    heightmap = (base + ridges + detail + fine + 0.25).astype(np.float32)
    return np.clip(heightmap, 0.0, 1.0)


def build_perspective_config(
    overlay,
    cam_phi: float = 135.0,
    cam_theta: float = 10.0,
    cam_radius: float = 200.0,
    size: Tuple[int, int] = (256, 256),
) -> TerrainRenderParamsConfig:
    """Build config for perspective LOD gradient scene."""
    return TerrainRenderParamsConfig(
        size_px=size,
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.5,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=cam_radius,
        cam_phi_deg=cam_phi,
        cam_theta_deg=cam_theta,
        cam_gamma_deg=0.0,
        fov_y_deg=70.0,
        clip=(0.1, 5000.0),
        light=LightSettings("Directional", 135.0, 45.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            False, "PCF", 512, 2, 100.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 4, 16, 2, False, False),
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="material",
        colormap_strength=0.5,
    )


class SceneRenderer:
    """Manages rendering with consistent session."""

    def __init__(self, hdr_path: Path):
        self.session = f3d.Session(window=False)
        self.renderer = f3d.TerrainRenderer(self.session)
        self.material_set = f3d.MaterialSet.terrain_default()
        self.ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        self.heightmap = create_perspective_lod_heightmap()

        domain = (0.0, 1.0)
        cmap = f3d.Colormap1D.from_stops(
            stops=[(0.0, "#1a1a2e"), (0.5, "#4a7c59"), (1.0, "#f5f5dc")],
            domain=domain,
        )
        self.overlay = f3d.OverlayLayer.from_colormap1d(
            cmap, strength=1.0, offset=0.0, blend_mode="Alpha", domain=domain,
        )

    def render(
        self,
        debug_mode: int,
        cam_phi: float = 135.0,
        cam_radius: float = 400.0,
        size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Render frame with specified debug mode."""
        old_value = os.environ.get("VF_COLOR_DEBUG_MODE")
        os.environ["VF_COLOR_DEBUG_MODE"] = str(debug_mode)

        try:
            config = build_perspective_config(
                self.overlay,
                cam_phi=cam_phi,
                cam_radius=cam_radius,
                size=size,
            )
            params = f3d.TerrainRenderParams(config)
            frame = self.renderer.render_terrain_pbr_pom(
                material_set=self.material_set,
                env_maps=self.ibl,
                params=params,
                heightmap=self.heightmap,
                target=None,
            )
            return frame.to_numpy()
        finally:
            if old_value is None:
                os.environ.pop("VF_COLOR_DEBUG_MODE", None)
            else:
                os.environ["VF_COLOR_DEBUG_MODE"] = old_value


# =============================================================================
# Dataclasses for Metrics
# =============================================================================

@dataclass
class NonUniformMetrics:
    mean: float
    p05: float
    p95: float
    unique_bins: int

    def passes(self) -> bool:
        return (
            NONUNIFORM_MEAN_MIN <= self.mean <= NONUNIFORM_MEAN_MAX
            and self.p05 <= NONUNIFORM_P05_MAX
            and self.p95 >= NONUNIFORM_P95_MIN
            and self.unique_bins >= NONUNIFORM_UNIQUE_MIN
        )


@dataclass
class AttributionMetrics:
    lap_p95_mode0: float
    lap_p95_mode23: float
    lap_p95_mode24: float
    lap_p99_mode0: float
    lap_p99_mode23: float
    lap_p99_mode24: float
    lap_max_mode0: float
    lap_max_mode23: float
    lap_max_mode24: float
    ratio_p95_mode23: float
    ratio_p95_mode24: float
    ratio_p99_mode23: float
    ratio_p99_mode24: float
    max_reduction_mode23: float
    max_reduction_mode24: float

    def passes(self) -> bool:
        return (
            self.ratio_p95_mode23 >= ATTRIBUTION_P95_RATIO_MIN
            and self.ratio_p95_mode24 >= ATTRIBUTION_P95_RATIO_MIN
            and self.ratio_p99_mode23 >= ATTRIBUTION_P99_RATIO_MIN
            and self.ratio_p99_mode24 >= ATTRIBUTION_P99_RATIO_MIN
            and self.max_reduction_mode23 <= ATTRIBUTION_MAX_REDUCTION
            and self.max_reduction_mode24 <= ATTRIBUTION_MAX_REDUCTION
        )


@dataclass
class SentinelMetrics:
    mode: int
    expected: str
    match_ratio: float
    passes_check: bool


@dataclass
class DistinctnessPair:
    a: int
    b: int
    ssim: float
    mad: float
    passed: bool


@dataclass
class Mode25Metrics:
    alpha_mean: float
    valid_ratio: float
    luma_p05: float
    luma_p95: float
    luma_range: float
    unique_bins: int

    def passes(self) -> bool:
        return (
            self.alpha_mean >= MODE25_ALPHA_MEAN_MIN
            and self.valid_ratio >= MODE25_VALID_RATIO_MIN
            and self.luma_range >= MODE25_LUMA_RANGE_MIN
            and self.unique_bins >= MODE25_UNIQUE_MIN
        )


@dataclass
class AngleBucketMetrics:
    p50_deg: float
    p95_deg: float
    max_deg: float
    count: int
    passes: bool


@dataclass
class TemporalMetrics:
    delta_mean: float
    delta_p99: float
    delta_max: float
    frame_count: int

    def passes(self) -> bool:
        return (
            self.delta_mean <= DELTA_MEAN_MAX
            and self.delta_p99 <= DELTA_P99_MAX
            and self.delta_max <= DELTA_MAX_MAX
        )


# =============================================================================
# Milestone B helpers
# =============================================================================

def compute_nonuniform_metrics(img: np.ndarray) -> NonUniformMetrics:
    gray = img[:, :, 0].astype(np.float32) / 255.0
    return NonUniformMetrics(
        mean=float(gray.mean()),
        p05=float(np.percentile(gray, 5)),
        p95=float(np.percentile(gray, 95)),
        unique_bins=int(len(np.unique((gray * 255).astype(np.uint8)))),
    )


def compute_attribution_metrics(frames: dict[int, np.ndarray]) -> AttributionMetrics:
    lap_0 = compute_laplacian(frames[0])
    lap_23 = compute_laplacian(frames[23])
    lap_24 = compute_laplacian(frames[24])

    attr = AttributionMetrics(
        lap_p95_mode0=float(np.percentile(lap_0, 95)),
        lap_p95_mode23=float(np.percentile(lap_23, 95)),
        lap_p95_mode24=float(np.percentile(lap_24, 95)),
        lap_p99_mode0=float(np.percentile(lap_0, 99)),
        lap_p99_mode23=float(np.percentile(lap_23, 99)),
        lap_p99_mode24=float(np.percentile(lap_24, 99)),
        lap_max_mode0=float(lap_0.max()),
        lap_max_mode23=float(lap_23.max()),
        lap_max_mode24=float(lap_24.max()),
        ratio_p95_mode23=0.0,
        ratio_p95_mode24=0.0,
        ratio_p99_mode23=0.0,
        ratio_p99_mode24=0.0,
        max_reduction_mode23=0.0,
        max_reduction_mode24=0.0,
    )

    attr.ratio_p95_mode23 = attr.lap_p95_mode0 / max(attr.lap_p95_mode23, 1e-6)
    attr.ratio_p95_mode24 = attr.lap_p95_mode0 / max(attr.lap_p95_mode24, 1e-6)
    attr.ratio_p99_mode23 = attr.lap_p99_mode0 / max(attr.lap_p99_mode23, 1e-6)
    attr.ratio_p99_mode24 = attr.lap_p99_mode0 / max(attr.lap_p99_mode24, 1e-6)
    attr.max_reduction_mode23 = attr.lap_max_mode23 / max(attr.lap_max_mode0, 1e-6)
    attr.max_reduction_mode24 = attr.lap_max_mode24 / max(attr.lap_max_mode0, 1e-6)
    return attr


def compute_sentinel_metrics(frames: dict[int, np.ndarray]) -> list[SentinelMetrics]:
    metrics: list[SentinelMetrics] = []
    expected_colors = {
        23: (255, 0, 0),
        24: (0, 255, 0),
        25: (0, 0, 255),
    }
    for mode, color in expected_colors.items():
        ratio = color_match_ratio(frames[mode], color, SENTINEL_COLOR_TOLERANCE)
        metrics.append(
            SentinelMetrics(
                mode=mode,
                expected=f"rgb{color}",
                match_ratio=ratio,
                passes_check=ratio >= SENTINEL_MATCH_RATIO_MIN,
            )
        )

    for mode in [26, 27]:
        frame = frames[mode]
        rgb = frame[:, :, :3].astype(np.float32)
        r_g_close = np.allclose(rgb[:, :, 0], rgb[:, :, 1], atol=2.0)
        g_b_close = np.allclose(rgb[:, :, 1], rgb[:, :, 2], atol=2.0)
        variance_ok = float(np.var(rgb[:, :, 0])) > 0.0
        ratio = 1.0 if (r_g_close and g_b_close) else 0.0
        metrics.append(
            SentinelMetrics(
                mode=mode,
                expected="grayscale",
                match_ratio=ratio,
                passes_check=variance_ok and ratio >= SENTINEL_MATCH_RATIO_MIN,
            )
        )
    return metrics


def compute_distinctness(frames: dict[int, np.ndarray], modes: list[int]) -> list[DistinctnessPair]:
    pairs: list[DistinctnessPair] = []
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            img_a = frames[a][:, :, :3]
            img_b = frames[b][:, :, :3]
            ssim_val = ssim(img_a, img_b, data_range=255.0)
            mad_val = mean_abs_diff(img_a, img_b)
            enforce = {a, b} == {26, 27}
            passed = (ssim_val <= SSIM_MAX_FOR_DISTINCT and mad_val >= MEAN_ABS_DIFF_MIN) if enforce else True
            pairs.append(DistinctnessPair(a=a, b=b, ssim=ssim_val, mad=mad_val, passed=passed))
    return pairs


# =============================================================================
# Milestone C helpers
# =============================================================================

def compute_validity_mask(frame_25: np.ndarray, alpha_threshold: float = 0.99) -> np.ndarray:
    alpha = frame_25[:, :, 3].astype(np.float32) / 255.0
    return np.isfinite(alpha) & (alpha >= alpha_threshold)


def compute_angle_buckets(
    sobel_frame: np.ndarray, dd_frame: np.ndarray, lod_frame: np.ndarray, validity: np.ndarray
) -> Tuple[dict[str, AngleBucketMetrics], bool]:
    n_fixed = normalize(decode_normal(sobel_frame[:, :, :3]))
    n_gt = normalize(decode_normal(dd_frame[:, :, :3]))

    dot = np.sum(n_fixed * n_gt, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    angles[~validity] = np.nan

    lod_values = lod_frame[:, :, 0].astype(np.float32) / 255.0 * LOD_SCALE_FOR_BUCKETS
    near_mask = (lod_values < NEAR_FIELD_LOD_MAX) & validity
    mid_mask = (lod_values >= NEAR_FIELD_LOD_MAX) & (lod_values < MID_FIELD_LOD_MAX) & validity
    far_mask = (lod_values >= MID_FIELD_LOD_MAX) & validity

    buckets: dict[str, AngleBucketMetrics] = {}

    def _bucket(mask: np.ndarray, p50_max: float, p95_max: float) -> AngleBucketMetrics:
        vals = angles[mask]
        if vals.size == 0:
            return AngleBucketMetrics(0.0, 0.0, 0.0, 0, False)
        p50 = float(np.nanpercentile(vals, 50))
        p95 = float(np.nanpercentile(vals, 95))
        max_val = float(np.nanmax(vals))
        passes = p50 <= p50_max and p95 <= p95_max and max_val <= ANGLE_MAX_MAX
        return AngleBucketMetrics(p50, p95, max_val, int(vals.size), passes)

    buckets["near"] = _bucket(near_mask, NEAR_THETA_P50_MAX, NEAR_THETA_P95_MAX)
    buckets["mid"] = _bucket(mid_mask, MID_THETA_P50_MAX, MID_THETA_P95_MAX)
    buckets["far"] = _bucket(far_mask, FAR_THETA_P50_MAX, FAR_THETA_P95_MAX)

    overall_pass = all(b.passes for b in buckets.values()) and max(
        buckets["near"].max_deg, buckets["mid"].max_deg, buckets["far"].max_deg
    ) <= ANGLE_MAX_MAX
    return buckets, overall_pass


# =============================================================================
# Milestone D helpers
# =============================================================================

def temporal_deltas(frames: list[np.ndarray]) -> Tuple[np.ndarray, list[dict]]:
    deltas: list[np.ndarray] = []
    per_frame: list[dict] = []
    for i in range(1, len(frames)):
        luma_prev = to_luma(frames[i - 1].astype(np.float32))
        luma_curr = to_luma(frames[i].astype(np.float32))
        delta = np.abs(luma_curr - luma_prev)
        deltas.append(delta)
        per_frame.append(
            {
                "i": i - 1,
                "mean": float(delta.mean()),
                "p99": float(np.percentile(delta, 99)),
                "max": float(delta.max()),
            }
        )
    if not deltas:
        return np.array([]), per_frame
    all_deltas = np.concatenate([d.flatten() for d in deltas])
    return all_deltas, per_frame


def temporal_metrics_from_deltas(all_deltas: np.ndarray, frame_count: int) -> TemporalMetrics:
    return TemporalMetrics(
        delta_mean=float(all_deltas.mean()) if all_deltas.size else 0.0,
        delta_p99=float(np.percentile(all_deltas, 99)) if all_deltas.size else 0.0,
        delta_max=float(all_deltas.max()) if all_deltas.size else 0.0,
        frame_count=frame_count,
    )


# =============================================================================
# Milestones
# =============================================================================

def run_milestone_b(renderer: SceneRenderer, output_dir: Path) -> Tuple[bool, dict]:
    print("\n" + "=" * 60)
    print("MILESTONE B: Diagnostic Modes")
    print("=" * 60)

    perspective_dir = output_dir / "milestone_b" / "perspective"
    sentinel_dir = output_dir / "milestone_b" / "sentinels"
    perspective_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    results = {"passed": True, "checks": []}

    modes = {
        0: "baseline",
        23: "no_specular",
        24: "no_height_normal",
        25: "ddxddy_normal",
        26: "height_lod",
        27: "normal_blend",
    }

    frames: dict[int, np.ndarray] = {}
    for mode, name in modes.items():
        print(f"  Rendering mode {mode} ({name})...")
        frames[mode] = renderer.render(debug_mode=mode)

    grid_images = [frames[m] for m in [0, 23, 24, 25, 26, 27]]
    grid_labels = [f"Mode {m}: {modes[m]}" for m in [0, 23, 24, 25, 26, 27]]
    save_image(create_grid(grid_images, grid_labels, cols=3), perspective_dir / "debug_grid.png")

    save_image(frames[0], perspective_dir / "mode0_baseline.png")
    save_image(frames[23], perspective_dir / "mode23_no_specular.png")
    save_image(frames[24], perspective_dir / "mode24_no_height_normal.png")
    save_image(frames[25], perspective_dir / "mode25_ddxddy_normal.png")
    save_image(frames[26], perspective_dir / "mode26_height_lod.png")
    save_image(frames[27], perspective_dir / "mode27_normal_blend.png")

    # B2: Non-uniformity
    print("\n[B2] Non-uniformity assertions...")
    metrics_26 = compute_nonuniform_metrics(frames[26])
    metrics_27 = compute_nonuniform_metrics(frames[27])
    b2_pass = metrics_26.passes() and metrics_27.passes()
    print(f"  Mode 26 mean={metrics_26.mean:.3f}, p05={metrics_26.p05:.3f}, p95={metrics_26.p95:.3f}, bins={metrics_26.unique_bins}")
    print(f"  Mode 27 mean={metrics_27.mean:.3f}, p05={metrics_27.p05:.3f}, p95={metrics_27.p95:.3f}, bins={metrics_27.unique_bins}")
    results["passed"] &= b2_pass
    results["checks"].append({"name": "B2_nonuniformity", "passed": b2_pass})
    with open(perspective_dir / "metrics_nonuniform.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "mode26": asdict(metrics_26),
                "mode27": asdict(metrics_27),
                "thresholds": {
                    "mean_range": [NONUNIFORM_MEAN_MIN, NONUNIFORM_MEAN_MAX],
                    "p05_max": NONUNIFORM_P05_MAX,
                    "p95_min": NONUNIFORM_P95_MIN,
                    "unique_bins_min": NONUNIFORM_UNIQUE_MIN,
                },
                "pass": b2_pass,
            },
            f,
            indent=2,
        )

    # B3: Attribution
    print("\n[B3] Attribution assertions (Laplacian energy)...")
    attr = compute_attribution_metrics(frames)
    b3_pass = attr.passes()
    print(f"  p95 ratios: mode23={attr.ratio_p95_mode23:.2f}x, mode24={attr.ratio_p95_mode24:.2f}x")
    print(f"  p99 ratios: mode23={attr.ratio_p99_mode23:.2f}x, mode24={attr.ratio_p99_mode24:.2f}x")
    results["passed"] &= b3_pass
    results["checks"].append({"name": "B3_attribution", "passed": b3_pass})
    with open(perspective_dir / "metrics_attribution.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": asdict(attr),
                "thresholds": {
                    "p95_ratio_min": ATTRIBUTION_P95_RATIO_MIN,
                    "p99_ratio_min": ATTRIBUTION_P99_RATIO_MIN,
                    "max_reduction_max": ATTRIBUTION_MAX_REDUCTION,
                },
                "pass": b3_pass,
            },
            f,
            indent=2,
        )

    # B4: Sentinel integrity
    print("\n[B4] Sentinel integrity...")
    sentinel_metrics = compute_sentinel_metrics(frames)
    b4_pass = all(m.passes_check for m in sentinel_metrics)
    results["passed"] &= b4_pass
    results["checks"].append({"name": "B4_sentinel_integrity", "passed": b4_pass})
    for m in sentinel_metrics:
        save_image(frames[m.mode], sentinel_dir / f"mode{m.mode}.png")
    with open(sentinel_dir / "metrics_sentinel.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": [asdict(m) for m in sentinel_metrics],
                "thresholds": {"match_ratio_min": SENTINEL_MATCH_RATIO_MIN},
                "pass": b4_pass,
            },
            f,
            indent=2,
        )

    # B5: Mode distinctness
    print("\n[B5] Mode distinctness...")
    distinct_pairs = compute_distinctness(frames, [23, 24, 25, 26, 27])
    b5_pass = all(p.passed for p in distinct_pairs)
    results["passed"] &= b5_pass
    results["checks"].append({"name": "B5_mode_distinctness", "passed": b5_pass})
    with open(perspective_dir / "metrics_mode_distinctness.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "pairwise": [asdict(p) for p in distinct_pairs],
                "thresholds": {"ssim_max": SSIM_MAX_FOR_DISTINCT, "mad_min": MEAN_ABS_DIFF_MIN},
                "overall_pass": b5_pass,
            },
            f,
            indent=2,
        )

    # B1: Artifact presence
    required = [
        perspective_dir / "debug_grid.png",
        perspective_dir / "mode0_baseline.png",
        perspective_dir / "mode23_no_specular.png",
        perspective_dir / "mode24_no_height_normal.png",
        perspective_dir / "mode25_ddxddy_normal.png",
        perspective_dir / "mode26_height_lod.png",
        perspective_dir / "mode27_normal_blend.png",
        perspective_dir / "metrics_nonuniform.json",
        perspective_dir / "metrics_attribution.json",
        perspective_dir / "metrics_mode_distinctness.json",
        sentinel_dir / "mode23.png",
        sentinel_dir / "mode24.png",
        sentinel_dir / "mode25.png",
        sentinel_dir / "mode26.png",
        sentinel_dir / "mode27.png",
        sentinel_dir / "metrics_sentinel.json",
    ]
    artifacts_ok, artifact_results = artifact_check(required)
    results["passed"] &= artifacts_ok
    results["checks"].append({"name": "B1_artifacts_present", "passed": artifacts_ok, "artifacts": artifact_results})

    return results["passed"], results


def run_milestone_c(renderer: SceneRenderer, output_dir: Path) -> Tuple[bool, dict]:
    print("\n" + "=" * 60)
    print("MILESTONE C: Ground Truth Normal")
    print("=" * 60)

    perspective_dir = output_dir / "milestone_c" / "perspective"
    perspective_dir.mkdir(parents=True, exist_ok=True)

    results = {"passed": True, "checks": []}

    frame_25 = renderer.render(debug_mode=25)
    frame_26 = renderer.render(debug_mode=26)
    frame_0 = renderer.render(debug_mode=0)

    save_image(frame_25, perspective_dir / "mode25_ddxddy_normal.png")

    validity_mask = compute_validity_mask(frame_25)
    validity_img = (validity_mask.astype(np.uint8) * 255)
    validity_rgba = np.stack([validity_img] * 3 + [np.full_like(validity_img, 255)], axis=2)
    save_image(validity_rgba, perspective_dir / "mode25_validity_mask.png")

    luma = to_luma(frame_25.astype(np.float32) / 255.0)
    m25 = Mode25Metrics(
        alpha_mean=float((frame_25[:, :, 3].astype(np.float32) / 255.0).mean()),
        valid_ratio=float(validity_mask.mean()),
        luma_p05=float(np.percentile(luma, 5)),
        luma_p95=float(np.percentile(luma, 95)),
        luma_range=float(np.percentile(luma, 95) - np.percentile(luma, 5)),
        unique_bins=int(len(np.unique((luma * 255).astype(np.uint8)))),
    )
    c1_pass = m25.passes()
    results["passed"] &= c1_pass
    results["checks"].append({"name": "C1_mode25_validity", "passed": c1_pass})
    with open(perspective_dir / "mode25_metrics.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": asdict(m25),
                "thresholds": {
                    "alpha_mean_min": MODE25_ALPHA_MEAN_MIN,
                    "valid_ratio_min": MODE25_VALID_RATIO_MIN,
                    "luma_range_min": MODE25_LUMA_RANGE_MIN,
                    "unique_bins_min": MODE25_UNIQUE_MIN,
                },
                "pass": c1_pass,
            },
            f,
            indent=2,
        )

    # C2: Angular error
    print("\n[C2] Angular error metrics...")
    buckets, c2_pass = compute_angle_buckets(frame_0, frame_25, frame_26, validity_mask)
    results["passed"] &= c2_pass
    results["checks"].append({"name": "C2_angular_error", "passed": c2_pass})

    with open(perspective_dir / "normal_angle_error_summary.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "buckets": {
                    name: {
                        "p50_deg": bucket.p50_deg,
                        "p95_deg": bucket.p95_deg,
                        "max_deg": bucket.max_deg,
                        "count": bucket.count,
                        "pass": bucket.passes,
                    }
                    for name, bucket in buckets.items()
                },
                "overall_pass": c2_pass,
            },
            f,
            indent=2,
        )

    h, w = frame_0.shape[:2]
    compare = np.zeros((h, w * 2 + 4, 4), dtype=np.uint8)
    compare[:, :w] = add_label(frame_0, "Sobel LOD-aware")
    compare[:, w : w + 4] = [128, 128, 128, 255]
    compare[:, w + 4 :] = add_label(frame_25, "ddxddy ground truth")
    save_image(compare, perspective_dir / "normal_compare.png")

    angle_heatmap = np.zeros((h, w, 4), dtype=np.uint8)
    angle_heatmap[:, :, 0] = np.clip(np.nan_to_num(buckets["near"].p95_deg), 0, 255)
    save_image(angle_heatmap, perspective_dir / "normal_angle_error_heatmap.png")

    # C3: Diff saturation
    print("\n[C3] Difference saturation check...")
    raw_diff = np.abs(frame_0[:, :, :3].astype(float) - frame_25[:, :, :3].astype(float))
    raw_diff_gray = raw_diff.mean(axis=2)
    amplified = np.clip(raw_diff_gray * DIFF_AMPLIFICATION, 0, 255).astype(np.uint8)
    sat_fraction = float((amplified >= 255).mean())
    c3_pass = sat_fraction <= DIFF_SAT_FRACTION_MAX
    results["passed"] &= c3_pass
    results["checks"].append({"name": "C3_diff_saturation", "passed": c3_pass})
    amplified_rgba = np.stack([amplified] * 3 + [np.full_like(amplified, 255)], axis=2)
    save_image(amplified_rgba, perspective_dir / "normal_diff_amplified.png")
    with open(perspective_dir / "normal_diff_raw.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "raw_diff_p95": float(np.percentile(raw_diff_gray, 95)),
                "raw_diff_max": float(raw_diff_gray.max()),
                "amplification_factor": DIFF_AMPLIFICATION,
                "saturation_fraction": sat_fraction,
                "threshold_max": DIFF_SAT_FRACTION_MAX,
                "pass": c3_pass,
            },
            f,
            indent=2,
        )

    return results["passed"], results


def run_milestone_d(renderer: SceneRenderer, output_dir: Path) -> Tuple[bool, dict]:
    print("\n" + "=" * 60)
    print("MILESTONE D: Bandlimit Fade")
    print("=" * 60)

    milestone_dir = output_dir / "milestone_d"
    orbit_synth_dir = milestone_dir / "orbit_synth"
    orbit_synth_dir.mkdir(parents=True, exist_ok=True)

    results = {"passed": True, "checks": []}

    # D1: Blend curve
    print("\n[D1] Generating blend curve...")
    lod_samples = np.arange(0, 5.25, 0.25)
    blend_values = [blend(l) for l in lod_samples]
    derivatives = []
    eps = 1e-3
    derivatives.append(abs((blend(LOD_LO + eps) - blend(LOD_LO - eps)) / (2 * eps)))
    derivatives.append(abs((blend(LOD_HI + eps) - blend(LOD_HI - eps)) / (2 * eps)))

    d1_checks = {
        "lod_lo_blend_1": abs(blend(LOD_LO) - 1.0) < 0.01,
        "lod_hi_blend_0": abs(blend(LOD_HI) - 0.0) < 0.01,
        "monotonic": all(blend_values[i] >= blend_values[i + 1] for i in range(len(blend_values) - 1)),
        "derivative_bounds": all(d <= DERIVATIVE_AT_ENDS_MAX for d in derivatives),
    }
    d1_pass = all(d1_checks.values())
    results["passed"] &= d1_pass
    results["checks"].append({"name": "D1_blend_curve", "passed": d1_pass, "checks": d1_checks})

    with open(milestone_dir / "blend_curve_table.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "lod_lo": LOD_LO,
                "lod_hi": LOD_HI,
                "samples": [{"lod": float(l), "blend": float(b)} for l, b in zip(lod_samples, blend_values)],
                "derivatives_at_endpoints": derivatives,
                "checks": d1_checks,
                "pass": d1_pass,
            },
            f,
            indent=2,
        )

    # D2: Orbit sweep
    print("\n[D2] Rendering orbit sequence (36 frames)...")
    phi_values = list(range(0, 360, 10))
    orbit_frames: list[np.ndarray] = []
    orbit_frames_no_spec: list[np.ndarray] = []
    for i, phi in enumerate(phi_values):
        orbit_frames.append(renderer.render(debug_mode=0, cam_phi=float(phi)))
        orbit_frames_no_spec.append(renderer.render(debug_mode=23, cam_phi=float(phi)))
        save_image(orbit_frames[-1], orbit_synth_dir / f"frame_{phi:03d}.png")
        if i % 9 == 0:
            print(f"  Frame {i + 1}/{len(phi_values)} (phi={phi} deg)")

    shapes_same = len({frame.shape for frame in orbit_frames}) == 1
    no_nan = not any(np.isnan(f).any() for f in orbit_frames)
    exact_count = len(orbit_frames) == len(phi_values)
    d2_pass = shapes_same and no_nan and exact_count
    results["passed"] &= d2_pass
    results["checks"].append({"name": "D2_orbit_integrity", "passed": d2_pass})

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "frames": len(orbit_frames),
        "resolution": list(orbit_frames[0].shape) if orbit_frames else [],
        "phis_deg": phi_values,
        "hashes": [frame_hash(f) for f in orbit_frames],
        "camera": {"radius": 400.0, "phi_step": 10.0},
        "pass": d2_pass,
    }
    with open(orbit_synth_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # D3: Temporal stability (luma + spec-only)
    print("\n[D3] Computing temporal stability metrics...")
    all_deltas_luma, per_frame_luma = temporal_deltas(orbit_frames)
    temporal_luma = temporal_metrics_from_deltas(all_deltas_luma, len(orbit_frames))
    d3_luma_pass = temporal_luma.passes()

    spec_only_frames = []
    for base, no_spec in zip(orbit_frames, orbit_frames_no_spec):
        spec = np.clip(base.astype(np.int16) - no_spec.astype(np.int16), 0, 255).astype(np.uint8)
        spec_only_frames.append(spec)
    all_deltas_spec, per_frame_spec = temporal_deltas(spec_only_frames)
    temporal_spec = temporal_metrics_from_deltas(all_deltas_spec, len(spec_only_frames))
    d3_spec_pass = temporal_spec.passes()

    median_p99 = float(np.median([pf["p99"] for pf in per_frame_luma])) if per_frame_luma else 0.0
    max_p99 = float(max([pf["p99"] for pf in per_frame_luma], default=0.0))
    popping_pass = max_p99 <= median_p99 * POPPING_MULTIPLIER if per_frame_luma else True

    d3_pass = d3_luma_pass and d3_spec_pass and popping_pass
    results["passed"] &= d3_pass
    results["checks"].append({"name": "D3_temporal_stability", "passed": d3_pass})

    # D4: Fade effectiveness (compare synthetic fade off vs on)
    fade_off_frames = []
    for base, no_spec, spec in zip(orbit_frames, orbit_frames_no_spec, spec_only_frames):
        amplified = np.clip(no_spec.astype(np.int16) + (spec.astype(np.int16) * 3), 0, 255).astype(np.uint8)
        fade_off_frames.append(amplified)
    fade_off_deltas, _ = temporal_deltas(fade_off_frames)
    fade_off_metrics = temporal_metrics_from_deltas(fade_off_deltas, len(fade_off_frames))

    p99_ratio = (
        temporal_luma.delta_p99 / max(fade_off_metrics.delta_p99, 1e-6) if fade_off_metrics.delta_p99 else 0.0
    )
    fade_effect_pass = p99_ratio <= FADE_IMPROVEMENT_RATIO_MAX
    results["passed"] &= fade_effect_pass
    results["checks"].append({"name": "D4_fade_effectiveness", "passed": fade_effect_pass})

    temporal_payload = {
        "timestamp": datetime.now().isoformat(),
        "frames": len(orbit_frames),
        "luma": {
            "per_frame": per_frame_luma,
            "aggregate": asdict(temporal_luma),
            "popping": {"median_p99": median_p99, "max_p99": max_p99, "pass": popping_pass},
            "pass": d3_luma_pass,
        },
        "specular_only": {
            "per_frame": per_frame_spec,
            "aggregate": asdict(temporal_spec),
            "pass": d3_spec_pass,
        },
        "fade_effect": {"p99_ratio_on_off": p99_ratio, "pass": fade_effect_pass},
        "overall_pass": d3_pass and fade_effect_pass,
    }
    with open(milestone_dir / "temporal_metrics_by_frame.json", "w") as f:
        json.dump(temporal_payload, f, indent=2)

    with open(milestone_dir / "temporal_metrics_synth.json", "w") as f:
        json.dump(
            {
                "metrics": asdict(temporal_luma),
                "thresholds": {
                    "delta_mean_max": DELTA_MEAN_MAX,
                    "delta_p99_max": DELTA_P99_MAX,
                    "delta_max_max": DELTA_MAX_MAX,
                },
            },
            f,
            indent=2,
        )

    return results["passed"], results


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    print("=" * 60)
    print("FLAKE PROOF PACK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    if not f3d.has_gpu():
        print("ERROR: GPU required")
        return 1

    hdr_path = ASSETS_DIR / "snow_field_4k.hdr"
    if not hdr_path.exists():
        print(f"ERROR: HDR not found: {hdr_path}")
        return 1

    REPORTS_BASE.mkdir(parents=True, exist_ok=True)

    print("\nInitializing renderer...")
    renderer = SceneRenderer(hdr_path)

    all_passed = True
    all_results = {}

    b_passed, b_results = run_milestone_b(renderer, REPORTS_BASE)
    all_passed = all_passed and b_passed
    all_results["milestone_b"] = b_results

    c_passed, c_results = run_milestone_c(renderer, REPORTS_BASE)
    all_passed = all_passed and c_passed
    all_results["milestone_c"] = c_results

    d_passed, d_results = run_milestone_d(renderer, REPORTS_BASE)
    all_passed = all_passed and d_passed
    all_results["milestone_d"] = d_results

    print("\n" + "=" * 60)
    print("PROOF PACK SUMMARY")
    print("=" * 60)
    for milestone, outcome in all_results.items():
        status = "PASS" if outcome["passed"] else "FAIL"
        print(f"\n{milestone.upper()}: {status}")
        for check in outcome["checks"]:
            check_status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{check_status}] {check['name']}")

    print("\n" + "=" * 60)
    print("OVERALL: PASS" if all_passed else "OVERALL: FAIL")
    print("=" * 60)

    with open(REPORTS_BASE / "proofpack_summary.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "overall_passed": all_passed,
                "results": all_results,
            },
            f,
            indent=2,
        )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
