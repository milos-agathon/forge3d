#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 - Dielectric Cook-Torrance GGX gallery generator.

Produces the milestone deliverables:
  - m1_brdf_gallery_ggx.png
  - m1_debug_ndf.png
  - m1_debug_g.png
  - m1_debug_spec_radiance.png
  - m1_meta.json

The renderer evaluates a gray dielectric (F0 = 0.04, baseColor = 0.5)
on a UV sphere lit by a single directional light (Li = 3.0) and saves
debug mosaics plus meta data/metrics for audit.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
from typing import Iterable, List, Sequence, Tuple

# Ensure the local forge3d Python package is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

try:
    import forge3d as f3d
except ImportError as exc:  # pragma: no cover - handled at runtime
    print("Error: forge3d module not available. Build the native extension via 'maturin develop --release'.")
    raise

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PIL = False
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


ROUGHNESS_VALUES: Tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
BASE_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)
F0: float = 0.04
LIGHT_INTENSITY: float = 3.0
LIGHT_DIR = np.array([0.5, 0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
CAMERA_POS = (0.0, 0.0, 2.0)
CAMERA_VIEW_DIR = (0.0, 0.0, 1.0)
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SPEC_THRESHOLD_LINEAR = 1e-4
N_DOT_H_CENTER = 1.0
N_DOT_H_EDGE = 1e-3  # approximate tail sample
LUM_RANGE = (75.0, 140.0)
TILE_WIDTH = 512
TILE_HEIGHT = 512
NUM_TILES = len(ROUGHNESS_VALUES)
GUTTER_PX = 16


@dataclass
class TileResult:
    roughness: float
    alpha: float
    caption: str
    full_tile: np.ndarray
    ndf_tile: np.ndarray
    g_tile: np.ndarray
    spec_tile: np.ndarray
    mean_srgb_luminance: float
    spec_threshold_ratio: float
    spec_threshold_value: float
    spec_max_linear: float
    ndf_center_raw: float
    ndf_edge_raw: float
    ndf_edge_theory: float
    d_peak_theory: float
    d_peak_delta: float
    g_center: float
    gpu_constants_hash: str


def ensure_rgba_u8(arr: np.ndarray) -> np.ndarray:
    """Validate that an array is RGBA uint8 and return a defensive copy."""
    data = np.asarray(arr)
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {data.shape}")
    if data.dtype != np.uint8:
        data = data.astype(np.uint8, copy=False)
    return data.copy()


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB float array in [0,1] to linear space."""
    a = 0.055
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + a) / (1.0 + a)) ** 2.4,
    )


def srgb_u8_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB buffer to linear floats in [0,1]."""
    srgb = rgb_u8.astype(np.float32) / 255.0
    return srgb_to_linear(srgb)


def ggx_ndf_value(alpha: float, n_dot_h: float) -> float:
    """Evaluate GGX normal distribution for given alpha and N·H."""
    a = max(alpha, 1e-4)
    a2 = a * a
    nh = max(n_dot_h, 0.0)
    denom = nh * nh * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom * denom)


def scan_linear_safety(tile_label: str, roughness: float, linear_rgb: np.ndarray) -> None:
    """Log safety stats: check finiteness and large values before encode."""
    flat = linear_rgb.reshape(-1, linear_rgb.shape[-1])
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    gt32_count = int((flat > 32.0).sum())
    max_val = float(np.nanmax(flat)) if flat.size else 0.0
    ok = nan_count == 0 and inf_count == 0 and gt32_count == 0
    print(
        f"[safety] tile={tile_label} r={roughness:.2f} ok={ok} "
        f"max_linear={max_val:.6f} nan={nan_count} inf={inf_count} gt32={gt32_count}"
    )


def compute_mean_srgb_luminance(rgb_u8: np.ndarray) -> float:
    """Compute mean luminance in sRGB/uint8 space."""
    rgb = rgb_u8.astype(np.float32)
    luminance = rgb @ LUMINANCE_WEIGHTS
    return float(luminance.mean())


def add_caption(tile: np.ndarray, text: str) -> np.ndarray:
    """Stamp a caption onto the tile (no-op if PIL is unavailable)."""
    if not HAS_PIL:
        return tile

    img = Image.fromarray(tile, mode="RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:  # pragma: no cover - fallback fonts
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFCompactDisplay.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max((img.width - text_w) // 2, 8)
    y = 8
    pad = 6
    draw.rectangle(
        [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img)


def _rgba_to_rgb(tile: np.ndarray) -> np.ndarray:
    rgba = ensure_rgba_u8(tile)
    if np.all(rgba[..., 3] == 255):
        return rgba[..., :3]
    pil_img = Image.fromarray(rgba, mode="RGBA")
    return np.asarray(pil_img.convert("RGB"))


def _validate_gutters(atlas: np.ndarray, tile_w: int, gutter: int, label: str) -> None:
    tile_count = len(ROUGHNESS_VALUES)
    for idx in range(1, tile_count):
        gx = idx * tile_w + (idx - 1) * gutter
        gutter_strip = atlas[:, gx:gx + gutter, :]
        if gutter_strip.shape[1] != gutter:
            raise ValueError(f"[{label}] Missing gutter {idx} width={gutter_strip.shape[1]}")
        if np.any(gutter_strip != 0):
            raise ValueError(f"[{label}] Non-black pixels in gutter {idx}")
        leading = gutter_strip[:, :min(4, gutter), :]
        if leading.size and int(leading.sum()) != 0:
            raise ValueError(f"[{label}] Brightness leak in gutter {idx}")
    for idx in range(tile_count - 1):
        x0 = (idx + 1) * tile_w + idx * gutter
        gutter_strip = atlas[:, x0:x0 + gutter, :]
        if np.any(gutter_strip != 0):
            raise ValueError(f"[{label}] Tile {idx} spills into gutter at x={x0}")


def make_row(tiles: Sequence[np.ndarray], gap: int) -> np.ndarray:
    """Lay tiles out in a single row separated by precise 16px black gutters."""
    tiles_rgb = [_rgba_to_rgb(tile) for tile in tiles]
    if not tiles_rgb:
        raise ValueError("No tiles provided for mosaic generation")
    tile_h, tile_w, _ = tiles_rgb[0].shape
    if tile_w != TILE_WIDTH or tile_h != TILE_HEIGHT:
        raise ValueError(f"Tiles must be {TILE_WIDTH}x{TILE_HEIGHT}, got {tile_w}x{tile_h}")
    total_w = len(tiles_rgb) * tile_w + (len(tiles_rgb) - 1) * GUTTER_PX
    atlas = np.zeros((tile_h, total_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles_rgb):
        start = i * (tile_w + GUTTER_PX)
        end = start + tile_w
        atlas[:, start:end, :] = tile
    _validate_gutters(atlas, tile_w, GUTTER_PX, "gallery")
    return atlas


def save_png(path: Path, rgba: np.ndarray) -> None:
    """Persist RGBA PNG using Pillow if available, otherwise numpy fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected image with 3 or 4 channels, got {arr.shape}")
    mode = "RGB" if arr.shape[2] == 3 else "RGBA"
    if HAS_PIL:
        Image.fromarray(arr.astype(np.uint8), mode=mode).save(path)
    else:  # pragma: no cover - fallback without PIL
        from imageio import imwrite  # type: ignore

        imwrite(path, arr)


def sample_center_edge(tile: np.ndarray) -> Tuple[float, float]:
    """Sample center and average rim grayscale values (expects RGBA)."""
    h, w, _ = tile.shape
    center = float(tile[h // 2, w // 2, 0]) / 255.0
    edge_samples = [
        tile[h // 2, 0, 0],
        tile[h // 2, w - 1, 0],
        tile[0, w // 2, 0],
        tile[h - 1, w // 2, 0],
    ]
    edge = float(np.mean(edge_samples) / 255.0)
    return center, edge


def compute_gpu_constants_hash(roughness: float) -> str:
    """Hash the GPU constant payload used for a tile."""
    alpha = roughness * roughness
    payload_values = [
        roughness,
        alpha,
        BASE_COLOR[0],
        BASE_COLOR[1],
        BASE_COLOR[2],
        F0,
        F0,
        F0,
        LIGHT_INTENSITY,
        float(LIGHT_DIR[0]),
        float(LIGHT_DIR[1]),
        float(LIGHT_DIR[2]),
        CAMERA_POS[0],
        CAMERA_POS[1],
        CAMERA_POS[2],
        1.0,  # exposure
    ]
    packed = struct.pack("<" + "f" * len(payload_values), *payload_values)
    return hashlib.sha256(packed).hexdigest()


def render_tiles(width: int, height: int, gap: int) -> Tuple[List[TileResult], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render all roughness tiles and build labeled mosaics."""
    if gap != GUTTER_PX:
        raise ValueError(f"Gutter must be {GUTTER_PX} pixels; got {gap}")
    if width != TILE_WIDTH or height != TILE_HEIGHT:
        raise ValueError(f"Tile dimensions must be {TILE_WIDTH}x{TILE_HEIGHT}; got {width}x{height}")
    full_tiles: List[np.ndarray] = []
    ndf_tiles: List[np.ndarray] = []
    g_tiles: List[np.ndarray] = []
    spec_tiles: List[np.ndarray] = []
    results: List[TileResult] = []
    threshold_value: float = SPEC_THRESHOLD_LINEAR

    for roughness in ROUGHNESS_VALUES:
        base_kwargs = dict(
            model="ggx",
            roughness=roughness,
            width=width,
            height=height,
            ndf_only=False,
            g_only=False,
            dfg_only=False,
            spec_only=False,
            roughness_visualize=False,
            exposure=1.0,
            light_intensity=LIGHT_INTENSITY,
            base_color=BASE_COLOR,
            clearcoat=0.0,
            clearcoat_roughness=0.0,
            sheen=0.0,
            sheen_tint=0.0,
            specular_tint=0.0,
            debug_dot_products=False,
            debug_lambert_only=False,
            debug_d=False,
            debug_spec_no_nl=False,
            debug_energy=False,
            debug_angle_sweep=False,
            debug_angle_component=2,
            debug_no_srgb=False,
            output_mode=1,
            metallic_override=0.0,
        )

        def render_variant(**overrides: object) -> np.ndarray:
            params = dict(base_kwargs)
            params.update(overrides)
            return ensure_rgba_u8(f3d.render_brdf_tile(**params))

        full = render_variant()
        ndf = render_variant(ndf_only=True)
        g = render_variant(g_only=True)
        spec = render_variant(spec_only=True)

        caption = f"GGX  r={roughness:.2f}  α={roughness * roughness:.4f}"
        full_linear = srgb_u8_to_linear(full[..., :3])
        scan_linear_safety("full", roughness, full_linear)
        full_mean = compute_mean_srgb_luminance(full[..., :3])

        spec_linear = srgb_u8_to_linear(spec[..., :3])
        scan_linear_safety("spec", roughness, spec_linear)
        spec_luminance = spec_linear @ LUMINANCE_WEIGHTS
        above = spec_luminance > threshold_value
        spec_ratio = float(above.mean()) if spec_luminance.size else 0.0
        spec_max = float(spec_luminance.max()) if spec_luminance.size else 0.0
        alpha_val = max(roughness * roughness, 1e-4)
        ndf_center_raw = ggx_ndf_value(alpha_val, N_DOT_H_CENTER)
        ndf_edge_raw = ggx_ndf_value(alpha_val, N_DOT_H_EDGE)
        d_peak_theory = 1.0 / (math.pi * (alpha_val * alpha_val))
        d_peak_delta = abs(ndf_center_raw - d_peak_theory)
        ndf_edge_theory = (alpha_val * alpha_val) / math.pi
        g_center, _ = sample_center_edge(g)
        constants_hash = compute_gpu_constants_hash(roughness)

        results.append(
            TileResult(
                roughness=roughness,
                alpha=roughness * roughness,
                caption=caption,
                full_tile=full,
                ndf_tile=ndf,
                g_tile=g,
                spec_tile=spec,
                mean_srgb_luminance=full_mean,
                spec_threshold_ratio=spec_ratio,
                spec_threshold_value=threshold_value,
                spec_max_linear=spec_max,
                ndf_center_raw=ndf_center_raw,
                ndf_edge_raw=ndf_edge_raw,
                ndf_edge_theory=ndf_edge_theory,
                d_peak_theory=d_peak_theory,
                d_peak_delta=d_peak_delta,
                g_center=g_center,
                gpu_constants_hash=constants_hash,
            )
        )

        full_tiles.append(add_caption(full, caption))
        ndf_tiles.append(add_caption(ndf, caption))
        g_tiles.append(add_caption(g, caption))
        spec_tiles.append(add_caption(spec, caption))

    gallery_full = make_row(full_tiles, gap)
    gallery_ndf = make_row(ndf_tiles, gap)
    gallery_g = make_row(g_tiles, gap)
    gallery_spec = make_row(spec_tiles, gap)
    return results, gallery_full, gallery_ndf, gallery_g, gallery_spec


def build_meta(results: Sequence[TileResult]) -> dict:
    """Assemble metadata payload for m1_meta.json."""
    luminance_ok = all(LUM_RANGE[0] <= t.mean_srgb_luminance <= LUM_RANGE[1] for t in results)
    spec_ratios = [t.spec_threshold_ratio for t in results]
    spec_monotonic = all(spec_ratios[i] <= spec_ratios[i + 1] + 1e-4 for i in range(len(spec_ratios) - 1))
    ndf_centers = [t.ndf_center_raw for t in results]
    ndf_peaked = all(ndf_centers[i] >= ndf_centers[i + 1] - 1e-4 for i in range(len(ndf_centers) - 1))

    tiles_meta = []
    for t in results:
        tiles_meta.append(
            {
                "roughness": round(t.roughness, 4),
                "alpha": round(t.alpha, 6),
                "caption": t.caption,
                "mean_srgb_luminance": round(t.mean_srgb_luminance, 3),
                "spec_threshold_ratio": round(t.spec_threshold_ratio, 6),
                "spec_threshold_linear": round(t.spec_threshold_value, 6),
                "spec_max_linear": round(t.spec_max_linear, 6),
                "ndf_center_raw": round(t.ndf_center_raw, 6),
                "ndf_edge_raw": round(t.ndf_edge_raw, 6),
                "ndf_edge_theory": round(t.ndf_edge_theory, 6),
                "modeled_values": {
                    "d_peak_theory": round(t.d_peak_theory, 6),
                    "d_peak_delta": round(t.d_peak_delta, 6),
                },
                "g_center": round(t.g_center, 6),
                "gpu_constants_hash": t.gpu_constants_hash,
            }
        )

    return {
        "description": "Milestone 1 GGX dielectric gallery",
        "tile_size": {
            "width": int(results[0].full_tile.shape[1]) if results else 0,
            "height": int(results[0].full_tile.shape[0]) if results else 0,
        },
        "roughness_values": list(ROUGHNESS_VALUES),
        "base_color": BASE_COLOR,
        "f0": F0,
        "light": {
            "direction": [float(d) for d in LIGHT_DIR],
            "radiance": float(LIGHT_INTENSITY),
        },
        "camera": {
            "position": CAMERA_POS,
            "view_dir": CAMERA_VIEW_DIR,
        },
        "tiles": tiles_meta,
        "spec_threshold_linear": round(results[0].spec_threshold_value if results else SPEC_THRESHOLD_LINEAR, 6),
        "spec_area_image": "m1_debug_spec_radiance.png",
        "acceptance": {
            "mean_luminance_in_range": luminance_ok,
            "spec_threshold_monotonic": spec_monotonic,
            "ndf_center_monotonic": ndf_peaked,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Milestone 1 GGX gallery and debug mosaics.")
    parser.add_argument("--tile-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(512, 512), help="Tile resolution (default: 512 512)")
    parser.add_argument("--gap", type=int, default=GUTTER_PX, help=f"Pixel gap between tiles (fixed at {GUTTER_PX})")
    parser.add_argument("--outdir", type=Path, default=Path("reports"), help="Output directory for artifacts (default: reports)")
    parser.add_argument("--meta", type=Path, default=None, help="Optional path for metadata JSON (default: OUTDIR/m1_meta.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height = args.tile_size
    if width <= 0 or height <= 0:
        raise ValueError("Tile dimensions must be positive")

    results, gallery_full, gallery_ndf, gallery_g, gallery_spec = render_tiles(width, height, args.gap)

    args.outdir.mkdir(parents=True, exist_ok=True)
    full_path = args.outdir / "m1_brdf_gallery_ggx.png"
    ndf_path = args.outdir / "m1_debug_ndf.png"
    g_path = args.outdir / "m1_debug_g.png"
    spec_path = args.outdir / "m1_debug_spec_radiance.png"

    save_png(full_path, gallery_full)
    save_png(ndf_path, gallery_ndf)
    save_png(g_path, gallery_g)
    save_png(spec_path, gallery_spec)

    meta_path = args.meta or (args.outdir / "m1_meta.json")

    meta_payload = build_meta(results)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, indent=2)

    print("\n=== Milestone 1 GGX Gallery Summary ===")
    print(f"Tiles: {len(results)}  Tile size: {width}x{height}")
    print(f"Outputs written to: {args.outdir.resolve()}")
    threshold_display = results[0].spec_threshold_value if results else SPEC_THRESHOLD_LINEAR
    for t in results:
        print(
            f"  r={t.roughness:.2f}  mean_lum={t.mean_srgb_luminance:6.2f}  "
            f"spec_area>{threshold_display:.6f}: {t.spec_threshold_ratio:6.3f}"
        )
    print("\nAcceptance checks:")
    print(f"  A1 luminance range ({LUM_RANGE[0]}, {LUM_RANGE[1]}): {meta_payload['acceptance']['mean_luminance_in_range']}")
    print(f"  A2 spec-area monotonic: {meta_payload['acceptance']['spec_threshold_monotonic']}")
    print(f"  A3 NDF peak monotonic decrease: {meta_payload['acceptance']['ndf_center_monotonic']}")
    print(f"Meta JSON -> {meta_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
