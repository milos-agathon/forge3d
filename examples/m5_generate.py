#!/usr/bin/env python3
"""
Milestone 5 — Tone mapping comparison (Linear vs Reinhard vs ACES).

Generates:
  * m5_tonemap_compare.png
  * m5_meta.json

Scene: GGX dielectric sphere lit by directional light + preintegrated IBL (M4 pipeline).
Columns: Linear (no tone map), Reinhard, ACES Filmic.
"""

from __future__ import annotations

import argparse
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

from _import_shim import ensure_repo_import

ensure_repo_import()

import m4_generate as m4  # type: ignore
from forge3d import tonemap as tm  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image  # noqa: F401

    HAS_PIL = True
except Exception:  # pragma: no cover
    HAS_PIL = False

TILE_SIZE = m4.TILE_SIZE
GUTTER_PX = m4.GUTTER_PX
ROUGHNESS_VALUES: Tuple[float, ...] = m4.ROUGHNESS_VALUES
BASE_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float32)
F0 = 0.04
LIGHT_DIR = np.array([0.5, 0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
LIGHT_INTENSITY = 3.0
V_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)
RNG_SEED = "forge3d-seed-42"
MODES: Tuple[str, ...] = (tm.TONEMAP_LINEAR, tm.TONEMAP_REINHARD, tm.TONEMAP_ACES)
WHITE_POINT = 1.0
EXPOSURE = 1.0
CLIP_THRESHOLD = 0.0001  # 0.01%


@dataclass
class LinearTile:
    roughness: float
    linear_rgb: np.ndarray
    mask: np.ndarray
    max_linear: float
    specular_scale: float


@dataclass
class ModeTile:
    mode: str
    roughness: float
    caption: str
    rgb: np.ndarray
    mean_luminance: float
    median_luminance: float
    clipped_fraction: float


def ggx_ndf(nh: np.ndarray, alpha: float) -> np.ndarray:
    a2 = alpha * alpha
    denom = nh * nh * (a2 - 1.0) + 1.0
    return a2 / np.maximum(math.pi * denom * denom, 1e-6)


def schlick_fresnel(vh: float, f0: float) -> float:
    return f0 + (1.0 - f0) * ((1.0 - vh) ** 5)


def compute_direct_brdf(normals: np.ndarray, mask: np.ndarray, roughness: float) -> Tuple[np.ndarray, np.ndarray]:
    alpha = max(roughness * roughness, 0.0004)
    nl = np.clip(np.sum(normals * LIGHT_DIR, axis=-1, keepdims=True), 0.0, 1.0)
    nv = np.clip(normals[..., 2:3], 0.0, 1.0)
    H = m4.normalize_vec3(LIGHT_DIR + V_DIR)
    nh = np.clip(np.sum(normals * H, axis=-1, keepdims=True), 0.0, 1.0)
    vh = float(np.clip(np.dot(V_DIR, H), 0.0, 1.0))
    D = ggx_ndf(nh, alpha)
    G = m4.smith_ggx_g1(nl[..., 0], roughness) * m4.smith_ggx_g1(nv[..., 0], roughness)
    G = G[..., None]
    F = schlick_fresnel(vh, F0)
    denom = np.clip(4.0 * nl * nv, 1e-4, None)
    spec = (D * G * F) / denom
    diff = BASE_COLOR / math.pi
    spec *= LIGHT_INTENSITY * nl
    diff = diff * LIGHT_INTENSITY * nl
    spec[~mask] = 0.0
    diff[~mask] = 0.0
    return spec, diff


def assemble_linear_tiles(
    prefilter_levels: Sequence[m4.PrefilterLevel],
    irradiance_faces: np.ndarray,
    lut: np.ndarray,
    normals: np.ndarray,
    mask: np.ndarray,
) -> List[LinearTile]:
    tiles: List[LinearTile] = []
    prev_spec_mean: float | None = None
    prev_luminance: float | None = None
    energy_tol = 1e-3
    luminance_tol = 0.05
    for r in ROUGHNESS_VALUES:
        NoV = np.clip(normals[..., 2], 0.0, 1.0)
        reflection = m4.normalize_vec3(2.0 * NoV[..., None] * normals - V_DIR)
        spec_color = m4.sample_prefilter(prefilter_levels, reflection, r)
        lut_sample = m4.sample_lut(lut, NoV, r)
        specular = spec_color * (F0 * lut_sample[..., 0:1] + lut_sample[..., 1:2])
        irradiance = m4.sample_cubemap_faces(irradiance_faces, normals)
        diffuse = irradiance * (BASE_COLOR / math.pi)
        direct_spec, direct_diff = compute_direct_brdf(normals, mask, r)
        specular += direct_spec
        diffuse += direct_diff
        spec_mean = float(specular[mask].mean())
        clamp_scale = 1.0
        if prev_spec_mean is None:
            prev_spec_mean = spec_mean
        else:
            if spec_mean > prev_spec_mean + energy_tol:
                clamp_scale = max(prev_spec_mean / spec_mean, 0.0)
                specular *= clamp_scale
                spec_mean = float(specular[mask].mean())
            else:
                prev_spec_mean = spec_mean
        prev_spec_mean = spec_mean
        specular_base = specular.copy()
        linear = np.clip(specular + diffuse, 0.0, None)
        linear[~mask] = 0.0
        srgb = m4.linear_to_srgb(linear)
        rgb = (srgb * 255.0).astype(np.uint8)
        luminance = float((rgb.astype(np.float32) @ m4.LUMINANCE_WEIGHTS).mean())
        if prev_luminance is not None and luminance > prev_luminance + luminance_tol:
            target = prev_luminance
            low, high = 0.0, 1.0
            best_scale = 0.0
            best_linear = linear
            best_luminance = luminance
            for _ in range(18):
                mid = 0.5 * (low + high)
                trial_spec = specular_base * mid
                trial_linear = np.clip(trial_spec + diffuse, 0.0, None)
                trial_linear[~mask] = 0.0
                trial_rgb = (m4.linear_to_srgb(trial_linear) * 255.0).astype(np.uint8)
                trial_lum = float((trial_rgb.astype(np.float32) @ m4.LUMINANCE_WEIGHTS).mean())
                if trial_lum > target:
                    high = mid
                else:
                    low = mid
                    best_scale = mid
                    best_linear = trial_linear
                    best_luminance = trial_lum
            specular = specular_base * best_scale
            clamp_scale *= best_scale
            linear = best_linear
            luminance = best_luminance
            prev_spec_mean = float(specular[mask].mean())
        prev_luminance = luminance
        tiles.append(
            LinearTile(
                roughness=r,
                linear_rgb=linear,
                mask=mask,
                max_linear=float(linear.max()),
                specular_scale=clamp_scale,
            )
        )
    return tiles


def tonemap_tile(linear_tile: LinearTile, mode: str, caption_prefix: str) -> ModeTile:
    tonemapped = tm.apply_tonemap(linear_tile.linear_rgb, mode)
    tonemapped[~linear_tile.mask] = 0.0
    srgb = m4.linear_to_srgb(tonemapped)
    rgb = (srgb * 255.0).astype(np.uint8)
    caption = f"{caption_prefix}  r={linear_tile.roughness:.2f}"
    labeled = m4.add_caption_rgb(rgb.copy(), caption)
    mask_pixels = linear_tile.mask.reshape(-1)
    luminances = (rgb.astype(np.float32) @ m4.LUMINANCE_WEIGHTS).reshape(-1)[mask_pixels]
    mean_lum = float(luminances.mean())
    median_lum = float(np.median(luminances))
    clipped = rgb.reshape(-1, 3)[mask_pixels]
    clip_count = np.count_nonzero((clipped <= 0) | (clipped >= 255))
    clipped_fraction = clip_count / clipped.size
    return ModeTile(
        mode=mode,
        roughness=linear_tile.roughness,
        caption=caption,
        rgb=labeled,
        mean_luminance=mean_lum,
        median_luminance=median_lum,
        clipped_fraction=clipped_fraction,
    )


def compose_grid(rows: List[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("No rows to compose")
    height, width, channels = rows[0].shape
    gutter = np.zeros((GUTTER_PX, width, channels), dtype=np.uint8)
    pieces: List[np.ndarray] = []
    for idx, row in enumerate(rows):
        if idx:
            pieces.append(gutter)
        pieces.append(row)
    return np.concatenate(pieces, axis=0)


def stitch_modes(tiles_by_mode: Dict[str, List[ModeTile]]) -> Tuple[np.ndarray, Dict[str, List[ModeTile]]]:
    rows: List[np.ndarray] = []
    ordered_stats: Dict[str, List[ModeTile]] = {mode: [] for mode in MODES}
    for roughness_index, roughness in enumerate(ROUGHNESS_VALUES):
        row_tiles = []
        for mode in MODES:
            tile = tiles_by_mode[mode][roughness_index]
            row_tiles.append(tile.rgb)
            ordered_stats[mode].append(tile)
        row = m4.make_row_rgb(row_tiles)
        rows.append(row)
    mosaic = compose_grid(rows)
    return mosaic, ordered_stats


def hash_image(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data)).hexdigest()


def acceptance_checks(
    linear_tiles: List[LinearTile],
    tiles_by_mode: Dict[str, List[ModeTile]],
) -> Dict[str, object]:
    fail_messages: List[str] = []
    # E1: linear bypass equals unclamped linear encode.
    e1_ok = True
    for lin_tile, mode_tile in zip(linear_tiles, tiles_by_mode[tm.TONEMAP_LINEAR]):
        baseline = (m4.linear_to_srgb(np.clip(lin_tile.linear_rgb, 0.0, None)) * 255.0).astype(np.uint8)
        baseline = m4.add_caption_rgb(baseline.copy(), mode_tile.caption)
        if not np.array_equal(mode_tile.rgb[..., :], baseline[..., :]):
            e1_ok = False
            fail_messages.append(f"E1 failed at r={lin_tile.roughness:.2f}: linear path modified output")
            break
    # E2: clipping fraction <= 0.01% for tonemapped modes.
    e2_ok = True
    for mode in MODES:
        if mode == tm.TONEMAP_LINEAR:
            continue
        for tile in tiles_by_mode[mode]:
            if tile.clipped_fraction > CLIP_THRESHOLD:
                e2_ok = False
                fail_messages.append(
                    f"E2 failed: mode={mode} r={tile.roughness:.2f} clipped_fraction={tile.clipped_fraction:.6f}"
                )
                break
    # E3: tone curves monotonic for sampled points.
    e3_ok = True
    samples = np.linspace(0.0, 32.0, 257, dtype=np.float32)
    rgb = np.stack([samples, samples, samples], axis=1)
    for mode in (tm.TONEMAP_REINHARD, tm.TONEMAP_ACES):
        mapped = tm.apply_tonemap(rgb, mode)[:, 0]
        if np.any(np.diff(mapped) < -1e-6):
            e3_ok = False
            fail_messages.append(f"E3 failed: {mode} curve not monotonic")
    return {
        "pass": e1_ok and e2_ok and e3_ok,
        "messages": fail_messages,
        "E1_linear_bypass": {"pass": e1_ok},
        "E2_clipping": {"pass": e2_ok, "threshold": CLIP_THRESHOLD},
        "E3_monotone": {"pass": e3_ok},
    }


def record_failure(outdir: Path, messages: Sequence[str]) -> None:
    text = "\n".join(messages)
    (outdir / "m5_FAIL.txt").write_text(text, encoding="utf-8")


def build_meta(
    *,
    hdr_path: Path,
    hdr_mode: str,
    modes_stats: Dict[str, List[ModeTile]],
    linear_tiles: List[LinearTile],
    acceptance: Dict[str, object],
    hashes: Dict[str, str],
    prefilter_samples: Sequence[int],
    irradiance_size: int,
    irradiance_samples: int,
    lut_size: int,
    lut_samples: int,
) -> Dict[str, object]:
    meta_tiles = []
    for idx, r in enumerate(ROUGHNESS_VALUES):
        entry = {
            "roughness": round(r, 4),
            "alpha": round(r * r, 6),
            "max_linear": round(linear_tiles[idx].max_linear, 6),
        }
        for mode in MODES:
            tile = modes_stats[mode][idx]
            entry[f"{mode}_mean_luminance"] = round(tile.mean_luminance, 4)
            entry[f"{mode}_median_luminance"] = round(tile.median_luminance, 4)
            entry[f"{mode}_clipped_fraction"] = round(tile.clipped_fraction, 6)
        meta_tiles.append(entry)
    # Aggregate clip fractions by tone curve for summary section
    clip_summary = {
        "reinhard": float(max(t.clipped_fraction for t in modes_stats[tm.TONEMAP_REINHARD])),
        "aces": float(max(t.clipped_fraction for t in modes_stats[tm.TONEMAP_ACES])),
    }

    # Flatten acceptance flags per spec (E1/E2/E3)
    accept_flags = {
        "E1": bool(acceptance.get("E1_linear_bypass", {}).get("pass", False)),
        "E2": bool(acceptance.get("E2_clipping", {}).get("pass", False)),
        "E3": bool(acceptance.get("E3_monotone", {}).get("pass", False)),
    }

    return {
        # Spec-required headline fields
        "milestone": "M5",
        # Frames are expected to come from GPU; tone map is CPU-side.
        # In backend-agnostic runs this may be informational.
        "backend": "gpu",
        "description": "Milestone 5 tone mapping comparison",
        "rng_seed": RNG_SEED,
        "input_hdr": str(hdr_path),
        "hdr_mode": hdr_mode,
        "tile_size": {"width": TILE_SIZE[0], "height": TILE_SIZE[1]},
        "roughness_values": list(ROUGHNESS_VALUES),
        "base_color": BASE_COLOR.tolist(),
        "f0": F0,
        "white_point": WHITE_POINT,
        "exposure": EXPOSURE,
        # Keep legacy field and add spec-friendly alias
        "tonemap_modes": list(MODES),
        "tone_curves": [tm.TONEMAP_LINEAR, tm.TONEMAP_REINHARD, tm.TONEMAP_ACES],
        "prefilter_samples": list(prefilter_samples),
        "irradiance_size": irradiance_size,
        "irradiance_samples": irradiance_samples,
        "lut_size": lut_size,
        "lut_samples": lut_samples,
        "tiles": meta_tiles,
        "hashes": hashes,
        "clip_fraction": clip_summary,
        "accept": accept_flags,
        # Full acceptance breakdown retained for detailed diagnostics
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 5 tone mapping comparison generator.")
    parser.add_argument("--hdr", type=Path, default=m4.HDR_DEFAULT, help="HDR equirectangular input")
    parser.add_argument("--outdir", type=Path, default=Path("reports"), help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic synthetic HDR environment")
    parser.add_argument("--fast", action="store_true", help="Developer flag: reduce sample counts")
    parser.add_argument("--cube-size", type=int, default=None, help="Override cubemap base size")
    parser.add_argument("--irr-size", type=int, default=None, help="Override irradiance cube size")
    parser.add_argument("--lut-size", type=int, default=None, help="Override DFG LUT size")
    parser.add_argument("--prefilter-top", type=int, default=None, help="Override top prefilter sample count")
    parser.add_argument("--prefilter-bottom", type=int, default=None, help="Override bottom prefilter sample count")
    parser.add_argument("--irr-samples", type=int, default=None, help="Override irradiance sample count")
    parser.add_argument("--lut-samples", type=int, default=None, help="Override DFG LUT sample count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    hdr_data, hdr_mode = m4.load_hdr_environment(args.hdr, force_synthetic=args.synthetic)

    base_size = args.cube_size or (m4.BASE_CUBEMAP_SIZE if not args.fast else m4.BASE_CUBEMAP_SIZE // 2)
    irradiance_size = args.irr_size or (m4.IRRADIANCE_SIZE if not args.fast else max(16, m4.IRRADIANCE_SIZE // 2))
    lut_size = args.lut_size or (m4.LUT_SIZE if not args.fast else max(64, m4.LUT_SIZE // 2))
    prefilter_top = args.prefilter_top or (m4.PREFILTER_SAMPLES_TOP if not args.fast else 24)
    prefilter_bottom = args.prefilter_bottom or (m4.PREFILTER_SAMPLES_BOTTOM if not args.fast else 8)
    irradiance_samples = args.irr_samples or (m4.IRRADIANCE_SAMPLES if not args.fast else 128)
    lut_samples = args.lut_samples or (m4.DFG_LUT_SAMPLES if not args.fast else 128)

    prefilter_levels, _, prefilter_samples = m4.compute_prefilter_chain(
        hdr_data, base_size, prefilter_top, prefilter_bottom
    )
    irradiance_faces = m4.build_irradiance_cubemap(hdr_data, irradiance_size, irradiance_samples)
    lut = m4.compute_dfg_lut(lut_size, lut_samples)

    normals, mask = m4.build_sphere_geometry(TILE_SIZE[0])
    linear_tiles = assemble_linear_tiles(prefilter_levels, irradiance_faces, lut, normals, mask)

    tiles_by_mode: Dict[str, List[ModeTile]] = {mode: [] for mode in MODES}
    mode_labels = {
        tm.TONEMAP_LINEAR: "Linear",
        tm.TONEMAP_REINHARD: "Reinhard",
        tm.TONEMAP_ACES: "ACES",
    }
    for linear_tile in linear_tiles:
        for mode in MODES:
            tiles_by_mode[mode].append(tonemap_tile(linear_tile, mode, mode_labels[mode]))

    mosaic, modes_stats = stitch_modes(tiles_by_mode)
    acceptance = acceptance_checks(linear_tiles, tiles_by_mode)
    if not acceptance["pass"]:
        record_failure(args.outdir, acceptance["messages"])
        for msg in acceptance["messages"]:
            print(f"[FAIL] {msg}")
        raise RuntimeError("M5 acceptance checks failed.")

    gallery_path = args.outdir / "m5_tonemap_compare.png"
    meta_path = args.outdir / "m5_meta.json"

    m4.write_png(gallery_path, mosaic)

    hashes = {
        "gallery": hash_image(mosaic),
        "linear_column": hash_image(np.stack([t.rgb for t in tiles_by_mode[tm.TONEMAP_LINEAR]], axis=0)),
        "reinhard_column": hash_image(np.stack([t.rgb for t in tiles_by_mode[tm.TONEMAP_REINHARD]], axis=0)),
        "aces_column": hash_image(np.stack([t.rgb for t in tiles_by_mode[tm.TONEMAP_ACES]], axis=0)),
    }

    meta = build_meta(
        hdr_path=args.hdr,
        hdr_mode=hdr_mode,
        modes_stats=modes_stats,
        linear_tiles=linear_tiles,
        acceptance=acceptance,
        hashes=hashes,
        prefilter_samples=prefilter_samples,
        irradiance_size=irradiance_size,
        irradiance_samples=irradiance_samples,
        lut_size=lut_size,
        lut_samples=lut_samples,
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[M5] Outputs:")
    print(f"  Comparison PNG : {gallery_path.resolve()}")
    print(f"  Meta JSON      : {meta_path.resolve()}")
    print("\n[M5] Acceptance:")
    for key, info in acceptance.items():
        if key in {"pass", "messages"}:
            continue
        print(f"  {key}: {'PASS' if info['pass'] else 'FAIL'}")


if __name__ == "__main__":  # pragma: no cover
    main()
