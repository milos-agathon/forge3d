#!/usr/bin/env python3
"""
style_match_eval_v2.py

Pure Python + NumPy + Pillow evaluation harness to compare a candidate render
against a reference render with fixed ROIs defined in a canonical 1920x1080
coordinate system, automatically scaled to the working image resolution.

Key upgrades vs v1:
- Handles mismatched image sizes via optional deterministic resize (default: cand_to_ref).
- Scales ROIs/exclusion from canonical 1920x1080 to working size.
- Excludes the "top-right uniform region" from ALL region metrics (including ROI_A/B/C)
  by splitting each ROI into subregions outside the exclusion and area-weight merging.

Dependencies:
  - numpy
  - pillow

Usage:
  python style_match_eval_v2.py \
    --ref /path/to/Gore_Range_Albers_1m.png \
    --cand /path/to/terrain_csm.png \
    --outdir reports/style_match

Outputs:
  - <outdir>/style_match_metrics.json
  - <outdir>/ab_side_by_side.png
  - <outdir>/diff_heatmap.png
  - <outdir>/style_match_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image


# -----------------------------
# Canonical spec coordinates
# -----------------------------

CANON_W = 1920
CANON_H = 1080

ROI_SPECS_CANON = {
    "ROI_A": {"x0": 140,  "x1": 820,  "y0": 260, "y1": 980},   # left/central shadowed ridges
    "ROI_B": {"x0": 760,  "x1": 1460, "y0": 220, "y1": 760},   # central ridge structure
    "ROI_C": {"x0": 1180, "x1": 1880, "y0": 620, "y1": 1060},  # lower-right textured slopes
}

EXCLUDE_CANON = {"x0": 1260, "x1": 1920, "y0": 0, "y1": 460}   # "top-right uniform region"


# -----------------------------
# Metrics parameters
# -----------------------------

DEFAULT_SSIM_WIN = 11  # odd
DEFAULT_SPECKLE_LUMA_THR = 0.90
DEFAULT_SPECKLE_VAR_THR = 5e-4


# -----------------------------
# Rect helper
# -----------------------------

@dataclass(frozen=True)
class Rect:
    x0: int
    x1: int
    y0: int
    y1: int

    def clamp(self, w: int, h: int) -> "Rect":
        x0 = max(0, min(self.x0, w))
        x1 = max(0, min(self.x1, w))
        y0 = max(0, min(self.y0, h))
        y1 = max(0, min(self.y1, h))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return Rect(x0, x1, y0, y1)

    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def is_empty(self) -> bool:
        return self.area() == 0

    def to_json(self) -> Dict[str, int]:
        return {"x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1}


def scale_rect_from_canon(r: Rect, w: int, h: int) -> Rect:
    sx = w / float(CANON_W)
    sy = h / float(CANON_H)
    return Rect(
        int(round(r.x0 * sx)),
        int(round(r.x1 * sx)),
        int(round(r.y0 * sy)),
        int(round(r.y1 * sy)),
    ).clamp(w, h)


# -----------------------------
# Image IO and conversions
# -----------------------------

def load_rgb_u8(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def resize_u8(rgb_u8: np.ndarray, w: int, h: int) -> np.ndarray:
    img = Image.fromarray(rgb_u8, mode="RGB")
    img2 = img.resize((w, h), resample=Image.Resampling.LANCZOS)
    return np.asarray(img2, dtype=np.uint8)


def srgb_u8_to_float(rgb_u8: np.ndarray) -> np.ndarray:
    return (rgb_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)


def linear_to_xyz(rgb_lin: np.ndarray) -> np.ndarray:
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    return np.tensordot(rgb_lin, M.T, axes=1)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

    fx = f(x); fy = f(y); fz = f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def rgb_srgb_to_lab(rgb_srgb01: np.ndarray) -> np.ndarray:
    return xyz_to_lab(linear_to_xyz(srgb_to_linear(rgb_srgb01)))


def luma_bt709(rgb_srgb01: np.ndarray) -> np.ndarray:
    r = rgb_srgb01[..., 0]
    g = rgb_srgb01[..., 1]
    b = rgb_srgb01[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)


# -----------------------------
# Fast box filter + SSIM
# -----------------------------

def integral_image_2d(img: np.ndarray) -> np.ndarray:
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    return ii


def box_filter(img: np.ndarray, win: int) -> np.ndarray:
    assert win >= 1 and win % 2 == 1
    r = win // 2
    padded = np.pad(img, ((r, r), (r, r)), mode="reflect")
    ii = integral_image_2d(padded)
    H, W = img.shape
    y0 = np.arange(0, H, dtype=np.int64)
    x0 = np.arange(0, W, dtype=np.int64)
    y1 = y0 + win
    x1 = x0 + win
    S = (ii[y1[:, None], x1[None, :]]
         - ii[y0[:, None], x1[None, :]]
         - ii[y1[:, None], x0[None, :]]
         + ii[y0[:, None], x0[None, :]])
    return (S / (win * win)).astype(np.float32)


def ssim_y_box(ref_y: np.ndarray, cand_y: np.ndarray, win: int) -> float:
    K1, K2 = 0.01, 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ref = ref_y.astype(np.float32)
    cand = cand_y.astype(np.float32)

    mu_x = box_filter(ref, win)
    mu_y = box_filter(cand, win)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = box_filter(ref * ref, win) - mu_x2
    sigma_y2 = box_filter(cand * cand, win) - mu_y2
    sigma_xy = box_filter(ref * cand, win) - mu_xy

    sigma_x2 = np.maximum(sigma_x2, 0.0)
    sigma_y2 = np.maximum(sigma_y2, 0.0)

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(np.mean(num / (den + 1e-12)))


# -----------------------------
# Edge + speckle
# -----------------------------

def sobel_edge_strength(y: np.ndarray) -> float:
    y = y.astype(np.float32)
    p = np.pad(y, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -1 * p[:-2, :-2] + 1 * p[:-2, 2:] +
        -2 * p[1:-1, :-2] + 2 * p[1:-1, 2:] +
        -1 * p[2:, :-2] + 1 * p[2:, 2:]
    )
    gy = (
         1 * p[:-2, :-2] + 2 * p[:-2, 1:-1] + 1 * p[:-2, 2:] +
        -1 * p[2:, :-2] + -2 * p[2:, 1:-1] + -1 * p[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


def local_variance_3x3(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    p = np.pad(y, ((1, 1), (1, 1)), mode="reflect")
    s = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    m = s / 9.0
    ss = (
        p[:-2, :-2]**2 + p[:-2, 1:-1]**2 + p[:-2, 2:]**2 +
        p[1:-1, :-2]**2 + p[1:-1, 1:-1]**2 + p[1:-1, 2:]**2 +
        p[2:, :-2]**2 + p[2:, 1:-1]**2 + p[2:, 2:]**2
    )
    ms = ss / 9.0
    return np.maximum(ms - m * m, 0.0).astype(np.float32)


def speckle_fraction(y: np.ndarray, luma_thr: float, var_thr: float) -> float:
    var = local_variance_3x3(y)
    mask = (y > luma_thr) & (var > var_thr)
    return float(np.mean(mask.astype(np.float32)))


# -----------------------------
# Region splitting to exclude E
# -----------------------------

def split_rect_excluding_topright_band(r: Rect, e: Rect) -> List[Rect]:
    """
    Return sub-rectangles covering R excluding the exclusion E, under the specific
    E definition (top-right band). This removes pixels where x>=E.x0 and y<E.y1.
    """
    out: List[Rect] = []
    if r.is_empty():
        return out

    # No Y overlap with the top band
    if r.y0 >= e.y1 or r.y1 <= e.y0:
        return [r]

    # Bottom region below E.y1 keeps full width of R
    if r.y1 > e.y1:
        out.append(Rect(r.x0, r.x1, max(r.y0, e.y1), r.y1))

    # Top band overlap: keep only left portion x < e.x0
    top_y0 = r.y0
    top_y1 = min(r.y1, e.y1)
    if top_y1 > top_y0:
        left_x0 = r.x0
        left_x1 = min(r.x1, e.x0)
        if left_x1 > left_x0:
            out.append(Rect(left_x0, left_x1, top_y0, top_y1))

    return [rr for rr in out if not rr.is_empty()]


# -----------------------------
# Metrics helpers
# -----------------------------

def crop(arr: np.ndarray, r: Rect) -> np.ndarray:
    return arr[r.y0:r.y1, r.x0:r.x1]


def luminance_percentiles(y_flat: np.ndarray) -> Dict[str, float]:
    pcts = [10, 50, 75, 90, 99]
    vals = np.percentile(y_flat, pcts).astype(np.float64)
    return {f"p{p}": float(v) for p, v in zip(pcts, vals)}


def deltae_stats(ref_lab: np.ndarray, cand_lab: np.ndarray) -> Dict[str, float]:
    d = ref_lab - cand_lab
    de = np.sqrt(np.sum(d * d, axis=-1)).astype(np.float32)  # CIE76
    return {"mean": float(np.mean(de)), "p95": float(np.percentile(de.reshape(-1), 95))}


def compute_region_metrics_simple(
    ref_rgb01: np.ndarray,
    cand_rgb01: np.ndarray,
    region: Rect,
    ssim_win: int,
    speckle_luma_thr: float,
    speckle_var_thr: float,
    compute_speckle: bool,
) -> Dict[str, object]:
    ref_crop = crop(ref_rgb01, region)
    cand_crop = crop(cand_rgb01, region)

    ref_y = luma_bt709(ref_crop)
    cand_y = luma_bt709(cand_crop)

    ref_lab = rgb_srgb_to_lab(ref_crop)
    cand_lab = rgb_srgb_to_lab(cand_crop)

    out: Dict[str, object] = {}
    out["rect"] = region.to_json()
    out["ssim_y"] = ssim_y_box(ref_y, cand_y, win=ssim_win)
    out["deltaE"] = deltae_stats(ref_lab, cand_lab)
    out["edge_strength_ref"] = sobel_edge_strength(ref_y)
    out["edge_strength_cand"] = sobel_edge_strength(cand_y)
    if compute_speckle:
        out["speckle_fraction_ref"] = speckle_fraction(ref_y, speckle_luma_thr, speckle_var_thr)
        out["speckle_fraction_cand"] = speckle_fraction(cand_y, speckle_luma_thr, speckle_var_thr)
        out["speckle_params"] = {"luma_threshold": speckle_luma_thr, "var_threshold": speckle_var_thr, "var_window": 3}
    return out


def merge_metrics_by_area(
    metrics: List[Tuple[Dict[str, object], int]],
    ref_y_flats: List[np.ndarray],
    cand_y_flats: List[np.ndarray],
) -> Dict[str, object]:
    total_area = sum(a for _, a in metrics)
    if total_area <= 0:
        return {"empty": True}

    def wavg(path: List[str]) -> float:
        s = 0.0
        for m, a in metrics:
            v = m
            for k in path:
                v = v[k]
            s += float(v) * a
        return s / total_area

    merged: Dict[str, object] = {
        "area_weighted": True,
        "subregions": [m["rect"] for m, _ in metrics],
        "ssim_y": wavg(["ssim_y"]),
        "deltaE": {"mean": wavg(["deltaE", "mean"]), "p95": wavg(["deltaE", "p95"])},
        "edge_strength_ref": wavg(["edge_strength_ref"]),
        "edge_strength_cand": wavg(["edge_strength_cand"]),
    }

    if "speckle_fraction_ref" in metrics[0][0]:
        merged["speckle_fraction_ref"] = wavg(["speckle_fraction_ref"])
        merged["speckle_fraction_cand"] = wavg(["speckle_fraction_cand"])
        merged["speckle_params"] = metrics[0][0].get("speckle_params")

    ref_all = np.concatenate(ref_y_flats, axis=0)
    cand_all = np.concatenate(cand_y_flats, axis=0)
    merged["luminance_percentiles_ref"] = luminance_percentiles(ref_all)
    merged["luminance_percentiles_cand"] = luminance_percentiles(cand_all)
    return merged


def compute_region_metrics_excluding(
    ref_rgb01: np.ndarray,
    cand_rgb01: np.ndarray,
    region: Rect,
    exclude: Rect,
    ssim_win: int,
    speckle_luma_thr: float,
    speckle_var_thr: float,
    compute_speckle: bool,
) -> Dict[str, object]:
    subs = split_rect_excluding_topright_band(region, exclude)
    subs = [s for s in subs if s.area() > 0]
    if not subs:
        return {"empty": True, "subregions": [], "exclude_rect": exclude.to_json(), "rect": region.to_json()}

    mlist: List[Tuple[Dict[str, object], int]] = []
    ref_y_flats: List[np.ndarray] = []
    cand_y_flats: List[np.ndarray] = []
    for sub in subs:
        m = compute_region_metrics_simple(
            ref_rgb01, cand_rgb01, sub, ssim_win,
            speckle_luma_thr, speckle_var_thr,
            compute_speckle=compute_speckle,
        )
        mlist.append((m, sub.area()))
        ref_y_flats.append(luma_bt709(crop(ref_rgb01, sub)).reshape(-1))
        cand_y_flats.append(luma_bt709(crop(cand_rgb01, sub)).reshape(-1))

    merged = merge_metrics_by_area(mlist, ref_y_flats, cand_y_flats)
    merged["exclude_rect"] = exclude.to_json()
    merged["rect"] = region.to_json()
    return merged


# -----------------------------
# Outputs
# -----------------------------

def save_ab_side_by_side(ref_u8: np.ndarray, cand_u8: np.ndarray, out_path: str) -> None:
    ref_img = Image.fromarray(ref_u8, mode="RGB")
    cand_img = Image.fromarray(cand_u8, mode="RGB")
    H = max(ref_img.height, cand_img.height)
    W = ref_img.width + cand_img.width
    out = Image.new("RGB", (W, H), (0, 0, 0))
    out.paste(ref_img, (0, 0))
    out.paste(cand_img, (ref_img.width, 0))
    out.save(out_path)


def save_diff_heatmap(ref_rgb01: np.ndarray, cand_rgb01: np.ndarray, out_path: str) -> None:
    ref_y = luma_bt709(ref_rgb01)
    cand_y = luma_bt709(cand_rgb01)
    diff = np.abs(ref_y - cand_y)
    p99 = float(np.percentile(diff.reshape(-1), 99))
    scale = max(0.25, p99)
    img = np.clip(diff / scale, 0.0, 1.0)
    u8 = (img * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_path)


def write_summary_md(metrics: Dict[str, object], out_path: str) -> None:
    lines = []
    lines.append("# Style match summary\n\n")
    meta = metrics["meta"]
    lines.append(f"- Generated: {meta['generated_utc']}\n")
    lines.append(f"- Reference: `{meta['ref_path']}`\n")
    lines.append(f"- Candidate: `{meta['cand_path']}`\n")
    lines.append(f"- Resize mode: `{meta['resize_mode']}`\n")
    if meta.get("resized"):
        lines.append(f"- Resized: {meta['resize_detail']}\n")
    lines.append(f"- Working size: {meta['frame_size']['width']}x{meta['frame_size']['height']}\n")
    lines.append(f"- SSIM window: {meta['ssim_window']}\n\n")

    lines.append("## Regions\n\n")
    for k, m in metrics["regions"].items():
        lines.append(f"### {k}\n")
        if m.get("empty"):
            lines.append("- (empty after exclusion)\n\n")
            continue
        lines.append(f"- SSIM(Y): {m['ssim_y']:.6f}\n")
        lines.append(f"- ΔE mean / p95: {m['deltaE']['mean']:.3f} / {m['deltaE']['p95']:.3f}\n")
        lines.append(f"- Edge strength (ref/cand): {m['edge_strength_ref']:.6f} / {m['edge_strength_cand']:.6f}\n")
        if "speckle_fraction_ref" in m:
            lines.append(f"- Speckle fraction (ref/cand): {m['speckle_fraction_ref']:.6f} / {m['speckle_fraction_cand']:.6f}\n")
        lines.append(f"- Luma p50 (ref/cand): {m['luminance_percentiles_ref']['p50']:.4f} / {m['luminance_percentiles_cand']['p50']:.4f}\n")
        lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference PNG path")
    ap.add_argument("--cand", required=True, help="Candidate PNG path")
    ap.add_argument("--outdir", default=os.path.join("reports", "style_match"), help="Output directory")
    ap.add_argument("--resize", choices=["cand_to_ref", "ref_to_cand", "none"], default="cand_to_ref",
                    help="If image sizes differ, deterministically resize one image")
    ap.add_argument("--ssim-win", type=int, default=DEFAULT_SSIM_WIN, help="Odd integer SSIM window size (box stats)")
    ap.add_argument("--speckle-luma-thr", type=float, default=DEFAULT_SPECKLE_LUMA_THR, help="Speckle luma threshold")
    ap.add_argument("--speckle-var-thr", type=float, default=DEFAULT_SPECKLE_VAR_THR, help="Speckle local variance threshold (3x3)")
    args = ap.parse_args()

    if args.ssim_win < 1 or args.ssim_win % 2 != 1:
        raise SystemExit("--ssim-win must be an odd integer >= 1")

    os.makedirs(args.outdir, exist_ok=True)

    ref_u8 = load_rgb_u8(args.ref)
    cand_u8 = load_rgb_u8(args.cand)

    resized = False
    resize_detail = ""

    if ref_u8.shape != cand_u8.shape:
        if args.resize == "none":
            raise SystemExit(f"Image sizes differ: ref={ref_u8.shape} cand={cand_u8.shape}. "
                             f"Use --resize cand_to_ref (default) or ref_to_cand.")
        if args.resize == "cand_to_ref":
            H, W = ref_u8.shape[0], ref_u8.shape[1]
            cand_u8 = resize_u8(cand_u8, W, H)
            resized = True
            resize_detail = f"cand resized to ref size {W}x{H} (LANCZOS)"
        else:
            H, W = cand_u8.shape[0], cand_u8.shape[1]
            ref_u8 = resize_u8(ref_u8, W, H)
            resized = True
            resize_detail = f"ref resized to cand size {W}x{H} (LANCZOS)"

    H, W = ref_u8.shape[0], ref_u8.shape[1]

    ref_rgb01 = srgb_u8_to_float(ref_u8)
    cand_rgb01 = srgb_u8_to_float(cand_u8)

    # Scale ROIs + exclude to working size
    rois_scaled: Dict[str, Rect] = {}
    for name, s in ROI_SPECS_CANON.items():
        r = Rect(s["x0"], s["x1"], s["y0"], s["y1"])
        rois_scaled[name] = scale_rect_from_canon(r, W, H)

    exclude_scaled = scale_rect_from_canon(Rect(**EXCLUDE_CANON), W, H)

    regions: Dict[str, object] = {}
    for name, rect in rois_scaled.items():
        regions[name] = compute_region_metrics_excluding(
            ref_rgb01, cand_rgb01,
            region=rect,
            exclude=exclude_scaled,
            ssim_win=args.ssim_win,
            speckle_luma_thr=args.speckle_luma_thr,
            speckle_var_thr=args.speckle_var_thr,
            compute_speckle=(name == "ROI_A"),
        )

    # Full frame exclusion: region is full frame
    full_rect = Rect(0, W, 0, H)
    regions["FULL_FRAME_EXCLUSION"] = compute_region_metrics_excluding(
        ref_rgb01, cand_rgb01,
        region=full_rect,
        exclude=exclude_scaled,
        ssim_win=args.ssim_win,
        speckle_luma_thr=args.speckle_luma_thr,
        speckle_var_thr=args.speckle_var_thr,
        compute_speckle=False,
    )

    metrics: Dict[str, object] = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ref_path": os.path.abspath(args.ref),
            "cand_path": os.path.abspath(args.cand),
            "resize_mode": args.resize,
            "resized": resized,
            "resize_detail": resize_detail,
            "frame_size": {"width": W, "height": H},
            "canonical_size": {"width": CANON_W, "height": CANON_H},
            "roi_specs_canonical_px": ROI_SPECS_CANON,
            "roi_specs_working_px": {k: v.to_json() for k, v in rois_scaled.items()},
            "exclude_canonical_px": EXCLUDE_CANON,
            "exclude_working_px": exclude_scaled.to_json(),
            "ssim_impl": "SSIM on luminance using box-filter local stats (integral image)",
            "ssim_window": int(args.ssim_win),
            "deltaE_impl": "CIE76 on Lab from sRGB(D65)",
            "edge_impl": "Mean Sobel gradient magnitude on luminance",
            "speckle_impl": "fraction(luma>thr AND local_var_3x3>thr) in ROI_A",
        },
        "regions": regions,
    }

    json_path = os.path.join(args.outdir, "style_match_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=False)

    save_ab_side_by_side(ref_u8, cand_u8, os.path.join(args.outdir, "ab_side_by_side.png"))
    save_diff_heatmap(ref_rgb01, cand_rgb01, os.path.join(args.outdir, "diff_heatmap.png"))
    write_summary_md(metrics, os.path.join(args.outdir, "style_match_summary.md"))

    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
