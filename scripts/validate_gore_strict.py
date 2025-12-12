#!/usr/bin/env python
"""Validator for GORE_STRICT_PROFILE terrain metrics."""
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image


# ---- GORE_STRICT_PROFILE ----

GORE_STRICT_PROFILE = {
    "luminance": {
        "quantiles": {
            1:  (0.07669176906347275, 0.010),
            5:  (0.10018510371446610, 0.010),
            25: (0.20886588096618652, 0.015),
            50: (0.34777727723121643, 0.015),
            75: (0.49392470717430115, 0.015),
            95: (0.64771070182323440, 0.020),
            99: (0.76940103113651510, 0.020),
        },
        "min_max": {
            "min": (0.050661176443099976, 0.010),
            "max": (0.846744298934936523, 0.030),
        },
        "dynamic_ratio": {
            "target": 6.7846999168396,
            "range":  (6.4, 7.2),
        },
        "crushed_fraction_max": 0.001,
        "blown_fraction_max":   0.001,
    },
    "bands": {
        # A: 0.05–0.20, B: 0.20–0.50, C: 0.50–0.80
        "pA": {"target": 0.2336175925925926, "range": (0.19, 0.27)},
        "pB": {"target": 0.5250212962962963, "range": (0.48, 0.57)},
        "pC": {"target": 0.23600462962962962, "range": (0.19, 0.27)},
    },
    "gradients": {
        "mean":   {"target": 0.050033506006002426, "range": (0.047, 0.053)},
        "median": {"target": 0.03174756467342377,  "range": (0.029, 0.035)},
        "q90":    {"target": 0.1179272413253784,   "range": (0.112, 0.125)},
        "q99":    {"target": 0.27155117601156237,  "range": (0.245, 0.300)},
    },
    "hsv": {
        "h_mean": {"target": 0.08701974093142302, "range": (0.075, 0.095)},
        "h_std":  {"target": 0.09980322610382295, "range": (0.080, 0.120),
                   "stretch": True},
        "s_mean": {"target": 0.3866623342037201,  "range": (0.36, 0.41)},
        "s_std":  {"target": 0.12991341948509216, "range": (0.10, 0.16)},
    },
    "band_hue": {
        "A": {"target": 0.03662301055707392, "range": (0.02, 0.07)},
        "B": {"target": 0.09872743318316927, "range": (0.06, 0.12)},
        "C": {"target": 0.11047631094540701, "range": (0.09, 0.13)},
        "monotone": True,
    },
}


# ---- Utility: color & gradients ----

def load_image(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    # sRGB luma (no gamma correction; consistent for both)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rgb: HxWx3, in [0,1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    s = np.zeros_like(v)
    nz = maxc != 0
    s[nz] = (maxc[nz] - minc[nz]) / maxc[nz]

    h = np.zeros_like(v)
    mask = maxc == r
    h[mask] = (g[mask] - b[mask]) / (maxc[mask] - minc[mask] + 1e-8)
    mask = maxc == g
    h[mask] = 2.0 + (b[mask] - r[mask]) / (maxc[mask] - minc[mask] + 1e-8)
    mask = maxc == b
    h[mask] = 4.0 + (r[mask] - g[mask]) / (maxc[mask] - minc[mask] + 1e-8)
    h = (h / 6.0) % 1.0
    return h, s, v


def luminance_gradients(L: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[:, 1:] = L[:, 1:] - L[:, :-1]
    gy[1:, :] = L[1:, :] - L[:-1, :]
    g = np.sqrt(gx * gx + gy * gy)
    return g


# ---- Validation helpers ----

@dataclass
class MetricResult:
    name: str
    value: float
    target: float
    lo: float
    hi: float
    pass_: bool
    stretch: bool = False


def check_range(name: str, value: float,
                target: float, lo: float, hi: float,
                stretch: bool = False) -> MetricResult:
    return MetricResult(
        name=name,
        value=float(value),
        target=float(target),
        lo=float(lo),
        hi=float(hi),
        pass_=(lo <= value <= hi),
        stretch=stretch,
    )


def summarize(results):
    hard = [r for r in results if not r.stretch]
    stretch = [r for r in results if r.stretch]

    def fmt(r: MetricResult) -> str:
        status = "PASS" if r.pass_ else "FAIL"
        return (f"{status:4} {r.name:15s} "
                f"value={r.value:.4f} target={r.target:.4f} "
                f"range=[{r.lo:.4f},{r.hi:.4f}]")

    print("=== HARD METRICS ===")
    for r in hard:
        print(fmt(r))
    if stretch:
        print("\n=== STRETCH METRICS (non-gating) ===")
        for r in stretch:
            print(fmt(r))

    n_fail = sum(1 for r in hard if not r.pass_)
    print(f"\nHard metrics failing: {n_fail}")
    return n_fail == 0


# ---- Main validation ----

def validate_image(path: str) -> None:
    rgb = load_image(path)
    H, W, _ = rgb.shape
    L = rgb_to_luminance(rgb)
    flat_L = L.reshape(-1)

    prof = GORE_STRICT_PROFILE
    res = []

    # Luminance min/max
    mn, mx = float(flat_L.min()), float(flat_L.max())
    tgt_min, tol_min = prof["luminance"]["min_max"]["min"]
    tgt_max, tol_max = prof["luminance"]["min_max"]["max"]
    res.append(check_range("L_min", mn, tgt_min,
                           tgt_min - tol_min, tgt_min + tol_min))
    res.append(check_range("L_max", mx, tgt_max,
                           tgt_max - tol_max, tgt_max + tol_max))

    # Quantiles
    qs = sorted(prof["luminance"]["quantiles"].keys())
    qvals = np.percentile(flat_L, qs)
    for p, v in zip(qs, qvals):
        tgt, tol = prof["luminance"]["quantiles"][p]
        res.append(check_range(f"L_q{p}", v, tgt,
                               tgt - tol, tgt + tol))

    # Crushed/blown
    crushed = np.mean(flat_L < 0.04)
    blown = np.mean(flat_L > 0.85)
    cf_max = prof["luminance"]["crushed_fraction_max"]
    bf_max = prof["luminance"]["blown_fraction_max"]
    res.append(check_range("crushed_frac", crushed, 0.0, 0.0, cf_max))
    res.append(check_range("blown_frac", blown, 0.0, 0.0, bf_max))

    # Dynamic ratio
    q10, q90 = np.percentile(flat_L, [10, 90])
    lo = flat_L <= q10
    hi = flat_L >= q90
    dyn = float(flat_L[hi].mean() / (flat_L[lo].mean() + 1e-8))
    tgt = prof["luminance"]["dynamic_ratio"]["target"]
    lo_r, hi_r = prof["luminance"]["dynamic_ratio"]["range"]
    res.append(check_range("dynamic_ratio", dyn, tgt, lo_r, hi_r))

    # Bands A/B/C
    mask_A = (L >= 0.05) & (L < 0.20)
    mask_B = (L >= 0.20) & (L < 0.50)
    mask_C = (L >= 0.50) & (L < 0.80)
    N = float(H * W)
    pA = mask_A.sum() / N
    pB = mask_B.sum() / N
    pC = mask_C.sum() / N
    for key, val in [("pA", pA), ("pB", pB), ("pC", pC)]:
        target = prof["bands"][key]["target"]
        lo_r, hi_r = prof["bands"][key]["range"]
        res.append(check_range(key, val, target, lo_r, hi_r))

    # Gradients
    g = luminance_gradients(L).reshape(-1)
    gm = g.mean()
    gmed = np.median(g)
    gq90, gq99 = np.percentile(g, [90, 99])
    grad_prof = prof["gradients"]
    res.append(check_range("g_mean", gm, grad_prof["mean"]["target"],
                           *grad_prof["mean"]["range"]))
    res.append(check_range("g_median", gmed, grad_prof["median"]["target"],
                           *grad_prof["median"]["range"]))
    res.append(check_range("g_q90", gq90, grad_prof["q90"]["target"],
                           *grad_prof["q90"]["range"]))
    res.append(check_range("g_q99", gq99, grad_prof["q99"]["target"],
                           *grad_prof["q99"]["range"]))

    # HSV stats
    h, s, v = rgb_to_hsv(rgb)
    flat_h = h.reshape(-1)
    flat_s = s.reshape(-1)
    hsv_prof = prof["hsv"]
    for name, arr in [("h_mean", flat_h), ("s_mean", flat_s)]:
        target = hsv_prof[name]["target"]
        lo_r, hi_r = hsv_prof[name]["range"]
        res.append(check_range(name, arr.mean(), target, lo_r, hi_r))

    # std metrics including stretch h_std
    for name, arr in [("h_std", flat_h), ("s_std", flat_s)]:
        target = hsv_prof[name]["target"]
        lo_r, hi_r = hsv_prof[name]["range"]
        stretch = bool(hsv_prof[name].get("stretch", False))
        res.append(check_range(name, arr.std(), target, lo_r, hi_r,
                               stretch=stretch))

    # Band-wise hue means
    band_prof = prof["band_hue"]
    for label, mask in [("A", mask_A), ("B", mask_B), ("C", mask_C)]:
        if mask.sum() == 0:
            val = float("nan")
        else:
            val = float(h[mask].mean())
        target = band_prof[label]["target"]
        lo_r, hi_r = band_prof[label]["range"]
        res.append(check_range(f"h_{label}", val, target, lo_r, hi_r))
    if band_prof.get("monotone", False):
        hA = res[-3].value
        hB = res[-2].value
        hC = res[-1].value
        mono_pass = (hA < hB < hC)
        res.append(MetricResult("h_monotone", float(hB), 0.0, 0.0, 1.0,
                                mono_pass))

    ok = summarize(res)
    if ok:
        print("\nOVERALL: PASS (all hard metrics within strict ranges)")
    else:
        print("\nOVERALL: FAIL (some hard metrics out of range)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True,
                    help="rendered image to validate")
    args = ap.parse_args()
    validate_image(args.img)


if __name__ == "__main__":
    main()
