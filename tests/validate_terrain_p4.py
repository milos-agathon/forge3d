#!/usr/bin/env python3
"""
Validation harness for terrain rendering milestones P1-P4.

Computes all metrics defined in docs/prompt.md and reports pass/fail status.
Metrics are computed only on non-water terrain pixels (R0 compliant).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    value: float | None = None
    expected: str = ""
    message: str = ""


@dataclass
class MilestoneResults:
    """Results for a milestone (P1, P2, P3, P4)."""
    name: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def add(self, name: str, passed: bool, value: float | None = None,
            expected: str = "", message: str = "") -> None:
        self.results.append(ValidationResult(name, passed, value, expected, message))


def load_image_linear(path: Path) -> np.ndarray:
    """Load image and convert sRGB to linear RGB."""
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    # sRGB to linear
    mask = img <= 0.04045
    img = np.where(mask, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return img


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute relative luminance: L = 0.2126R + 0.7152G + 0.0722B."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV. Input shape (..., 3), output shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    # Hue
    h = np.zeros_like(maxc)
    mask_r = (maxc == r) & (delta > 0)
    mask_g = (maxc == g) & (delta > 0) & ~mask_r
    mask_b = (maxc == b) & (delta > 0) & ~mask_r & ~mask_g

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    h = h / 6.0  # Normalize to [0, 1]

    # Saturation
    s = np.zeros_like(maxc)
    s[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]

    # Value
    v = maxc

    return np.stack([h, s, v], axis=-1)


def compute_gradient_magnitude(luminance: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using forward differences."""
    dy = np.diff(luminance, axis=0, prepend=luminance[:1, :])
    dx = np.diff(luminance, axis=1, prepend=luminance[:, :1])
    return np.sqrt(dx**2 + dy**2)


def create_water_mask(rgb: np.ndarray, luminance: np.ndarray) -> np.ndarray:
    """Create water mask based on blue dominance and saturation (R0)."""
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Water: blue-ish hue (0.5-0.7), high saturation, moderate value
    blue_hue = (h > 0.5) & (h < 0.75)
    high_sat = s > 0.3
    blue_dominant = rgb[..., 2] > rgb[..., 0] * 1.2

    water = blue_hue & high_sat & blue_dominant
    return water


def validate_luminance_quantiles(
    luminance: np.ndarray,
    mask: np.ndarray,
    ref_quantiles: dict[str, float],
    tolerance: float = 0.03
) -> list[ValidationResult]:
    """Validate luminance quantiles against reference (R1, P2-L, P3-L, P4-L)."""
    results = []
    L = luminance[mask]

    for q_name, q_ref in ref_quantiles.items():
        q_val = float(q_name.replace("q", "")) / 100.0
        q_computed = float(np.quantile(L, q_val))
        passed = abs(q_computed - q_ref) <= tolerance
        results.append(ValidationResult(
            name=f"L-{q_name}",
            passed=passed,
            value=q_computed,
            expected=f"{q_ref:.3f} ± {tolerance:.3f}",
            message=f"quantile {q_name}: {q_computed:.3f} (ref: {q_ref:.3f})"
        ))

    return results


def validate_p4(image_path: Path, verbose: bool = True) -> dict[str, MilestoneResults]:
    """Run full P1-P4 validation on terrain image."""
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    rgb = load_image_linear(image_path)
    luminance = compute_luminance(rgb)
    water_mask = create_water_mask(rgb, luminance)
    terrain_mask = ~water_mask

    if verbose:
        print(f"Image: {image_path}")
        print(f"Shape: {rgb.shape}")
        print(f"Terrain pixels: {terrain_mask.sum()} ({100*terrain_mask.mean():.1f}%)")
        print(f"Water pixels: {water_mask.sum()} ({100*water_mask.mean():.1f}%)")
        print()

    # Terrain-only arrays
    L = luminance[terrain_mask]
    hsv = rgb_to_hsv(rgb)
    H = hsv[..., 0][terrain_mask]
    S = hsv[..., 1][terrain_mask]
    gradient = compute_gradient_magnitude(luminance)
    G = gradient[terrain_mask]

    # ═══════════════════════════════════════════════════════════════════════════
    # GORE_STRICT_PROFILE - DO NOT MODIFY THESE VALUES
    # ═══════════════════════════════════════════════════════════════════════════
    GORE = {
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
            "min": (0.050661176443099976, 0.010),
            "max": (0.846744298934936523, 0.030),
            "dynamic_ratio": (6.7846999168396, (6.4, 7.2)),
            "crushed_max": 0.001,
            "blown_max": 0.001,
        },
        "bands": {
            "pA": (0.2336175925925926, (0.19, 0.27)),
            "pB": (0.5250212962962963, (0.48, 0.57)),
            "pC": (0.23600462962962962, (0.19, 0.27)),
        },
        "gradients": {
            "mean":   (0.050033506006002426, (0.047, 0.053)),
            "median": (0.03174756467342377, (0.029, 0.035)),
            "q90":    (0.1179272413253784, (0.112, 0.125)),
            "q99":    (0.27155117601156237, (0.245, 0.300)),
        },
        "hsv": {
            "h_mean": (0.08701974093142302, (0.075, 0.095)),
            "h_std":  (0.09980322610382295, (0.080, 0.120)),  # stretch goal
            "s_mean": (0.3866623342037201, (0.36, 0.41)),
            "s_std":  (0.12991341948509216, (0.10, 0.16)),
        },
        "band_hue": {
            "A": (0.03662301055707392, (0.02, 0.07)),
            "B": (0.09872743318316927, (0.06, 0.12)),
            "C": (0.11047631094540701, (0.09, 0.13)),
        },
    }

    # Legacy format for validate_luminance_quantiles
    ref_quantiles = {
        "q01": 0.07669176906347275,
        "q05": 0.10018510371446610,
        "q25": 0.20886588096618652,
        "q50": 0.34777727723121643,
        "q75": 0.49392470717430115,
        "q95": 0.64771070182323440,
        "q99": 0.76940103113651510,
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # P1: Baseline sanity checks
    # ═══════════════════════════════════════════════════════════════════════════
    p1 = MilestoneResults("P1-Baseline")

    # Basic luminance bounds (GORE_STRICT)
    L_min, L_max = float(L.min()), float(L.max())
    min_target, min_tol = GORE["luminance"]["min"]
    max_target, max_tol = GORE["luminance"]["max"]
    p1.add("L_min", min_target - min_tol <= L_min <= min_target + min_tol, L_min,
           f"{min_target:.3f} ± {min_tol:.3f}")
    p1.add("L_max", max_target - max_tol <= L_max <= max_target + max_tol, L_max,
           f"{max_target:.3f} ± {max_tol:.3f}")

    # Crushed/blown (GORE_STRICT: <0.1%)
    crushed = float((L < 0.04).mean())
    blown = float((L > 0.85).mean())
    p1.add("crushed", crushed < GORE["luminance"]["crushed_max"], crushed, "<0.1%")
    p1.add("blown", blown < GORE["luminance"]["blown_max"], blown, "<0.1%")

    # ═══════════════════════════════════════════════════════════════════════════
    # P2: Luminance & Dynamic Range Lock
    # ═══════════════════════════════════════════════════════════════════════════
    p2 = MilestoneResults("P2-Luminance")

    # P2-L quantiles (GORE_STRICT tolerances per quantile)
    for q_pct, (q_target, q_tol) in GORE["luminance"]["quantiles"].items():
        q_val = q_pct / 100.0
        q_computed = float(np.quantile(L, q_val))
        passed = abs(q_computed - q_target) <= q_tol
        p2.add(f"L-q{q_pct:02d}", passed, q_computed, f"{q_target:.4f} ± {q_tol:.3f}")

    # Dynamic ratio (GORE_STRICT: 6.4-7.2)
    q10, q90_val = np.quantile(L, 0.10), np.quantile(L, 0.90)
    mean_hi = float(L[L >= q90_val].mean())
    mean_lo = float(L[L <= q10].mean()) if (L <= q10).sum() > 0 else 0.001
    ratio = mean_hi / max(mean_lo, 0.001)
    ratio_target, (ratio_lo, ratio_hi) = GORE["luminance"]["dynamic_ratio"]
    p2.add("dynamic_ratio", ratio_lo <= ratio <= ratio_hi, ratio, f"{ratio_lo}-{ratio_hi}")

    # Stricter crushed/blown (GORE_STRICT: <0.1%)
    p2.add("crushed_strict", (L < 0.05).mean() < GORE["luminance"]["crushed_max"], 
           float((L < 0.05).mean()), "<0.1%")
    p2.add("blown_strict", (L > 0.85).mean() < GORE["luminance"]["blown_max"], 
           float((L > 0.85).mean()), "<0.1%")

    # P2-G gradient stats (GORE_STRICT)
    g_mean, g_median = float(G.mean()), float(np.median(G))
    g_q90, g_q99 = float(np.quantile(G, 0.90)), float(np.quantile(G, 0.99))
    for name, val in [("mean", g_mean), ("median", g_median), ("q90", g_q90), ("q99", g_q99)]:
        target, (lo, hi) = GORE["gradients"][name]
        p2.add(f"g_{name}", lo <= val <= hi, val, f"{lo}-{hi}")

    # P2-Band occupancy (GORE_STRICT)
    band_A = (L >= 0.05) & (L < 0.20)  # Shadows
    band_B = (L >= 0.20) & (L < 0.50)  # Midtones
    band_C = (L >= 0.50) & (L < 0.80)  # Highlights
    pA, pB, pC = band_A.mean(), band_B.mean(), band_C.mean()
    for name, val in [("pA", pA), ("pB", pB), ("pC", pC)]:
        target, (lo, hi) = GORE["bands"][name]
        p2.add(name, lo <= val <= hi, float(val), f"{lo}-{hi}")

    # ═══════════════════════════════════════════════════════════════════════════
    # P3: Gradient Structure & Global Hue
    # ═══════════════════════════════════════════════════════════════════════════
    p3 = MilestoneResults("P3-Structure")

    # P3-L dynamic ratio (GORE_STRICT: 6.4-7.2)
    p3.add("ratio_tight", ratio_lo <= ratio <= ratio_hi, ratio, f"{ratio_lo}-{ratio_hi}")

    # P3-G gradients (GORE_STRICT)
    for name, val in [("mean", g_mean), ("median", g_median), ("q90", g_q90), ("q99", g_q99)]:
        target, (lo, hi) = GORE["gradients"][name]
        p3.add(f"g_{name}_p3", lo <= val <= hi, val, f"{lo}-{hi}")

    # P3-G band-wise gradients
    G_A = G[band_A]
    G_B = G[band_B]
    G_C = G[band_C]
    g_A_mean = float(G_A.mean()) if len(G_A) > 0 else 0
    g_B_mean = float(G_B.mean()) if len(G_B) > 0 else 0
    g_C_mean = float(G_C.mean()) if len(G_C) > 0 else 0
    # Band gradients should follow: g_A < g_B, g_C <= g_B
    p3.add("g_A_mean", g_A_mean < g_B_mean, g_A_mean, f"< g_B({g_B_mean:.3f})")
    p3.add("g_B_mean", 0.040 <= g_B_mean <= 0.060, g_B_mean, "0.040-0.060")
    p3.add("g_C_mean", g_C_mean <= g_B_mean + 0.005, g_C_mean, f"<= {g_B_mean + 0.005:.3f}")

    # P3-T spatial uniformity (3x3 tiles)
    h, w = luminance.shape
    tile_h, tile_w = h // 3, w // 3
    tile_medians = []
    for i in range(3):
        for j in range(3):
            tile_lum = luminance[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tile_mask = terrain_mask[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            if tile_mask.sum() > 0:
                tile_medians.append(float(np.median(tile_lum[tile_mask])))
    if tile_medians:
        mu_tiles = float(np.mean(tile_medians))
        sigma_tiles = float(np.std(tile_medians))
        tile_range = max(tile_medians) - min(tile_medians)
        p3.add("mu_tiles", 0.32 <= mu_tiles <= 0.40, mu_tiles, "0.32-0.40")
        p3.add("sigma_tiles", 0.07 <= sigma_tiles <= 0.12, sigma_tiles, "0.07-0.12")
        p3.add("tile_range", tile_range >= 0.20, tile_range, ">=0.20")

    # P3-C global hue/saturation (GORE_STRICT)
    h_mean, h_std = float(H.mean()), float(H.std())
    s_mean, s_std = float(S.mean()), float(S.std())
    
    h_target, (h_lo, h_hi) = GORE["hsv"]["h_mean"]
    p3.add("h_mean", h_lo <= h_mean <= h_hi, h_mean, f"{h_lo}-{h_hi}")
    
    # h_std: STRETCH GOAL - log but warn if outside range
    h_std_target, (h_std_lo, h_std_hi) = GORE["hsv"]["h_std"]
    h_std_in_range = h_std_lo <= h_std <= h_std_hi
    p3.add("h_std", h_std_in_range, h_std, f"{h_std_lo}-{h_std_hi} [STRETCH]")
    
    s_target, (s_lo, s_hi) = GORE["hsv"]["s_mean"]
    p3.add("s_mean", s_lo <= s_mean <= s_hi, s_mean, f"{s_lo}-{s_hi}")
    
    s_std_target, (s_std_lo, s_std_hi) = GORE["hsv"]["s_std"]
    p3.add("s_std", s_std_lo <= s_std <= s_std_hi, s_std, f"{s_std_lo}-{s_std_hi}")

    # P3-C band-wise hue monotonicity
    H_A = H[band_A]
    H_B = H[band_B]
    H_C = H[band_C]
    h_A_mean = float(H_A.mean()) if len(H_A) > 0 else 0
    h_B_mean = float(H_B.mean()) if len(H_B) > 0 else 0
    h_C_mean = float(H_C.mean()) if len(H_C) > 0 else 0
    p3.add("hue_monotone", h_A_mean < h_B_mean < h_C_mean, None,
           f"h_A({h_A_mean:.3f}) < h_B({h_B_mean:.3f}) < h_C({h_C_mean:.3f})")

    # ═══════════════════════════════════════════════════════════════════════════
    # P4: Valley Structure, Midtones & Hue Variation (FOCUS)
    # ═══════════════════════════════════════════════════════════════════════════
    p4 = MilestoneResults("P4-Valley/Hue")

    # P4-L band occupancy (GORE_STRICT)
    for name, val in [("pA", pA), ("pB", pB), ("pC", pC)]:
        target, (lo, hi) = GORE["bands"][name]
        p4.add(f"{name}_p4", lo <= val <= hi, float(val), f"{lo}-{hi}")

    # P4-L mid quantiles (GORE_STRICT)
    for q_pct in [25, 50, 75]:
        q_target, q_tol = GORE["luminance"]["quantiles"][q_pct]
        q_computed = float(np.quantile(L, q_pct / 100.0))
        passed = abs(q_computed - q_target) <= q_tol
        p4.add(f"q{q_pct}_p4", passed, q_computed, f"{q_target:.4f} ± {q_tol:.3f}")

    # P4-G valley gradients
    g_A_q90 = float(np.quantile(G_A, 0.90)) if len(G_A) > 0 else 0
    p4.add("g_A_mean_p4", g_A_mean < g_B_mean, g_A_mean, f"< g_B({g_B_mean:.3f})")
    p4.add("g_A_q90", g_A_q90 >= 0.060, g_A_q90, ">=0.060")
    p4.add("g_ordering", g_A_mean <= g_B_mean and g_C_mean <= g_B_mean + 0.005, 
           g_B_mean, f"g_A <= g_B, g_C <= g_B+0.005")

    # P4-C hue variation (GORE_STRICT)
    p4.add("h_mean_p4", h_lo <= h_mean <= h_hi, h_mean, f"{h_lo}-{h_hi}")
    p4.add("h_std_p4", h_std_in_range, h_std, f"{h_std_lo}-{h_std_hi} [STRETCH]")

    # P4-C band-wise hue (GORE_STRICT)
    for band_name, h_band_mean in [("A", h_A_mean), ("B", h_B_mean), ("C", h_C_mean)]:
        target, (lo, hi) = GORE["band_hue"][band_name]
        p4.add(f"h_{band_name}_range", lo <= h_band_mean <= hi, h_band_mean, f"{lo}-{hi}")

    # P4-C hue monotonicity (GORE_STRICT: h_A < h_B < h_C)
    p4.add("hue_monotone_p4", h_A_mean < h_B_mean < h_C_mean, None,
           f"h_A({h_A_mean:.3f}) < h_B({h_B_mean:.3f}) < h_C({h_C_mean:.3f})")

    milestones = {"P1": p1, "P2": p2, "P3": p3, "P4": p4}

    # Print results
    if verbose:
        for ms_name, ms in milestones.items():
            status = "✓ PASS" if ms.passed else "✗ FAIL"
            print(f"\n{'='*60}")
            print(f"{ms.name}: {status}")
            print('='*60)
            for r in ms.results:
                mark = "✓" if r.passed else "✗"
                val_str = f"{r.value:.4f}" if r.value is not None else "N/A"
                print(f"  {mark} {r.name:20s}: {val_str:10s} (expected: {r.expected})")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        all_passed = all(ms.passed for ms in milestones.values())
        for ms_name, ms in milestones.items():
            status = "PASS" if ms.passed else "FAIL"
            n_pass = sum(1 for r in ms.results if r.passed)
            n_total = len(ms.results)
            print(f"  {ms_name}: {status} ({n_pass}/{n_total})")
        print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")

    return milestones


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_terrain_p4.py <image_path>")
        print("Example: python validate_terrain_p4.py examples/output/terrain_csm.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    milestones = validate_p4(image_path)

    # Exit with code based on P4 pass status (focus of this task)
    all_passed = all(ms.passed for ms in milestones.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
