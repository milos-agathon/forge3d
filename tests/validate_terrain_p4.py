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

    # Reference quantiles (R1)
    ref_quantiles = {
        "q01": 0.077, "q05": 0.100, "q25": 0.209, "q50": 0.348,
        "q75": 0.494, "q95": 0.648, "q99": 0.769
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # P1: Baseline sanity checks
    # ═══════════════════════════════════════════════════════════════════════════
    p1 = MilestoneResults("P1-Baseline")

    # Basic luminance bounds (R1)
    L_min, L_max = float(L.min()), float(L.max())
    p1.add("L_min", 0.04 <= L_min <= 0.08, L_min, "0.04-0.08")
    p1.add("L_max", 0.80 <= L_max <= 0.90, L_max, "0.80-0.90")

    # Crushed/blown (R1)
    crushed = float((L < 0.04).mean())
    blown = float((L > 0.85).mean())
    p1.add("crushed", crushed < 0.005, crushed, "<0.5%")
    p1.add("blown", blown < 0.005, blown, "<0.5%")

    # ═══════════════════════════════════════════════════════════════════════════
    # P2: Luminance & Dynamic Range Lock
    # ═══════════════════════════════════════════════════════════════════════════
    p2 = MilestoneResults("P2-Luminance")

    # P2-L quantiles (±0.030)
    for r in validate_luminance_quantiles(luminance, terrain_mask, ref_quantiles, 0.030):
        p2.results.append(r)

    # Dynamic ratio (P2-L)
    q10, q90_val = np.quantile(L, 0.10), np.quantile(L, 0.90)
    mean_hi = float(L[L >= q90_val].mean())
    mean_lo = float(L[L <= q10].mean()) if (L <= q10).sum() > 0 else 0.001
    ratio = mean_hi / max(mean_lo, 0.001)
    p2.add("dynamic_ratio", 5.5 <= ratio <= 8.0, ratio, "5.5-8.0")

    # Stricter crushed/blown (P2-L)
    p2.add("crushed_strict", (L < 0.05).mean() < 0.001, float((L < 0.05).mean()), "<0.1%")
    p2.add("blown_strict", (L > 0.85).mean() < 0.001, float((L > 0.85).mean()), "<0.1%")

    # P2-G gradient stats
    g_mean, g_median = float(G.mean()), float(np.median(G))
    g_q90, g_q99 = float(np.quantile(G, 0.90)), float(np.quantile(G, 0.99))
    p2.add("g_mean", 0.045 <= g_mean <= 0.055, g_mean, "0.045-0.055")
    p2.add("g_median", 0.027 <= g_median <= 0.040, g_median, "0.027-0.040")
    p2.add("g_q90", 0.10 <= g_q90 <= 0.14, g_q90, "0.10-0.14")
    p2.add("g_q99", 0.23 <= g_q99 <= 0.30, g_q99, "0.23-0.30")

    # P2-Band occupancy
    band_A = (L >= 0.05) & (L < 0.20)  # Shadows
    band_B = (L >= 0.20) & (L < 0.50)  # Midtones
    band_C = (L >= 0.50) & (L < 0.80)  # Highlights
    pA, pB, pC = band_A.mean(), band_B.mean(), band_C.mean()
    p2.add("pA", 0.18 <= pA <= 0.30, float(pA), "0.18-0.30")
    p2.add("pB", 0.45 <= pB <= 0.60, float(pB), "0.45-0.60")
    p2.add("pC", 0.10 <= pC <= 0.25, float(pC), "0.10-0.25")

    # ═══════════════════════════════════════════════════════════════════════════
    # P3: Gradient Structure & Global Hue
    # ═══════════════════════════════════════════════════════════════════════════
    p3 = MilestoneResults("P3-Structure")

    # P3-L tighter dynamic ratio
    p3.add("ratio_tight", 6.4 <= ratio <= 7.2, ratio, "6.4-7.2")

    # P3-G gradients (tighter)
    p3.add("g_mean_p3", 0.047 <= g_mean <= 0.053, g_mean, "0.047-0.053")
    p3.add("g_median_p3", 0.029 <= g_median <= 0.035, g_median, "0.029-0.035")
    p3.add("g_q90_p3", 0.112 <= g_q90 <= 0.125, g_q90, "0.112-0.125")
    p3.add("g_q99_p3", 0.245 <= g_q99 <= 0.300, g_q99, "0.245-0.300")

    # P3-G band-wise gradients
    G_A = gradient[terrain_mask][band_A]
    G_B = gradient[terrain_mask][band_B]
    G_C = gradient[terrain_mask][band_C]
    g_A_mean = float(G_A.mean()) if len(G_A) > 0 else 0
    g_B_mean = float(G_B.mean()) if len(G_B) > 0 else 0
    g_C_mean = float(G_C.mean()) if len(G_C) > 0 else 0
    p3.add("g_A_mean", 0.044 <= g_A_mean <= 0.055, g_A_mean, "0.044-0.055")
    p3.add("g_B_mean", 0.045 <= g_B_mean <= 0.055, g_B_mean, "0.045-0.055")
    p3.add("g_C_mean", 0.035 <= g_C_mean <= 0.050, g_C_mean, "0.035-0.050")
    p3.add("g_C_le_g_B", g_C_mean <= g_B_mean + 0.005, g_C_mean, f"<= {g_B_mean + 0.005:.3f}")

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

    # P3-C global hue/saturation
    h_mean, h_std = float(H.mean()), float(H.std())
    s_mean, s_std = float(S.mean()), float(S.std())
    p3.add("h_mean", 0.075 <= h_mean <= 0.095, h_mean, "0.075-0.095")
    p3.add("h_std", 0.055 <= h_std <= 0.120, h_std, "0.055-0.120")
    p3.add("s_mean", 0.36 <= s_mean <= 0.41, s_mean, "0.36-0.41")
    p3.add("s_std", 0.10 <= s_std <= 0.15, s_std, "0.10-0.15")

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

    # P4-L band occupancy (strict)
    p4.add("pA_p4", 0.22 <= pA <= 0.26, float(pA), "0.22-0.26")
    p4.add("pB_p4", 0.49 <= pB <= 0.57, float(pB), "0.49-0.57")
    p4.add("pC_p4", 0.20 <= pC <= 0.26, float(pC), "0.20-0.26")

    # P4-L mid quantiles (strict)
    q25 = float(np.quantile(L, 0.25))
    q50 = float(np.quantile(L, 0.50))
    q75 = float(np.quantile(L, 0.75))
    p4.add("q25_p4", 0.20 <= q25 <= 0.22, q25, "0.20-0.22")
    p4.add("q50_p4", 0.34 <= q50 <= 0.37, q50, "0.34-0.37")
    p4.add("q75_p4", 0.48 <= q75 <= 0.51, q75, "0.48-0.51")

    # P4-G valley gradients
    g_A_q90 = float(np.quantile(G_A, 0.90)) if len(G_A) > 0 else 0
    p4.add("g_A_mean_p4", 0.024 <= g_A_mean <= 0.034, g_A_mean, "0.024-0.034")
    p4.add("g_A_q90", g_A_q90 >= 0.060, g_A_q90, ">=0.060")
    p4.add("g_A_le_g_B", g_A_mean <= g_B_mean, g_A_mean, f"<= {g_B_mean:.3f}")
    p4.add("g_B_le_g_C_plus", g_B_mean <= g_C_mean + 0.010, g_B_mean, f"<= {g_C_mean + 0.010:.3f}")

    # P4-C hue variation
    p4.add("h_mean_p4", 0.075 <= h_mean <= 0.095, h_mean, "0.075-0.095")
    p4.add("h_std_p4", 0.060 <= h_std <= 0.130, h_std, "0.060-0.130")

    # P4-C band B (midtones) hue
    h_B_std = float(H_B.std()) if len(H_B) > 0 else 0
    p4.add("h_B_mean", 0.08 <= h_B_mean <= 0.11, h_B_mean, "0.08-0.11")
    p4.add("h_B_std", 0.08 <= h_B_std <= 0.15, h_B_std, "0.08-0.15")

    # P4-C hue progression
    p4.add("h_A_range", 0.02 <= h_A_mean <= 0.07, h_A_mean, "0.02-0.07")
    p4.add("h_B_range", 0.06 <= h_B_mean <= 0.11, h_B_mean, "0.06-0.11")
    p4.add("h_C_range", 0.09 <= h_C_mean <= 0.13, h_C_mean, "0.09-0.13")
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
