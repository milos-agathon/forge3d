#!/usr/bin/env python3
"""
Terrain Render Validation Script

Implements R0-R8 metrics from the terrain rendering spec.
All metrics are computed on non-water pixels only.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import colorsys


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    value: float
    expected: str
    message: str = ""


@dataclass
class ValidationReport:
    """Complete validation report for an image."""
    image_path: str
    results: List[ValidationResult] = field(default_factory=list)
    
    def add(self, name: str, passed: bool, value: float, expected: str, msg: str = ""):
        self.results.append(ValidationResult(name, passed, value, expected, msg))
    
    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines = [f"\n{'='*60}", f"Validation Report: {self.image_path}", f"{'='*60}"]
        lines.append(f"Overall: {passed}/{total} tests passed\n")
        
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            lines.append(f"{status}: {r.name}")
            lines.append(f"       Value: {r.value:.4f}, Expected: {r.expected}")
            if r.message:
                lines.append(f"       {r.message}")
        
        return "\n".join(lines)


def load_image(path: str) -> np.ndarray:
    """Load image as float32 RGB in [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance L = 0.2126*R + 0.7152*G + 0.0722*B."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def build_nonwater_mask(rgb: np.ndarray, land_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    R0.1: Build non-water mask.
    Water is detected as: (B - max(R,G) > 0.15) & (B > 0.35)
    """
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    water = (B - np.maximum(R, G) > 0.15) & (B > 0.35)
    
    if land_mask is not None:
        nonwater = land_mask & ~water
    else:
        nonwater = ~water
    
    return nonwater


def rgb_to_hsv_vectorized(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB array to HSV arrays (h, s, v) each in [0,1]."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    v = maxc
    
    delta = maxc - minc
    s = np.where(maxc > 0, delta / maxc, 0)
    
    # Hue calculation
    h = np.zeros_like(maxc)
    
    # When delta > 0
    mask_r = (maxc == R) & (delta > 0)
    mask_g = (maxc == G) & (delta > 0)
    mask_b = (maxc == B) & (delta > 0)
    
    h[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2
    h[mask_b] = ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4
    
    h = h / 6.0  # Normalize to [0, 1]
    h = h % 1.0  # Handle negative values
    
    return h, s, v


def compute_gradient_magnitude(L: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude of luminance."""
    gx = np.zeros_like(L)
    gy = np.zeros_like(L)
    gx[:, :-1] = L[:, 1:] - L[:, :-1]
    gy[:-1, :] = L[1:, :] - L[:-1, :]
    return np.sqrt(gx**2 + gy**2)


def validate_r1_luminance(L: np.ndarray, mask: np.ndarray, report: ValidationReport):
    """R1: Luminance & tone curve validation."""
    L_masked = L[mask]
    
    # Reference quantiles
    ref_q = {1: 0.077, 5: 0.099, 25: 0.205, 50: 0.356, 75: 0.498, 95: 0.650, 99: 0.772}
    
    # R1.1 - Luminance bounds
    L_min, L_max = L_masked.min(), L_masked.max()
    report.add("R1.1a L_min", 0.04 <= L_min <= 0.08, L_min, "[0.04, 0.08]")
    report.add("R1.1b L_max", 0.80 <= L_max <= 0.90, L_max, "[0.80, 0.90]")
    
    # R1.2 - Quantile matching
    for q_pct, ref_val in ref_q.items():
        q_val = np.percentile(L_masked, q_pct)
        diff = abs(q_val - ref_val)
        report.add(f"R1.2 q{q_pct:02d}", diff <= 0.030, q_val, f"{ref_val}±0.030")
    
    # R1.3 - Dynamic range ratio
    q_low = np.quantile(L_masked, 0.10)
    q_high = np.quantile(L_masked, 0.90)
    dark = L_masked[L_masked <= q_low]
    bright = L_masked[L_masked >= q_high]
    ratio = bright.mean() / dark.mean() if dark.mean() > 0 else 999
    report.add("R1.3 dynamic_ratio", 5.5 <= ratio <= 8.0, ratio, "[5.5, 8.0]")
    
    # R1.4 - No crushed/blown
    crushed_pct = 100 * np.sum(L_masked < 0.04) / len(L_masked)
    blown_pct = 100 * np.sum(L_masked > 0.85) / len(L_masked)
    report.add("R1.4a crushed<0.04", crushed_pct <= 0.5, crushed_pct, "≤0.5%")
    report.add("R1.4b blown>0.85", blown_pct <= 0.5, blown_pct, "≤0.5%")


def validate_r2_gradient(L: np.ndarray, mask: np.ndarray, report: ValidationReport):
    """R2: Local contrast & terrain detail validation."""
    g = compute_gradient_magnitude(L)
    g_masked = g[mask]
    
    g_mean = g_masked.mean()
    g_median = np.median(g_masked)
    g_q90 = np.percentile(g_masked, 90)
    g_q99 = np.percentile(g_masked, 99)
    
    report.add("R2.1a grad_mean", 0.0433 <= g_mean <= 0.0586, g_mean, "[0.0433, 0.0586]")
    report.add("R2.1b grad_median", 0.028 <= g_median <= 0.040, g_median, "[0.028, 0.040]")
    report.add("R2.1c grad_q90", 0.10 <= g_q90 <= 0.14, g_q90, "[0.10, 0.14]")
    report.add("R2.1d grad_q99", 0.23 <= g_q99 <= 0.30, g_q99, "[0.23, 0.30]")


def validate_r3_color(rgb: np.ndarray, mask: np.ndarray, report: ValidationReport):
    """R3: Global color statistics validation."""
    h, s, v = rgb_to_hsv_vectorized(rgb)
    h_masked = h[mask]
    s_masked = s[mask]
    
    h_mean = h_masked.mean()
    h_std = h_masked.std()
    s_mean = s_masked.mean()
    s_std = s_masked.std()
    
    # R3.1 - Hue band
    report.add("R3.1a hue_mean", 0.055 <= h_mean <= 0.085, h_mean, "[0.055, 0.085]")
    report.add("R3.1b hue_std", 0.025 <= h_std <= 0.060, h_std, "[0.025, 0.060]")
    
    # Purple-magenta check
    purple_mask = (h_masked >= 0.70) | (h_masked <= 0.05)  # Wrap around
    purple_mask = ((h_masked >= 0.70) & (h_masked <= 0.95)) & (s_masked > 0.20)
    purple_pct = 100 * np.sum(purple_mask) / len(h_masked)
    report.add("R3.1c purple_pct", purple_pct <= 1.0, purple_pct, "≤1%")
    
    # R3.2 - Saturation
    report.add("R3.2a sat_mean", 0.34 <= s_mean <= 0.40, s_mean, "[0.34, 0.40]")
    report.add("R3.2b sat_std", 0.09 <= s_std <= 0.13, s_std, "[0.09, 0.13]")


def validate_r4_color_bands(rgb: np.ndarray, L: np.ndarray, mask: np.ndarray, report: ValidationReport):
    """R4: Color by luminance band validation."""
    h, s, v = rgb_to_hsv_vectorized(rgb)
    
    # Define bands
    bands = [
        (1, 0.05, 0.20, (0.40, 0.52), (0.02, 0.06)),  # shadows
        (2, 0.20, 0.50, (0.34, 0.43), (0.05, 0.09)),  # midtones
        (3, 0.50, 0.80, (0.20, 0.30), (0.09, 0.13)),  # highlights
    ]
    
    prev_h_mean = -1
    for band_num, L_lo, L_hi, s_range, h_range in bands:
        band_mask = mask & (L >= L_lo) & (L < L_hi)
        if np.sum(band_mask) < 100:
            report.add(f"R4.1 band{band_num}_sat", False, 0, "insufficient pixels")
            continue
        
        s_band = s[band_mask]
        h_band = h[band_mask]
        s_mean = s_band.mean()
        h_mean = h_band.mean()
        
        # R4.1 - Per-band saturation
        report.add(f"R4.1 band{band_num}_sat", 
                   s_range[0] <= s_mean <= s_range[1], s_mean, f"[{s_range[0]}, {s_range[1]}]")
        
        # R4.2 - Per-band hue
        report.add(f"R4.2 band{band_num}_hue",
                   h_range[0] <= h_mean <= h_range[1], h_mean, f"[{h_range[0]}, {h_range[1]}]")
        
        # Monotonic hue check
        if prev_h_mean >= 0:
            report.add(f"R4.2 hue_mono_{band_num-1}->{band_num}",
                       h_mean > prev_h_mean, h_mean, f">{prev_h_mean:.4f}")
        prev_h_mean = h_mean


def validate_image(image_path: str, reference_mode: bool = False) -> ValidationReport:
    """Run all validations on an image."""
    report = ValidationReport(image_path)
    
    rgb = load_image(image_path)
    L = compute_luminance(rgb)
    mask = build_nonwater_mask(rgb)
    
    water_pct = 100 * (1 - np.sum(mask) / mask.size)
    print(f"Water pixels excluded: {water_pct:.1f}%")
    
    validate_r1_luminance(L, mask, report)
    validate_r2_gradient(L, mask, report)
    validate_r3_color(rgb, mask, report)
    validate_r4_color_bands(rgb, L, mask, report)
    
    return report


def compare_images(ref_path: str, target_path: str):
    """Compare reference and target images."""
    print(f"\n{'='*60}")
    print("REFERENCE IMAGE ANALYSIS")
    print(f"{'='*60}")
    ref_report = validate_image(ref_path, reference_mode=True)
    print(ref_report.summary())
    
    print(f"\n{'='*60}")
    print("TARGET IMAGE ANALYSIS")
    print(f"{'='*60}")
    target_report = validate_image(target_path)
    print(target_report.summary())
    
    # Summary comparison
    ref_passed = sum(1 for r in ref_report.results if r.passed)
    target_passed = sum(1 for r in target_report.results if r.passed)
    total = len(ref_report.results)
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Reference: {ref_passed}/{total} tests passed")
    print(f"Target:    {target_passed}/{total} tests passed")
    
    return ref_report, target_report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_terrain.py <image_path> [reference_path]")
        print("       python validate_terrain.py <target_path> --compare <reference_path>")
        sys.exit(1)
    
    if "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        target_path = sys.argv[1]
        ref_path = sys.argv[idx + 1]
        compare_images(ref_path, target_path)
    elif len(sys.argv) == 3:
        compare_images(sys.argv[2], sys.argv[1])
    else:
        report = validate_image(sys.argv[1])
        print(report.summary())
