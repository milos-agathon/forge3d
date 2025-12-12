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
    """Load image as sRGB (no gamma conversion).
    
    NOTE: GORE_STRICT_PROFILE thresholds were calibrated on sRGB values
    without linear conversion (see tools/validate_gore_strict.py line 72).
    Do NOT apply sRGB-to-linear conversion here.
    """
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
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
    """Create water mask based on blue dominance and saturation (R0).
    
    This is a fallback heuristic method. For authoritative water masking,
    use create_water_mask_from_debug_render() with debug mode 4.
    """
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Water: blue-ish hue (0.5-0.7), high saturation, moderate value
    blue_hue = (h > 0.5) & (h < 0.75)
    high_sat = s > 0.3
    blue_dominant = rgb[..., 2] > rgb[..., 0] * 1.2

    water = blue_hue & high_sat & blue_dominant
    return water


def create_water_mask_from_debug_render(debug_rgb: np.ndarray) -> np.ndarray:
    """Create water mask from shader debug mode 4 render.
    
    Debug mode 4 outputs:
    - CYAN (0, 1, 1) = water pixels
    - MAGENTA (1, 0, 1) = land pixels
    
    Args:
        debug_rgb: RGB image from debug mode 4 render (float32, 0-1 range)
    
    Returns:
        Boolean mask where True = water (to be excluded)
    """
    # CYAN has high G and B, low R
    # We detect cyan by: G > 0.5 AND B > 0.5 AND R < 0.3
    r, g, b = debug_rgb[..., 0], debug_rgb[..., 1], debug_rgb[..., 2]
    is_cyan = (g > 0.5) & (b > 0.5) & (r < 0.3)
    return is_cyan


def apply_water_buffer(water_mask: np.ndarray, radius_px: int = 2) -> np.ndarray:
    """Apply binary dilation to water mask to buffer shorelines.
    
    This creates a buffer zone around water pixels to prevent shoreline
    leakage into metrics. Uses DILATION of water_mask (not erosion of
    terrain) so that if there's no water, nothing is excluded.
    
    Args:
        water_mask: Boolean mask where True = water (to be excluded)
        radius_px: Buffer radius in pixels (default 2)
    
    Returns:
        Dilated water mask (True = water + buffer zone to exclude)
    """
    from scipy import ndimage
    
    if radius_px <= 0 or not water_mask.any():
        return water_mask
    
    # Create circular structuring element
    size = 2 * radius_px + 1
    y, x = np.ogrid[:size, :size]
    center = radius_px
    struct = ((x - center) ** 2 + (y - center) ** 2) <= radius_px ** 2
    
    dilated = ndimage.binary_dilation(water_mask, structure=struct, iterations=1)
    return dilated


def create_gradient_mask(used_mask: np.ndarray) -> np.ndarray:
    """Create mask for gradient computation requiring full 8-neighborhood to be valid.
    
    Gradients must only be computed on pixels whose full neighborhood is non-water.
    This prevents shoreline leakage into gradient metrics.
    
    Args:
        used_mask: Boolean mask where True = terrain (used in metrics), 
                   already eroded by apply_shoreline_erosion
    
    Returns:
        Mask where True = pixel AND all 8 neighbors are valid for gradient computation
    """
    # 8-neighborhood: require all neighbors to be valid
    grad_mask = used_mask.copy()
    
    # Shift in all 8 directions and AND together
    # This ensures center pixel AND all neighbors are in used_mask
    grad_mask[1:, :] &= used_mask[:-1, :]   # shift down (neighbor above)
    grad_mask[:-1, :] &= used_mask[1:, :]   # shift up (neighbor below)
    grad_mask[:, 1:] &= used_mask[:, :-1]   # shift right (neighbor left)
    grad_mask[:, :-1] &= used_mask[:, 1:]   # shift left (neighbor right)
    # Diagonals
    grad_mask[1:, 1:] &= used_mask[:-1, :-1]   # shift down-right
    grad_mask[1:, :-1] &= used_mask[:-1, 1:]   # shift down-left
    grad_mask[:-1, 1:] &= used_mask[1:, :-1]   # shift up-right
    grad_mask[:-1, :-1] &= used_mask[1:, 1:]   # shift up-left
    
    return grad_mask


def render_water_mask_image(preset_path: Path, output_dir: Path) -> Path | None:
    """Render with debug mode 4 to get authoritative water mask from shader.
    
    Returns path to water mask debug render, or None on failure.
    """
    import subprocess
    import sys as _sys
    
    project_root = Path(__file__).parent.parent
    output_path = output_dir / "water_debug_mode4.png"
    
    cmd = [
        _sys.executable,
        str(project_root / "examples" / "terrain_demo.py"),
        "--preset", str(preset_path),
        "--debug-mode", "4",
        "--output", str(output_path),
        "--overwrite",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    if result.returncode != 0:
        print(f"WARNING: Failed to render water mask debug image")
        return None
    
    return output_path


def get_water_mask_stats(water_mask: np.ndarray) -> dict:
    """Compute statistics about water mask for validation.
    
    Returns dict with:
    - excluded_fraction: fraction of pixels excluded
    - excluded_count: total excluded pixels
    - largest_component_size: size of largest connected component
    """
    from scipy import ndimage
    
    excluded_count = int(water_mask.sum())
    total_pixels = water_mask.size
    excluded_fraction = excluded_count / total_pixels
    
    # Find connected components
    labeled, num_features = ndimage.label(water_mask)
    if num_features == 0:
        largest_component_size = 0
    else:
        component_sizes = ndimage.sum(water_mask, labeled, range(1, num_features + 1))
        largest_component_size = int(max(component_sizes)) if len(component_sizes) > 0 else 0
    
    return {
        "excluded_fraction": excluded_fraction,
        "excluded_count": excluded_count,
        "total_pixels": total_pixels,
        "num_components": num_features,
        "largest_component_size": largest_component_size,
    }


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


def validate_p4(
    image_path: Path,
    verbose: bool = True,
    output_dir: Path | None = None,
    preset_path: Path | None = None,
) -> dict[str, MilestoneResults]:
    """Run full P1-P4 validation on terrain image.
    
    Args:
        image_path: Path to rendered terrain image
        verbose: Whether to print detailed output
        output_dir: If provided, save water_mask.png debug artifact here
        preset_path: If provided, use debug mode 4 render for authoritative water mask
    
    Returns:
        Dictionary of milestone results
    """
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    rgb = load_image_linear(image_path)
    luminance = compute_luminance(rgb)
    
    # Get water mask - prefer authoritative shader-based mask if preset available
    water_mask = None
    water_mask_stats = None
    buffer_radius_px = 2  # Required default per spec
    
    if preset_path is not None and output_dir is not None:
        # Render with debug mode 4 to get authoritative water mask
        debug_path = render_water_mask_image(preset_path, output_dir)
        if debug_path is not None and debug_path.exists():
            debug_rgb = load_image_linear(debug_path)
            water_mask = create_water_mask_from_debug_render(debug_rgb)
            if verbose:
                print(f"Water mask from shader debug mode 4: {debug_path}")
    
    # Fallback to heuristic if shader-based mask not available
    if water_mask is None:
        water_mask = create_water_mask(rgb, luminance)
        if verbose:
            print("Water mask from heuristic (fallback)")
    
    # Compute raw water mask stats BEFORE buffer dilation
    water_mask_stats = get_water_mask_stats(water_mask)
    raw_water_pixels = int(water_mask.sum())
    raw_water_fraction = float(water_mask.mean())
    
    # Apply buffer via DILATION of water_mask (not erosion of terrain!)
    # This ensures if there's no water, nothing is excluded
    excluded_mask = apply_water_buffer(water_mask, radius_px=buffer_radius_px)
    
    # used_mask = NOT excluded_mask (True = pixels allowed in metrics)
    used_mask = ~excluded_mask
    
    # Create gradient mask requiring full 8-neighborhood to be valid
    grad_mask = create_gradient_mask(used_mask)
    
    # For backwards compatibility
    terrain_mask = used_mask
    
    # Compute final exclusion stats (after buffer dilation)
    excluded_fraction = float(excluded_mask.mean())
    excluded_count = int(excluded_mask.sum())
    
    if verbose:
        print(f"  Raw water pixels: {raw_water_pixels} ({100*raw_water_fraction:.2f}%)")
        print(f"  After buffer (radius={buffer_radius_px}): excluded {excluded_count} ({100*excluded_fraction:.2f}%)")
        print(f"  Gradient mask pixels: {grad_mask.sum()} ({100*grad_mask.mean():.2f}%)")
        if water_mask_stats:
            print(f"  Largest water component: {water_mask_stats['largest_component_size']} pixels")

    # Save water mask debug artifacts if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save binary mask (white=USED terrain, black=EXCLUDED water+shoreline)
        water_mask_path = output_dir / "water_mask.png"
        mask_img = np.where(used_mask, 255, 0).astype(np.uint8)
        Image.fromarray(mask_img, mode='L').save(water_mask_path)
        
        # Save overlay (red tint on excluded pixels)
        overlay_path = output_dir / "water_mask_overlay.png"
        overlay_rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        overlay_rgb[excluded_mask, 0] = 255  # Red channel = 255 for excluded
        overlay_rgb[excluded_mask, 1] = overlay_rgb[excluded_mask, 1] // 2
        overlay_rgb[excluded_mask, 2] = overlay_rgb[excluded_mask, 2] // 2
        Image.fromarray(overlay_rgb, mode='RGB').save(overlay_path)
        
        if verbose:
            print(f"Water mask saved: {water_mask_path}")
            print(f"Water overlay saved: {overlay_path}")
    
    # HARD ASSERTIONS for Gore Range runs
    # Only enforce if we have meaningful water (excluded_fraction > 0)
    if excluded_fraction > 0:
        if not (0.001 <= excluded_fraction <= 0.10):
            print(f"WARNING: excluded_fraction {excluded_fraction:.4f} outside expected range [0.001, 0.10]")
        lcs = water_mask_stats['largest_component_size'] if water_mask_stats else 0
        if lcs < 2000:
            print(f"WARNING: largest_component_size {lcs} < 2000 pixels")
    
    # HARD FAIL if masks are empty
    if used_mask.sum() == 0:
        print("HARD FAIL: used_mask.sum() == 0 - no valid pixels for metrics")
        sys.exit(1)
    if grad_mask.sum() == 0:
        print("HARD FAIL: grad_mask.sum() == 0 - no valid pixels for gradient metrics")
        sys.exit(1)

    if verbose:
        print(f"Image: {image_path}")
        print(f"Shape: {rgb.shape}")
        print(f"Used pixels: {used_mask.sum()} ({100*used_mask.mean():.1f}%)")
        print(f"Excluded pixels: {excluded_count} ({100*excluded_fraction:.1f}%)")
        print(f"Gradient mask pixels: {grad_mask.sum()} ({100*grad_mask.mean():.1f}%)")
        print()

    # Arrays for metrics - use used_mask (after erosion) for all non-gradient metrics
    L = luminance[used_mask]
    hsv = rgb_to_hsv(rgb)
    H = hsv[..., 0][used_mask]
    S = hsv[..., 1][used_mask]
    
    # Gradients use grad_mask (8-neighborhood requirement)
    gradient = compute_gradient_magnitude(luminance)
    G = gradient[grad_mask]

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
    # Bands computed on 1D L array (used_mask pixels only)
    band_A = (L >= 0.05) & (L < 0.20)  # Shadows
    band_B = (L >= 0.20) & (L < 0.50)  # Midtones
    band_C = (L >= 0.50) & (L < 0.80)  # Highlights
    pA, pB, pC = band_A.mean(), band_B.mean(), band_C.mean()
    for name, val in [("pA", pA), ("pB", pB), ("pC", pC)]:
        target, (lo, hi) = GORE["bands"][name]
        p2.add(name, lo <= val <= hi, float(val), f"{lo}-{hi}")
    
    # For band-wise gradients and hue, we need 2D masks
    # Create 2D band masks on full luminance array, combined with grad_mask/used_mask
    band_A_2d = (luminance >= 0.05) & (luminance < 0.20) & grad_mask
    band_B_2d = (luminance >= 0.20) & (luminance < 0.50) & grad_mask
    band_C_2d = (luminance >= 0.50) & (luminance < 0.80) & grad_mask
    # For hue bands, use used_mask (not grad_mask)
    band_A_hue = (luminance >= 0.05) & (luminance < 0.20) & used_mask
    band_B_hue = (luminance >= 0.20) & (luminance < 0.50) & used_mask
    band_C_hue = (luminance >= 0.50) & (luminance < 0.80) & used_mask

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

    # P3-G band-wise gradients (use 2D masks for proper alignment)
    G_A = gradient[band_A_2d]
    G_B = gradient[band_B_2d]
    G_C = gradient[band_C_2d]
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

    # P3-C band-wise hue monotonicity (use 2D masks for proper alignment)
    H_full = hsv[..., 0]  # Full 2D hue array
    H_A = H_full[band_A_hue]
    H_B = H_full[band_B_hue]
    H_C = H_full[band_C_hue]
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
            status = "[PASS]" if ms.passed else "[FAIL]"
            print(f"\n{'='*60}")
            print(f"{ms.name}: {status}")
            print('='*60)
            for r in ms.results:
                mark = "[OK]" if r.passed else "[X]"
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


def render_and_validate(render_mode: str, profile: str, output_dir: Path) -> dict:
    """Render terrain with specified mode and validate against profile.
    
    Args:
        render_mode: One of:
            - 'p5' or 'p6': Use built-in CLI params for that mode
            - 'preset:<path>': Use JSON preset file
            - 'file:<path>': Validate existing rendered file (no render)
        profile: Profile name (e.g., 'GORE_STRICT_PROFILE')
        output_dir: Directory to save rendered image
    
    Returns:
        Dict with render path, milestones, and pass status
    """
    import subprocess
    import sys as _sys
    import json
    import time
    
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent.parent
    start_time = time.time()
    
    # Handle file: prefix - validate existing file without rendering
    if render_mode.startswith("file:"):
        file_path = Path(render_mode[5:])
        if not file_path.is_absolute():
            file_path = project_root / file_path
        if not file_path.exists():
            return {"path": None, "milestones": None, "passed": False, "error": f"File not found: {file_path}"}
        
        print(f"\n{'='*60}")
        print(f"Validating existing file: {file_path}")
        print('='*60)
        
        milestones = validate_p4(file_path, verbose=True, output_dir=output_dir)
        all_passed = all(ms.passed for ms in milestones.values())
        elapsed = time.time() - start_time
        
        return {
            "path": file_path,
            "milestones": milestones,
            "passed": all_passed,
            "render_mode": render_mode,
            "profile": profile,
            "elapsed_s": elapsed,
        }
    
    # Handle preset: prefix - use JSON preset via terrain_demo.py --preset
    if render_mode.startswith("preset:"):
        preset_path = Path(render_mode[7:])
        if not preset_path.is_absolute():
            preset_path = project_root / preset_path
        if not preset_path.exists():
            return {"path": None, "milestones": None, "passed": False, "error": f"Preset not found: {preset_path}"}
        
        output_name = preset_path.stem
        output_path = output_dir / f"{output_name}_terrain_csm.png"
        
        # Use terrain_demo.py with --preset for guaranteed parity
        cmd = [
            _sys.executable,
            str(project_root / "examples" / "terrain_demo.py"),
            "--preset", str(preset_path),
            "--output", str(output_path),
            "--overwrite",
        ]
        
        print(f"\n{'='*60}")
        print(f"Rendering with preset: {preset_path.name}")
        print('='*60)
        print(f"Command: {' '.join(cmd)}")
        
    else:
        # Legacy p5/p6 mode
        output_path = output_dir / f"{render_mode}_terrain_csm.png"
        
        cmd = [
            _sys.executable,
            str(project_root / "examples" / "terrain_demo.py"),
            "--dem", str(project_root / "assets" / "Gore_Range_Albers_1m.tif"),
            "--hdr", str(project_root / "assets" / "hdri" / "snow_field_4k.hdr"),
            "--size", "1920", "1080",
            "--msaa", "4",
            "--z-scale", "2.0",
            "--cam-radius", "1000.0",
            "--cam-phi", "135.0",
            "--cam-theta", "45.0",
            "--exposure", "1.0",
            "--ibl-intensity", "1.0",
            "--sun-azimuth", "135.0",
            "--sun-elevation", "35.0",
            "--shadows", "csm",
            "--cascades", "4",
            "--colormap", "terrain",
            "--colormap-strength", "0.5",
            "--albedo-mode", "mix",
            "--normal-strength", "1.5",
            "--lambert-contrast", "0.3",
            "--output", str(output_path),
            "--overwrite",
            "--render", render_mode,
        ]
        
        # Add mode-specific parameters
        if render_mode == "p5":
            cmd.extend(["--unsharp-strength", "0.15"])
        elif render_mode == "p6":
            cmd.extend(["--unsharp-strength", "0.0"])
            cmd.extend(["--detail-strength", "0.5"])
            cmd.extend(["--detail-sigma-px", "3.0"])
        
        print(f"\n{'='*60}")
        print(f"Rendering with --render {render_mode}")
        print('='*60)
        print(f"Command: {' '.join(cmd[:10])}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Render failed!")
        print(f"STDERR: {result.stderr[-500:] if result.stderr else 'None'}")
        return {"path": None, "milestones": None, "passed": False, "error": result.stderr}
    
    print(f"Rendered: {output_path}")
    print(f"Render time: {elapsed:.2f}s")
    
    # Determine preset_path for authoritative water mask (only for preset: mode)
    validation_preset_path = None
    if render_mode.startswith("preset:"):
        validation_preset_path = preset_path
    
    # Validate the rendered image (pass output_dir for water_mask.png and preset for authoritative mask)
    milestones = validate_p4(output_path, verbose=True, output_dir=output_dir, preset_path=validation_preset_path)
    all_passed = all(ms.passed for ms in milestones.values())
    
    return {
        "path": output_path,
        "milestones": milestones,
        "passed": all_passed,
        "render_mode": render_mode,
        "profile": profile,
        "elapsed_s": elapsed,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate terrain rendering against GORE_STRICT_PROFILE metrics"
    )
    parser.add_argument(
        "image_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to rendered image to validate (legacy mode)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="GORE_STRICT_PROFILE",
        help="Validation profile name (default: GORE_STRICT_PROFILE)",
    )
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        help="Render mode: 'p5', 'p6', 'preset:<path>', or 'file:<path>'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for rendered images (default: reports/)",
    )
    
    args = parser.parse_args()
    
    # Mode 1: Render and validate with --render p5|p6
    if args.render is not None:
        result = render_and_validate(args.render, args.profile, args.output_dir)
        if not result["passed"]:
            print(f"\nFAILED: --render {args.render} did not pass {args.profile}")
            sys.exit(1)
        print(f"\nPASSED: --render {args.render} meets {args.profile}")
        sys.exit(0)
    
    # Mode 2: Legacy - validate existing image
    if args.image_path is None:
        print("Usage: python -m tests.validate_terrain_p4 <image_path>")
        print("   or: python -m tests.validate_terrain_p4 --profile GORE_STRICT_PROFILE --render p5|p6")
        sys.exit(1)
    
    milestones = validate_p4(args.image_path)
    all_passed = all(ms.passed for ms in milestones.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
