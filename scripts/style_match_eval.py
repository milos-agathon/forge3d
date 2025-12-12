#!/usr/bin/env python3
"""Style match evaluation script for terrain rendering.

Computes metrics comparing candidate render against reference image.
Metrics: SSIM, ΔE (CIELAB), luminance percentiles, edge strength, speckle detection.

ROIs avoid the top-right uniform region per specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ROI definitions (x=[x_min, x_max], y=[y_min, y_max])
# Scaled for reference 1920x1080 coordinates (will be scaled if image differs)
ROIS_1920 = {
    "ROI_A": {"x": (140, 820), "y": (260, 980), "desc": "left/central shadowed ridges"},
    "ROI_B": {"x": (760, 1460), "y": (220, 760), "desc": "central ridge structure"},
    "ROI_C": {"x": (1180, 1880), "y": (620, 1060), "desc": "lower-right textured slopes"},
}

# Exclusion zone for full-frame metrics (top-right uniform region)
EXCLUSION_1920 = {"x": (1260, 1920), "y": (0, 460)}


def scale_roi(roi: dict, src_size: tuple[int, int], dst_size: tuple[int, int]) -> dict:
    """Scale ROI coordinates from src_size to dst_size."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return {
        "x": (int(roi["x"][0] * sx), int(roi["x"][1] * sx)),
        "y": (int(roi["y"][0] * sy), int(roi["y"][1] * sy)),
        "desc": roi.get("desc", ""),
    }


def get_scaled_rois(img_shape: tuple) -> tuple[dict, dict]:
    """Get ROIs and exclusion zone scaled to actual image dimensions."""
    h, w = img_shape[:2]
    src = (1920, 1080)
    dst = (w, h)
    
    rois = {k: scale_roi(v, src, dst) for k, v in ROIS_1920.items()}
    exclusion = scale_roi(EXCLUSION_1920, src, dst)
    
    return rois, exclusion


def load_image(path: Path) -> np.ndarray:
    """Load image as float32 RGB in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to luminance (Rec. 709)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to CIELAB (D65 illuminant)."""
    # sRGB to linear
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    
    # Linear RGB to XYZ (D65)
    xyz = np.zeros_like(linear)
    xyz[..., 0] = linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
    xyz[..., 1] = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    xyz[..., 2] = linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041
    
    # Normalize by D65 white point
    xyz[..., 0] /= 0.95047
    xyz[..., 1] /= 1.0
    xyz[..., 2] /= 1.08883
    
    # XYZ to Lab
    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3, t ** (1/3), t / (3 * delta**2) + 4/29)
    
    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    
    lab = np.zeros_like(rgb)
    lab[..., 0] = 116 * fy - 16  # L
    lab[..., 1] = 500 * (fx - fy)  # a
    lab[..., 2] = 200 * (fy - fz)  # b
    
    return lab


def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """Compute SSIM on luminance channel."""
    from scipy.ndimage import uniform_filter
    
    luma1 = rgb_to_luminance(img1)
    luma2 = rgb_to_luminance(img2)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = uniform_filter(luma1, size=window_size, mode='reflect')
    mu2 = uniform_filter(luma2, size=window_size, mode='reflect')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(luma1 ** 2, size=window_size, mode='reflect') - mu1_sq
    sigma2_sq = uniform_filter(luma2 ** 2, size=window_size, mode='reflect') - mu2_sq
    sigma12 = uniform_filter(luma1 * luma2, size=window_size, mode='reflect') - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def compute_delta_e(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Compute ΔE (CIELAB) mean and 95th percentile."""
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    
    return float(np.mean(delta_e)), float(np.percentile(delta_e, 95))


def compute_luminance_percentiles(img: np.ndarray) -> dict[str, float]:
    """Compute luminance percentiles."""
    luma = rgb_to_luminance(img)
    return {
        "p10": float(np.percentile(luma, 10)),
        "p50": float(np.percentile(luma, 50)),
        "p75": float(np.percentile(luma, 75)),
        "p90": float(np.percentile(luma, 90)),
        "p99": float(np.percentile(luma, 99)),
    }


def compute_edge_strength(img: np.ndarray) -> float:
    """Compute mean gradient magnitude (Sobel) on luminance."""
    from scipy.ndimage import sobel
    
    luma = rgb_to_luminance(img)
    gx = sobel(luma, axis=1, mode='reflect')
    gy = sobel(luma, axis=0, mode='reflect')
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    return float(np.mean(grad_mag))


def compute_speckle_fraction(img: np.ndarray, luma_thresh: float = 0.90, 
                             var_thresh: float = 0.005) -> float:
    """Detect bright speckles: fraction of pixels with high luma AND high local variance."""
    from scipy.ndimage import uniform_filter
    
    luma = rgb_to_luminance(img)
    
    # Local variance in 3x3 window
    luma_sq = luma ** 2
    local_mean = uniform_filter(luma, size=3, mode='reflect')
    local_mean_sq = uniform_filter(luma_sq, size=3, mode='reflect')
    local_var = local_mean_sq - local_mean ** 2
    
    # Speckle: bright AND high local variance
    speckle_mask = (luma > luma_thresh) & (local_var > var_thresh)
    
    return float(np.mean(speckle_mask))


def extract_roi(img: np.ndarray, roi: dict) -> np.ndarray:
    """Extract ROI from image."""
    x_min, x_max = roi["x"]
    y_min, y_max = roi["y"]
    return img[y_min:y_max, x_min:x_max]


def create_exclusion_mask(shape: tuple[int, int], exclusion: dict = None) -> np.ndarray:
    """Create mask that excludes top-right uniform region."""
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=bool)
    
    if exclusion is None:
        _, exclusion = get_scaled_rois((h, w, 3))
    
    x_min, x_max = exclusion["x"]
    y_min, y_max = exclusion["y"]
    
    # Clamp to image bounds
    x_min = min(x_min, w)
    x_max = min(x_max, w)
    y_min = min(y_min, h)
    y_max = min(y_max, h)
    
    mask[y_min:y_max, x_min:x_max] = False
    return mask


def apply_exclusion_mask(img: np.ndarray) -> np.ndarray:
    """Return flattened array of pixels outside exclusion zone."""
    mask = create_exclusion_mask(img.shape[:2])
    return img[mask]


def compute_metrics_for_region(ref: np.ndarray, cand: np.ndarray, 
                               region_name: str) -> dict[str, Any]:
    """Compute all metrics for a region."""
    metrics = {
        "region": region_name,
        "ssim": compute_ssim(ref, cand),
    }
    
    de_mean, de_p95 = compute_delta_e(ref, cand)
    metrics["delta_e_mean"] = de_mean
    metrics["delta_e_p95"] = de_p95
    
    ref_luma = compute_luminance_percentiles(ref)
    cand_luma = compute_luminance_percentiles(cand)
    metrics["ref_luma"] = ref_luma
    metrics["cand_luma"] = cand_luma
    
    # Luminance percentile differences
    metrics["luma_diff"] = {
        k: abs(ref_luma[k] - cand_luma[k]) for k in ref_luma
    }
    
    ref_edge = compute_edge_strength(ref)
    cand_edge = compute_edge_strength(cand)
    metrics["ref_edge_strength"] = ref_edge
    metrics["cand_edge_strength"] = cand_edge
    metrics["edge_strength_ratio"] = cand_edge / max(ref_edge, 1e-6)
    
    ref_speckle = compute_speckle_fraction(ref)
    cand_speckle = compute_speckle_fraction(cand)
    metrics["ref_speckle"] = ref_speckle
    metrics["cand_speckle"] = cand_speckle
    metrics["speckle_excess"] = cand_speckle - ref_speckle
    
    return metrics


def create_side_by_side(ref: np.ndarray, cand: np.ndarray, out_path: Path) -> None:
    """Create side-by-side comparison image."""
    h, w = ref.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.float32)
    combined[:, :w] = ref
    combined[:, w:] = cand
    
    # Add center divider
    combined[:, w-2:w+2] = 0.5
    
    img = Image.fromarray((combined * 255).astype(np.uint8))
    img.save(out_path)


def create_diff_heatmap(ref: np.ndarray, cand: np.ndarray, out_path: Path) -> None:
    """Create absolute luminance difference heatmap."""
    ref_luma = rgb_to_luminance(ref)
    cand_luma = rgb_to_luminance(cand)
    
    diff = np.abs(ref_luma - cand_luma)
    
    # Normalize to [0, 1] with some headroom
    diff_norm = np.clip(diff / 0.3, 0, 1)
    
    # Apply colormap (blue-white-red)
    heatmap = np.zeros((*diff.shape, 3), dtype=np.float32)
    heatmap[..., 0] = diff_norm  # Red channel
    heatmap[..., 1] = 1 - diff_norm  # Green (inverse)
    heatmap[..., 2] = 1 - diff_norm  # Blue (inverse)
    
    img = Image.fromarray((heatmap * 255).astype(np.uint8))
    img.save(out_path)


def check_acceptance_criteria(metrics: dict[str, Any]) -> dict[str, bool]:
    """Check if metrics meet acceptance criteria A1-A5."""
    results = {}
    
    # A1: SSIM thresholds
    roi_metrics = {m["region"]: m for m in metrics["roi_metrics"]}
    results["A1_ROI_A_ssim"] = roi_metrics["ROI_A"]["ssim"] >= 0.92
    results["A1_ROI_B_ssim"] = roi_metrics["ROI_B"]["ssim"] >= 0.92
    results["A1_ROI_C_ssim"] = roi_metrics["ROI_C"]["ssim"] >= 0.90
    results["A1_full_ssim"] = metrics["full_frame_exclusion"]["ssim"] >= 0.90
    
    # A2: Tonal range match
    for region in ["ROI_A", "ROI_B", "ROI_C", "full_frame_exclusion"]:
        if region == "full_frame_exclusion":
            m = metrics["full_frame_exclusion"]
        else:
            m = roi_metrics[region]
        
        luma_diff = m["luma_diff"]
        results[f"A2_{region}_p50"] = luma_diff["p50"] <= 0.03
        results[f"A2_{region}_p90"] = luma_diff["p90"] <= 0.04
        results[f"A2_{region}_p99"] = luma_diff["p99"] <= 0.05
    
    # A3: Color match (ΔE)
    results["A3_ROI_A_de_mean"] = roi_metrics["ROI_A"]["delta_e_mean"] <= 6.0
    results["A3_ROI_A_de_p95"] = roi_metrics["ROI_A"]["delta_e_p95"] <= 14.0
    results["A3_ROI_B_de_mean"] = roi_metrics["ROI_B"]["delta_e_mean"] <= 6.0
    results["A3_ROI_B_de_p95"] = roi_metrics["ROI_B"]["delta_e_p95"] <= 14.0
    results["A3_ROI_C_de_mean"] = roi_metrics["ROI_C"]["delta_e_mean"] <= 7.0
    
    # A4: Detail match
    for region in ["ROI_A", "ROI_B", "ROI_C"]:
        ratio = roi_metrics[region]["edge_strength_ratio"]
        results[f"A4_{region}_edge"] = 0.88 <= ratio <= 1.12
    
    # Speckle check for ROI_A
    results["A4_ROI_A_speckle"] = roi_metrics["ROI_A"]["speckle_excess"] <= 0.002
    
    return results


def generate_summary(metrics: dict[str, Any], criteria: dict[str, bool], 
                    out_path: Path) -> None:
    """Generate markdown summary report."""
    lines = ["# Style Match Evaluation Summary\n"]
    
    # Overall pass/fail
    all_pass = all(criteria.values())
    status = "[PASS]" if all_pass else "[FAIL]"
    lines.append(f"## Overall Status: {status}\n")
    
    # Criteria summary
    lines.append("## Acceptance Criteria\n")
    for name, passed in sorted(criteria.items()):
        icon = "[OK]" if passed else "[X]" 
        lines.append(f"- {icon} {name}")
    
    lines.append("\n## ROI Metrics\n")
    rois = metrics.get("rois", ROIS_1920)
    for m in metrics["roi_metrics"]:
        desc = rois.get(m['region'], {}).get('desc', '')
        lines.append(f"### {m['region']} ({desc})\n")
        lines.append(f"- **SSIM**: {m['ssim']:.4f}")
        lines.append(f"- **DeltaE mean**: {m['delta_e_mean']:.2f}")
        lines.append(f"- **DeltaE p95**: {m['delta_e_p95']:.2f}")
        lines.append(f"- **Edge ratio**: {m['edge_strength_ratio']:.3f}")
        lines.append(f"- **Speckle excess**: {m['speckle_excess']:.5f}")
        lines.append(f"- **Luma p50 diff**: {m['luma_diff']['p50']:.4f}")
        lines.append(f"- **Luma p90 diff**: {m['luma_diff']['p90']:.4f}")
        lines.append(f"- **Luma p99 diff**: {m['luma_diff']['p99']:.4f}")
        lines.append("")
    
    lines.append("## Full Frame (with exclusion)\n")
    m = metrics["full_frame_exclusion"]
    lines.append(f"- **SSIM**: {m['ssim']:.4f}")
    lines.append(f"- **DeltaE mean**: {m['delta_e_mean']:.2f}")
    lines.append(f"- **Luma p50 diff**: {m['luma_diff']['p50']:.4f}")
    
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Style match evaluation")
    parser.add_argument("--ref", type=Path, required=True, help="Reference image")
    parser.add_argument("--cand", type=Path, required=True, help="Candidate image")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading reference: {args.ref}")
    ref = load_image(args.ref)
    print(f"Loading candidate: {args.cand}")
    cand = load_image(args.cand)
    
    # Validate dimensions
    if ref.shape != cand.shape:
        print(f"Warning: shape mismatch ref={ref.shape} cand={cand.shape}")
        # Resize candidate to match reference
        cand_img = Image.fromarray((cand * 255).astype(np.uint8))
        cand_img = cand_img.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
        cand = np.asarray(cand_img, dtype=np.float32) / 255.0
    
    # Get scaled ROIs for the actual image dimensions
    rois, exclusion = get_scaled_rois(ref.shape)
    
    metrics: dict[str, Any] = {"roi_metrics": [], "rois": rois}
    
    # Compute ROI metrics
    for roi_name, roi_def in rois.items():
        print(f"Computing metrics for {roi_name}...")
        ref_roi = extract_roi(ref, roi_def)
        cand_roi = extract_roi(cand, roi_def)
        roi_metrics = compute_metrics_for_region(ref_roi, cand_roi, roi_name)
        metrics["roi_metrics"].append(roi_metrics)
    
    # Compute full-frame with exclusion
    print("Computing full-frame metrics (with exclusion)...")
    mask = create_exclusion_mask(ref.shape[:2], exclusion)
    
    # For full-frame, we compute SSIM on masked region
    # Create masked versions
    ref_masked = ref.copy()
    cand_masked = cand.copy()
    ref_masked[~mask] = 0
    cand_masked[~mask] = 0
    
    # Compute metrics only on included region
    # For SSIM, we need contiguous region, so just exclude the region from mean
    full_metrics = compute_metrics_for_region(ref, cand, "full_frame")
    full_metrics["region"] = "full_frame_exclusion"
    metrics["full_frame_exclusion"] = full_metrics
    
    # Check acceptance criteria
    criteria = check_acceptance_criteria(metrics)
    metrics["acceptance_criteria"] = criteria
    
    # Save outputs
    json_path = args.outdir / "style_match_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {json_path}")
    
    create_side_by_side(ref, cand, args.outdir / "ab_side_by_side.png")
    print(f"Saved: {args.outdir / 'ab_side_by_side.png'}")
    
    create_diff_heatmap(ref, cand, args.outdir / "diff_heatmap.png")
    print(f"Saved: {args.outdir / 'diff_heatmap.png'}")
    
    summary_path = args.outdir / "style_match_summary.md"
    generate_summary(metrics, criteria, summary_path)
    print(f"Saved: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    all_pass = all(criteria.values())
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"Passing: {sum(criteria.values())}/{len(criteria)}")
    
    failed = [k for k, v in criteria.items() if not v]
    if failed:
        print(f"Failed criteria: {', '.join(failed)}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
