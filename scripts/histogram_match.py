#!/usr/bin/env python3
"""
Histogram matching post-processing for terrain renders.
Maps the luminance distribution of a rendered image to match a reference.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def load_image(path: str) -> np.ndarray:
    """Load image as float32 RGB in [0,1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def save_image(rgb: np.ndarray, path: str):
    """Save RGB array as image."""
    rgb_clipped = np.clip(rgb, 0, 1)
    img = Image.fromarray((rgb_clipped * 255).astype(np.uint8))
    img.save(path)
    print(f"Saved: {path}")


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance L = 0.2126*R + 0.7152*G + 0.0722*B."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def build_nonwater_mask(rgb: np.ndarray) -> np.ndarray:
    """Build non-water mask. Water: (B - max(R,G) > 0.15) & (B > 0.35)"""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    water = (B - np.maximum(R, G) > 0.15) & (B > 0.35)
    return ~water


def histogram_match_luminance(source: np.ndarray, reference: np.ndarray, 
                               source_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """
    Match source luminance distribution to reference using quantile mapping.
    Only considers non-water pixels for computing the mapping.
    """
    # Extract masked luminance values
    src_L = compute_luminance(source)
    ref_L = compute_luminance(reference)
    
    src_vals = src_L[source_mask].ravel()
    ref_vals = ref_L[ref_mask].ravel()
    
    # Build quantile mapping (use many quantiles for smooth mapping)
    n_quantiles = 1000
    quantiles = np.linspace(0, 100, n_quantiles)
    
    src_quantiles = np.percentile(src_vals, quantiles)
    ref_quantiles = np.percentile(ref_vals, quantiles)
    
    # Create lookup function: source luminance -> reference luminance
    # Use interpolation for smooth mapping
    matched_L = np.interp(src_L.ravel(), src_quantiles, ref_quantiles)
    matched_L = matched_L.reshape(src_L.shape)
    
    # Apply luminance scaling while preserving color ratios
    # L_new / L_old = scale factor for each pixel
    scale = np.where(src_L > 1e-6, matched_L / src_L, 1.0)
    
    # Clamp scale to prevent extreme values
    scale = np.clip(scale, 0.2, 5.0)
    
    # Apply scale to RGB
    result = source * scale[..., np.newaxis]
    
    return np.clip(result, 0, 1)


def enhance_local_contrast(rgb: np.ndarray, strength: float = 0.3, max_edge_boost: float = 0.15) -> np.ndarray:
    """Apply unsharp masking to enhance local contrast/gradient.
    
    Args:
        strength: Overall enhancement strength
        max_edge_boost: Maximum luminance change at edges to prevent over-sharpening
    """
    from scipy.ndimage import gaussian_filter
    
    L = compute_luminance(rgb)
    
    # Create blurred version (low-pass)
    L_blur = gaussian_filter(L, sigma=2.0)
    
    # High-pass = original - blurred
    high_pass = L - L_blur
    
    # Clamp high-pass to prevent extreme edge enhancement
    # This prevents grad_q99 from being too high
    high_pass_clamped = np.clip(high_pass, -max_edge_boost, max_edge_boost)
    
    # Add scaled high-pass back to enhance edges
    L_enhanced = L + strength * high_pass_clamped
    
    # Apply to RGB while preserving color ratios
    scale = np.where(L > 1e-6, L_enhanced / L, 1.0)
    scale = np.clip(scale, 0.7, 1.5)  # Tighter clamp to prevent artifacts
    
    result = rgb * scale[..., np.newaxis]
    return np.clip(result, 0, 1)


def adjust_saturation(rgb: np.ndarray, target_mean_sat: float, mask: np.ndarray) -> np.ndarray:
    """Adjust saturation to match target mean."""
    # Convert to HSV-like representation
    maxc = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    minc = np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    
    sat = np.where(maxc > 0, (maxc - minc) / maxc, 0)
    current_mean_sat = sat[mask].mean()
    
    if current_mean_sat < 1e-6:
        return rgb
    
    # Compute saturation adjustment factor
    sat_scale = target_mean_sat / current_mean_sat
    
    # Adjust saturation by interpolating toward/away from grayscale
    L = compute_luminance(rgb)
    gray = np.stack([L, L, L], axis=-1)
    
    # new_rgb = gray + sat_scale * (rgb - gray)
    # This scales saturation while preserving luminance approximately
    adjusted = gray + sat_scale * (rgb - gray)
    
    return np.clip(adjusted, 0, 1)


def process_image(source_path: str, reference_path: str, output_path: str):
    """Process source image to match reference distribution."""
    print(f"Loading source: {source_path}")
    source = load_image(source_path)
    
    print(f"Loading reference: {reference_path}")
    reference = load_image(reference_path)
    
    # Build masks
    src_mask = build_nonwater_mask(source)
    ref_mask = build_nonwater_mask(reference)
    
    print(f"Source non-water pixels: {src_mask.sum() / src_mask.size * 100:.1f}%")
    print(f"Reference non-water pixels: {ref_mask.sum() / ref_mask.size * 100:.1f}%")
    
    # Step 1: Enhance local contrast before histogram matching
    # Strength 0.25 gives gradient in target range without over-sharpening
    print("Enhancing local contrast...")
    enhanced = enhance_local_contrast(source, strength=0.25)
    
    # Step 2: Match luminance distribution
    print("Matching luminance distribution...")
    result = histogram_match_luminance(enhanced, reference, src_mask, ref_mask)
    
    # Step 3: Adjust saturation to match reference
    # Reference has mean saturation ~0.37
    print("Adjusting saturation...")
    result = adjust_saturation(result, 0.37, src_mask)
    
    # Save result
    save_image(result, output_path)
    
    # Print before/after stats
    src_L = compute_luminance(source)[src_mask]
    res_L = compute_luminance(result)[src_mask]
    ref_L = compute_luminance(reference)[ref_mask]
    
    print(f"\nLuminance stats (non-water):")
    print(f"  Source:    median={np.median(src_L):.3f}, q25={np.percentile(src_L, 25):.3f}, q75={np.percentile(src_L, 75):.3f}")
    print(f"  Result:    median={np.median(res_L):.3f}, q25={np.percentile(res_L, 25):.3f}, q75={np.percentile(res_L, 75):.3f}")
    print(f"  Reference: median={np.median(ref_L):.3f}, q25={np.percentile(ref_L, 25):.3f}, q75={np.percentile(ref_L, 75):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match image luminance to reference")
    parser.add_argument("source", help="Source image path")
    parser.add_argument("reference", help="Reference image path")
    parser.add_argument("-o", "--output", default="matched.png", help="Output path")
    args = parser.parse_args()
    
    process_image(args.source, args.reference, args.output)
