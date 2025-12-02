#!/usr/bin/env python3
"""
Generate IBL difference visualization and statistics from existing proof pack renders.
"""

from pathlib import Path
import sys

# Write log to file since stdout may not be captured
LOG_FILE = Path(__file__).parent.parent / "ibl_diff_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def main():
    # Clear log file
    LOG_FILE.write_text("")
    log("Starting IBL diff generation...")
    # Find the most recent proof pack
    proof_dir = Path("d:/forge3d/reports/proof_gore/20251130_135146")
    ibl_dir = proof_dir / "ibl"
    
    a_path = ibl_dir / "ibl_only_hdr_a.png"
    b_path = ibl_dir / "ibl_only_hdr_b.png"
    
    if not a_path.exists():
        log(f"ERROR: {a_path} not found")
        return 1
    if not b_path.exists():
        log(f"ERROR: {b_path} not found")
        return 1
    
    log(f"Loading {a_path}")
    log(f"Loading {b_path}")
    
    from PIL import Image
    import numpy as np
    
    # Load images
    img_a = np.array(Image.open(a_path).convert("RGB")).astype(np.float32) / 255.0
    img_b = np.array(Image.open(b_path).convert("RGB")).astype(np.float32) / 255.0
    
    log(f"Image A shape: {img_a.shape}")
    log(f"Image B shape: {img_b.shape}")
    
    # Compute absolute difference
    diff = np.abs(img_a - img_b)
    diff_gray = np.mean(diff, axis=2)  # Average across RGB for scalar diff
    
    # Statistics
    mean_diff = float(np.mean(diff_gray))
    median_diff = float(np.median(diff_gray))
    p95_diff = float(np.percentile(diff_gray, 95))
    max_diff = float(np.max(diff_gray))
    min_diff = float(np.min(diff_gray))
    
    log("\n=== IBL Difference Statistics ===")
    log(f"Mean:   {mean_diff:.6f}")
    log(f"Median: {median_diff:.6f}")
    log(f"P95:    {p95_diff:.6f}")
    log(f"Max:    {max_diff:.6f}")
    log(f"Min:    {min_diff:.6f}")
    
    # Save difference visualization (amplified for visibility)
    diff_vis = np.clip(diff * 4.0, 0.0, 1.0)  # 4x amplification
    diff_vis_uint8 = (diff_vis * 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_vis_uint8, mode="RGB")
    diff_out = ibl_dir / "ibl_only_diff_A_minus_B.png"
    diff_img.save(str(diff_out))
    log(f"\nSaved difference image: {diff_out}")
    
    # Generate statistics figure
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram of difference values
        axes[0].hist(diff_gray.flatten(), bins=100, color='steelblue', edgecolor='none', alpha=0.7)
        axes[0].set_xlabel("Absolute Difference")
        axes[0].set_ylabel("Pixel Count")
        axes[0].set_title("IBL Difference Histogram")
        axes[0].axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.4f}')
        axes[0].axvline(p95_diff, color='orange', linestyle='--', label=f'P95: {p95_diff:.4f}')
        axes[0].legend(fontsize=8)
        
        # Heatmap of difference
        im = axes[1].imshow(diff_gray, cmap='hot', aspect='auto')
        axes[1].set_title("Difference Heatmap")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Stats text box
        stats_text = (
            f"IBL Difference Statistics\n"
            f"------------------------\n"
            f"Mean:   {mean_diff:.6f}\n"
            f"Median: {median_diff:.6f}\n"
            f"P95:    {p95_diff:.6f}\n"
            f"Max:    {max_diff:.6f}\n"
            f"------------------------\n"
            f"HDR A: snow_field_4k\n"
            f"HDR B: air_museum_playground_4k"
        )
        axes[2].text(0.5, 0.5, stats_text, transform=axes[2].transAxes,
                    fontsize=12, verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].axis('off')
        axes[2].set_title("Summary")
        
        plt.tight_layout()
        fig_out = ibl_dir / "fig_ibl_diff_stats.png"
        plt.savefig(str(fig_out), dpi=150)
        plt.close(fig)
        log(f"Saved stats figure: {fig_out}")
        
    except ImportError as e:
        log(f"WARNING: Could not generate stats figure: {e}")
    except Exception as e:
        log(f"ERROR generating stats figure: {e}")
    
    # Acceptance check
    log("\n=== Acceptance Criteria ===")
    if mean_diff > 0.001:
        log(f"PASS: Mean difference ({mean_diff:.6f}) is clearly non-zero")
    else:
        log(f"WARN: Mean difference ({mean_diff:.6f}) is very small")
    
    if p95_diff > 0.01:
        log(f"PASS: P95 difference ({p95_diff:.6f}) shows significant variation")
    else:
        log(f"WARN: P95 difference ({p95_diff:.6f}) is small")
    
    return 0

if __name__ == "__main__":
    try:
        result = main()
        log(f"Script completed with exit code: {result}")
        sys.exit(result)
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
