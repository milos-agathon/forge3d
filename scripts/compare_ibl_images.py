#!/usr/bin/env python3
"""Quick comparison of IBL outputs to verify HDRI swap changes lighting."""
from PIL import Image
import numpy as np
from pathlib import Path

a_path = Path("reports/proof_gore/ibl_only_hdri_A.png")
b_path = Path("reports/proof_gore/ibl_only_hdri_B.png")

if not a_path.exists():
    print(f"ERROR: {a_path} does not exist")
    exit(1)
if not b_path.exists():
    print(f"ERROR: {b_path} does not exist")
    exit(1)

a = np.array(Image.open(a_path))
b = np.array(Image.open(b_path))

print(f"Image A shape: {a.shape}")
print(f"Image B shape: {b.shape}")

diff = np.abs(a.astype(float) - b.astype(float))
print(f"Mean pixel difference: {diff.mean():.4f}")
print(f"Max pixel difference: {diff.max():.4f}")
print(f"P95 pixel difference: {np.percentile(diff, 95):.4f}")
print(f"Files byte-identical: {np.array_equal(a, b)}")

# Save diff image (amplified for visibility)
diff_vis = np.clip(diff * 3.0, 0.0, 255.0).astype(np.uint8)
diff_path = Path("reports/proof_gore/ibl_only_diff_A_minus_B.png")
Image.fromarray(diff_vis).save(diff_path)
print(f"Diff image saved to: {diff_path}")

# Acceptance criteria from todo.md:
# - A and B are not near-identical
# - Difference stats are clearly non-zero across terrain
if diff.mean() > 1.0:
    print("\n✓ PASS: IBL responds to HDRI changes (mean diff > 1.0)")
else:
    print("\n✗ FAIL: IBL appears unresponsive (mean diff <= 1.0)")
