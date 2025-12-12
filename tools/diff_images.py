#!/usr/bin/env python3
"""Compute image diffs for P6.1 CLI override precedence proof."""
from PIL import Image
import numpy as np

p5 = np.array(Image.open('reports/p5/p5_terrain_csm.png'))
p5f = np.array(Image.open('reports/p5/p5_forced_colorspace.png'))
p6_preset = np.array(Image.open('reports/p6/p6_preset_terrain_csm.png'))

d1 = np.abs(p5.astype(float) - p5f.astype(float))
d2 = np.abs(p5f.astype(float) - p6_preset.astype(float))

print("=== P6.1 CLI Override Precedence Proof ===")
print(f"P5 vs P5-forced: mean={d1.mean():.4f}, max={d1.max():.0f}, nonzero_pixels={np.count_nonzero(d1.sum(axis=2))}")
print(f"P5-forced vs P6-preset: mean={d2.mean():.4f}, max={d2.max():.0f}, nonzero_pixels={np.count_nonzero(d2.sum(axis=2))}")

# Acceptance criteria
accept_d1 = d1.mean() > 0 or d1.max() > 0 or np.count_nonzero(d1.sum(axis=2)) > 0
accept_d2 = d2.mean() == 0 and d2.max() == 0 and np.count_nonzero(d2.sum(axis=2)) == 0

print()
print("=== Acceptance Criteria ===")
print(f"Diff(P5 vs P5-forced) != 0: {'PASS' if accept_d1 else 'FAIL'}")
print(f"Diff(P5-forced vs P6) == 0: {'PASS' if accept_d2 else 'FAIL'}")
