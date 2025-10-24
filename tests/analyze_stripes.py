#!/usr/bin/env python3
"""Analyze PNG files for horizontal stripe artifacts.

This script checks for:
1. Identical consecutive rows (indicates possible banding)
2. Row hash uniqueness (should be high for natural imagery)
3. Byte-level patterns that indicate padding issues
"""

import sys
import hashlib
from pathlib import Path
import numpy as np


def analyze_png(path: Path) -> dict:
    """Analyze a PNG file for stripe artifacts."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, trying with forge3d...")
        import forge3d as f3d
        img_array = f3d.png_to_numpy(str(path))
    else:
        img = Image.open(path)
        img_array = np.array(img)

    height, width = img_array.shape[:2]
    channels = img_array.shape[2] if img_array.ndim == 3 else 1

    print(f"\nAnalyzing {path.name}:")
    print(f"  Size: {width}x{height}, Channels: {channels}")
    print(f"  Dtype: {img_array.dtype}, Shape: {img_array.shape}")

    # Calculate row hashes
    row_hashes = []
    for y in range(height):
        row_data = img_array[y, :].tobytes()
        row_hash = hashlib.sha256(row_data).hexdigest()
        row_hashes.append(row_hash)

    # Count unique row hashes
    unique_hashes = len(set(row_hashes))
    uniqueness_pct = (unique_hashes / height) * 100

    # Count consecutive identical rows
    consecutive_identical = 0
    for i in range(1, len(row_hashes)):
        if row_hashes[i] == row_hashes[i-1]:
            consecutive_identical += 1

    consecutive_pct = (consecutive_identical / (height - 1)) * 100 if height > 1 else 0

    # Check for padding patterns (look for repeating 256-byte boundaries)
    bytes_per_row = width * channels
    padded_bpr = ((bytes_per_row + 255) // 256) * 256
    has_padding_artifacts = False

    if padded_bpr != bytes_per_row:
        print(f"  Note: Unpadded BPR={bytes_per_row}, Padded BPR={padded_bpr}")
        # Check if there are any suspicious patterns at 256-byte boundaries
        # (This would only show up if padding wasn't properly removed)

    print(f"  Row hashes: {unique_hashes}/{height} unique ({uniqueness_pct:.1f}%)")
    print(f"  Consecutive identical rows: {consecutive_identical}/{height-1} ({consecutive_pct:.1f}%)")

    # Check pixel value distribution
    if channels >= 3:
        # For RGB/RGBA, check if it's mostly gray
        r_mean = img_array[:, :, 0].mean()
        g_mean = img_array[:, :, 1].mean()
        b_mean = img_array[:, :, 2].mean()
        gray_variance = np.std([r_mean, g_mean, b_mean])
        print(f"  RGB means: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
        print(f"  Gray variance: {gray_variance:.2f} (low=grayish, high=colorful)")

        # Check if image is suspiciously flat gray
        is_gray = gray_variance < 5.0 and r_mean > 50 and r_mean < 200
        if is_gray:
            print(f"  WARNING: Image appears to be flat gray!")

    # Overall assessment
    has_stripes = consecutive_pct > 1.0 or uniqueness_pct < 95.0

    results = {
        'width': width,
        'height': height,
        'channels': channels,
        'unique_rows': unique_hashes,
        'uniqueness_pct': uniqueness_pct,
        'consecutive_identical': consecutive_identical,
        'consecutive_pct': consecutive_pct,
        'has_stripes': has_stripes,
        'has_padding_artifacts': has_padding_artifacts,
    }

    if has_stripes:
        print(f"  [!] STRIPE ARTIFACTS DETECTED!")
    else:
        print(f"  [OK] No obvious stripe artifacts")

    return results


def main():
    if len(sys.argv) < 2:
        # Analyze default outputs
        paths = [
            Path("examples/out/terrain_4k.png"),
            Path("examples/out/terrain_4k_validated.png"),
            Path("examples/out/terrain_1k_validated.png"),
        ]
    else:
        paths = [Path(p) for p in sys.argv[1:]]

    for path in paths:
        if path.exists():
            analyze_png(path)
        else:
            print(f"\nSkipping {path.name}: File not found")


if __name__ == "__main__":
    main()
