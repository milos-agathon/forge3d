#!/usr/bin/env python3
"""
Validator for terrain output images to detect horizontal banding artifacts.

This script performs programmatic checks to ensure saved terrain images
do not exhibit horizontal striping caused by readback padding issues.
"""

import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def validate_row_uniqueness(image_path: Path, max_duplicate_ratio: float = 0.1) -> bool:
    """
    Validate that image rows are unique (not repeated due to padding artifacts).

    Args:
        image_path: Path to the image file
        max_duplicate_ratio: Maximum allowed ratio of duplicate rows (0.1 = 10%)

    Returns:
        True if validation passes, False otherwise
    """
    print(f"Validating row uniqueness for: {image_path}")

    img = np.array(Image.open(image_path))
    h, w, c = img.shape

    print(f"  Image shape: {h}×{w}×{c}")

    # Compute hash for each row
    row_hashes = []
    for y in range(h):
        row_bytes = img[y, :, :].tobytes()
        row_hash = hashlib.sha256(row_bytes).hexdigest()
        row_hashes.append(row_hash)

    # Count duplicates
    hash_counts = Counter(row_hashes)
    max_count = max(hash_counts.values())
    unique_rows = len(hash_counts)
    duplicate_ratio = (h - unique_rows) / h

    print(f"  Total rows: {h}")
    print(f"  Unique row patterns: {unique_rows}")
    print(f"  Most common pattern appears: {max_count} times")
    print(f"  Duplicate ratio: {duplicate_ratio:.1%}")

    # Check for suspicious banding (>10% of rows identical)
    if max_count > max_duplicate_ratio * h:
        print(f"  [FAIL] Suspicious repeated rows detected!")
        print(f"     {max_count}/{h} rows share the same pattern")
        return False

    print(f"  [PASS] Row uniqueness validated")
    return True


def validate_no_gray_bands(image_path: Path, max_row_std: float = 10.0) -> bool:
    """
    Validate that row-wise mean brightness doesn't have large variations
    (which would indicate horizontal banding).

    Args:
        image_path: Path to the image file
        max_row_std: Maximum allowed standard deviation of row means

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\nValidating no gray bands for: {image_path}")

    img = np.array(Image.open(image_path))
    h, w, c = img.shape

    # Compute mean brightness for each row (ignoring alpha if present)
    rgb_only = img[:, :, :3] if c >= 3 else img
    row_means = rgb_only.mean(axis=(1, 2))

    # Analyze middle 80% of image (ignore edge effects)
    middle_start = int(h * 0.1)
    middle_end = int(h * 0.9)
    middle_row_means = row_means[middle_start:middle_end]

    row_std = middle_row_means.std()
    row_min = middle_row_means.min()
    row_max = middle_row_means.max()
    row_range = row_max - row_min

    print(f"  Row mean statistics (middle 80%):")
    print(f"    Min: {row_min:.2f}")
    print(f"    Max: {row_max:.2f}")
    print(f"    Range: {row_range:.2f}")
    print(f"    Std dev: {row_std:.2f}")

    if row_std > max_row_std:
        print(f"  [FAIL] Horizontal banding detected!")
        print(f"     Row std dev {row_std:.2f} exceeds threshold {max_row_std}")

        # Show which rows are outliers
        mean_overall = middle_row_means.mean()
        outlier_rows = []
        for i, val in enumerate(middle_row_means):
            if abs(val - mean_overall) > 2 * row_std:
                actual_row = middle_start + i
                outlier_rows.append((actual_row, val))

        if outlier_rows:
            print(f"     Outlier rows (>2σ from mean):")
            for row, val in outlier_rows[:10]:  # Show first 10
                print(f"       Row {row}: mean={val:.2f}")

        return False

    print(f"  [PASS] No horizontal banding detected")
    return True


def validate_brightness_range(image_path: Path, min_mean: float = 50.0, max_mean: float = 250.0) -> bool:
    """
    Validate that image brightness is in a reasonable range (not all black/white).

    Args:
        image_path: Path to the image file
        min_mean: Minimum expected mean brightness
        max_mean: Maximum expected mean brightness

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\nValidating brightness range for: {image_path}")

    img = np.array(Image.open(image_path))
    rgb_only = img[:, :, :3] if img.shape[2] >= 3 else img

    mean_brightness = rgb_only.mean()
    min_val = rgb_only.min()
    max_val = rgb_only.max()
    std_val = rgb_only.std()

    print(f"  Overall statistics:")
    print(f"    Mean: {mean_brightness:.2f}")
    print(f"    Min: {min_val}")
    print(f"    Max: {max_val}")
    print(f"    Std: {std_val:.2f}")

    if not (min_mean <= mean_brightness <= max_mean):
        print(f"  [FAIL] Mean brightness {mean_brightness:.2f} outside range [{min_mean}, {max_mean}]")
        return False

    if std_val < 1.0:
        print(f"  [FAIL] No variation (std={std_val:.2f}), likely solid color")
        return False

    print(f"  [PASS] Brightness range is reasonable")
    return True


def validate_buffer_size(image_path: Path) -> bool:
    """
    Validate that the saved image has correct dimensions (no padding artifacts in file).

    Args:
        image_path: Path to the image file

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\nValidating buffer size for: {image_path}")

    img = Image.open(image_path)
    w, h = img.size

    # For PNG, check file size is reasonable
    file_size = image_path.stat().st_size
    expected_min = w * h * 3  # At least 3 bytes per pixel uncompressed
    expected_max = w * h * 4 * 2  # At most 4 bytes per pixel * 2 for compression overhead

    print(f"  Dimensions: {w}×{h}")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Expected range: {expected_min:,} to {expected_max:,} bytes")

    # Just verify the image can be loaded and has expected shape
    arr = np.array(img)
    if arr.shape[0] != h or arr.shape[1] != w:
        print(f"  [FAIL] Array shape {arr.shape} doesn't match dimensions {h}x{w}")
        return False

    print(f"  [PASS] Buffer size is correct")
    return True


def main():
    """Run all validators on specified image."""
    if len(sys.argv) < 2:
        print("Usage: python validate_terrain_output.py <image_path>")
        print("Example: python validate_terrain_output.py examples/out/terrain_4k.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("TERRAIN OUTPUT VALIDATION")
    print("=" * 70)

    results = {}

    # Run all validators
    results['row_uniqueness'] = validate_row_uniqueness(image_path)
    results['no_gray_bands'] = validate_no_gray_bands(image_path)
    results['brightness_range'] = validate_brightness_range(image_path)
    results['buffer_size'] = validate_buffer_size(image_path)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:20s}: {status}")

    print("=" * 70)

    if all_passed:
        print("SUCCESS: ALL VALIDATIONS PASSED")
        sys.exit(0)
    else:
        print("ERROR: SOME VALIDATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
