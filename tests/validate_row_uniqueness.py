#!/usr/bin/env python3
"""
Row uniqueness validator for stripe detection.
Tests acceptance criteria from task.xml.
"""
import hashlib
import sys
from pathlib import Path

try:
    import imageio.v3 as iio
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install imageio numpy")
    sys.exit(1)


def validate_image(path: Path) -> dict:
    """
    Validate an image for horizontal stripe artifacts.

    Returns dict with:
    - unique_rows: int - number of unique row hashes
    - total_rows: int - total rows
    - uniqueness_pct: float - percentage of unique rows
    - consecutive_dupes: int - number of consecutive duplicate rows
    - duplicate_pct: float - percentage of duplicate rows
    - passed: bool - True if no stripes detected
    """
    img = iio.imread(str(path))
    h, w = img.shape[0], img.shape[1]

    # Compute SHA256 hash for each row
    rows = [hashlib.sha256(img[y, :, :].tobytes()).hexdigest() for y in range(h)]

    # Count unique rows
    unique_rows = len(set(rows))
    uniqueness_pct = (unique_rows / h) * 100.0

    # Count consecutive duplicate rows
    consecutive_dupes = sum(1 for i in range(1, h) if rows[i] == rows[i-1])
    duplicate_pct = (consecutive_dupes / (h-1)) * 100.0 if h > 1 else 0.0

    # Pass criteria: >=95% unique, <1% duplicates
    passed = uniqueness_pct >= 95.0 and duplicate_pct < 1.0

    return {
        'unique_rows': unique_rows,
        'total_rows': h,
        'uniqueness_pct': uniqueness_pct,
        'consecutive_dupes': consecutive_dupes,
        'duplicate_pct': duplicate_pct,
        'passed': passed,
        'width': w,
        'height': h,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_row_uniqueness.py <image.png>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    result = validate_image(path)

    print(f"\nValidating: {path.name}")
    print(f"  Size: {result['width']}x{result['height']}")
    print(f"  Unique rows: {result['unique_rows']}/{result['total_rows']} ({result['uniqueness_pct']:.1f}%)")
    print(f"  Consecutive duplicates: {result['consecutive_dupes']} ({result['duplicate_pct']:.1f}%)")
    print()

    if result['passed']:
        print("[PASS] No stripe artifacts detected")
        return 0
    else:
        print("[FAIL] Stripe artifacts detected")
        print(f"  - Uniqueness: {result['uniqueness_pct']:.1f}% (need >=95%)")
        print(f"  - Duplicates: {result['duplicate_pct']:.1f}% (need <1%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
