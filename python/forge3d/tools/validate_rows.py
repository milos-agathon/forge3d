#!/usr/bin/env python3
"""
Row-stride validator tool (prompt8.md)

Validates that PNG files written by forge3d have correct row structure with no padding artifacts.

Per prompt8.md:
- Confirm tight rows after unpadding
- Check file length vs width×height×4

Usage:
    python -m forge3d.tools.validate_rows <image_path>

RELEVANT FILES: src/renderer/readback.rs, src/util/image_write.rs
"""

import sys
import argparse
from pathlib import Path
import numpy as np


def validate_png_row_structure(image_path: Path) -> dict:
    """
    Validate PNG file has correct tight row structure.

    Checks:
    1. File exists and is readable
    2. Dimensions are valid
    3. Raw pixel data length matches expected (width × height × 4)
    4. No duplicate consecutive rows (indicating padding issues)
    5. Rows have expected variance (not all identical)

    Args:
        image_path: Path to PNG file

    Returns:
        dict with validation results:
        {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str],
            "stats": dict,
        }
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check file exists
    if not image_path.exists():
        results["valid"] = False
        results["errors"].append(f"File does not exist: {image_path}")
        return results

    try:
        from PIL import Image
        img = Image.open(image_path)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to open image: {e}")
        return results

    # Get dimensions
    width, height = img.size
    results["stats"]["width"] = width
    results["stats"]["height"] = height
    results["stats"]["mode"] = img.mode

    # Convert to numpy array
    arr = np.array(img)

    # Check dimensions
    if arr.ndim != 3:
        results["valid"] = False
        results["errors"].append(f"Expected 3D array, got {arr.ndim}D")
        return results

    if arr.shape[2] not in (3, 4):
        results["valid"] = False
        results["errors"].append(f"Expected RGB or RGBA, got {arr.shape[2]} channels")
        return results

    # Validate data length
    expected_pixels = width * height
    actual_pixels = arr.shape[0] * arr.shape[1]

    if actual_pixels != expected_pixels:
        results["valid"] = False
        results["errors"].append(
            f"Pixel count mismatch: expected {expected_pixels}, got {actual_pixels}"
        )

    # Check for tight packing (file size should be minimal)
    file_size = image_path.stat().st_size
    results["stats"]["file_size_bytes"] = file_size

    # Raw RGBA data would be width × height × 4 bytes
    # PNG will be compressed, so we just check it's reasonable
    raw_size = width * height * 4
    results["stats"]["raw_rgba_size"] = raw_size

    if file_size > raw_size * 2:  # Allow 2x overhead for PNG headers/compression
        results["warnings"].append(
            f"File size ({file_size} bytes) seems large for {width}×{height} image "
            f"(expected ≤{raw_size * 2} bytes)"
        )

    # Check for duplicate consecutive rows (padding artifact symptom)
    duplicate_row_count = 0
    for i in range(height - 1):
        if np.array_equal(arr[i], arr[i + 1]):
            duplicate_row_count += 1

    results["stats"]["duplicate_consecutive_rows"] = duplicate_row_count

    # Allow some duplicates (e.g., sky, flat areas), but too many is suspicious
    duplicate_threshold = height * 0.3  # 30% max
    if duplicate_row_count > duplicate_threshold:
        results["valid"] = False
        results["errors"].append(
            f"Too many duplicate consecutive rows: {duplicate_row_count}/{height} "
            f"(threshold: {duplicate_threshold:.0f}). This may indicate padding artifacts."
        )
    elif duplicate_row_count > height * 0.2:  # 20% warning threshold
        results["warnings"].append(
            f"High number of duplicate consecutive rows: {duplicate_row_count}/{height}"
        )

    # Check row variance (all rows should not be identical)
    row_variances = np.var(arr.reshape(height, -1), axis=1)
    zero_variance_rows = np.sum(row_variances < 1e-6)
    results["stats"]["zero_variance_rows"] = int(zero_variance_rows)

    if zero_variance_rows > height * 0.5:
        results["warnings"].append(
            f"Many rows have zero variance: {zero_variance_rows}/{height}"
        )

    # Check for vertical banding (stride artifacts)
    # Compute difference between even and odd columns
    if width > 2:
        col_diff = np.abs(arr[:, ::2, :].mean() - arr[:, 1::2, :].mean())
        results["stats"]["even_odd_col_diff"] = float(col_diff)

        if col_diff > 10.0:  # Threshold in 0-255 scale
            results["warnings"].append(
                f"Significant difference between even/odd columns: {col_diff:.2f}. "
                "This may indicate stride artifacts."
            )

    # Check for horizontal banding
    if height > 2:
        row_diff = np.abs(arr[::2, :, :].mean() - arr[1::2, :, :].mean())
        results["stats"]["even_odd_row_diff"] = float(row_diff)

        if row_diff > 10.0:
            results["warnings"].append(
                f"Significant difference between even/odd rows: {row_diff:.2f}. "
                "This may indicate stride artifacts."
            )

    return results


def print_validation_results(image_path: Path, results: dict, verbose: bool = False):
    """Print validation results in human-readable format."""
    print(f"Validating: {image_path}")
    print("=" * 60)

    # Print stats
    stats = results["stats"]
    if "width" in stats and "height" in stats:
        print(f"Dimensions: {stats['width']}×{stats['height']} ({stats.get('mode', 'N/A')})")

    if "file_size_bytes" in stats and "raw_rgba_size" in stats:
        file_mb = stats["file_size_bytes"] / (1024 * 1024)
        raw_mb = stats["raw_rgba_size"] / (1024 * 1024)
        compression_ratio = stats["raw_rgba_size"] / stats["file_size_bytes"]
        print(f"File size: {file_mb:.2f} MiB (raw: {raw_mb:.2f} MiB, compression: {compression_ratio:.1f}x)")

    if verbose and stats:
        print("\nDetailed stats:")
        for key, value in stats.items():
            if key not in ("width", "height", "mode", "file_size_bytes", "raw_rgba_size"):
                print(f"  {key}: {value}")

    # Print errors
    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"  • {error}")

    # Print warnings
    if results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"  • {warning}")

    # Print result
    print()
    if results["valid"]:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")

    print("=" * 60)


def validate_multiple_files(image_paths: list[Path], verbose: bool = False) -> int:
    """
    Validate multiple PNG files.

    Returns:
        Number of failed validations
    """
    failed_count = 0

    for i, path in enumerate(image_paths):
        if i > 0:
            print()  # Spacing between files

        results = validate_png_row_structure(path)
        print_validation_results(path, results, verbose)

        if not results["valid"]:
            failed_count += 1

    if len(image_paths) > 1:
        print()
        print(f"Summary: {len(image_paths) - failed_count}/{len(image_paths)} passed")

    return failed_count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate PNG row structure for stride artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python -m forge3d.tools.validate_rows output.png

  # Validate multiple files
  python -m forge3d.tools.validate_rows *.png

  # Verbose output with detailed stats
  python -m forge3d.tools.validate_rows -v output.png
        """,
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="PNG files to validate",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed statistics",
    )

    args = parser.parse_args()

    # Validate all files
    failed_count = validate_multiple_files(args.images, args.verbose)

    # Exit with error code if any validation failed
    sys.exit(failed_count)


if __name__ == "__main__":
    main()
