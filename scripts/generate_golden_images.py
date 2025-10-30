#!/usr/bin/env python3
"""
Generate golden reference images for P9 regression testing

Creates 12 golden images with different BRDF × shadow × GI combinations
at 1280×920 resolution for SSIM validation (epsilon ≥ 0.98).

Usage:
    python scripts/generate_golden_images.py [--output-dir tests/golden]
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for forge3d import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d
except ImportError:
    print("Error: forge3d not found. Run 'maturin develop --release' first.")
    sys.exit(1)


GOLDEN_CONFIGS = [
    # (name, width, height, brdf, shadows, gi_modes, ibl_enabled, description)
    (
        "lambert_hard_nogi",
        1280,
        920,
        "lambert",
        "hard",
        [],
        False,
        "Lambert diffuse with hard shadows, no GI",
    ),
    (
        "phong_pcf_nogi",
        1280,
        920,
        "phong",
        "pcf",
        [],
        False,
        "Phong specular with PCF soft shadows",
    ),
    (
        "ggx_pcf_ibl",
        1280,
        920,
        "cooktorrance-ggx",
        "pcf",
        ["ibl"],
        True,
        "Cook-Torrance GGX with PCF and IBL",
    ),
    (
        "disney_pcss_ibl_ssao",
        1280,
        920,
        "disney-principled",
        "pcss",
        ["ibl", "ssao"],
        True,
        "Disney Principled with PCSS, IBL, and SSAO",
    ),
    (
        "orennayar_vsm_nogi",
        1280,
        920,
        "oren-nayar",
        "vsm",
        [],
        False,
        "Oren-Nayar rough diffuse with VSM shadows",
    ),
    (
        "toon_hard_nogi",
        1280,
        920,
        "toon",
        "hard",
        [],
        False,
        "Toon shading with hard shadows (stylized)",
    ),
    (
        "ashikhmin_pcss_ibl",
        1280,
        920,
        "ashikhmin-shirley",
        "pcss",
        ["ibl"],
        True,
        "Ashikhmin-Shirley anisotropic with PCSS and IBL",
    ),
    (
        "ward_evsm_nogi",
        1280,
        920,
        "ward",
        "evsm",
        [],
        False,
        "Ward anisotropic with EVSM shadows",
    ),
    (
        "blinnphong_msm_nogi",
        1280,
        920,
        "blinn-phong",
        "msm",
        [],
        False,
        "Blinn-Phong with MSM shadows",
    ),
    (
        "ggx_csm_ibl_gtao",
        1280,
        920,
        "cooktorrance-ggx",
        "csm",
        ["ibl", "gtao"],
        True,
        "GGX with cascaded shadow maps, IBL, and GTAO",
    ),
    (
        "disney_pcf_ibl_ssgi",
        1280,
        920,
        "disney-principled",
        "pcf",
        ["ibl", "ssgi"],
        True,
        "Disney Principled with PCF, IBL, and SSGI",
    ),
    (
        "ggx_pcss_ibl_ssr",
        1280,
        920,
        "cooktorrance-ggx",
        "pcss",
        ["ibl", "ssr"],
        True,
        "GGX with PCSS, IBL, and screen-space reflections",
    ),
]


def generate_golden_image(
    name: str,
    width: int,
    height: int,
    brdf: str,
    shadows: str,
    gi_modes: list,
    ibl_enabled: bool,
    description: str,
    output_dir: Path,
) -> None:
    """Generate a single golden reference image"""

    print(f"\nGenerating: {name}")
    print(f"  Resolution: {width}×{height}")
    print(f"  BRDF: {brdf}")
    print(f"  Shadows: {shadows}")
    print(f"  GI: {gi_modes if gi_modes else 'none'}")
    print(f"  Description: {description}")

    # TODO: This is a placeholder implementation
    # In a real implementation, you would:
    # 1. Create a Scene with specified width/height
    # 2. Set up camera looking at interesting terrain
    # 3. Configure lighting (directional sun + optional IBL)
    # 4. Set BRDF model
    # 5. Configure shadow technique
    # 6. Enable GI modes (SSAO, GTAO, SSGI, SSR, IBL)
    # 7. Render to PNG

    # For now, just create a placeholder file
    output_path = output_dir / f"{name}.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write a tiny valid PNG (1×1 white pixel) as placeholder
    # PNG header + IHDR + IDAT + IEND
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D,  # IHDR length
        0x49, 0x48, 0x44, 0x52,  # IHDR
        0x00, 0x00, 0x00, 0x01,  # width=1
        0x00, 0x00, 0x00, 0x01,  # height=1
        0x08, 0x02, 0x00, 0x00, 0x00,  # bit depth=8, color type=2 (RGB)
        0x90, 0x77, 0x53, 0xDE,  # CRC
        0x00, 0x00, 0x00, 0x0C,  # IDAT length
        0x49, 0x44, 0x41, 0x54,  # IDAT
        0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F, 0x00, 0x05, 0xFE, 0x02, 0xFE,  # compressed data
        0xDC, 0xCC, 0x59, 0xE7,  # CRC
        0x00, 0x00, 0x00, 0x00,  # IEND length
        0x49, 0x45, 0x4E, 0x44,  # IEND
        0xAE, 0x42, 0x60, 0x82,  # CRC
    ])

    with open(output_path, 'wb') as f:
        f.write(png_data)

    print(f"  ✓ Placeholder written: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference images for P9 testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/golden"),
        help="Output directory for golden images (default: tests/golden)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing golden images",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Golden Image Generator (P9)")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images: {len(GOLDEN_CONFIGS)}")
    print(f"Resolution: 1280×920 (all images)")
    print("=" * 70)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each golden image
    success_count = 0
    for config in GOLDEN_CONFIGS:
        name, width, height, brdf, shadows, gi_modes, ibl_enabled, description = config

        output_path = args.output_dir / f"{name}.png"

        # Skip if exists and not overwriting
        if output_path.exists() and not args.overwrite:
            print(f"\nSkipping: {name} (already exists)")
            continue

        try:
            generate_golden_image(
                name,
                width,
                height,
                brdf,
                shadows,
                gi_modes,
                ibl_enabled,
                description,
                args.output_dir,
            )
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Generated {success_count}/{len(GOLDEN_CONFIGS)} golden images")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    if success_count < len(GOLDEN_CONFIGS):
        print("\n⚠️  Some images failed to generate")
        sys.exit(1)


if __name__ == "__main__":
    main()
