#!/usr/bin/env python3
"""
Mipmap generation demo for forge3d

Demonstrates CPU mipmap generation with different methods and settings,
and shows how to save and analyze mipmap pyramids.
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import os
import numpy as np
import forge3d as f3d
import forge3d.texture as tex


def create_test_texture(width=256, height=256):
    x = np.linspace(0, 10 * np.pi, width)
    y = np.linspace(0, 10 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    pattern = (
        np.sin(X) * np.cos(Y) * 0.5 +
        np.sin(X * 4) * np.cos(Y * 4) * 0.3 +
        np.sin(X * 16) * np.cos(Y * 16) * 0.2
    )
    pattern = (pattern + 1.0) * 0.5
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :, 0] = pattern
    rgba[:, :, 1] = pattern * 0.8
    rgba[:, :, 2] = pattern * 0.6
    rgba[:, :, 3] = 1.0
    return rgba


def demo_basic_mipmap_generation():
    print("=== Basic Mipmap Generation Demo ===")
    texture_data = create_test_texture(256, 256)
    print(f"Created test texture: {texture_data.shape}, range: [{texture_data.min():.3f}, {texture_data.max():.3f}]")

    # Valid methods in forge3d.texture: 'box' (CPU), 'gpu'/'weighted' fall back to 'box'
    methods = ["box", "gpu", "weighted"]
    outdir = "mipmap_demo_output"
    os.makedirs(outdir, exist_ok=True)

    for method in methods:
        print(f"\n--- Testing {method} method ---")
        try:
            pyramid = tex.generate_mipmaps(texture_data, method=method, max_levels=None, gamma_aware=True)
            print(f"Generated {len(pyramid)} mipmap levels using {method}")
            for i, level in enumerate(pyramid[:5]):
                print(f"  Level {i}: {level.shape[1]}x{level.shape[0]} pixels")
                # Save first few levels
                u8 = (np.clip(level, 0.0, 1.0) * 255).astype(np.uint8)
                f3d.numpy_to_png(f"{outdir}/mip{i}_{method}.png", u8)
        except Exception as e:
            print(f"  ERROR: {method} method failed: {e}")


def demo_gamma_aware_mipmaps():
    print("\n=== Gamma-Aware vs Linear Mipmapping Demo ===")
    size = 128
    checker = np.zeros((size, size, 4), dtype=np.float32)
    s = 8
    for i in range(0, size, s * 2):
        for j in range(0, size, s * 2):
            checker[i:i+s, j:j+s] = [1.0, 1.0, 1.0, 1.0]
            checker[i+s:i+2*s, j+s:j+2*s] = [1.0, 1.0, 1.0, 1.0]

    try:
        linear = tex.generate_mipmaps(checker, method="box", gamma_aware=False)
        gamma = tex.generate_mipmaps(checker, method="box", gamma_aware=True)
        outdir = "mipmap_demo_output"
        os.makedirs(outdir, exist_ok=True)
        lin_u8 = (np.clip(linear[3], 0.0, 1.0) * 255).astype(np.uint8)
        gam_u8 = (np.clip(gamma[3], 0.0, 1.0) * 255).astype(np.uint8)
        f3d.numpy_to_png(f"{outdir}/checker_linear.png", lin_u8)
        f3d.numpy_to_png(f"{outdir}/checker_gamma.png", gam_u8)
        print("Saved checker_linear.png and checker_gamma.png")
    except Exception as e:
        print(f"  ERROR: Gamma comparison failed: {e}")


def main():
    print("forge3d Mipmap Generation Demo")
    print("==============================")
    try:
        demo_basic_mipmap_generation()
        demo_gamma_aware_mipmaps()
        print("\n=== Demo Complete ===")
        print("Images saved to mipmap_demo_output/")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
