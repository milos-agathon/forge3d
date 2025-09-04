#!/usr/bin/env python3
"""
Sampler configuration demo for forge3d

Demonstrates sampler mode combinations and how to create sampler
descriptors. Rendering steps are skipped if the compiled extension
is not available.
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import os
import numpy as np

try:
    import forge3d as f3d
    HAVE_F3D = True
except Exception as e:
    print(f"forge3d import warning: {e}")
    HAVE_F3D = False


def demo_sampler_modes_listing():
    print("=== Available Sampler Modes ===")
    try:
        modes = f3d.list_sampler_modes()
        print(f"Found {len(modes)} sampler combinations\n")

        from collections import defaultdict
        by_address = defaultdict(list)
        for m in modes:
            by_address[m["address_mode"]].append(m)

        for addr in sorted(by_address.keys()):
            print(f"Address Mode: {addr}")
            print("-" * 40)
            for m in by_address[addr]:
                print(
                    f"  {m['name']:22} | filter: {m['mag_filter']}/{m['min_filter']:7} | mip: {m['mip_filter']}"
                )
            print()
    except Exception as e:
        print(f"ERROR: {e}")


def demo_basic_sampler_creation():
    print("=== Basic Sampler Creation ===")
    # Combined tokens (address_filter_mip)
    test = [
        "clamp_nearest_nearest",
        "clamp_linear_linear",
        "repeat_linear_linear",
        "mirror_nearest_linear",
    ]
    for token in test:
        try:
            addr, flt, mip = token.split("_", 2)
            desc = f3d.make_sampler(addr, flt, mip)
            print(f"  OK   {token:22} -> {desc}")
        except Exception as e:
            print(f"  ERR  {token:22} -> {e}")


def demo_sampler_visual_comparison():
    print("\n=== Sampler Visual Comparison (terrain) ===")
    if not HAVE_F3D or not hasattr(f3d, "Renderer"):
        print("Skipping: compiled extension not available")
        return

    size = 32
    heights = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    for i in range(size):
        for j in range(size):
            d = max(abs(i - c), abs(j - c))
            heights[i, j] = max(0, (c - d)) / c * 0.5

    outdir = "sampler_demo_output"
    os.makedirs(outdir, exist_ok=True)

    tokens = ["clamp_nearest_nearest", "clamp_linear_linear", "repeat_linear_linear"]
    for token in tokens:
        try:
            print(f"  Render with {token} ...")
            r = f3d.Renderer(256, 256)
            r.upload_height_r32f(heights)
            # Note: actual sampler wiring is internal; demo focuses on API consistency.
            rgba = r.render_terrain_rgba()
            f3d.numpy_to_png(os.path.join(outdir, f"terrain_{token}.png"), rgba)
        except Exception as e:
            print(f"  ERR  {token} -> {e}")


def demo_address_mode_behavior():
    print("\n=== Address Mode Behavior ===")
    tips = {
        "clamp": "Clamp to edge; avoids tiling seams (UI, terrain).",
        "repeat": "Tile texture; seamless patterns.",
        "mirror": "Mirror repeat; symmetric tiling without hard seams.",
    }
    for k, v in tips.items():
        print(f"  {k:6} -> {v}")


def demo_filter_mode_comparison():
    print("\n=== Filter Mode Comparison ===")
    print("Filter Mode Characteristics:")
    info = {
        "nearest": {"desc": "Point sampling", "best": "UI/pixel art"},
        "linear": {"desc": "Bilinear filtering", "best": "Photos/terrain"},
    }
    for k, d in info.items():
        print(f"  {k:7}: {d['desc']} (best: {d['best']})")

    try:
        modes = f3d.list_sampler_modes()
        nearest = [m for m in modes if m["mag_filter"] == "nearest"]
        linear = [m for m in modes if m["mag_filter"] == "linear"]
        print(f"Available: nearest={len(nearest)}  linear={len(linear)}")
    except Exception as e:
        print(f"ERROR: {e}")


def demo_mipmap_filter_effects():
    print("\n=== Mipmap Filter Effects ===")
    print("  nearest: sharp transitions between mips")
    print("  linear : smooth (trilinear) transitions")
    try:
        modes = f3d.list_sampler_modes()
        mn = [m for m in modes if m["mip_filter"] == "nearest"]
        ml = [m for m in modes if m["mip_filter"] == "linear"]
        print(f"  combinations -> nearest={len(mn)}, linear={len(ml)}")
    except Exception as e:
        print(f"ERROR: {e}")


def demo_sampler_recommendations():
    print("\n=== Sampler Recommendations ===")
    recs = {
        "Terrain/Heightmaps": ("clamp_linear_linear", "Smooth terrain; no edge seams"),
        "Repeating Textures": ("repeat_linear_linear", "Seamless tiling"),
        "UI/HUD Elements": ("clamp_nearest_nearest", "Crisp pixels"),
        "Pixel Art": ("clamp_nearest_nearest", "No blurring"),
        "Symmetric Patterns": ("mirror_linear_linear", "Mirror tiling"),
    }
    for k, (mode, why) in recs.items():
        print(f"  {k:20} -> {mode:22} : {why}")


def main():
    print("forge3d Sampler Configuration Demo")
    print("==================================")
    try:
        demo_sampler_modes_listing()
        demo_basic_sampler_creation()
        demo_sampler_visual_comparison()
        demo_address_mode_behavior()
        demo_filter_mode_comparison()
        demo_mipmap_filter_effects()
        demo_sampler_recommendations()
        print("\n=== Demo Complete ===")
        print("Check sampler_demo_output/ for images (if rendered).")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
