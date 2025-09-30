#!/usr/bin/env python3
"""
Small end-to-end example demonstrating:
- Weighted OIT composition for points + lines
- Picking to R32Uint ID buffer and reporting center pixel ID

Run:
    python examples/vector_oit_and_pick_demo.py
"""
import sys
import os
import numpy as np
import forge3d as f3d


def main(width: int = 640, height: int = 360) -> int:
    # Optional: configure point impostors
    f3d.set_point_shape_mode(5)        # 0=circle, 4=texture, 5=sphere impostor
    f3d.set_point_lod_threshold(24.0)  # fall back to circle below 24 px

    if not f3d.is_weighted_oit_available():
        print("Weighted OIT is not available on this build/platform; exiting.")
        return 0

    rgba, pick_id = f3d.vector_oit_and_pick_demo(int(width), int(height))
    out = os.path.abspath("vector_oit_demo.png")
    f3d.numpy_to_png(out, rgba)
    print(f"Saved: {out}")
    print(f"Center pick id: {pick_id}")
    return 0


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    raise SystemExit(main(w, h))
